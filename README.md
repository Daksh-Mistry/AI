# Adversarial Claim Scoring Engine
### Multi-agent argumentation analysis with mathematical scoring

> **Not a chatbot.** A structured argumentation analysis engine with mathematical scoring underneath.

---

## Table of Contents
1. [System Architecture](#architecture)
2. [How to Run Locally](#local-setup)
3. [Google Cloud Deployment](#gcp-deployment)
4. [Configuration Reference](#config)
5. [Testing](#testing)
6. [Pipeline Layer Reference](#pipeline)
7. [Future Improvements](#future)

---

## System Architecture <a name="architecture"></a>

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ADVERSARIAL CLAIM ENGINE                         │
│                                                                       │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌─────────────┐  │
│  │ Document │   │  Dual-Agent│   │ Embed +    │   │ Filter      │  │
│  │ Ingestion│──▶│  Claim Gen │──▶│ Cluster    │──▶│ Top X%      │  │
│  │ LangChain│   │ (Aff + Neg)│   │ DBSCAN     │   │ Imbalance   │  │
│  │ ChromaDB │   │ Gemini     │   │ text-emb   │   │ Check       │  │
│  └──────────┘   └────────────┘   └────────────┘   └──────┬──────┘  │
│                                                            │         │
│  ┌──────────┐   ┌────────────┐   ┌────────────┐          │         │
│  │  Judge   │   │ Aggregation│   │ Multi-Dim  │          │         │
│  │ Gemini   │◀──│ Dot Product│◀──│ Scoring    │◀─────────┘         │
│  │ 2.5 Pro  │   │ Gap + Conf │   │ Gemini     │                     │
│  └────┬─────┘   └────────────┘   │ Flash      │                     │
│       │                           └────────────┘                     │
│       ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              STREAMLIT FRONTEND (Split-Screen UI)             │   │
│  │  Upload │ Question │ Config ││ Claims ││ Verdict │ Judge      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

Google Cloud Deployment:
  Streamlit App ──▶ Cloud Run (Container)
  LLM Calls     ──▶ Vertex AI (Gemini Flash + Pro)
  Embeddings    ──▶ Vertex AI (text-embedding-004)
  Vector Store  ──▶ ChromaDB (local in container, or Cloud Storage mount)
```

### File Structure
```
adversarial-claim-engine/
├── app.py                  # Streamlit frontend
├── backend/
│   ├── config.py           # All tunable parameters
│   ├── ingestion.py        # Layer 1: Document → ChromaDB
│   ├── agents.py           # Layer 2: Dual-agent claim generation
│   ├── clustering.py       # Layer 3: Embedding + DBSCAN/KMeans merge
│   ├── filtering.py        # Layer 4: Top-X% + imbalance check
│   ├── scoring.py          # Layer 5: Multi-dimensional LLM scoring
│   ├── aggregation.py      # Layer 6: Weighted dot product + verdict
│   ├── judge.py            # Layer 7: Gemini Pro synthesis
│   └── graph.py            # Layer 8: LangGraph orchestration
├── tests/
│   └── test_pipeline.py    # Full test suite (no API calls needed)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## How to Run Locally <a name="local-setup"></a>

### Prerequisites
- Python 3.11+
- Google Cloud account with Vertex AI enabled
- `gcloud` CLI installed

### Step 1: Clone and install
```bash
git clone <repo>
cd adversarial-claim-engine

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Authenticate with Google Cloud
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable run.googleapis.com
```

### Step 3: Set environment variables
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"     # or nearest region
```

### Step 4: Run the app
```bash
# From project root
PYTHONPATH=backend streamlit run app.py
```
Open: http://localhost:8501

### Step 5: Run tests (no API required)
```bash
python -m pytest tests/ -v
# OR
PYTHONPATH=backend python tests/test_pipeline.py
```

---

## Google Cloud Deployment <a name="gcp-deployment"></a>

### Architecture on GCP
```
User Browser
     │
     ▼
Cloud Run (Streamlit container, port 8080)
     │
     ├──▶ Vertex AI — Gemini 2.5 Flash (claim generation + scoring)
     ├──▶ Vertex AI — Gemini 2.5 Pro   (judge synthesis)
     └──▶ Vertex AI — text-embedding-004 (document + claim embeddings)
          ChromaDB runs in-memory / ephemeral in container
```

### Step-by-Step Deployment

#### 1. Set project variables
```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export SERVICE_NAME="adversarial-claim-engine"
export IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"
```

#### 2. Enable APIs
```bash
gcloud services enable \
  aiplatform.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  containerregistry.googleapis.com \
  --project=$PROJECT_ID
```

#### 3. Build and push container
```bash
# Build locally and push
gcloud builds submit --tag $IMAGE --project=$PROJECT_ID

# OR build with Cloud Build (no local Docker needed)
gcloud builds submit \
  --config cloudbuild.yaml \
  --project=$PROJECT_ID
```

#### 4. Deploy to Cloud Run
```bash
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --max-instances 3 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION \
  --allow-unauthenticated \
  --project=$PROJECT_ID
```

#### 5. Grant Vertex AI access to Cloud Run service account
```bash
# Get the service account used by Cloud Run
SA="$(gcloud run services describe $SERVICE_NAME \
  --platform managed --region $REGION \
  --format 'value(spec.template.spec.serviceAccountName)' \
  --project=$PROJECT_ID)"

# Grant Vertex AI User role
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SA" \
  --role="roles/aiplatform.user"
```

#### 6. Get the service URL
```bash
gcloud run services describe $SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --format 'value(status.url)' \
  --project=$PROJECT_ID
```

### Optional: Persistent ChromaDB with Cloud Storage
For production persistence across container restarts:
```bash
# Mount GCS bucket as FUSE filesystem
gcloud storage buckets create gs://$PROJECT_ID-chromadb --location=$REGION

# Add flag to Cloud Run deploy:
--set-env-vars CHROMA_PERSIST_DIR=/mnt/chromadb
--add-volume name=chromadb,type=cloud-storage,bucket=$PROJECT_ID-chromadb
--add-volume-mount volume=chromadb,mount-path=/mnt/chromadb
```

---

## Configuration Reference <a name="config"></a>

All parameters live in `backend/config.py`. Override via `RuntimeConfig` or Streamlit sidebar.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `chunk_size` | 800 | 400–2000 | Larger = more context per chunk, fewer chunks |
| `chunk_overlap` | 100 | 0–300 | Higher = better paragraph boundary preservation |
| `retrieval_k` | 6 | 3–15 | More chunks retrieved per RAG query |
| `clustering_method` | dbscan | dbscan/kmeans | DBSCAN: auto clusters; KMeans: fixed count |
| `dbscan_eps` | 0.35 | 0.25–0.5 | Lower = less merging; Higher = more aggressive |
| `dbscan_min_samples` | 2 | 1–3 | 1 = every isolated point is its own cluster |
| `filter_percent` | 0.25 | 0.2–0.35 | Top 25% of merged claims kept |
| `max_claims` | 50 | 10–100 | Hard cap per side |
| `weights` | [5,3,1] | experiment | [s1_relevance, s2_confidence, s3_specificity] |
| `discard_threshold` | 0.2 | 0.1–0.3 | Discard if ANY dimension < threshold |
| `imbalance_ratio` | 5.0 | 3–10 | N/P or P/N > this → no_conclusion |
| `conclusion_thresh` | 0.1 | 0.05–0.2 | avg gap < this → no_conclusion |
| `confidence_denom` | 0.3 | 0.2–0.5 | Scales confidence %; lower = inflates faster |

### Adding a Scoring Dimension
In `config.py`, add to `SCORING_DIMENSIONS`:
```python
{
    "id": "s4",
    "name": "Temporal Relevance",
    "weight": 2,
    "description": "How recent is the evidence? Claims with dates within 5 years score higher."
}
```
No other code changes needed — scoring and aggregation are fully dimension-agnostic.

---

## Testing <a name="testing"></a>

```bash
# All tests (no Vertex AI API calls required)
PYTHONPATH=backend python -m pytest tests/ -v

# Individual test class
PYTHONPATH=backend python -m pytest tests/test_pipeline.py::TestAggregation -v
```

### Test cases covered:
- `TestConfig` — RuntimeConfig defaults and overrides
- `TestFiltering` — specificity scoring, top-%, imbalance, balancing
- `TestClustering` — DBSCAN/KMeans labels, merge reduces count (pure math)
- `TestAggregation` — composite scores, affirmative/negative wins, no-conclusion triggers
- `TestAgentParser` — numbered claim extraction, short claim rejection
- `TestExpectedOutputFormat` — output schema validation

### Manual calibration test cases (run with real API):
```bash
# Test 1: Biased corporate document → should lean against or no_conclusion
# Test 2: Neutral scientific document → balanced verdict or no_conclusion
# Test 3: Document with overwhelming one-sided evidence → high confidence verdict
# Test 4: Equal evidence both sides → no_conclusion
# Test 5: Vague document with no specifics → most claims discarded, no_conclusion
```

---

## Pipeline Layer Reference <a name="pipeline"></a>

| Layer | File | Input | Output |
|-------|------|-------|--------|
| 1. Ingestion | `ingestion.py` | Document file | ChromaDB vectorstore |
| 2. Claim Gen | `agents.py` | Vectorstore + question | Raw claim lists (both sides) |
| 3. Clustering | `clustering.py` | Raw claims | Merged unique claims |
| 4. Filtering | `filtering.py` | Merged claims | Top-X% balanced claims |
| 5. Scoring | `scoring.py` | Filtered claims | Scored claim dicts (s1, s2, s3) |
| 6. Aggregation | `aggregation.py` | Scored claims | Verdict + confidence + gap |
| 7. Judge | `judge.py` | Aggregation result | Narrative synthesis |
| 8. Orchestration | `graph.py` | Initial state | Final EngineState |

---

## Future Improvements <a name="future"></a>

1. **External grounding** — Vertex AI Search Grounding to anchor factual claims beyond the document (partially solves the "document confidence ≠ objective truth" limitation)
2. **Streaming UI** — Stream pipeline progress step-by-step via `st.status` rather than blocking on full pipeline
3. **Persistent sessions** — Cloud Firestore to store analysis results for sharing/revisiting
4. **Async scoring** — Parallelize LLM scoring calls with `asyncio` (currently sequential, bottleneck for large claim sets)
5. **Dimension presets** — Pre-configured weight matrices for Legal, Scientific, Political domains
6. **User score override** — Let users manually adjust individual claim scores and re-run aggregation (full human-in-the-loop)
7. **Multi-document support** — Ingest multiple documents into separate ChromaDB collections; compare cross-document claims
8. **Confidence calibration** — Replace linear confidence formula with calibrated sigmoid based on empirical test cases
9. **Claim provenance** — Track which document chunk each claim was extracted from; show source on hover
10. **Web grounding toggle** — Optional second pass where Negative agent challenges claims using live web search
