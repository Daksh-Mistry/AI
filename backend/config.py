"""
config.py — Central configuration for Adversarial Claim Scoring Engine
All parameters tunable without touching core code.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ─────────────────────────────────────────────
# DOCUMENT INGESTION
# ─────────────────────────────────────────────
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 100
RETRIEVAL_K: int = 6  # chunks retrieved per RAG query

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
FLASH_MODEL: str = "gemini-1.5-flash"
PRO_MODEL: str = "gemini-1.5-pro"
EMBEDDING_MODEL: str = "text-embedding-004"

# ─────────────────────────────────────────────
# CLUSTERING (DBSCAN)
# ─────────────────────────────────────────────
CLUSTERING_METHOD: str = "dbscan"  # "dbscan" | "kmeans"
DBSCAN_EPS: float = 0.35           # cosine distance neighbourhood radius (0.25–0.5)
DBSCAN_MIN_SAMPLES: int = 2        # min points per cluster
KMEANS_N_CLUSTERS: int = 10        # used only if CLUSTERING_METHOD = "kmeans"

# ─────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────
FILTER_PERCENT: float = 0.25       # top X% of merged claims kept (0.2–0.35)
MAX_CLAIMS: int = 50               # hard cap per side after balancing

# ─────────────────────────────────────────────
# SCORING DIMENSIONS
# Each entry: (name, weight, description)
# Add/remove rows to extend scoring dimensions.
# ─────────────────────────────────────────────
SCORING_DIMENSIONS: List[dict] = [
    {
        "id": "s1",
        "name": "Logical Relevance",
        "weight": 5,
        "description": "How directly does this claim answer the user's actual question?"
    },
    {
        "id": "s2",
        "name": "Document Confidence",
        "weight": 3,
        "description": "How strongly does the source document support this specific claim?"
    },
    {
        "id": "s3",
        "name": "Specificity",
        "weight": 1,
        "description": "Is the claim concrete? Contains numbers, dates, named entities?"
    },
]

# Derived: weights list in dimension order
WEIGHTS: List[int] = [d["weight"] for d in SCORING_DIMENSIONS]
WEIGHT_SUM: float = float(sum(WEIGHTS))  # normalisation denominator

# ─────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────
DISCARD_THRESHOLD: float = 0.2    # discard claim if ANY dimension < this
IMBALANCE_RATIO: float = 5.0      # sides differing by this factor → no-conclusion
CONCLUSION_THRESH: float = 0.1    # avg gap < this → no-conclusion
CONFIDENCE_DENOM: float = 0.3     # scales confidence percentage (tune experimentally)
MIN_CLAIMS_PER_SIDE: int = 1      # minimum surviving claims; below → no-conclusion

# ─────────────────────────────────────────────
# GOOGLE CLOUD / VERTEX AI
# ─────────────────────────────────────────────
GCP_PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")  # read from environment
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
CHROMA_PERSIST_DIR: str = "./chroma_db"  # only used if persist mode is enabled

# ─────────────────────────────────────────────
# RUNTIME OVERRIDES (used by Streamlit sidebar)
# ─────────────────────────────────────────────
@dataclass
class RuntimeConfig:
    """
    Instance-level config that can be overridden from UI without
    touching the module-level defaults above.
    """
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    retrieval_k: int = RETRIEVAL_K
    clustering_method: str = CLUSTERING_METHOD
    dbscan_eps: float = DBSCAN_EPS
    dbscan_min_samples: int = DBSCAN_MIN_SAMPLES
    kmeans_n_clusters: int = KMEANS_N_CLUSTERS
    filter_percent: float = FILTER_PERCENT
    max_claims: int = MAX_CLAIMS
    weights: List[int] = field(default_factory=lambda: list(WEIGHTS))
    discard_threshold: float = DISCARD_THRESHOLD
    imbalance_ratio: float = IMBALANCE_RATIO
    conclusion_thresh: float = CONCLUSION_THRESH
    confidence_denom: float = CONFIDENCE_DENOM

    @property
    def weight_sum(self) -> float:
        return float(sum(self.weights))
