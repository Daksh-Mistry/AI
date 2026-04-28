"""
app.py — Adversarial Claim Scoring Engine — Streamlit Frontend

Split-screen verdict dashboard with full transparency:
  - Upload document (PDF / TXT / DOCX)
  - Enter question
  - Sidebar: full config controls
  - Main: verdict banner + split-screen claims + judge analysis
"""

import os
import sys
import tempfile
import logging

import streamlit as st
import vertexai
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()  # Load .env file if present

from config import RuntimeConfig, SCORING_DIMENSIONS, GCP_PROJECT_ID, GCP_LOCATION
from graph import build_graph
from ingestion import ingest_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Initialise Vertex AI SDK with explicit project & location ─────────────────
if GCP_PROJECT_ID:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    logger.info("Vertex AI initialised: project=%s, location=%s", GCP_PROJECT_ID, GCP_LOCATION)
else:
    logger.warning("GOOGLE_CLOUD_PROJECT not set — Vertex AI will use default credentials")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarial Claim Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .verdict-banner {
    border-radius: 8px;
    padding: 20px 32px;
    margin: 16px 0;
    text-align: center;
    font-weight: 700;
    font-size: 1.4rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .verdict-aff  { background: #1a3d2b; color: #4ade80; border: 1px solid #166534; }
  .verdict-neg  { background: #3d1a1a; color: #f87171; border: 1px solid #991b1b; }
  .verdict-none { background: #2d2b1a; color: #fbbf24; border: 1px solid #92400e; }
  .claim-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    line-height: 1.5;
  }
  .score-pill {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
  }
  .score-high  { background: #166534; color: #4ade80; }
  .score-mid   { background: #92400e; color: #fbbf24; }
  .score-low   { background: #7f1d1d; color: #f87171; }
  .dim-scores  { font-size: 0.75rem; color: #64748b; font-family: 'IBM Plex Mono', monospace; }
  .section-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin: 20px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e293b;
  }
  .reasoning-block {
    background: #0f172a;
    border-left: 3px solid #334155;
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 0.86rem;
    color: #94a3b8;
    margin: 8px 0;
    font-style: italic;
  }
  .pipeline-log {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #475569;
    max-height: 200px;
    overflow-y: auto;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 0.70:
        return "score-high"
    elif score >= 0.40:
        return "score-mid"
    return "score-low"


def render_claim_card(claim_dict: dict, idx: int, dims: list) -> None:
    composite = claim_dict.get("composite", 0.0)
    claim_text = claim_dict.get("claim", "")
    reasoning  = claim_dict.get("reasoning", "")
    css_class  = score_color(composite)

    dim_parts = []
    for d in dims:
        val = claim_dict.get(d["id"], 0.0)
        dim_parts.append(f"{d['id']}={val:.2f}")
    dim_str = "  |  ".join(dim_parts)

    st.markdown(f"""
    <div class="claim-card">
      <span class="score-pill {css_class}">{composite:.4f}</span>
      <strong>#{idx}</strong> {claim_text}
      <div class="dim-scores" style="margin-top:6px">{dim_str}</div>
      {'<div class="reasoning-block">' + reasoning + '</div>' if reasoning else ''}
    </div>
    """, unsafe_allow_html=True)


def render_verdict_banner(verdict: str, confidence: int) -> None:
    if verdict == "affirmative":
        cls  = "verdict-aff"
        icon = "✅"
        label = f"VERDICT: AFFIRMATIVE — Confidence {confidence}%"
    elif verdict == "negative":
        cls  = "verdict-neg"
        icon = "❌"
        label = f"VERDICT: NEGATIVE — Confidence {confidence}%"
    else:
        cls  = "verdict-none"
        icon = "⚠️"
        label = "VERDICT: NO CONCLUSION"
    st.markdown(f'<div class="verdict-banner {cls}">{icon} {label}</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR: CONFIG CONTROLS
# ─────────────────────────────────────────────
def build_config_from_sidebar() -> RuntimeConfig:
    st.sidebar.header("⚙️ Pipeline Configuration")

    with st.sidebar.expander("📄 Document Ingestion", expanded=False):
        chunk_size    = st.slider("Chunk size (chars)", 400, 2000, 800, 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 300, 100, 25)
        retrieval_k   = st.slider("RAG retrieval K", 3, 15, 6)

    with st.sidebar.expander("🔵 Clustering", expanded=False):
        method = st.selectbox("Method", ["dbscan", "kmeans"], index=0)
        eps       = st.slider("DBSCAN eps",         0.10, 0.80, 0.35, 0.05)
        min_samp  = st.slider("DBSCAN min_samples", 1,    5,    2)
        k_clusters= st.slider("K-Means clusters",   3,    30,   10)

    with st.sidebar.expander("🔽 Filtering", expanded=False):
        filter_pct = st.slider("Top % of claims", 10, 60, 25, 5)
        max_claims = st.slider("Max claims per side", 5, 100, 50, 5)

    with st.sidebar.expander("⚖️ Scoring Weights", expanded=True):
        w_labels = [f"{d['name']} ({d['id']})" for d in SCORING_DIMENSIONS]
        w_defaults = [d["weight"] for d in SCORING_DIMENSIONS]
        weights = []
        for label, default in zip(w_labels, w_defaults):
            w = st.slider(label, 1, 10, default)
            weights.append(w)

    with st.sidebar.expander("🚦 Thresholds", expanded=False):
        discard_thresh    = st.slider("Discard threshold (min dim score)", 0.05, 0.50, 0.20, 0.05)
        imbalance_ratio   = st.slider("Imbalance ratio (× multiplier)",   2.0, 20.0, 5.0, 0.5)
        conclusion_thresh = st.slider("Conclusion gap threshold",          0.01, 0.30, 0.10, 0.01)
        conf_denom        = st.slider("Confidence denominator",            0.10, 0.80, 0.30, 0.05)

    return RuntimeConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=retrieval_k,
        clustering_method=method,
        dbscan_eps=eps,
        dbscan_min_samples=min_samp,
        kmeans_n_clusters=k_clusters,
        filter_percent=filter_pct / 100,
        max_claims=max_claims,
        weights=weights,
        discard_threshold=discard_thresh,
        imbalance_ratio=imbalance_ratio,
        conclusion_thresh=conclusion_thresh,
        confidence_denom=conf_denom,
    )


# ─────────────────────────────────────────────
# RESULT RENDERER
# ─────────────────────────────────────────────
def render_results(state: dict) -> None:
    agg    = state.get("agg_result", {})
    judge  = state.get("judge_output", {})
    verdict  = agg.get("verdict", "no_conclusion")
    conf     = agg.get("confidence", 0)
    avg_aff  = agg.get("avg_aff")
    avg_neg  = agg.get("avg_neg")
    gap      = agg.get("gap")
    aff_top  = agg.get("aff_top", [])
    neg_top  = agg.get("neg_top", [])
    aff_all  = agg.get("aff_all", [])
    neg_all  = agg.get("neg_all", [])

    # ── Verdict Banner ────────────────────────────────────────────────────────
    render_verdict_banner(verdict, conf)

    # ── Score summary ─────────────────────────────────────────────────────────
    if avg_aff is not None and avg_neg is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Affirmative avg score", f"{avg_aff:.4f}")
        c2.metric("Negative avg score",    f"{avg_neg:.4f}")
        c3.metric("Score gap",             f"{gap:.4f}")

    st.markdown("---")

    # ── Split screen: top claims ───────────────────────────────────────────────
    col_aff, col_neg = st.columns(2, gap="large")

    with col_aff:
        st.markdown('<div class="section-header">✅ Supporting Claims (Top 5)</div>',
                    unsafe_allow_html=True)
        if aff_top:
            for i, c in enumerate(aff_top, 1):
                render_claim_card(c, i, SCORING_DIMENSIONS)
        else:
            st.info("No affirmative claims survived scoring.")

    with col_neg:
        st.markdown('<div class="section-header">❌ Opposing Claims (Top 5)</div>',
                    unsafe_allow_html=True)
        if neg_top:
            for i, c in enumerate(neg_top, 1):
                render_claim_card(c, i, SCORING_DIMENSIONS)
        else:
            st.info("No negative claims survived scoring.")

    # ── Judge narrative ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">⚖️ Judge Analysis</div>',
                unsafe_allow_html=True)

    if judge:
        exp = judge.get("verdict_explanation", "")
        if exp:
            st.markdown(f'<div class="reasoning-block">{exp}</div>',
                        unsafe_allow_html=True)

        if verdict != "no_conclusion":
            j1, j2 = st.columns(2)
            with j1:
                decisive = judge.get("decisive_claims", [])
                if decisive:
                    st.markdown("**Decisive claims (winning side)**")
                    for item in decisive:
                        st.markdown(
                            f"- **[{item.get('score', '?')}]** {item.get('claim', '')} "
                            f"— _{item.get('why_decisive', '')}_"
                        )
            with j2:
                ack = judge.get("acknowledged_claims", [])
                if ack:
                    st.markdown("**Acknowledged (losing side)**")
                    for item in ack:
                        st.markdown(
                            f"- **[{item.get('score', '?')}]** {item.get('claim', '')} "
                            f"— _{item.get('why_acknowledged', '')}_"
                        )
            st.markdown(f"**What the winner could not prove:** "
                        f"{judge.get('weak_points_winner', '')}")
            st.markdown(f"**Uncertainty:** {judge.get('uncertainty_note', '')}")
        else:
            # No-conclusion report
            reason = agg.get("no_conclusion_reason", "")
            if reason:
                st.warning(f"**Why no conclusion:** {reason}")
            for side, key in [("Affirmative", "strongest_affirmative"),
                               ("Negative",   "strongest_negative")]:
                items = judge.get(key, [])
                if items:
                    st.markdown(f"**Strongest {side} claims:**")
                    for item in items:
                        st.markdown(
                            f"- [{item.get('score', '?')}] {item.get('claim', '')} "
                            f"— _{item.get('note', '')}_"
                        )
            wtc = judge.get("what_would_change_verdict", "")
            if wtc:
                st.info(f"**What would tip the balance:** {wtc}")

    # ── Full transparency: all scored claims ───────────────────────────────────
    st.markdown("---")
    with st.expander("🔍 Full Transparency — All Scored Claims", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"**All Affirmative ({len(aff_all)} claims)**")
            for i, c in enumerate(aff_all, 1):
                render_claim_card(c, i, SCORING_DIMENSIONS)
        with t2:
            st.markdown(f"**All Negative ({len(neg_all)} claims)**")
            for i, c in enumerate(neg_all, 1):
                render_claim_card(c, i, SCORING_DIMENSIONS)

    # ── Pipeline log ───────────────────────────────────────────────────────────
    logs = state.get("pipeline_log", [])
    if logs:
        with st.expander("📋 Pipeline Log", expanded=False):
            log_text = "\n".join(f"[{i+1}] {l}" for i, l in enumerate(logs))
            st.markdown(f'<div class="pipeline-log">{log_text}</div>',
                        unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <h1 style="font-family:'IBM Plex Mono',monospace; font-size:1.8rem; margin-bottom:0">
      ⚖️ Adversarial Claim Scoring Engine
    </h1>
    <p style="color:#64748b; margin-top:4px; font-size:0.9rem">
      Multi-agent argumentation analysis with mathematical scoring · Not a chatbot.
    </p>
    """, unsafe_allow_html=True)

    # Sidebar config
    cfg = build_config_from_sidebar()

    # Main input area
    col_upload, col_q = st.columns([1, 2])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt", "docx"],
            help="PDF, plain text, or Word document",
        )

    with col_q:
        question = st.text_area(
            "Analytical Question",
            height=100,
            placeholder="e.g. Has BJP improved India's economic performance since 2014?",
            help="Frame as a yes/no or evaluative question for best results.",
        )
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # Run pipeline
    if run_btn:
        if not uploaded:
            st.error("Please upload a document.")
            return
        if not question.strip():
            st.error("Please enter an analytical question.")
            return

        # Save upload to temp file
        suffix = "." + uploaded.name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getbuffer())
            tmp_path = f.name

        # Run pipeline with progress indicators
        progress = st.progress(0, text="Ingesting document…")
        try:
            vectorstore = ingest_document(tmp_path, cfg)
            progress.progress(15, text="Document ingested. Building graph…")

            graph = build_graph()
            from graph import EngineState
            initial = EngineState(
                question=question,
                vectorstore=vectorstore,
                cfg=cfg,
                aff_raw=[], neg_raw=[],
                aff_merged=[], neg_merged=[],
                aff_filtered=[], neg_filtered=[],
                aff_scored=[], neg_scored=[],
                agg_result={}, judge_output={},
                pipeline_log=[], error=None,
            )

            progress.progress(20, text="Generating claims (dual agents)…")
            # Run graph (blocking)
            final_state = graph.invoke(initial)
            progress.progress(100, text="Analysis complete!")

            if final_state.get("error"):
                st.error(f"Pipeline error: {final_state['error']}")
            else:
                render_results(final_state)

        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            logger.exception("Pipeline failure")
        finally:
            os.unlink(tmp_path)
            progress.empty()


if __name__ == "__main__":
    main()
