"""
graph.py — LangGraph Orchestration

Wires all pipeline layers into a stateful directed graph:

  generate → cluster → filter → score → aggregate
                                            │
                              ┌─────────────┤
                              ↓             ↓
                          no_concl        judge
                              └─────┬──────┘
                                   END

State is passed immutably through each node.
"""

import logging
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END

from ingestion import ingest_document
from agents import generate_both_sides
from clustering import embed_and_merge
from filtering import filter_top_percent, check_imbalance, balance_sides
from scoring import score_claims
from aggregation import aggregate
from judge import run_judge
from config import RuntimeConfig, SCORING_DIMENSIONS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────
class EngineState(TypedDict):
    # Inputs
    question: str
    vectorstore: Any
    cfg: Optional[RuntimeConfig]

    # Pipeline outputs (accumulated)
    aff_raw: List[str]
    neg_raw: List[str]
    aff_merged: List[str]
    neg_merged: List[str]
    aff_filtered: List[str]
    neg_filtered: List[str]
    aff_scored: List[Dict[str, Any]]
    neg_scored: List[Dict[str, Any]]

    # Aggregation
    agg_result: Dict[str, Any]

    # Judge output
    judge_output: Dict[str, Any]

    # Pipeline meta
    pipeline_log: List[str]
    error: Optional[str]


# ─────────────────────────────────────────────
# NODE: GENERATE
# ─────────────────────────────────────────────
def node_generate(state: EngineState) -> EngineState:
    logger.info("NODE: generate")
    cfg = state.get("cfg") or RuntimeConfig()
    try:
        sides = generate_both_sides(state["vectorstore"], state["question"], cfg)
        return {
            **state,
            "aff_raw": sides["affirmative"],
            "neg_raw": sides["negative"],
            "pipeline_log": state.get("pipeline_log", []) + [
                f"Generated: {len(sides['affirmative'])} affirmative, "
                f"{len(sides['negative'])} negative raw claims."
            ],
        }
    except Exception as e:
        return {**state, "error": f"generate: {e}",
                "aff_raw": [], "neg_raw": []}


# ─────────────────────────────────────────────
# NODE: CLUSTER (embed + merge)
# ─────────────────────────────────────────────
def node_cluster(state: EngineState) -> EngineState:
    logger.info("NODE: cluster")
    cfg = state.get("cfg") or RuntimeConfig()
    try:
        aff_merged = embed_and_merge(state["aff_raw"], cfg)
        neg_merged = embed_and_merge(state["neg_raw"], cfg)
        return {
            **state,
            "aff_merged": aff_merged,
            "neg_merged": neg_merged,
            "pipeline_log": state.get("pipeline_log", []) + [
                f"Merged: {len(aff_merged)} affirmative, "
                f"{len(neg_merged)} negative unique claims."
            ],
        }
    except Exception as e:
        return {**state, "error": f"cluster: {e}",
                "aff_merged": state.get("aff_raw", []),
                "neg_merged": state.get("neg_raw", [])}


# ─────────────────────────────────────────────
# NODE: FILTER (top-X%)
# ─────────────────────────────────────────────
def node_filter(state: EngineState) -> EngineState:
    logger.info("NODE: filter")
    cfg = state.get("cfg") or RuntimeConfig()

    aff_tuples = filter_top_percent(state["aff_merged"], cfg)
    neg_tuples = filter_top_percent(state["neg_merged"], cfg)

    aff_filtered = [c for c, _ in aff_tuples]
    neg_filtered = [c for c, _ in neg_tuples]

    # Imbalance check BEFORE balancing
    imbalance_reason = check_imbalance(aff_filtered, neg_filtered, cfg)
    if imbalance_reason:
        # Pre-populate agg_result to trigger no_conclusion routing
        return {
            **state,
            "aff_filtered": aff_filtered,
            "neg_filtered": neg_filtered,
            "aff_scored": [],
            "neg_scored": [],
            "agg_result": {
                "verdict": "no_conclusion",
                "confidence": 0,
                "aff_top": [],
                "neg_top": [],
                "aff_all": [],
                "neg_all": [],
                "no_conclusion_reason": imbalance_reason,
            },
            "pipeline_log": state.get("pipeline_log", []) + [
                f"IMBALANCE: {imbalance_reason}"
            ],
        }

    aff_bal, neg_bal = balance_sides(aff_filtered, neg_filtered, cfg)
    return {
        **state,
        "aff_filtered": aff_bal,
        "neg_filtered": neg_bal,
        "pipeline_log": state.get("pipeline_log", []) + [
            f"Filtered + balanced: {len(aff_bal)} affirmative, "
            f"{len(neg_bal)} negative claims per side."
        ],
    }


# ─────────────────────────────────────────────
# NODE: SCORE (multi-dimensional)
# ─────────────────────────────────────────────
def node_score(state: EngineState) -> EngineState:
    logger.info("NODE: score")
    cfg = state.get("cfg") or RuntimeConfig()

    # Skip if already routed to no_conclusion by filter node
    if state.get("agg_result", {}).get("verdict") == "no_conclusion":
        return state

    aff_scored = score_claims(
        state["aff_filtered"], state["question"], cfg, SCORING_DIMENSIONS
    )
    neg_scored = score_claims(
        state["neg_filtered"], state["question"], cfg, SCORING_DIMENSIONS
    )
    return {
        **state,
        "aff_scored": aff_scored,
        "neg_scored": neg_scored,
        "pipeline_log": state.get("pipeline_log", []) + [
            f"Scored: {len(aff_scored)} aff, {len(neg_scored)} neg claims survived scoring."
        ],
    }


# ─────────────────────────────────────────────
# NODE: AGGREGATE
# ─────────────────────────────────────────────
def node_aggregate(state: EngineState) -> EngineState:
    logger.info("NODE: aggregate")
    cfg = state.get("cfg") or RuntimeConfig()

    # Skip if already no_conclusion
    if state.get("agg_result", {}).get("verdict") == "no_conclusion":
        return state

    agg_result = aggregate(
        state.get("aff_scored", []),
        state.get("neg_scored", []),
        cfg,
        SCORING_DIMENSIONS,
    )
    return {
        **state,
        "agg_result": agg_result,
        "pipeline_log": state.get("pipeline_log", []) + [
            f"Aggregation verdict: {agg_result['verdict']} "
            f"(confidence={agg_result.get('confidence', 0)}%)"
        ],
    }


# ─────────────────────────────────────────────
# NODE: JUDGE
# ─────────────────────────────────────────────
def node_judge(state: EngineState) -> EngineState:
    logger.info("NODE: judge")
    judge_output = run_judge(
        question=state["question"],
        agg_result=state["agg_result"],
    )
    return {
        **state,
        "judge_output": judge_output,
        "pipeline_log": state.get("pipeline_log", []) + ["Judge synthesis complete."],
    }


# ─────────────────────────────────────────────
# NODE: NO CONCLUSION
# ─────────────────────────────────────────────
def node_no_conclusion(state: EngineState) -> EngineState:
    logger.info("NODE: no_conclusion")
    judge_output = run_judge(
        question=state["question"],
        agg_result=state["agg_result"],
        data_note=state["agg_result"].get("no_conclusion_reason"),
    )
    return {
        **state,
        "judge_output": judge_output,
        "pipeline_log": state.get("pipeline_log", []) + [
            "No-conclusion path executed."
        ],
    }


# ─────────────────────────────────────────────
# ROUTING CONDITION
# ─────────────────────────────────────────────
def route_verdict(state: EngineState) -> str:
    verdict = state.get("agg_result", {}).get("verdict", "no_conclusion")
    return "conclude" if verdict != "no_conclusion" else "no_conclusion"


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────
def build_graph():
    """
    Compile and return the LangGraph StateGraph.

    Graph topology:
        generate → cluster → filter → score → aggregate
                                                    ├── [conclude]    → judge    → END
                                                    └── [no_conclusion] → no_concl → END
    """
    g = StateGraph(EngineState)

    # Register nodes
    g.add_node("generate",    node_generate)
    g.add_node("cluster",     node_cluster)
    g.add_node("filter",      node_filter)
    g.add_node("score",       node_score)
    g.add_node("aggregate",   node_aggregate)
    g.add_node("judge",       node_judge)
    g.add_node("no_concl",    node_no_conclusion)

    # Entry point
    g.set_entry_point("generate")

    # Linear edges
    g.add_edge("generate",  "cluster")
    g.add_edge("cluster",   "filter")
    g.add_edge("filter",    "score")
    g.add_edge("score",     "aggregate")

    # Conditional branching after aggregation
    g.add_conditional_edges(
        "aggregate",
        route_verdict,
        {
            "conclude":       "judge",
            "no_conclusion":  "no_concl",
        },
    )

    # Terminal edges
    g.add_edge("judge",    END)
    g.add_edge("no_concl", END)

    return g.compile()


# ─────────────────────────────────────────────
# CONVENIENCE: run full pipeline
# ─────────────────────────────────────────────
def run_pipeline(
    file_path: str,
    question: str,
    cfg: Optional[RuntimeConfig] = None,
) -> EngineState:
    """
    End-to-end convenience function: ingest file → run graph → return state.
    """
    from ingestion import ingest_document as _ingest
    cfg = cfg or RuntimeConfig()
    vectorstore = _ingest(file_path, cfg)
    graph = build_graph()
    initial_state = EngineState(
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
    return graph.invoke(initial_state)
