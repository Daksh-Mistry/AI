"""
aggregation.py — Layer 7: Score Aggregation + Verdict Logic

Applies weighted dot product math from the design doc:
  raw_score  = Σ(weight_i × score_i)
  norm_score = raw_score / weight_sum

Then:
  avg_A, avg_B → gap → imbalance check → threshold check → verdict
"""

import logging
from typing import List, Dict, Any, Optional

from config import RuntimeConfig, SCORING_DIMENSIONS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# COMPOSITE SCORE (per claim)
# ─────────────────────────────────────────────
def composite_score(
    scores: Dict[str, Any],
    cfg: Optional[RuntimeConfig] = None,
    dimensions: Optional[List[dict]] = None,
) -> float:
    """
    Weighted dot product, normalised to [0, 1].

    raw  = Σ(weight_i × score_i)
    norm = raw / sum(all weights)
    """
    cfg = cfg or RuntimeConfig()
    dims = dimensions or SCORING_DIMENSIONS
    weights = cfg.weights

    raw = sum(
        weights[i] * float(scores.get(d["id"], 0.0))
        for i, d in enumerate(dims)
    )
    return round(raw / cfg.weight_sum, 6)


# ─────────────────────────────────────────────
# AGGREGATION ENGINE
# ─────────────────────────────────────────────
def aggregate(
    aff_scores: List[Dict[str, Any]],
    neg_scores: List[Dict[str, Any]],
    cfg: Optional[RuntimeConfig] = None,
    dimensions: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """
    Full aggregation pipeline returning structured verdict dict.

    Returns:
        {
            verdict:           'affirmative' | 'negative' | 'no_conclusion',
            confidence:        int 0–100  (present if verdict != no_conclusion),
            avg_aff:           float,
            avg_neg:           float,
            gap:               float,
            aff_top:           list of top-5 aff claim dicts with composite score,
            neg_top:           list of top-5 neg claim dicts with composite score,
            no_conclusion_reason: str  (present if no_conclusion),
        }
    """
    cfg = cfg or RuntimeConfig()
    dims = dimensions or SCORING_DIMENSIONS

    # ── 1. Add composite score to every claim dict ──────────────────────────
    for s in aff_scores:
        s["composite"] = composite_score(s, cfg, dims)
    for s in neg_scores:
        s["composite"] = composite_score(s, cfg, dims)

    # ── 2. Sort both sides descending ────────────────────────────────────────
    aff_sorted = sorted(aff_scores, key=lambda x: x["composite"], reverse=True)
    neg_sorted = sorted(neg_scores, key=lambda x: x["composite"], reverse=True)

    # ── 3. Empty-side check ──────────────────────────────────────────────────
    if not aff_sorted or not neg_sorted:
        side = "affirmative" if not aff_sorted else "negative"
        return _no_conclusion(
            reason=f"The {side} side produced zero scoreable claims.",
            aff_sorted=aff_sorted,
            neg_sorted=neg_sorted,
        )

    # ── 4. Imbalance check ───────────────────────────────────────────────────
    ratio = max(len(aff_sorted), len(neg_sorted)) / min(len(aff_sorted), len(neg_sorted))
    if ratio >= cfg.imbalance_ratio:
        return _no_conclusion(
            reason=(
                f"Data imbalance: {len(aff_sorted)} affirmative vs "
                f"{len(neg_sorted)} negative claims "
                f"(ratio {ratio:.1f}x ≥ threshold {cfg.imbalance_ratio:.1f}x). "
                "Cannot make a fair comparison."
            ),
            aff_sorted=aff_sorted,
            neg_sorted=neg_sorted,
        )

    # ── 5. Balance: equal counts both sides ──────────────────────────────────
    n = min(len(aff_sorted), len(neg_sorted))
    aff_top_n = aff_sorted[:n]
    neg_top_n = neg_sorted[:n]

    # ── 6. Average per side ──────────────────────────────────────────────────
    avg_aff = sum(s["composite"] for s in aff_top_n) / n
    avg_neg = sum(s["composite"] for s in neg_top_n) / n
    gap = abs(avg_aff - avg_neg)

    logger.info(
        "Aggregation: avg_aff=%.4f, avg_neg=%.4f, gap=%.4f (threshold=%.3f)",
        avg_aff, avg_neg, gap, cfg.conclusion_thresh,
    )

    # ── 7. Conclusion threshold check ────────────────────────────────────────
    if gap < cfg.conclusion_thresh:
        return _no_conclusion(
            reason=(
                f"Score gap ({gap:.4f}) is below the conclusion threshold "
                f"({cfg.conclusion_thresh:.3f}). Both sides are too evenly matched "
                "to support a definitive verdict."
            ),
            aff_sorted=aff_sorted,
            neg_sorted=neg_sorted,
            avg_aff=avg_aff,
            avg_neg=avg_neg,
            gap=gap,
        )

    # ── 8. Verdict ───────────────────────────────────────────────────────────
    verdict = "affirmative" if avg_aff > avg_neg else "negative"

    # Confidence: linear interpolation scaled by CONFIDENCE_DENOM
    confidence = min(100, int((gap / cfg.confidence_denom) * 100))

    result = {
        "verdict":    verdict,
        "confidence": confidence,
        "avg_aff":    round(avg_aff, 6),
        "avg_neg":    round(avg_neg, 6),
        "gap":        round(gap, 6),
        "aff_top":    aff_sorted[:5],
        "neg_top":    neg_sorted[:5],
        "aff_all":    aff_sorted,
        "neg_all":    neg_sorted,
        "no_conclusion_reason": None,
    }

    logger.info("Verdict: %s (confidence=%d%%)", verdict, confidence)
    return result


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _no_conclusion(
    reason: str,
    aff_sorted: list = None,
    neg_sorted: list = None,
    avg_aff: float = None,
    avg_neg: float = None,
    gap: float = None,
) -> Dict[str, Any]:
    result = {
        "verdict":             "no_conclusion",
        "confidence":          0,
        "avg_aff":             round(avg_aff, 6) if avg_aff is not None else None,
        "avg_neg":             round(avg_neg, 6) if avg_neg is not None else None,
        "gap":                 round(gap, 6) if gap is not None else None,
        "aff_top":             (aff_sorted or [])[:5],
        "neg_top":             (neg_sorted or [])[:5],
        "aff_all":             aff_sorted or [],
        "neg_all":             neg_sorted or [],
        "no_conclusion_reason": reason,
    }
    logger.info("No conclusion: %s", reason)
    return result
