"""
filtering.py — Layer 5: Filtering

Fast rule-based specificity scoring to select top X% of merged claims
before the expensive multi-dimensional LLM scoring step.
Also handles: imbalance check and claim-count balancing.
"""

import re
import logging
from typing import List, Optional, Tuple

from config import RuntimeConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# QUICK SPECIFICITY SCORER (rule-based)
# ─────────────────────────────────────────────
_UNIT_PATTERN = re.compile(
    r"\b(crore|billion|million|trillion|km|kg|GW|MW|%|lakh|"
    r"Rs\.?|INR|USD|EUR|GBP)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN    = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER_PATTERN  = re.compile(r"\b\d[\d,\.]*\b")
_NAMED_ENT_PAT   = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")  # two-word proper nouns


def specificity_score(claim: str) -> float:
    """
    Rule-based fast scorer (0.0–1.0).
    Used only for filtering, NOT for the final scoring pipeline.

    Scoring:
    • Base score           : 0.20
    • Has any number       : +0.20
    • Has year             : +0.10
    • Has unit/scale       : +0.15
    • Has named entity     : +0.10
    • Length ≥ 8 words     : +0.10
    • Length ≥ 15 words    : +0.05 (bonus for detail)
    • Ends with a period   : +0.05 (complete sentence proxy)
    • Has quoted figure    : +0.05 (e.g. "47,000 km")
    """
    score = 0.20

    if _NUMBER_PATTERN.search(claim):
        score += 0.20
    if _YEAR_PATTERN.search(claim):
        score += 0.10
    if _UNIT_PATTERN.search(claim):
        score += 0.15
    if _NAMED_ENT_PAT.search(claim):
        score += 0.10

    word_count = len(claim.split())
    if word_count >= 8:
        score += 0.10
    if word_count >= 15:
        score += 0.05

    if claim.rstrip().endswith("."):
        score += 0.05

    # Bonus: contains both a number AND a unit in the same claim
    if _NUMBER_PATTERN.search(claim) and _UNIT_PATTERN.search(claim):
        score += 0.05

    return min(round(score, 4), 1.0)


# ─────────────────────────────────────────────
# TOP-X% SELECTOR
# ─────────────────────────────────────────────
def filter_top_percent(
    claims: List[str],
    cfg: Optional[RuntimeConfig] = None,
) -> List[Tuple[str, float]]:
    """
    Return the top FILTER_PERCENT of claims by specificity score,
    capped at MAX_CLAIMS.

    Returns:
        List of (claim_text, quick_score) tuples, sorted descending.
    """
    cfg = cfg or RuntimeConfig()

    if not claims:
        return []

    scored = [(c, specificity_score(c)) for c in claims]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_n = max(1, int(len(scored) * cfg.filter_percent))
    top_n = min(top_n, cfg.max_claims)

    selected = scored[:top_n]
    logger.info(
        "Filter: %d → %d claims (top %.0f%%, max=%d)",
        len(claims), len(selected), cfg.filter_percent * 100, cfg.max_claims,
    )
    return selected


# ─────────────────────────────────────────────
# IMBALANCE CHECK
# ─────────────────────────────────────────────
def check_imbalance(
    aff_claims: List,
    neg_claims: List,
    cfg: Optional[RuntimeConfig] = None,
) -> Optional[str]:
    """
    Returns a no-conclusion reason string if imbalance detected, else None.

    Triggers:
    1. One side is completely empty.
    2. count ratio exceeds IMBALANCE_RATIO.
    3. Either side has fewer than MIN_CLAIMS_PER_SIDE.
    """
    from config import MIN_CLAIMS_PER_SIDE
    cfg = cfg or RuntimeConfig()

    n_aff = len(aff_claims)
    n_neg = len(neg_claims)

    if n_aff == 0 and n_neg == 0:
        return "Both sides produced zero claims — cannot analyse."

    if n_aff == 0:
        return "Affirmative side produced zero claims — cannot balance."

    if n_neg == 0:
        return "Negative side produced zero claims — cannot balance."

    if n_aff < MIN_CLAIMS_PER_SIDE or n_neg < MIN_CLAIMS_PER_SIDE:
        return (
            f"Insufficient claims: affirmative={n_aff}, negative={n_neg}. "
            f"Minimum required per side: {MIN_CLAIMS_PER_SIDE}."
        )

    ratio = max(n_aff, n_neg) / min(n_aff, n_neg)
    if ratio >= cfg.imbalance_ratio:
        return (
            f"Severe data imbalance: {n_aff} affirmative vs {n_neg} negative claims "
            f"(ratio {ratio:.1f}x ≥ threshold {cfg.imbalance_ratio}x)."
        )

    return None


# ─────────────────────────────────────────────
# BALANCING: ensure equal claim counts
# ─────────────────────────────────────────────
def balance_sides(
    aff_claims: List[str],
    neg_claims: List[str],
    cfg: Optional[RuntimeConfig] = None,
) -> Tuple[List[str], List[str]]:
    """
    Truncate both sides to min(len_aff, len_neg, MAX_CLAIMS).
    Call AFTER imbalance check passes.
    """
    cfg = cfg or RuntimeConfig()
    n = min(len(aff_claims), len(neg_claims), cfg.max_claims)
    return aff_claims[:n], neg_claims[:n]
