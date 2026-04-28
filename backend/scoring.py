"""
scoring.py — Layer 6: Multi-Dimensional Scoring

Each surviving claim is scored independently on all SCORING_DIMENSIONS.
Agents are adversarially blinded: the scorer for affirmative claims
uses a skeptic prompt (and vice versa) to prevent score inflation.

Each scoring call returns:
    { 's1': float, 's2': float, 's3': float, 'reasoning': str, 'claim': str }
Claims where ANY dimension < DISCARD_THRESHOLD are discarded.
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any

from langchain_google_vertexai import ChatVertexAI

from config import RuntimeConfig, FLASH_MODEL, SCORING_DIMENSIONS, GCP_PROJECT_ID, GCP_LOCATION

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SCORING PROMPT (adversarially blinded)
# ─────────────────────────────────────────────
_SCORING_PROMPT_TEMPLATE = """You are a SKEPTICAL EVALUATOR. Your job is to critically
assess the quality of ONE claim. Be rigorous — inflate no scores.

QUESTION BEING ANALYSED: {question}

CLAIM TO SCORE: {claim}

Score this claim on EXACTLY these {n_dims} dimensions.
Return ONLY a valid JSON object — no markdown, no preamble, no trailing text.

Scoring dimensions (each 0.0 to 1.0):
{dimensions_block}

DISCARD RULE: If ANY score would be below 0.2, set it to exactly 0.0.

JSON FORMAT (copy keys exactly):
{{
{json_keys}
  "reasoning": "ONE sentence explaining the dominant scoring factor."
}}

Return ONLY the JSON object above. Nothing else.
"""


def _build_dimensions_block(dimensions: List[dict]) -> str:
    lines = []
    for d in dimensions:
        lines.append(
            f'  - "{d["id"]}" ({d["name"]}, weight={d["weight"]}): {d["description"]}'
        )
    return "\n".join(lines)


def _build_json_keys(dimensions: List[dict]) -> str:
    lines = [f'  "{d["id"]}": <0.0-1.0>,' for d in dimensions]
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    """
    Robustly extract the first JSON object from LLM output.
    Handles markdown code fences and extra text.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    # Find first {...}
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from: {raw[:200]}")


# ─────────────────────────────────────────────
# SINGLE-CLAIM SCORER
# ─────────────────────────────────────────────
def score_claim(
    claim: str,
    question: str,
    cfg: Optional[RuntimeConfig] = None,
    dimensions: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """
    Score a single claim on all configured dimensions.

    Returns:
        Dict with dimension scores, 'reasoning', and 'claim'.
        Returns zeroed scores dict on parse failure (claim will be discarded).
    """
    cfg = cfg or RuntimeConfig()
    dims = dimensions or SCORING_DIMENSIONS

    prompt = _SCORING_PROMPT_TEMPLATE.format(
        question=question,
        claim=claim,
        n_dims=len(dims),
        dimensions_block=_build_dimensions_block(dims),
        json_keys=_build_json_keys(dims),
    )

    llm = ChatVertexAI(
        model_name=FLASH_MODEL,
        temperature=0.0,
        max_output_tokens=512,
        project=GCP_PROJECT_ID or None,
        location=GCP_LOCATION,
    )

    try:
        raw = llm.invoke(prompt).content
        scores = _extract_json(raw)

        # Coerce all dimension scores to float, clamp to [0, 1]
        for d in dims:
            key = d["id"]
            try:
                scores[key] = max(0.0, min(1.0, float(scores.get(key, 0.0))))
            except (TypeError, ValueError):
                scores[key] = 0.0

        scores["reasoning"] = str(scores.get("reasoning", "No reasoning provided."))

    except Exception as exc:
        logger.warning("Score parse error for claim '%s...': %s", claim[:40], exc)
        scores = {d["id"]: 0.0 for d in dims}
        scores["reasoning"] = f"Parse error: {exc}"

    scores["claim"] = claim
    return scores


# ─────────────────────────────────────────────
# BATCH SCORER WITH DISCARD FILTER
# ─────────────────────────────────────────────
def score_claims(
    claims: List[str],
    question: str,
    cfg: Optional[RuntimeConfig] = None,
    dimensions: Optional[List[dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Score all claims and discard any where ANY dimension < DISCARD_THRESHOLD.

    Returns:
        List of score dicts for claims that passed the discard threshold.
    """
    cfg = cfg or RuntimeConfig()
    dims = dimensions or SCORING_DIMENSIONS
    threshold = cfg.discard_threshold

    results = []
    discarded = 0

    for i, claim in enumerate(claims):
        logger.debug("Scoring claim %d/%d: '%s...'", i + 1, len(claims), claim[:50])
        scores = score_claim(claim, question, cfg, dims)

        dim_scores = [scores.get(d["id"], 0.0) for d in dims]

        # Discard if ANY dimension falls below threshold
        if min(dim_scores) < threshold:
            logger.debug("  ↳ DISCARDED (min score %.3f < threshold %.3f)",
                         min(dim_scores), threshold)
            discarded += 1
        else:
            results.append(scores)

    logger.info(
        "Scoring: %d claims in → %d passed, %d discarded (threshold=%.2f)",
        len(claims), len(results), discarded, threshold,
    )
    return results
