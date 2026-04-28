"""
judge.py — Layer 8: Judge / Synthesizer

Receives full aggregation result and generates a transparent,
human-readable verdict narrative using Gemini 2.5 Pro.

The Judge does NOT change the mathematical verdict — it translates
the numbers into a reasoned explanation and surfaces the reasoning
trail for user verification.
"""

import json
import logging
from typing import Dict, Any, Optional

from langchain_google_vertexai import ChatVertexAI

from config import PRO_MODEL, GCP_PROJECT_ID, GCP_LOCATION

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# JUDGE PROMPT — structured JSON output
# ─────────────────────────────────────────────
_JUDGE_PROMPT = """You are an IMPARTIAL JUDGE synthesising a structured argumentation analysis.
Your role is to explain the mathematical scoring result in clear, reasoned language.
You do NOT override the verdict — you explain it.

═══════════════════════════════════════════════════════
ANALYSIS INPUTS
═══════════════════════════════════════════════════════
Question analysed: {question}

Mathematical verdict: {verdict}
Confidence: {confidence}%
Average score — Affirmative side: {avg_aff:.4f}
Average score — Negative side:    {avg_neg:.4f}
Score gap: {gap:.4f}

TOP AFFIRMATIVE CLAIMS (with composite scores):
{aff_claims_text}

TOP NEGATIVE CLAIMS (with composite scores):
{neg_claims_text}

Data note: {data_note}
═══════════════════════════════════════════════════════

Generate a structured analysis. Return ONLY valid JSON — no markdown, no preamble.

{{
  "verdict_explanation": "2 sentences: why did this verdict emerge from the math?",
  "decisive_claims": [
    {{"claim": "...", "score": 0.0, "why_decisive": "one sentence"}}
  ],
  "acknowledged_claims": [
    {{"claim": "...", "score": 0.0, "why_acknowledged": "one sentence"}}
  ],
  "weak_points_winner": "What the winning side could NOT prove or left uncertain.",
  "weak_points_loser": "What the losing side's strongest available argument was.",
  "uncertainty_note": "One sentence on what remains unresolved by this document.",
  "data_quality_note": "{data_note}"
}}

Rules:
- decisive_claims: top 3 from the WINNING side (or affirmative side if no_conclusion).
- acknowledged_claims: top 3 from the OTHER side.
- All claim texts must be verbatim from the inputs above.
- Scores must be verbatim floats from the inputs above.
- Be factual. Reference specific claims and scores.
- Return ONLY the JSON object.
"""

_NO_CONCLUSION_PROMPT = """You are an IMPARTIAL JUDGE. The mathematical scoring system
could not reach a verdict for the following reason:

{no_conclusion_reason}

Question: {question}

Available top claims from AFFIRMATIVE side:
{aff_claims_text}

Available top claims from NEGATIVE side:
{neg_claims_text}

Generate a structured no-conclusion report. Return ONLY valid JSON:
{{
  "verdict_explanation": "2 sentences explaining why no conclusion is possible.",
  "strongest_affirmative": [
    {{"claim": "...", "score": 0.0, "note": "one sentence"}}
  ],
  "strongest_negative": [
    {{"claim": "...", "score": 0.0, "note": "one sentence"}}
  ],
  "what_would_change_verdict": "What additional evidence would tip the balance?",
  "uncertainty_note": "Key unresolved question from this analysis.",
  "data_quality_note": "{no_conclusion_reason}"
}}

Return ONLY the JSON object.
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _format_claims_text(claims: list) -> str:
    if not claims:
        return "  (none)"
    lines = []
    for i, c in enumerate(claims[:5], 1):
        score = c.get("composite", 0.0)
        text = c.get("claim", "")[:120]
        lines.append(f"  {i}. [{score:.4f}] {text}")
    return "\n".join(lines)


def _safe_json(raw: str) -> dict:
    import re
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"verdict_explanation": raw, "parse_error": True}


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def run_judge(
    question: str,
    agg_result: Dict[str, Any],
    data_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate the Judge's narrative synthesis.

    Args:
        question:    The original user question.
        agg_result:  Output from aggregation.aggregate().
        data_note:   Optional note about data quality warnings.

    Returns:
        Structured dict with verdict explanation, decisive/acknowledged claims,
        weak points, and uncertainty notes.
    """
    verdict = agg_result.get("verdict", "no_conclusion")
    aff_top = agg_result.get("aff_top", [])
    neg_top = agg_result.get("neg_top", [])
    data_note = data_note or agg_result.get("no_conclusion_reason", "None")

    aff_text = _format_claims_text(aff_top)
    neg_text = _format_claims_text(neg_top)

    llm = ChatVertexAI(
        model_name=PRO_MODEL,
        temperature=0.1,
        max_output_tokens=2048,
        project=GCP_PROJECT_ID or None,
        location=GCP_LOCATION,
    )

    if verdict == "no_conclusion":
        prompt = _NO_CONCLUSION_PROMPT.format(
            no_conclusion_reason=agg_result.get("no_conclusion_reason", "Unknown reason."),
            question=question,
            aff_claims_text=aff_text,
            neg_claims_text=neg_text,
        )
    else:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            verdict=verdict,
            confidence=agg_result.get("confidence", 0),
            avg_aff=agg_result.get("avg_aff", 0.0),
            avg_neg=agg_result.get("avg_neg", 0.0),
            gap=agg_result.get("gap", 0.0),
            aff_claims_text=aff_text,
            neg_claims_text=neg_text,
            data_note=data_note,
        )

    logger.info("Running Judge with model=%s, verdict=%s", PRO_MODEL, verdict)
    raw = llm.invoke(prompt).content
    result = _safe_json(raw)

    # Attach metadata
    result["_verdict"] = verdict
    result["_confidence"] = agg_result.get("confidence", 0)
    result["_avg_aff"] = agg_result.get("avg_aff")
    result["_avg_neg"] = agg_result.get("avg_neg")
    result["_gap"] = agg_result.get("gap")

    return result
