"""
agents.py — Layer 2: Dual-Agent Claim Generation

Affirmative agent  → claims that SUPPORT the question answer
Negative agent     → claims that OPPOSE / complicate the answer

Both are strictly RAG-grounded: they ONLY use retrieved document
context and cannot introduce training-data knowledge.
"""

import logging
import re
from typing import List, Optional

from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

from config import RuntimeConfig, FLASH_MODEL, GCP_PROJECT_ID, GCP_LOCATION

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
_AFFIRMATIVE_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a strict ADVOCATE. Your ONLY source of information is the
document excerpts provided below. Do NOT use any outside knowledge.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Your task: Extract every atomic factual claim from the document that SUPPORTS
an affirmative answer to the question above.

STRICT RULES:
1. One claim per numbered line. Example: "1. Road network expanded by 47,000 km between 2014–2024."
2. Each claim must be ATOMIC — one single verifiable statement. No compound claims.
3. Each claim must be SPECIFIC — include numbers, dates, names, percentages where available.
4. REJECT vague assertions like "things improved significantly."
5. ONLY use information present in the document context above.
6. If a claim cannot be verified from the context, do NOT include it.
7. Do NOT write any preamble or explanation — output the numbered list ONLY.

ATOMIC SUPPORTING CLAIMS:
"""
)

_NEGATIVE_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a strict CRITIC. Your ONLY source of information is the
document excerpts provided below. Do NOT use any outside knowledge.

DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Your task: Extract every atomic factual claim from the document that OPPOSES,
complicates, or undermines an affirmative answer to the question above.

STRICT RULES:
1. One claim per numbered line. Example: "1. Unemployment rose by 3.2% in the same period."
2. Each claim must be ATOMIC — one single verifiable statement. No compound claims.
3. Each claim must be SPECIFIC — include numbers, dates, names, percentages where available.
4. REJECT vague assertions like "there were many problems."
5. ONLY use information present in the document context above.
6. If a claim cannot be verified from the context, do NOT include it.
7. Do NOT write any preamble or explanation — output the numbered list ONLY.

ATOMIC OPPOSING CLAIMS:
"""
)


# ─────────────────────────────────────────────
# CLAIM PARSER
# ─────────────────────────────────────────────
def _parse_claims(raw_text: str) -> List[str]:
    """
    Extract numbered claims from LLM output.
    Strips numbering, blank lines, and any non-claim artifacts.
    """
    claims = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match lines starting with a digit (numbered list)
        match = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if match:
            claim_text = match.group(1).strip()
            # Reject obviously vague claims (< 5 words)
            if len(claim_text.split()) >= 5:
                claims.append(claim_text)
    return claims


# ─────────────────────────────────────────────
# RETRIEVAL-AUGMENTED CLAIM GENERATION
# ─────────────────────────────────────────────
def _build_rag_chain(vectorstore: Chroma, cfg: RuntimeConfig, prompt_template: PromptTemplate):
    """Build a RetrievalQA chain with the given prompt."""
    llm = ChatVertexAI(
        model_name=FLASH_MODEL,
        temperature=0.2,   # low temp for factual extraction
        max_output_tokens=2048,
        project=GCP_PROJECT_ID or None,
        location=GCP_LOCATION,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg.retrieval_k},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False,
    )
    return chain


def generate_claims(
    vectorstore: Chroma,
    question: str,
    side: str,  # "affirmative" | "negative"
    cfg: Optional[RuntimeConfig] = None,
) -> List[str]:
    """
    Generate RAG-grounded atomic claims from document.

    Args:
        vectorstore:  ChromaDB instance (already populated).
        question:     The user's analytical question.
        side:         "affirmative" or "negative".
        cfg:          RuntimeConfig (uses defaults if None).

    Returns:
        List of clean claim strings.
    """
    cfg = cfg or RuntimeConfig()

    template = (
        _AFFIRMATIVE_TEMPLATE if side.lower() in ("affirmative", "aff", "positive")
        else _NEGATIVE_TEMPLATE
    )

    logger.info("Generating %s claims for question: '%s'", side, question[:60])
    chain = _build_rag_chain(vectorstore, cfg, template)

    result = chain.invoke({"query": question})
    raw_text = result.get("result", "")

    claims = _parse_claims(raw_text)
    logger.info("Generated %d raw %s claims", len(claims), side)
    return claims


# ─────────────────────────────────────────────
# CONVENIENCE: generate both sides at once
# ─────────────────────────────────────────────
def generate_both_sides(
    vectorstore: Chroma,
    question: str,
    cfg: Optional[RuntimeConfig] = None,
) -> dict:
    """
    Returns dict with keys 'affirmative' and 'negative', each a list of claims.
    """
    cfg = cfg or RuntimeConfig()
    return {
        "affirmative": generate_claims(vectorstore, question, "affirmative", cfg),
        "negative":    generate_claims(vectorstore, question, "negative",    cfg),
    }
