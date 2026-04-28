"""
ingestion.py — Layer 1: Document Ingestion (RAG)

Loads PDF / TXT / DOCX, chunks with overlap, embeds via
text-embedding-004, stores in ChromaDB.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import RuntimeConfig, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LOADER REGISTRY
# ─────────────────────────────────────────────
_LOADERS = {
    ".pdf":  PDFPlumberLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
}


def _get_loader(file_path: str):
    suffix = Path(file_path).suffix.lower()
    cls = _LOADERS.get(suffix)
    if cls is None:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {list(_LOADERS.keys())}"
        )
    return cls(file_path)


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def ingest_document(
    file_path: str,
    cfg: Optional[RuntimeConfig] = None,
    collection_name: str = "claim_engine",
) -> Chroma:
    """
    Parse → chunk → embed → store.

    Args:
        file_path:          Absolute path to the document.
        cfg:                RuntimeConfig instance (uses module defaults if None).
        persist_directory:  ChromaDB storage path (uses config default if None).
        collection_name:    ChromaDB collection name.

    Returns:
        A Chroma vectorstore ready to be used as a retriever.
    """
    cfg = cfg or RuntimeConfig()

    logger.info("Loading document: %s", file_path)
    loader = _get_loader(file_path)
    docs = loader.load()
    logger.info("Loaded %d page(s)/section(s)", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks (size=%d, overlap=%d)",
                len(chunks), cfg.chunk_size, cfg.chunk_overlap)

    embedder = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)

    # In-memory Chroma — no persist_directory.
    # Cloud Run containers are ephemeral; each request ingests its own
    # document fresh, so disk persistence adds no value and causes issues.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=collection_name,
    )
    logger.info("ChromaDB vectorstore built (in-memory). Collection: '%s'", collection_name)
    return vectorstore


def load_existing_vectorstore(
    persist_directory: Optional[str] = None,
    collection_name: str = "claim_engine",
) -> Chroma:
    """
    Load a previously persisted ChromaDB store (for resuming sessions).
    """
    persist_dir = persist_directory or CHROMA_PERSIST_DIR
    embedder = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder,
        collection_name=collection_name,
    )
