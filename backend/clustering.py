"""
clustering.py — Layer 3 & 4: Embedding + Clustering + Merging (Deduplication)

Converts claims → 768-dim vectors → clusters via DBSCAN or K-Means →
selects centroid-nearest representative from each cluster.
No LLM needed — pure mathematical deduplication.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from langchain_google_vertexai import VertexAIEmbeddings

from config import RuntimeConfig, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────
def embed_claims(claims: List[str]) -> np.ndarray:
    """
    Convert claims to L2-normalised 768-dim vectors via text-embedding-004.
    L2 normalisation makes Euclidean distance equivalent to cosine distance.

    Returns:
        np.ndarray of shape (n_claims, 768), float32.
    """
    if not claims:
        return np.empty((0, 768), dtype=np.float32)

    embedder = VertexAIEmbeddings(model_name=EMBEDDING_MODEL)
    raw_vectors = embedder.embed_documents(claims)
    vectors = np.array(raw_vectors, dtype=np.float32)
    vectors = normalize(vectors, norm="l2")   # cosine ≡ 1 - (u·v) after normalisation
    logger.info("Embedded %d claims → shape %s", len(claims), vectors.shape)
    return vectors


# ─────────────────────────────────────────────
# CLUSTERING ALGORITHMS
# ─────────────────────────────────────────────
def _cluster_dbscan(
    vectors: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """
    DBSCAN: automatically discovers cluster count.
    Isolated claims become their own single-item clusters (label = -1 treated
    separately so they are not discarded).

    Returns: array of integer labels, shape (n_claims,).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=-1)
    labels = db.fit_predict(vectors)

    # Reassign noise points (label=-1) to unique cluster IDs so they
    # survive as singleton representative claims.
    noise_mask = labels == -1
    max_label = labels.max() if labels.max() >= 0 else -1
    for i, is_noise in enumerate(noise_mask):
        if is_noise:
            max_label += 1
            labels[i] = max_label

    logger.info("DBSCAN: eps=%.3f, min_samples=%d → %d clusters",
                eps, min_samples, len(set(labels)))
    return labels


def _cluster_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    K-Means: fixed number of clusters. Use when document size is known.
    """
    n_clusters = min(n_clusters, len(vectors))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(vectors)
    logger.info("K-Means: k=%d clusters", n_clusters)
    return labels


# ─────────────────────────────────────────────
# MERGING: select representative per cluster
# ─────────────────────────────────────────────
def _select_representative(
    claims: List[str],
    vectors: np.ndarray,
    cluster_indices: np.ndarray,
) -> Tuple[str, np.ndarray]:
    """
    For a cluster, compute the centroid vector, then find the claim
    whose vector has the minimum Euclidean distance to the centroid.
    That claim is the mathematical representative of the entire cluster.

    Returns:
        (representative_claim_text, centroid_vector)
    """
    cluster_vecs = vectors[cluster_indices]
    centroid = cluster_vecs.mean(axis=0)
    distances = np.linalg.norm(cluster_vecs - centroid, axis=1)
    best_idx = cluster_indices[np.argmin(distances)]
    return claims[best_idx], centroid


def merge_claims(
    claims: List[str],
    vectors: np.ndarray,
    cfg: Optional[RuntimeConfig] = None,
) -> List[str]:
    """
    Cluster claims and return one representative per cluster.

    Args:
        claims:   Original claim strings.
        vectors:  L2-normalised embedding matrix, shape (n_claims, dim).
        cfg:      RuntimeConfig; uses defaults if None.

    Returns:
        List of representative claim strings (deduplicated).
    """
    cfg = cfg or RuntimeConfig()

    if len(claims) == 0:
        return []

    if len(claims) == 1:
        return claims[:]

    # Choose clustering algorithm
    if cfg.clustering_method.lower() == "kmeans":
        labels = _cluster_kmeans(vectors, cfg.kmeans_n_clusters)
    else:
        labels = _cluster_dbscan(vectors, cfg.dbscan_eps, cfg.dbscan_min_samples)

    # Collect one representative per cluster
    representatives = []
    for label in sorted(set(labels)):
        idxs = np.where(labels == label)[0]
        rep_claim, _ = _select_representative(claims, vectors, idxs)
        representatives.append(rep_claim)

    logger.info(
        "Merged %d raw claims → %d unique representatives (method=%s)",
        len(claims), len(representatives), cfg.clustering_method,
    )
    return representatives


# ─────────────────────────────────────────────
# CONVENIENCE: embed + merge in one call
# ─────────────────────────────────────────────
def embed_and_merge(
    claims: List[str],
    cfg: Optional[RuntimeConfig] = None,
) -> List[str]:
    """Embed claims and return deduplicated representatives."""
    if not claims:
        return []
    vectors = embed_claims(claims)
    return merge_claims(claims, vectors, cfg)
