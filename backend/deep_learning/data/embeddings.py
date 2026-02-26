"""
FinBERT Embedding Pipeline.

Extracts 768-dimensional CLS token vectors from ProsusAI/finbert and reduces
them to compact representations via IncrementalPCA for downstream consumption
by the Temporal Fusion Transformer.

Usage (standalone backfill):
    python -m deep_learning.data.embeddings --backfill --days 180
"""

from __future__ import annotations

import argparse
import logging
import struct
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 768


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_finbert_encoder():
    """Load ProsusAI/finbert as a bare encoder (no classification head)."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoModel, AutoTokenizer

    model_name = "ProsusAI/finbert"
    logger.info("Loading FinBERT encoder: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    logger.info("FinBERT encoder loaded (%d parameters)", sum(p.numel() for p in model.parameters()))
    return tokenizer, model


# ---------------------------------------------------------------------------
# Single / batch embedding extraction
# ---------------------------------------------------------------------------

def extract_embedding(text: str, *, max_length: int = 512) -> np.ndarray:
    """Return the [CLS] hidden-state vector (768-d, float32) for *text*."""
    import torch

    tokenizer, model = _load_finbert_encoder()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    cls_vec: np.ndarray = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    return cls_vec.astype(np.float32)


def extract_embeddings_batch(
    texts: Sequence[str],
    *,
    max_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Vectorised extraction for a list of texts.

    Returns:
        ndarray of shape (N, 768) with float32 CLS vectors.
    """
    import torch

    tokenizer, model = _load_finbert_encoder()
    all_vecs: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start: start + batch_size])
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls_vecs = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        all_vecs.append(cls_vecs)

    return np.concatenate(all_vecs, axis=0)


# ---------------------------------------------------------------------------
# PCA dimensionality reduction
# ---------------------------------------------------------------------------

def fit_pca(
    embeddings: np.ndarray,
    n_components: int = 32,
    save_path: Optional[str] = None,
) -> "IncrementalPCA":
    """Fit IncrementalPCA on a matrix of (N, 768) embeddings."""
    from sklearn.decomposition import IncrementalPCA
    import joblib

    logger.info("Fitting PCA: %s -> %d components on %d samples", embeddings.shape, n_components, len(embeddings))
    pca = IncrementalPCA(n_components=n_components)
    pca.fit(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    logger.info("PCA explained variance: %.4f", explained)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, path)
        logger.info("PCA model saved to %s", path)

    return pca


def load_pca(path: str) -> "IncrementalPCA":
    import joblib
    return joblib.load(path)


def reduce_embeddings(
    embeddings: np.ndarray,
    pca: "IncrementalPCA",
) -> np.ndarray:
    """Project (N, 768) embeddings down to (N, pca.n_components) via PCA."""
    return pca.transform(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Serialisation helpers (DB storage as bytes)
# ---------------------------------------------------------------------------

def embedding_to_bytes(vec: np.ndarray) -> bytes:
    """Pack a 1-D float32 array into a compact binary blob."""
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dim: int = _EMBEDDING_DIM) -> np.ndarray:
    """Unpack binary blob back to float32 array."""
    return np.frombuffer(data, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# DB backfill logic
# ---------------------------------------------------------------------------

def backfill_embeddings(
    *,
    days: int = 180,
    pca_dim: int = 32,
    batch_size: int = 64,
) -> dict[str, int]:
    """
    Compute and store FinBERT embeddings for all unprocessed news articles.

    Steps:
        1. Query news_processed rows that lack a corresponding news_embeddings row.
        2. Extract CLS embeddings in batches.
        3. Fit or load PCA model.
        4. Store reduced embeddings in news_embeddings table.
    """
    from app.db import SessionLocal
    from app.models import NewsProcessed, NewsRaw
    from deep_learning.config import get_tft_config

    cfg = get_tft_config()
    pca_path = Path(cfg.embedding.pca_model_path)

    stats = {"embedded": 0, "skipped": 0, "pca_fitted": False}

    with SessionLocal() as session:
        from sqlalchemy import text as sa_text

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        rows = (
            session.query(
                NewsProcessed.id,
                NewsProcessed.cleaned_text,
                NewsRaw.title,
                NewsRaw.description,
            )
            .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
            .filter(NewsRaw.published_at >= cutoff)
            .order_by(NewsProcessed.id.asc())
            .all()
        )

        if not rows:
            logger.info("No articles to embed")
            return stats

        logger.info("Found %d articles for embedding extraction", len(rows))

        ids: list[int] = []
        texts: list[str] = []
        for row in rows:
            text = str(row.cleaned_text or f"{row.title} {row.description or ''}")[:1200]
            if len(text.strip()) < 10:
                stats["skipped"] += 1
                continue
            ids.append(int(row.id))
            texts.append(text)

        if not texts:
            return stats

        full_embeddings = extract_embeddings_batch(texts, batch_size=batch_size)
        logger.info("Extracted %d embeddings, shape=%s", len(full_embeddings), full_embeddings.shape)

        if pca_path.exists():
            pca = load_pca(str(pca_path))
            logger.info("Loaded existing PCA model from %s", pca_path)
        else:
            pca = fit_pca(full_embeddings, n_components=pca_dim, save_path=str(pca_path))
            stats["pca_fitted"] = True

        reduced = reduce_embeddings(full_embeddings, pca)

        try:
            from app.models import NewsEmbedding
        except ImportError:
            logger.error("NewsEmbedding model not found - run DB migration first")
            return stats

        for idx, article_id in enumerate(ids):
            existing = session.query(NewsEmbedding).filter(
                NewsEmbedding.news_processed_id == article_id
            ).first()
            if existing:
                stats["skipped"] += 1
                continue

            emb = NewsEmbedding(
                news_processed_id=article_id,
                embedding_full=embedding_to_bytes(full_embeddings[idx]),
                embedding_pca=embedding_to_bytes(reduced[idx]),
                pca_version=f"pca{pca_dim}_v1",
            )
            session.add(emb)
            stats["embedded"] += 1

            if stats["embedded"] % 200 == 0:
                session.commit()
                logger.info("Committed %d embeddings so far", stats["embedded"])

        session.commit()

    logger.info("Embedding backfill complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# Daily aggregation helper
# ---------------------------------------------------------------------------

def aggregate_daily_embeddings(
    article_embeddings: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aggregate multiple article embeddings into a single daily vector.

    Uses confidence-weighted mean when weights are provided,
    otherwise a simple mean.
    """
    if article_embeddings.ndim == 1:
        return article_embeddings

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        w = w / (w.sum() + 1e-9)
        return (article_embeddings * w).sum(axis=0).astype(np.float32)

    return article_embeddings.mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="FinBERT embedding backfill")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.backfill:
        from app.db import init_db
        init_db()
        backfill_embeddings(days=args.days, pca_dim=args.pca_dim, batch_size=args.batch_size)
