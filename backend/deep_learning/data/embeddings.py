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
def load_finbert_classifier():
    """
    Load ProsusAI/finbert once per process.

    Shared by the sentiment pipeline (`app.ai_engine`) and the embedding
    extractor below so a process that needs both keeps a single ~440 MB model
    copy; the embedding path uses the base-model view of this classifier.
    """
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "ProsusAI/finbert"
    logger.info("Loading FinBERT classifier: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if hasattr(model, "eval"):
        model.eval()
    logger.info("FinBERT classifier loaded")
    return tokenizer, model


@lru_cache(maxsize=1)
def _load_finbert_encoder():
    """Bare encoder view of the shared FinBERT classifier (no second copy)."""
    tokenizer, classifier = load_finbert_classifier()
    return tokenizer, classifier.base_model


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
    with torch.inference_mode():
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
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

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
        with torch.inference_mode():
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
    import warnings
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
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
    """
    Unpack binary blob back to a float32 array of length ``dim``.

    The DB may contain vectors stored with a different PCA dimension
    (e.g. old 32-dim rows mixed with new 8-dim rows after a config
    change).  We normalise every vector to ``dim`` elements:
      - If the stored vector is longer  → truncate to the first ``dim`` values.
      - If the stored vector is shorter → zero-pad to ``dim`` values.
    This prevents ``np.stack`` from raising "all arrays must have the
    same shape" when multiple vectors are aggregated for the same day.
    """
    arr = np.frombuffer(data, dtype=np.float32).copy()
    if len(arr) == dim:
        return arr
    if len(arr) > dim:
        return arr[:dim]
    # shorter than expected — zero-pad
    padded = np.zeros(dim, dtype=np.float32)
    padded[: len(arr)] = arr
    return padded


# ---------------------------------------------------------------------------
# DB backfill logic
# ---------------------------------------------------------------------------

def _insert_embedding_batch(
    session,
    payloads: Sequence[dict[str, object]],
) -> tuple[int, int]:
    """Insert a batch without per-row existence queries.

    ``news_processed_id`` is unique, so PostgreSQL and SQLite can safely
    ignore a concurrent duplicate at the database boundary. The fallback
    keeps the function usable with other SQLAlchemy dialects while still
    reducing existence checks to one query per batch.

    Returns:
        ``(inserted_count, skipped_count)`` for the attempted payloads.
    """
    if not payloads:
        return 0, 0

    from app.models import NewsEmbedding

    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as dialect_insert
    else:
        article_ids = [int(payload["news_processed_id"]) for payload in payloads]
        existing_ids = {
            int(row[0])
            for row in session.query(NewsEmbedding.news_processed_id)
            .filter(NewsEmbedding.news_processed_id.in_(article_ids))
            .all()
        }
        pending = [
            NewsEmbedding(**payload)
            for payload in payloads
            if int(payload["news_processed_id"]) not in existing_ids
        ]
        session.add_all(pending)
        return len(pending), len(payloads) - len(pending)

    statement = (
        dialect_insert(NewsEmbedding)
        .values(list(payloads))
        .on_conflict_do_nothing(
            index_elements=[NewsEmbedding.news_processed_id]
        )
    )
    try:
        result = session.execute(statement.returning(NewsEmbedding.news_processed_id))
        inserted = len(result.scalars().all())
    except Exception:
        result = session.execute(statement)
        inserted = max(0, int(result.rowcount or 0))
    return inserted, len(payloads) - inserted


def backfill_embeddings(
    *,
    days: int = 180,
    pca_dim: int = 8,
    batch_size: int = 64,
) -> dict[str, int]:
    """
    Compute and store FinBERT embeddings for all unprocessed news articles.

    Steps:
        1. Query news_processed rows that lack a corresponding news_embeddings row.
        2. Extract CLS embeddings in batches.
        3. Fit or load PCA model.
        4. Store reduced embeddings in race-safe batches.
    """
    from app.db import SessionLocal
    from app.models import NewsEmbedding, NewsProcessed, NewsRaw
    from deep_learning.config import get_tft_config

    cfg = get_tft_config()
    pca_path = Path(cfg.embedding.pca_model_path)
    target_pca_version = f"pca{pca_dim}_v1"

    stats = {"embedded": 0, "skipped": 0, "pca_fitted": False, "reprojected": 0}

    with SessionLocal() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        rows = (
            session.query(
                NewsProcessed.id,
                NewsProcessed.cleaned_text,
                NewsRaw.title,
                NewsRaw.description,
            )
            .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
            .outerjoin(
                NewsEmbedding,
                NewsEmbedding.news_processed_id == NewsProcessed.id,
            )
            .filter(NewsRaw.published_at >= cutoff)
            .filter(NewsEmbedding.id.is_(None))
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

        if not pca_path.exists():
            repo_id = getattr(getattr(cfg, "training", None), "hf_model_repo", None)
            if repo_id:
                try:
                    from deep_learning.models.hub import download_pca_artifact

                    download_pca_artifact(pca_path.parent, repo_id)
                except Exception as exc:
                    logger.debug("Optional Hub PCA fetch skipped: %s", exc)

        pca = load_pca(str(pca_path)) if pca_path.exists() else None
        existing_components = (
            int(getattr(pca, "n_components_", getattr(pca, "n_components", 0)) or 0)
            if pca is not None
            else 0
        )
        existing_samples = int(getattr(pca, "n_samples_seen_", 0)) if pca is not None else 0
        if (
            pca is not None
            and (
                (existing_components == pca_dim and existing_samples > pca_dim)
                # Older serialized/custom PCA implementations may not expose
                # sklearn's fitted-state metadata; preserve compatibility and
                # validate the transformed shape below.
                or existing_components == 0
            )
        ):
            logger.info("Loaded existing PCA model from %s", pca_path)
        else:
            hist_rows = (
                session.query(NewsEmbedding.embedding_full)
                .filter(NewsEmbedding.embedding_full.isnot(None))
                .order_by(NewsEmbedding.id.desc())
                .limit(2000)
                .all()
            )
            hist_vecs = []
            for h_row in hist_rows:
                raw_blob = h_row[0] if isinstance(h_row, tuple) else getattr(h_row, "embedding_full", None)
                if not raw_blob:
                    continue
                arr = np.frombuffer(raw_blob, dtype=np.float32).copy()
                if len(arr) == _EMBEDDING_DIM and np.all(np.isfinite(arr)):
                    hist_vecs.append(arr)

            fit_matrix = (
                np.vstack([np.stack(hist_vecs), full_embeddings])
                if hist_vecs
                else full_embeddings
            )

            # IncrementalPCA fitted with n_samples <= n_components either raises
            # or emits an empty-slice warning / produces fewer than pca_dim axes.
            if len(fit_matrix) <= pca_dim:
                logger.info(
                    "Deferring PCA fit: need more than %d samples for %d components; have %d",
                    pca_dim,
                    pca_dim,
                    len(fit_matrix),
                )
                stats["pca_deferred"] = True
                return stats
            pca = fit_pca(fit_matrix, n_components=pca_dim, save_path=str(pca_path))
            stats["pca_fitted"] = True

            # Align legacy/mismatched PCA rows onto the newly fitted orthonormal basis.
            mismatch_rows = (
                session.query(NewsEmbedding.id, NewsEmbedding.embedding_full)
                .filter(NewsEmbedding.embedding_full.isnot(None))
                .filter(
                    NewsEmbedding.pca_version.is_(None)
                    | (NewsEmbedding.pca_version != target_pca_version)
                )
                .order_by(NewsEmbedding.id.asc())
                .all()
            )
            reproj_ids: list[int] = []
            reproj_vecs: list[np.ndarray] = []
            for m_row in mismatch_rows:
                row_id = int(m_row[0] if isinstance(m_row, tuple) else m_row.id)
                raw_blob = m_row[1] if isinstance(m_row, tuple) else m_row.embedding_full
                if not raw_blob:
                    continue
                arr = np.frombuffer(raw_blob, dtype=np.float32).copy()
                if len(arr) == _EMBEDDING_DIM and np.all(np.isfinite(arr)):
                    reproj_ids.append(row_id)
                    reproj_vecs.append(arr)

            if reproj_vecs:
                reproj_reduced = reduce_embeddings(np.stack(reproj_vecs), pca)
                updates = [
                    {
                        "id": row_id,
                        "embedding_pca": embedding_to_bytes(reproj_reduced[idx]),
                        "pca_version": target_pca_version,
                    }
                    for idx, row_id in enumerate(reproj_ids)
                ]
                for start in range(0, len(updates), 200):
                    session.bulk_update_mappings(NewsEmbedding, updates[start : start + 200])
                    session.commit()
                stats["reprojected"] = len(updates)
                logger.info(
                    "Reprojected %d existing embeddings to %s",
                    len(updates),
                    target_pca_version,
                )

        reduced = reduce_embeddings(full_embeddings, pca)

        payloads = [
            {
                "news_processed_id": article_id,
                "embedding_full": embedding_to_bytes(full_embeddings[idx]),
                "embedding_pca": embedding_to_bytes(reduced[idx]),
                "pca_version": target_pca_version,
            }
            for idx, article_id in enumerate(ids)
        ]

        # Keep transactions bounded while reducing the previous N existence
        # queries to one database-level upsert per persistence batch.
        persist_batch_size = 200
        for start in range(0, len(payloads), persist_batch_size):
            batch = payloads[start : start + persist_batch_size]
            inserted, skipped = _insert_embedding_batch(session, batch)
            stats["embedded"] += inserted
            stats["skipped"] += skipped
            session.commit()
            logger.info(
                "Committed embedding batch: %d inserted, %d skipped, %d total",
                inserted,
                skipped,
                stats["embedded"],
            )

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
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.backfill:
        from app.db import init_db
        init_db()
        backfill_embeddings(days=args.days, pca_dim=args.pca_dim, batch_size=args.batch_size)
