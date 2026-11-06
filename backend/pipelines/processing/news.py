"""
Process news_raw -> news_processed with deterministic dedup.

Faz 2: Reproducible news pipeline - second stage.
Applies canonicalization, language detection, and deterministic dedup.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import case, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import NewsRaw, NewsProcessed
from app.utils import canonical_title, canonical_title_batch, clean_text
from app.data_manager import detect_language
from app.db import get_db_type

logger = logging.getLogger(__name__)


def compute_canonical_title_hash(title: str, *, canonical_title_str: Optional[str] = None) -> str:
    """
    Compute hash of canonical title.
    
    Args:
        title: Raw title string
        canonical_title_str: Optional precomputed canonical title string
        
    Returns:
        sha256 hex64 of canonical title
    """
    canon = canonical_title_str if canonical_title_str is not None else canonical_title(title)
    return hashlib.sha256(canon.encode()).hexdigest()


def compute_dedup_key(
    url_hash: Optional[str],
    source: str,
    canonical_title_hash: str,
) -> str:
    """
    Compute deterministic dedup key.
    
    Priority:
        1. url_hash if not None (URL is the best identifier)
        2. sha256(source + "|" + canonical_title_hash) as fallback
    
    Args:
        url_hash: Hash of normalized URL (may be None)
        source: Source name (e.g., "google_news")
        canonical_title_hash: Hash of canonical title
        
    Returns:
        sha256 hex64 dedup key
    """
    if url_hash:
        return url_hash
    
    # Fallback: combine source + title hash
    fallback_input = f"{source}|{canonical_title_hash}"
    return hashlib.sha256(fallback_input.encode()).hexdigest()


def normalize_publisher(value: Optional[str]) -> Optional[str]:
    """Normalize whitespace while retaining a user-facing publisher name."""
    normalized = " ".join(str(value or "").split()).strip()
    return normalized[:300] or None


def compute_content_dedup_key(
    title: str,
    publisher: Optional[str],
    published_at: datetime,
    *,
    canonical_title_str: Optional[str] = None,
) -> str:
    """Content identity that is stable across Google wrapper/query URLs."""
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    publication_date = published_at.astimezone(timezone.utc).date().isoformat()
    c_title = canonical_title_str if canonical_title_str is not None else canonical_title(title)
    identity = "|".join(
        (
            c_title,
            (normalize_publisher(publisher) or "unknown").casefold(),
            publication_date,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def process_single_raw(
    session: Session,
    raw: NewsRaw,
    run_id: uuid.UUID,
    *,
    canonical_cache: Optional[dict[str, int]] = None,
    _is_duplicate_out: Optional[list[bool]] = None,
) -> Optional[int]:
    """
    Process a single NewsRaw into NewsProcessed.
    
    Args:
        session: Database session
        raw: NewsRaw object to process
        run_id: Pipeline run UUID
        
    Returns:
        processed_id if inserted, None if duplicate
    """
    # Canonicalize once
    canon = canonical_title(raw.title)
    canon_hash = compute_canonical_title_hash(raw.title, canonical_title_str=canon)
    
    # Clean text (title + description)
    cleaned = clean_text(raw.title)
    if raw.description:
        cleaned += " " + clean_text(raw.description)
    cleaned = cleaned[:5000]  # Reasonable limit
    
    publisher = normalize_publisher(raw.publisher)
    dedup = compute_content_dedup_key(raw.title, publisher, raw.published_at, canonical_title_str=canon)
    
    language = detect_language(cleaned)
    language_confidence = None
    
    try:
        # Isolate a bad/contended article without rolling back the other rows
        # already processed in this batch.
        with session.begin_nested():
            raw.publisher = publisher
            if canonical_cache is not None and dedup in canonical_cache:
                duplicate_of_id = canonical_cache[dedup]
            else:
                canonical = (
                    session.query(NewsProcessed)
                    .filter(
                        NewsProcessed.dedup_key == dedup,
                        NewsProcessed.duplicate_of_id.is_(None),
                    )
                    .first()
                )
                duplicate_of_id = canonical.id if canonical is not None else None
                if canonical_cache is not None and duplicate_of_id is not None:
                    canonical_cache[dedup] = duplicate_of_id

            row_key = dedup if duplicate_of_id is None else hashlib.sha256(
                f"duplicate|{raw.id}|{dedup}".encode("utf-8")
            ).hexdigest()
            processed = NewsProcessed(
                raw_id=raw.id,
                canonical_title=canon[:500],
                canonical_title_hash=canon_hash,
                cleaned_text=cleaned,
                dedup_key=row_key,
                dedup_version="content_v2",
                duplicate_of_id=duplicate_of_id,
                language=language,
                language_confidence=language_confidence,
                run_id=run_id,
            )
            session.add(processed)
            session.flush()
            processed_id = processed.id
            if canonical_cache is not None and duplicate_of_id is None and processed_id is not None:
                canonical_cache[dedup] = processed_id
            if _is_duplicate_out is not None:
                _is_duplicate_out.append(duplicate_of_id is not None)
        return processed_id
            
    except Exception as e:
        logger.debug(f"Process raw article failed: {e}")
        return None


def process_raw_to_processed(
    session: Session,
    run_id: uuid.UUID,
    batch_size: int = 100,
) -> dict:
    """
    Process unprocessed raw articles.
    
    Finds news_raw records that don't have corresponding news_processed,
    canonicalizes them, and inserts to news_processed with dedup.
    
    Args:
        session: Database session
        run_id: Pipeline run UUID
        batch_size: Number of records to process per batch
        
    Returns:
        dict with stats:
            - processed: Total items attempted
            - inserted: New items in news_processed
            - duplicates: Skipped due to dedup_key conflict
    """
    stats = {
        "processed": 0,
        "inserted": 0,
        "duplicates": 0,
    }
    
    # Find unprocessed raw articles
    # LEFT JOIN to find raw records without processed counterparts
    unprocessed_query = (
        session.query(NewsRaw)
        .outerjoin(NewsProcessed, NewsRaw.id == NewsProcessed.raw_id)
        .filter(NewsProcessed.id.is_(None))
        .order_by(NewsRaw.id)
    )
    
    total = unprocessed_query.count()
    logger.info(f"[run_id={run_id}] Found {total} unprocessed raw articles")
    
    if total == 0:
        return stats
    
    canonical_cache: dict[str, int] = {}
    failed_raw_ids: set[int] = set()
    previous_batch_ids: Optional[list[int]] = None

    # The query shrinks after every commit. Always consume its first page;
    # offset pagination here used to skip every second batch.
    while True:
        query = unprocessed_query
        if failed_raw_ids:
            query = query.filter(~NewsRaw.id.in_(failed_raw_ids))
        batch = query.limit(batch_size).all()
        
        if not batch:
            break

        batch_ids = [r.id for r in batch]
        if previous_batch_ids is not None and batch_ids == previous_batch_ids:
            logger.error("No progress made processing raw articles batch; terminating loop to avoid hang.")
            break
        previous_batch_ids = batch_ids

        # Batch-canonicalize titles using fast Rayon-parallel Rust kernel
        batch_canons = canonical_title_batch([r.title for r in batch])
        batch_canon_hashes = [
            compute_canonical_title_hash(r.title, canonical_title_str=c)
            for r, c in zip(batch, batch_canons)
        ]
        batch_publishers = [normalize_publisher(r.publisher) for r in batch]
        batch_dedup_keys = [
            compute_content_dedup_key(
                raw.title,
                pub,
                raw.published_at,
                canonical_title_str=canon,
            )
            for raw, pub, canon in zip(batch, batch_publishers, batch_canons)
        ]

        # Pre-populate canonical_cache for this batch in a single SQL query
        if isinstance(session, Session):
            missing_keys = [k for k in set(batch_dedup_keys) if k not in canonical_cache]
            if missing_keys:
                existing_canonicals = (
                    session.query(NewsProcessed.dedup_key, NewsProcessed.id)
                    .filter(
                        NewsProcessed.dedup_key.in_(missing_keys),
                        NewsProcessed.duplicate_of_id.is_(None),
                    )
                    .all()
                )
                for d_key, p_id in existing_canonicals:
                    canonical_cache[d_key] = p_id
        
        # Batch accumulation: avoid per-row session.begin_nested() and flush()
        new_canonicals: list[tuple[NewsRaw, NewsProcessed, str]] = []
        known_duplicates: list[NewsProcessed] = []
        intra_batch_duplicates: list[tuple[NewsRaw, str, str, str, str, Optional[str], Optional[float]]] = []
        batch_canonical_keys: dict[str, NewsProcessed] = {}

        for raw, publisher, canon, canon_hash, dedup in zip(
            batch, batch_publishers, batch_canons, batch_canon_hashes, batch_dedup_keys
        ):
            stats["processed"] += 1
            raw.publisher = publisher

            cleaned = clean_text(raw.title)
            if raw.description:
                cleaned += " " + clean_text(raw.description)
            cleaned = cleaned[:5000]

            language = detect_language(cleaned)
            language_confidence = None

            if dedup in canonical_cache:
                duplicate_of_id = canonical_cache[dedup]
                row_key = hashlib.sha256(f"duplicate|{raw.id}|{dedup}".encode("utf-8")).hexdigest()
                proc = NewsProcessed(
                    raw_id=raw.id,
                    canonical_title=canon[:500],
                    canonical_title_hash=canon_hash,
                    cleaned_text=cleaned,
                    dedup_key=row_key,
                    dedup_version="content_v2",
                    duplicate_of_id=duplicate_of_id,
                    language=language,
                    language_confidence=language_confidence,
                    run_id=run_id,
                )
                known_duplicates.append(proc)
                stats["duplicates"] += 1
            elif dedup in batch_canonical_keys:
                intra_batch_duplicates.append(
                    (raw, canon, canon_hash, cleaned, dedup, language, language_confidence)
                )
                stats["duplicates"] += 1
            else:
                proc = NewsProcessed(
                    raw_id=raw.id,
                    canonical_title=canon[:500],
                    canonical_title_hash=canon_hash,
                    cleaned_text=cleaned,
                    dedup_key=dedup,
                    dedup_version="content_v2",
                    duplicate_of_id=None,
                    language=language,
                    language_confidence=language_confidence,
                    run_id=run_id,
                )
                new_canonicals.append((raw, proc, dedup))
                batch_canonical_keys[dedup] = proc
                stats["inserted"] += 1

        cache_snapshot = dict(canonical_cache)
        try:
            for _, proc, _ in new_canonicals:
                session.add(proc)
            for proc in known_duplicates:
                session.add(proc)

            session.flush()

            for _, proc, dedup in new_canonicals:
                canonical_cache[dedup] = proc.id

            if intra_batch_duplicates:
                for raw, canon, canon_hash, cleaned, dedup, lang, conf in intra_batch_duplicates:
                    canonical_id = canonical_cache.get(dedup)
                    row_key = hashlib.sha256(f"duplicate|{raw.id}|{dedup}".encode("utf-8")).hexdigest()
                    proc = NewsProcessed(
                        raw_id=raw.id,
                        canonical_title=canon[:500],
                        canonical_title_hash=canon_hash,
                        cleaned_text=cleaned,
                        dedup_key=row_key,
                        dedup_version="content_v2",
                        duplicate_of_id=canonical_id,
                        language=lang,
                        language_confidence=conf,
                        run_id=run_id,
                    )
                    session.add(proc)
                session.flush()

            session.commit()
        except Exception as e:
            logger.warning(f"Batch processing flush failed, falling back to per-row: {e}")
            session.rollback()
            canonical_cache.clear()
            canonical_cache.update(cache_snapshot)
            # Rollback stats increment for this batch and run row-by-row fallback
            stats["processed"] -= len(batch)
            stats["inserted"] -= len(new_canonicals)
            stats["duplicates"] -= (len(known_duplicates) + len(intra_batch_duplicates))
            for raw in batch:
                stats["processed"] += 1
                dup_flag = []
                result = process_single_raw(
                    session,
                    raw,
                    run_id,
                    canonical_cache=canonical_cache if isinstance(session, Session) else None,
                    _is_duplicate_out=dup_flag,
                )
                if result is not None:
                    if dup_flag and dup_flag[0]:
                        stats["duplicates"] += 1
                    else:
                        stats["inserted"] += 1
                else:
                    failed_raw_ids.add(raw.id)
                    stats["duplicates"] += 1
            session.commit()
    
    logger.info(
        f"[run_id={run_id}] Processing complete: "
        f"{stats['processed']} processed, {stats['inserted']} inserted, "
        f"{stats['duplicates']} duplicates"
    )
    
    return stats


def backfill_content_dedup(session: Session, *, dry_run: bool = True) -> dict:
    """Idempotently mark historical duplicates without deleting audit rows."""
    rows = (
        session.query(NewsProcessed, NewsRaw)
        .join(NewsRaw, NewsRaw.id == NewsProcessed.raw_id)
        .order_by(NewsRaw.published_at.asc(), NewsProcessed.id.asc())
        .all()
    )
    canonical_by_key: dict[str, int] = {}
    duplicate_relationship_updates = 0
    version_updates = 0
    publisher_updates = 0
    processed_updates: list[dict] = []
    raw_updates: list[dict] = []
    for processed, raw in rows:
        publisher = normalize_publisher(raw.publisher)
        if publisher is None and isinstance(raw.raw_payload, dict):
            source = raw.raw_payload.get("source")
            publisher = normalize_publisher(source.get("name") if isinstance(source, dict) else source)
        key = compute_content_dedup_key(raw.title, publisher, raw.published_at)
        canonical_id = canonical_by_key.setdefault(key, processed.id)
        desired_duplicate = None if canonical_id == processed.id else canonical_id
        duplicate_changed = processed.duplicate_of_id != desired_duplicate
        version_changed = processed.dedup_version != "content_v2"
        if duplicate_changed:
            duplicate_relationship_updates += 1
        if version_changed:
            version_updates += 1
        if not dry_run and (duplicate_changed or version_changed):
            # Keep every mapping the same shape so SQLAlchemy can send one
            # executemany batch instead of flushing thousands of individual
            # UPDATE statements over the production DB connection.
            processed_updates.append(
                {
                    "id": processed.id,
                    "duplicate_of_id": desired_duplicate,
                    "dedup_version": "content_v2",
                }
            )
        if raw.publisher != publisher:
            publisher_updates += 1
            if not dry_run:
                raw_updates.append({"id": raw.id, "publisher": publisher})
    if not dry_run:
        # UPDATE executemany is still serialized by several PostgreSQL drivers.
        # CASE-based chunks keep the same transaction semantics while reducing
        # a historical backfill from tens of thousands of round trips to a few
        # bounded statements.
        chunk_size = 500
        for start in range(0, len(raw_updates), chunk_size):
            chunk = raw_updates[start:start + chunk_size]
            ids = [item["id"] for item in chunk]
            publisher_by_id = {item["id"]: item["publisher"] for item in chunk}
            session.execute(
                update(NewsRaw)
                .where(NewsRaw.id.in_(ids))
                .values(publisher=case(publisher_by_id, value=NewsRaw.id))
            )
        for start in range(0, len(processed_updates), chunk_size):
            chunk = processed_updates[start:start + chunk_size]
            ids = [item["id"] for item in chunk]
            duplicate_by_id = {item["id"]: item["duplicate_of_id"] for item in chunk}
            session.execute(
                update(NewsProcessed)
                .where(NewsProcessed.id.in_(ids))
                .values(
                    duplicate_of_id=case(duplicate_by_id, value=NewsProcessed.id),
                    dedup_version="content_v2",
                )
            )
        session.commit()
    return {
        "rows_scanned": len(rows),
        "duplicate_updates": duplicate_relationship_updates,
        "dedup_version_updates": version_updates,
        "publisher_updates": publisher_updates,
        "dry_run": dry_run,
    }
