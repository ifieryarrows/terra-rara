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

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import NewsRaw, NewsProcessed
from app.utils import canonical_title, clean_text
from app.data_manager import detect_language
from app.db import get_db_type

logger = logging.getLogger(__name__)


def compute_canonical_title_hash(title: str) -> str:
    """
    Compute hash of canonical title.
    
    Args:
        title: Raw title string
        
    Returns:
        sha256 hex64 of canonical_title(title)
    """
    canon = canonical_title(title)
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


def compute_content_dedup_key(title: str, publisher: Optional[str], published_at: datetime) -> str:
    """Content identity that is stable across Google wrapper/query URLs."""
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    publication_date = published_at.astimezone(timezone.utc).date().isoformat()
    identity = "|".join(
        (
            canonical_title(title),
            (normalize_publisher(publisher) or "unknown").casefold(),
            publication_date,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def process_single_raw(
    session: Session,
    raw: NewsRaw,
    run_id: uuid.UUID,
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
    # Canonicalize
    canon = canonical_title(raw.title)
    canon_hash = compute_canonical_title_hash(raw.title)
    
    # Clean text (title + description)
    cleaned = clean_text(raw.title)
    if raw.description:
        cleaned += " " + clean_text(raw.description)
    cleaned = cleaned[:5000]  # Reasonable limit
    
    publisher = normalize_publisher(raw.publisher)
    dedup = compute_content_dedup_key(raw.title, publisher, raw.published_at)
    
    language = detect_language(cleaned)
    language_confidence = None
    
    try:
        # Isolate a bad/contended article without rolling back the other rows
        # already processed in this batch.
        with session.begin_nested():
            raw.publisher = publisher
            canonical = (
                session.query(NewsProcessed)
                .filter(
                    NewsProcessed.dedup_key == dedup,
                    NewsProcessed.duplicate_of_id.is_(None),
                )
                .first()
            )
            duplicate_of_id = canonical.id if canonical is not None else None
            row_key = dedup if canonical is None else hashlib.sha256(
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
    
    # The query shrinks after every commit. Always consume its first page;
    # offset pagination here used to skip every second batch.
    while True:
        batch = unprocessed_query.limit(batch_size).all()
        
        if not batch:
            break
        
        for raw in batch:
            stats["processed"] += 1
            
            result = process_single_raw(session, raw, run_id)
            
            if result:
                processed = session.get(NewsProcessed, result)
                if processed is not None and processed.duplicate_of_id is not None:
                    stats["duplicates"] += 1
                else:
                    stats["inserted"] += 1
            else:
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
    duplicate_updates = 0
    publisher_updates = 0
    for processed, raw in rows:
        publisher = normalize_publisher(raw.publisher)
        if publisher is None and isinstance(raw.raw_payload, dict):
            source = raw.raw_payload.get("source")
            publisher = normalize_publisher(source.get("name") if isinstance(source, dict) else source)
        key = compute_content_dedup_key(raw.title, publisher, raw.published_at)
        canonical_id = canonical_by_key.setdefault(key, processed.id)
        desired_duplicate = None if canonical_id == processed.id else canonical_id
        if processed.duplicate_of_id != desired_duplicate or processed.dedup_version != "content_v2":
            duplicate_updates += 1
            if not dry_run:
                processed.duplicate_of_id = desired_duplicate
                processed.dedup_version = "content_v2"
        if raw.publisher != publisher:
            publisher_updates += 1
            if not dry_run:
                raw.publisher = publisher
    if not dry_run:
        session.commit()
    return {
        "rows_scanned": len(rows),
        "duplicate_updates": duplicate_updates,
        "publisher_updates": publisher_updates,
        "dry_run": dry_run,
    }
