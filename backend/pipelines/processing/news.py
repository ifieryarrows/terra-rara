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
    
    # Compute dedup key
    dedup = compute_dedup_key(
        url_hash=raw.url_hash,
        source=raw.source or "unknown",
        canonical_title_hash=canon_hash,
    )
    
    # Detect language (optional, use simple heuristic for now)
    # Full langdetect is slow; Faz 3 can improve this
    language = "en"  # Assume English for now
    language_confidence = None
    
    try:
        db_type = get_db_type()
        
        if db_type == "postgresql":
            stmt = pg_insert(NewsProcessed).values(
                raw_id=raw.id,
                canonical_title=canon[:500],
                canonical_title_hash=canon_hash,
                cleaned_text=cleaned,
                dedup_key=dedup,
                language=language,
                language_confidence=language_confidence,
                run_id=run_id,
            ).on_conflict_do_nothing(index_elements=["dedup_key"])
            
            result = session.execute(stmt)
            
            if result.rowcount > 0:
                return raw.id  # Successfully inserted
            return None  # Duplicate
            
        else:
            # SQLite fallback
            processed = NewsProcessed(
                raw_id=raw.id,
                canonical_title=canon[:500],
                canonical_title_hash=canon_hash,
                cleaned_text=cleaned,
                dedup_key=dedup,
                language=language,
                language_confidence=language_confidence,
                run_id=run_id,
            )
            session.add(processed)
            session.flush()
            return processed.id
            
    except Exception as e:
        logger.debug(f"Process raw article failed: {e}")
        session.rollback()
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
    
    # Process in batches
    offset = 0
    while True:
        batch = unprocessed_query.limit(batch_size).offset(offset).all()
        
        if not batch:
            break
        
        for raw in batch:
            stats["processed"] += 1
            
            result = process_single_raw(session, raw, run_id)
            
            if result:
                stats["inserted"] += 1
            else:
                stats["duplicates"] += 1
        
        session.commit()
        offset += batch_size
        
        if offset >= total:
            break
    
    logger.info(
        f"[run_id={run_id}] Processing complete: "
        f"{stats['processed']} processed, {stats['inserted']} inserted, "
        f"{stats['duplicates']} duplicates"
    )
    
    return stats
