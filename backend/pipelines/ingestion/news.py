"""
News ingestion to news_raw table.

Faz 2: Reproducible news pipeline - first stage.
Fetches from RSS/API and stores raw data for audit trail.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.models import NewsRaw
from app.settings import get_settings
from app.utils import normalize_url, clean_text
from app.rss_ingest import fetch_google_news, fetch_google_news_concurrent
from app.db import get_db_type

logger = logging.getLogger(__name__)


def normalize_publisher(value) -> Optional[str]:
    """Normalize provider-specific publisher shapes into one DB field."""
    if isinstance(value, dict):
        value = value.get("name") or value.get("title")
    normalized = " ".join(str(value or "").split()).strip()
    return normalized[:300] or None


def compute_url_hash(url: Optional[str]) -> Optional[str]:
    """
    Compute deterministic hash of normalized URL.
    
    Args:
        url: Raw URL string (may be None or empty)
        
    Returns:
        sha256 hex64 of normalized URL, or None if URL is empty/invalid
    """
    if not url or not url.strip():
        return None
    
    normalized = normalize_url(url)
    if not normalized:
        return None
    
    return hashlib.sha256(normalized.encode()).hexdigest()


def insert_raw_article(
    session: Session,
    url: Optional[str],
    title: Optional[str],
    description: Optional[str],
    source: str,
    source_feed: str,
    published_at: Optional[datetime],
    run_id: uuid.UUID,
    raw_payload: Optional[dict] = None,
    publisher: Optional[str] = None,
) -> tuple[Optional[int], str]:
    """
    Insert single article to news_raw.
    
    Uses ON CONFLICT DO NOTHING for url_hash to handle duplicates gracefully.
    
    Args:
        session: Database session
        url: Article URL (can be None)
        title: Article title (will use fallback if empty)
        description: Article description
        source: Source name (e.g., "google_news", "newsapi")
        source_feed: Exact feed URL or query string
        published_at: Publication timestamp (will use now if None)
        run_id: Pipeline run UUID
        raw_payload: Original response fragment for debugging
        
    Returns:
        Tuple of (raw_id, status) where status is 'inserted', 'duplicate', or 'error'
    """
    fetched_at = datetime.now(timezone.utc)
    
    # Ensure title is never None/empty (NOT NULL constraint)
    if not title or not title.strip():
        title = "(untitled)"
    title = clean_text(title)[:500]  # Truncate to column limit
    
    # Ensure published_at is never None (NOT NULL constraint)
    if not published_at:
        published_at = fetched_at
    
    url_hash = compute_url_hash(url)
    publisher = normalize_publisher(publisher)
    
    try:
        db_type = get_db_type(session)
        
        if db_type == "postgresql":
            # Use INSERT ... ON CONFLICT RETURNING id for PostgreSQL in one round trip
            stmt = pg_insert(NewsRaw).values(
                url=url,
                url_hash=url_hash,
                title=title,
                description=description[:2000] if description else None,
                source=source,
                publisher=publisher,
                source_feed=source_feed[:500] if source_feed else None,
                published_at=published_at,
                fetched_at=fetched_at,
                run_id=run_id,
                raw_payload=raw_payload,
            )
            
            # Only conflict on url_hash if it's not None
            if url_hash:
                stmt = stmt.on_conflict_do_nothing(index_elements=["url_hash"])
            
            stmt = stmt.returning(NewsRaw.id)
            result = session.execute(stmt)
            row = result.fetchone()
            if row:
                return (row[0], "inserted")
            return (None, "duplicate")
            
        else:
            # SQLite fallback: pre-check url_hash for duplicate then flush
            if url_hash:
                existing = session.query(NewsRaw.id).filter(NewsRaw.url_hash == url_hash).first()
                if existing is not None:
                    return (None, "duplicate")
            article = NewsRaw(
                url=url,
                url_hash=url_hash,
                title=title,
                description=description[:2000] if description else None,
                source=source,
                publisher=publisher,
                source_feed=source_feed[:500] if source_feed else None,
                published_at=published_at,
                fetched_at=fetched_at,
                run_id=run_id,
                raw_payload=raw_payload,
            )
            session.add(article)
            session.flush()
            return (article.id, "inserted")
            
    except Exception as e:
        # Distinguish between duplicate (23505) and other errors
        pgcode = getattr(getattr(e, "orig", None), "pgcode", None)
        err_msg = str(e).lower()
        if pgcode == "23505" or "unique constraint" in err_msg:
            return (None, "duplicate")
        logger.warning(f"Insert raw article failed (pgcode={pgcode}): {e}")
        session.rollback()
        return (None, "error")


def insert_raw_articles_batch(
    session: Session,
    articles: list[dict],
    run_id: uuid.UUID,
    default_source: str = "google_news",
    default_source_feed: Optional[str] = None,
) -> dict[str, int]:
    """
    Batch insert raw articles into news_raw in a single database round trip.
    
    Deduplicates intra-batch url_hashes in memory and uses
    INSERT ... ON CONFLICT (url_hash) DO NOTHING RETURNING id.
    
    Args:
        session: Database session
        articles: List of article dicts
        run_id: Pipeline run UUID
        default_source: Fallback source name
        default_source_feed: Fallback source feed
        
    Returns:
        Dict with keys: inserted, duplicates, errors
    """
    stats = {"inserted": 0, "duplicates": 0, "errors": 0}
    if not articles:
        return stats
        
    fetched_at = datetime.now(timezone.utc)
    seen_hashes: set[str] = set()
    to_insert: list[dict] = []
    
    for art in articles:
        url = art.get("url")
        title = art.get("title")
        if not title or not str(title).strip():
            title = "(untitled)"
        title = clean_text(str(title))[:500]
        
        published_at = art.get("published_at") or fetched_at
        url_hash = compute_url_hash(url)
        publisher = normalize_publisher(
            art.get("publisher") or art.get("source") or art.get("author")
        )
        source = art.get("source_name") or art.get("source") or default_source
        source_feed = art.get("source_feed") or default_source_feed
        raw_payload = art.get("raw_payload")
        description = art.get("description")
        
        if url_hash is not None:
            if url_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(url_hash)
            
        to_insert.append({
            "url": url,
            "url_hash": url_hash,
            "title": title,
            "description": description[:2000] if description else None,
            "source": source,
            "publisher": publisher,
            "source_feed": source_feed[:500] if source_feed else None,
            "published_at": published_at,
            "fetched_at": fetched_at,
            "run_id": run_id,
            "raw_payload": raw_payload,
        })
        
    if not to_insert:
        return stats

    try:
        db_type = get_db_type(session)
        if db_type == "postgresql":
            stmt = pg_insert(NewsRaw).values(to_insert)
            stmt = stmt.on_conflict_do_nothing(index_elements=["url_hash"])
            stmt = stmt.returning(NewsRaw.id)
            result = session.execute(stmt)
            inserted_rows = result.fetchall()
            inserted_count = len(inserted_rows)
            stats["inserted"] += inserted_count
            stats["duplicates"] += len(to_insert) - inserted_count
        else:
            # SQLite batch: query existing hashes in one IN query and insert remainder
            batch_hashes = [d["url_hash"] for d in to_insert if d.get("url_hash")]
            existing_hashes: set[str] = set()
            if batch_hashes:
                rows = session.query(NewsRaw.url_hash).filter(NewsRaw.url_hash.in_(batch_hashes)).all()
                existing_hashes = {r[0] for r in rows if r[0]}
            new_objects: list[NewsRaw] = []
            for d in to_insert:
                h = d.get("url_hash")
                if h and h in existing_hashes:
                    stats["duplicates"] += 1
                else:
                    new_objects.append(NewsRaw(**d))
                    if h:
                        existing_hashes.add(h)
            if new_objects:
                session.add_all(new_objects)
                session.flush()
                stats["inserted"] += len(new_objects)
    except Exception as e:
        logger.warning(f"Batch raw articles insert failed, falling back to per-row insertion: {e}")
        session.rollback()
        for item in to_insert:
            _, status = insert_raw_article(
                session=session,
                url=item["url"],
                title=item["title"],
                description=item["description"],
                source=item["source"],
                source_feed=item["source_feed"],
                published_at=item["published_at"],
                run_id=run_id,
                raw_payload=item["raw_payload"],
                publisher=item["publisher"],
            )
            if status == "inserted":
                stats["inserted"] += 1
            elif status == "duplicate":
                stats["duplicates"] += 1
            else:
                stats["errors"] += 1

    return stats


def ingest_news_to_raw(
    session: Session,
    run_id: uuid.UUID,
    sources: Optional[list[str]] = None,
) -> dict:
    """
    Ingest news from all sources into news_raw.
    
    Currently supports:
    - google_news: RSS feed from Google News
    - newsapi: NewsAPI.org (if API key configured)
    
    Args:
        session: Database session
        run_id: Pipeline run UUID
        sources: List of source types to fetch (default: all configured)
        
    Returns:
        dict with stats:
            - fetched: Total items fetched from sources
            - inserted: New items inserted to news_raw
            - duplicates: Skipped due to url_hash conflict
            - errors: Items that failed to insert
    """
    settings = get_settings()
    sources = sources or ["google_news"]
    
    # Add newsapi if key is configured
    if settings.newsapi_key and "newsapi" not in sources:
        sources.append("newsapi")
    
    stats = {
        "fetched": 0,
        "inserted": 0,
        "duplicates": 0,
        "errors": 0,
        "sources": sources,
    }
    
    # Strategic queries for copper market
    QUERIES = [
        "copper supply deficit",
        "copper price forecast",
        "copper mining production",
        "copper demand China",
        "copper EV battery",
        "Freeport-McMoRan copper",
        "BHP copper",
        "Rio Tinto copper",
    ]
    
    logger.info(f"[run_id={run_id}] Ingesting news from {sources} with {len(QUERIES)} queries")
    
    for source in sources:
        if source == "google_news":
            try:
                query_results = fetch_google_news_concurrent(
                    queries=QUERIES,
                    language=settings.news_language,
                )
                articles_to_batch = []
                for query, articles in query_results.items():
                    stats["fetched"] += len(articles)
                    for article in articles:
                        art = dict(article)
                        art["source_name"] = "google_news"
                        art["source_feed"] = f"google_news:{query}"
                        art["raw_payload"] = {"query": query, "source": article.get("source")}
                        art["publisher"] = article.get("source")
                        articles_to_batch.append(art)
                
                batch_stats = insert_raw_articles_batch(
                    session=session,
                    articles=articles_to_batch,
                    run_id=run_id,
                    default_source="google_news",
                )
                stats["inserted"] += batch_stats["inserted"]
                stats["duplicates"] += batch_stats["duplicates"]
                stats["errors"] += batch_stats["errors"]
            except Exception as e:
                logger.warning(f"Error in concurrent google_news ingestion: {e}")
                stats["errors"] += 1
                    
        elif source == "newsapi" and settings.newsapi_key:
            # NewsAPI implementation - reuse existing fetch
            from app.data_manager import fetch_newsapi_articles
            
            api_articles_to_batch = []
            for query in QUERIES[:3]:  # Limit API calls
                try:
                    articles = fetch_newsapi_articles(
                        api_key=settings.newsapi_key,
                        query=query,
                        language=settings.news_language,
                        lookback_days=settings.lookback_days,
                    )
                    
                    stats["fetched"] += len(articles)
                    
                    for article in articles:
                        art = dict(article)
                        art["source_name"] = "newsapi"
                        art["source_feed"] = f"newsapi:{query}"
                        art["raw_payload"] = {
                            "query": query,
                            "author": article.get("author"),
                            "source": article.get("source"),
                        }
                        art["publisher"] = article.get("source") or article.get("author")
                        api_articles_to_batch.append(art)
                            
                except Exception as e:
                    logger.warning(f"Error fetching newsapi for '{query}': {e}")
                    stats["errors"] += 1
            
            if api_articles_to_batch:
                batch_stats = insert_raw_articles_batch(
                    session=session,
                    articles=api_articles_to_batch,
                    run_id=run_id,
                    default_source="newsapi",
                )
                stats["inserted"] += batch_stats["inserted"]
                stats["duplicates"] += batch_stats["duplicates"]
                stats["errors"] += batch_stats["errors"]
    
    session.commit()
    
    logger.info(
        f"[run_id={run_id}] News ingestion complete: "
        f"{stats['fetched']} fetched, {stats['inserted']} inserted, "
        f"{stats['duplicates']} duplicates"
    )
    
    return stats
