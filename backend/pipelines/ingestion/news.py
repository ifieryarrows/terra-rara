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
from app.rss_ingest import fetch_google_news
from app.db import get_db_type

logger = logging.getLogger(__name__)


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
    title: str,
    description: Optional[str],
    source: str,
    source_feed: str,
    published_at: datetime,
    run_id: uuid.UUID,
    raw_payload: Optional[dict] = None,
) -> Optional[int]:
    """
    Insert single article to news_raw.
    
    Uses ON CONFLICT DO NOTHING for url_hash to handle duplicates gracefully.
    
    Args:
        session: Database session
        url: Article URL (can be None)
        title: Article title
        description: Article description
        source: Source name (e.g., "google_news", "newsapi")
        source_feed: Exact feed URL or query string
        published_at: Publication timestamp (UTC)
        run_id: Pipeline run UUID
        raw_payload: Original response fragment for debugging
        
    Returns:
        raw_id if inserted, None if duplicate or error
    """
    if not title or not title.strip():
        return None
    
    title = clean_text(title)[:500]  # Truncate to column limit
    url_hash = compute_url_hash(url)
    
    try:
        db_type = get_db_type()
        
        if db_type == "postgresql":
            # Use INSERT ... ON CONFLICT for PostgreSQL
            stmt = pg_insert(NewsRaw).values(
                url=url,
                url_hash=url_hash,
                title=title,
                description=description[:2000] if description else None,
                source=source,
                source_feed=source_feed[:500] if source_feed else None,
                published_at=published_at,
                run_id=run_id,
                raw_payload=raw_payload,
            )
            
            # Only conflict on url_hash if it's not None
            if url_hash:
                stmt = stmt.on_conflict_do_nothing(index_elements=["url_hash"])
            
            result = session.execute(stmt)
            
            if result.rowcount > 0:
                # Get the inserted ID
                # For PostgreSQL, we need to query it
                row = session.execute(
                    text("SELECT id FROM news_raw WHERE url_hash = :hash ORDER BY id DESC LIMIT 1"),
                    {"hash": url_hash}
                ).fetchone()
                return row[0] if row else None
            
            return None  # Duplicate
            
        else:
            # SQLite fallback - simple insert with error handling
            article = NewsRaw(
                url=url,
                url_hash=url_hash,
                title=title,
                description=description[:2000] if description else None,
                source=source,
                source_feed=source_feed[:500] if source_feed else None,
                published_at=published_at,
                run_id=run_id,
                raw_payload=raw_payload,
            )
            session.add(article)
            session.flush()
            return article.id
            
    except Exception as e:
        logger.debug(f"Insert raw article failed: {e}")
        session.rollback()
        return None


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
            for query in QUERIES:
                try:
                    articles = fetch_google_news(
                        query=query,
                        language=settings.news_language,
                    )
                    
                    stats["fetched"] += len(articles)
                    
                    for article in articles:
                        raw_id = insert_raw_article(
                            session=session,
                            url=article.get("url"),
                            title=article.get("title", ""),
                            description=article.get("description"),
                            source="google_news",
                            source_feed=f"google_news:{query}",
                            published_at=article.get("published_at", datetime.now(timezone.utc)),
                            run_id=run_id,
                            raw_payload={"query": query, "source": article.get("source")},
                        )
                        
                        if raw_id:
                            stats["inserted"] += 1
                        else:
                            stats["duplicates"] += 1
                            
                except Exception as e:
                    logger.warning(f"Error fetching {source} for '{query}': {e}")
                    stats["errors"] += 1
                    
        elif source == "newsapi" and settings.newsapi_key:
            # NewsAPI implementation - reuse existing fetch
            from app.data_manager import fetch_newsapi_articles
            
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
                        raw_id = insert_raw_article(
                            session=session,
                            url=article.get("url"),
                            title=article.get("title", ""),
                            description=article.get("description"),
                            source="newsapi",
                            source_feed=f"newsapi:{query}",
                            published_at=article.get("published_at", datetime.now(timezone.utc)),
                            run_id=run_id,
                            raw_payload={"query": query, "author": article.get("author")},
                        )
                        
                        if raw_id:
                            stats["inserted"] += 1
                        else:
                            stats["duplicates"] += 1
                            
                except Exception as e:
                    logger.warning(f"Error fetching newsapi for '{query}': {e}")
                    stats["errors"] += 1
    
    session.commit()
    
    logger.info(
        f"[run_id={run_id}] News ingestion complete: "
        f"{stats['fetched']} fetched, {stats['inserted']} inserted, "
        f"{stats['duplicates']} duplicates"
    )
    
    return stats
