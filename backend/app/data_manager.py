"""
Data Manager: News and price data ingestion.

Handles:
- NewsAPI fetching (if API key provided)
- RSS feed fallback (Google News)
- Fuzzy deduplication for RSS noise
- Multi-symbol yfinance price ingestion
- Language filtering for FinBERT compatibility

Usage:
    python -m app.data_manager --fetch
    python -m app.data_manager --fetch --news-only
    python -m app.data_manager --fetch --prices-only
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import yfinance as yf
from rapidfuzz import fuzz
from langdetect import detect, LangDetectException
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db, get_db_type
from app.models import NewsArticle, PriceBar
from app.settings import get_settings
from app.rss_ingest import fetch_google_news
from app.utils import (
    clean_text,
    canonical_title,
    normalize_url,
    generate_dedup_key,
    truncate_text,
)
from app.lock import pipeline_lock

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_upsert_stmt(table, values: dict, index_elements: list, update_set: dict = None):
    """Create database-agnostic upsert statement."""
    db_type = get_db_type()
    
    if db_type == "postgresql":
        stmt = pg_insert(table).values(**values)
        if update_set:
            stmt = stmt.on_conflict_do_update(index_elements=index_elements, set_=update_set)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)
    else:
        # SQLite
        stmt = sqlite_insert(table).values(**values)
        if update_set:
            stmt = stmt.on_conflict_do_update(index_elements=index_elements, set_=update_set)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=index_elements)
    
    return stmt


# =============================================================================
# NewsAPI Fetching
# =============================================================================

def fetch_newsapi_articles(
    api_key: str,
    query: str,
    language: str = "en",
    lookback_days: int = 30,
    page_size: int = 100
) -> list[dict]:
    """
    Fetch articles from NewsAPI.
    
    Note: Free plan limits to ~1 month of history.
    """
    logger.info(f"Fetching from NewsAPI: query='{query}', language={language}")
    
    # Calculate date range
    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=min(lookback_days, 30))  # API limit
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": api_key,
        "q": query,
        "language": language,
        "from": from_date.strftime("%Y-%m-%d"),
        "to": to_date.strftime("%Y-%m-%d"),
        "sortBy": "publishedAt",
        "pageSize": page_size,
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
        
        articles = []
        for item in data.get("articles", []):
            try:
                published_str = item.get("publishedAt", "")
                published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00")) if published_str else datetime.now(timezone.utc)
                
                articles.append({
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "source": item.get("source", {}).get("name", ""),
                    "author": item.get("author", ""),
                    "published_at": published_at,
                })
            except Exception as e:
                logger.debug(f"Error parsing NewsAPI article: {e}")
                continue
        
        logger.info(f"Fetched {len(articles)} articles from NewsAPI")
        return articles
        
    except requests.RequestException as e:
        logger.error(f"NewsAPI request failed: {e}")
        return []


# =============================================================================
# Language Detection
# =============================================================================

def detect_language(text: str) -> Optional[str]:
    """Detect language of text. Returns None if detection fails."""
    if not text or len(text) < 20:
        return None
    
    try:
        return detect(text)
    except LangDetectException:
        return None


def filter_by_language(
    articles: list[dict],
    target_language: str = "en"
) -> tuple[list[dict], int]:
    """
    Filter articles by language.
    
    Returns:
        Tuple of (filtered_articles, num_filtered_out)
    """
    filtered = []
    filtered_out = 0
    
    for article in articles:
        # Try to detect from title + description
        text = f"{article.get('title', '')} {article.get('description', '')}"
        lang = detect_language(text)
        
        if lang is None or lang == target_language:
            filtered.append(article)
        else:
            filtered_out += 1
            logger.debug(f"Filtered out ({lang}): {article.get('title', '')[:50]}")
    
    if filtered_out > 0:
        logger.info(f"Language filter: kept {len(filtered)}, filtered out {filtered_out}")
    
    return filtered, filtered_out


# =============================================================================
# Fuzzy Deduplication
# =============================================================================

def get_recent_titles(
    session: Session,
    window_hours: int = 48
) -> list[str]:
    """Get canonical titles from recent articles for fuzzy dedup."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    
    articles = session.query(NewsArticle.canonical_title).filter(
        NewsArticle.published_at >= cutoff,
        NewsArticle.canonical_title.isnot(None)
    ).all()
    
    return [a[0] for a in articles if a[0]]


def is_fuzzy_duplicate(
    title: str,
    existing_titles: list[str],
    threshold: int = 85
) -> bool:
    """
    Check if title is too similar to existing titles.
    Uses token_set_ratio for robust matching.
    """
    if not title or not existing_titles:
        return False
    
    canon = canonical_title(title)
    
    for existing in existing_titles:
        similarity = fuzz.token_set_ratio(canon, existing)
        if similarity >= threshold:
            logger.debug(f"Fuzzy duplicate ({similarity}%): '{title[:50]}...'")
            return True
    
    return False


# =============================================================================
# News Ingestion
# =============================================================================

def ingest_news(session: Session) -> dict:
    """
    Ingest news from all configured sources.
    
    Returns:
        Dict with stats: imported, duplicates, language_filtered, fuzzy_filtered
    """
    settings = get_settings()
    
    stats = {
        "imported": 0,
        "duplicates": 0,
        "language_filtered": 0,
        "fuzzy_filtered": 0,
        "source": "unknown",
    }
    
    # Collect articles from sources
    all_articles = []
    
    # Try NewsAPI first if key is available
    if settings.newsapi_key:
        articles = fetch_newsapi_articles(
            api_key=settings.newsapi_key,
            query=settings.news_query,
            language=settings.news_language,
            lookback_days=settings.lookback_days,
        )
        if articles:
            all_articles.extend(articles)
            stats["source"] = "newsapi"
    
    # RSS fallback/supplement
    if not all_articles or not settings.newsapi_key:
        rss_articles = fetch_google_news(
            query=settings.news_query,
            language=settings.news_language,
        )
        all_articles.extend(rss_articles)
        stats["source"] = "rss" if not settings.newsapi_key else "newsapi+rss"
    
    if not all_articles:
        logger.warning("No articles fetched from any source")
        return stats
    
    logger.info(f"Total articles fetched: {len(all_articles)}")
    
    # Language filter
    all_articles, lang_filtered = filter_by_language(
        all_articles,
        target_language=settings.news_language
    )
    stats["language_filtered"] = lang_filtered
    
    # Get recent titles for fuzzy dedup
    recent_titles = get_recent_titles(
        session,
        window_hours=settings.fuzzy_dedup_window_hours
    )
    
    # Process articles
    for article in all_articles:
        try:
            title = clean_text(article.get("title", ""))
            if not title:
                continue
            
            # Fuzzy dedup check
            if is_fuzzy_duplicate(
                title,
                recent_titles,
                threshold=settings.fuzzy_dedup_threshold
            ):
                stats["fuzzy_filtered"] += 1
                continue
            
            # Prepare fields
            description = clean_text(article.get("description", ""))
            content = clean_text(article.get("content", ""))
            url = normalize_url(article.get("url", ""))
            source = article.get("source", "Unknown")
            author = article.get("author", "")
            published_at = article.get("published_at", datetime.now(timezone.utc))
            
            # Generate keys
            dedup_key = generate_dedup_key(
                url=url,
                title=title,
                published_at=published_at,
                source=source
            )
            canon_title = canonical_title(title)
            
            # Upsert
            stmt = get_upsert_stmt(
                NewsArticle,
                values={
                    "dedup_key": dedup_key,
                    "title": truncate_text(title, 500),
                    "canonical_title": truncate_text(canon_title, 500),
                    "description": truncate_text(description, 2000) if description else None,
                    "content": truncate_text(content, 10000) if content else None,
                    "url": url or None,
                    "source": source,
                    "author": author or None,
                    "language": settings.news_language,
                    "published_at": published_at,
                    "fetched_at": datetime.now(timezone.utc),
                },
                index_elements=["dedup_key"]
            )
            
            result = session.execute(stmt)
            
            if result.rowcount > 0:
                stats["imported"] += 1
                # Add to recent titles for this batch
                recent_titles.append(canon_title)
            else:
                stats["duplicates"] += 1
                
        except Exception as e:
            logger.warning(f"Error processing article: {e}")
            continue
    
    session.commit()
    
    logger.info(
        f"News ingestion complete: "
        f"{stats['imported']} imported, "
        f"{stats['duplicates']} duplicates, "
        f"{stats['fuzzy_filtered']} fuzzy filtered, "
        f"{stats['language_filtered']} language filtered"
    )
    
    return stats


# =============================================================================
# Price Ingestion
# =============================================================================

def ingest_prices(session: Session) -> dict:
    """
    Ingest price data for all configured symbols.
    
    Returns:
        Dict with stats per symbol
    """
    settings = get_settings()
    symbols = settings.symbols_list
    
    stats = {}
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=settings.lookback_days)
    
    for symbol in symbols:
        logger.info(f"Fetching prices for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d"
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                stats[symbol] = {"imported": 0, "duplicates": 0, "error": "no_data"}
                continue
            
            imported = 0
            duplicates = 0
            
            for date_idx, row in df.iterrows():
                try:
                    # Convert index to datetime
                    if hasattr(date_idx, 'to_pydatetime'):
                        bar_date = date_idx.to_pydatetime()
                    else:
                        bar_date = date_idx
                    
                    # Ensure timezone
                    if bar_date.tzinfo is None:
                        bar_date = bar_date.replace(tzinfo=timezone.utc)
                    
                    # Upsert
                    stmt = get_upsert_stmt(
                        PriceBar,
                        values={
                            "symbol": symbol,
                            "date": bar_date,
                            "open": float(row.get("Open", 0)) if row.get("Open") else None,
                            "high": float(row.get("High", 0)) if row.get("High") else None,
                            "low": float(row.get("Low", 0)) if row.get("Low") else None,
                            "close": float(row["Close"]),
                            "volume": float(row.get("Volume", 0)) if row.get("Volume") else None,
                            "adj_close": float(row.get("Adj Close", row["Close"])),
                            "fetched_at": datetime.now(timezone.utc),
                        },
                        index_elements=["symbol", "date"],
                        update_set={
                            "close": float(row["Close"]),
                            "adj_close": float(row.get("Adj Close", row["Close"])),
                            "fetched_at": datetime.now(timezone.utc),
                        }
                    )
                    
                    result = session.execute(stmt)
                    
                    if result.rowcount > 0:
                        imported += 1
                    else:
                        duplicates += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing price bar: {e}")
                    continue
            
            session.commit()
            
            stats[symbol] = {"imported": imported, "duplicates": duplicates}
            logger.info(f"{symbol}: {imported} bars imported, {duplicates} updated")
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            stats[symbol] = {"imported": 0, "duplicates": 0, "error": str(e)}
    
    return stats


# =============================================================================
# Main Entry Point
# =============================================================================

def fetch_all(
    news: bool = True,
    prices: bool = True
) -> dict:
    """
    Run full data ingestion pipeline.
    
    Args:
        news: Whether to fetch news
        prices: Whether to fetch prices
    
    Returns:
        Combined stats dict
    """
    logger.info("Starting data ingestion pipeline...")
    
    results = {
        "news": None,
        "prices": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    with SessionLocal() as session:
        if news:
            results["news"] = ingest_news(session)
        
        if prices:
            results["prices"] = ingest_prices(session)
    
    logger.info("Data ingestion complete")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch news and price data"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Run data fetch"
    )
    parser.add_argument(
        "--news-only",
        action="store_true",
        help="Fetch only news"
    )
    parser.add_argument(
        "--prices-only",
        action="store_true",
        help="Fetch only prices"
    )
    parser.add_argument(
        "--no-lock",
        action="store_true",
        help="Skip pipeline lock (for testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.fetch:
        parser.print_help()
        return
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Determine what to fetch
    fetch_news = not args.prices_only
    fetch_prices = not args.news_only
    
    # Run with or without lock
    if args.no_lock:
        results = fetch_all(news=fetch_news, prices=fetch_prices)
    else:
        try:
            with pipeline_lock():
                results = fetch_all(news=fetch_news, prices=fetch_prices)
        except RuntimeError as e:
            logger.error(f"Could not acquire lock: {e}")
            logger.info("Another pipeline process may be running. Use --no-lock to bypass.")
            return
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATA INGESTION SUMMARY")
    print("=" * 50)
    
    if results.get("news"):
        news = results["news"]
        print(f"\nNews ({news.get('source', 'unknown')}):")
        print(f"  - Imported: {news.get('imported', 0)}")
        print(f"  - Duplicates: {news.get('duplicates', 0)}")
        print(f"  - Fuzzy filtered: {news.get('fuzzy_filtered', 0)}")
        print(f"  - Language filtered: {news.get('language_filtered', 0)}")
    
    if results.get("prices"):
        print("\nPrices:")
        for symbol, stats in results["prices"].items():
            status = f"{stats.get('imported', 0)} imported"
            if stats.get("error"):
                status = f"ERROR: {stats['error']}"
            print(f"  - {symbol}: {status}")
    
    print(f"\nTimestamp: {results.get('timestamp', 'N/A')}")


if __name__ == "__main__":
    main()

