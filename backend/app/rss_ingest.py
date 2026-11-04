"""
RSS feed ingestion with Google News RSS support.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus

import feedparser
import requests
from requests.adapters import HTTPAdapter
from dateutil import parser as dateutil_parser

from app.settings import get_settings
from app.utils import clean_text, normalize_url

logger = logging.getLogger(__name__)


def build_google_news_rss_url(query: str, language: str = "en") -> str:
    """
    Build a Google News RSS URL for a search query.
    
    Args:
        query: Search query (e.g., "copper price")
        language: Language code (e.g., "en")
    
    Returns:
        Google News RSS URL
    """
    encoded_query = quote_plus(query)
    # Google News RSS format
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl={language}&gl=US&ceid=US:{language}"
    return url


def parse_rss_date(date_str: str) -> Optional[datetime]:
    """Parse RSS date string to datetime."""
    if not date_str:
        return None
    
    try:
        dt = dateutil_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def fetch_rss_feed(
    url: str,
    max_items: int = 100,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """
    Fetch and parse an RSS feed.
    
    Args:
        url: RSS feed URL
        max_items: Maximum number of items to return
        session: Optional requests.Session for connection pooling
    
    Returns:
        List of article dicts with keys: title, url, published_at, source, description
    """
    logger.info(f"Fetching RSS feed: {url}")
    
    try:
        # feedparser.parse(url) performs its own urllib request without a
        # configurable timeout. A stalled RSS host can therefore hold the
        # pipeline worker and its advisory lock indefinitely. Bound both
        # connection setup and response reads, then give feedparser the
        # downloaded bytes so parsing remains unchanged.
        http_client = session if session is not None else requests
        response = http_client.get(url, timeout=(5.0, 15.0))
        response.raise_for_status()
        feed = feedparser.parse(
            response.content,
            response_headers={
                "content-location": response.url,
                "content-type": response.headers.get("Content-Type", ""),
            },
        )
        
        if feed.bozo and feed.bozo_exception:
            logger.warning(f"RSS feed parsing warning: {feed.bozo_exception}")
        
        articles = []
        
        for entry in feed.entries[:max_items]:
            try:
                # Extract fields
                title = entry.get("title", "")
                link = entry.get("link", "")
                published = entry.get("published", entry.get("updated", ""))
                source = entry.get("source", {}).get("title", "")
                
                # Google News wraps the actual source in the title
                # Format: "Article Title - Source Name"
                if not source and " - " in title:
                    parts = title.rsplit(" - ", 1)
                    if len(parts) == 2:
                        title, source = parts
                
                # Get description/summary
                description = entry.get("summary", entry.get("description", ""))
                
                # Clean content
                title = clean_text(title)
                description = clean_text(description)
                
                if not title:
                    continue
                
                # Parse date
                published_at = parse_rss_date(published)
                if not published_at:
                    published_at = datetime.now(timezone.utc)
                
                articles.append({
                    "title": title,
                    "url": normalize_url(link) if link else None,
                    "published_at": published_at,
                    "source": source or "Google News",
                    "description": description or None,
                })
                
            except Exception as e:
                logger.debug(f"Error parsing RSS entry: {e}")
                continue
        
        logger.info(f"Fetched {len(articles)} articles from RSS")
        return articles
        
    except Exception as e:
        logger.error(f"Failed to fetch RSS feed: {e}")
        return []


def fetch_google_news(
    query: Optional[str] = None,
    language: Optional[str] = None,
    max_items: int = 100,
    session: Optional[requests.Session] = None,
) -> list[dict]:
    """
    Fetch articles from Google News RSS.
    
    Args:
        query: Search query. If None, uses settings.
        language: Language code. If None, uses settings.
        max_items: Maximum articles to fetch
        session: Optional requests.Session for connection pooling
    
    Returns:
        List of article dicts
    """
    settings = get_settings()
    
    query = query or settings.news_query
    language = language or settings.news_language
    
    url = build_google_news_rss_url(query, language)
    return fetch_rss_feed(url, max_items, session=session)


def fetch_google_news_concurrent(
    queries: list[str],
    language: Optional[str] = None,
    max_items: int = 100,
    max_workers: int = 8,
) -> dict[str, list[dict]]:
    """
    Fetch multiple Google News queries concurrently using a connection pool.
    
    Args:
        queries: List of search queries
        language: Language code
        max_items: Max articles per query
        max_workers: Maximum concurrent workers
        
    Returns:
        Dict mapping query string to list of article dicts
    """
    if not queries:
        return {}

    num_workers = min(len(queries), max_workers)
    results: dict[str, list[dict]] = {}

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        })
        adapter = HTTPAdapter(
            pool_connections=num_workers,
            pool_maxsize=num_workers,
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_query = {
                executor.submit(
                    fetch_google_news,
                    query=q,
                    language=language,
                    max_items=max_items,
                    session=session,
                ): q
                for q in queries
            }
            for future in as_completed(future_to_query):
                q = future_to_query[future]
                try:
                    results[q] = future.result()
                except Exception as e:
                    logger.warning(f"Error fetching Google News for '{q}': {e}")
                    results[q] = []

    return results


def fetch_custom_rss_feeds(
    urls: list[str],
    max_items_per_feed: int = 50
) -> list[dict]:
    """
    Fetch articles from multiple custom RSS feeds.
    
    Args:
        urls: List of RSS feed URLs
        max_items_per_feed: Max items per feed
    
    Returns:
        Combined list of article dicts
    """
    all_articles = []
    
    for url in urls:
        try:
            articles = fetch_rss_feed(url, max_items_per_feed)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Failed to fetch RSS {url}: {e}")
    
    return all_articles


