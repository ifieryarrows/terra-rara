"""
RSS feed ingestion with Google News RSS support.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote_plus

import feedparser
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
    max_items: int = 100
) -> list[dict]:
    """
    Fetch and parse an RSS feed.
    
    Args:
        url: RSS feed URL
        max_items: Maximum number of items to return
    
    Returns:
        List of article dicts with keys: title, url, published_at, source, description
    """
    logger.info(f"Fetching RSS feed: {url}")
    
    try:
        feed = feedparser.parse(url)
        
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
    max_items: int = 100
) -> list[dict]:
    """
    Fetch articles from Google News RSS.
    
    Args:
        query: Search query. If None, uses settings.
        language: Language code. If None, uses settings.
        max_items: Maximum articles to fetch
    
    Returns:
        List of article dicts
    """
    settings = get_settings()
    
    query = query or settings.news_query
    language = language or settings.news_language
    
    url = build_google_news_rss_url(query, language)
    return fetch_rss_feed(url, max_items)


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

