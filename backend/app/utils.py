"""
Utility functions for data normalization and deduplication.
"""

import hashlib
import re
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from bs4 import BeautifulSoup


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    if not text:
        return ""
    return " ".join(text.split())


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def clean_text(text: str) -> str:
    """Clean text by removing HTML and normalizing whitespace."""
    if not text:
        return ""
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text.strip()


def canonical_title(title: str) -> str:
    """
    Create a canonical version of a title for fuzzy dedup comparison.
    - Lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    if not title:
        return ""
    # Lowercase
    title = title.lower()
    # Remove punctuation (keep alphanumeric and spaces)
    title = re.sub(r"[^\w\s]", " ", title)
    # Normalize whitespace
    title = normalize_whitespace(title)
    return title


def normalize_url(url: str) -> str:
    """
    Normalize a URL by:
    - Removing tracking parameters (utm_*, fbclid, etc.)
    - Removing fragments
    - Lowercasing the domain
    - Sorting query parameters
    """
    if not url:
        return ""
    
    try:
        parsed = urlparse(url)
        
        # Lowercase domain
        netloc = parsed.netloc.lower()
        
        # Remove tracking parameters
        tracking_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "ref", "source", "mc_cid", "mc_eid"
        }
        query_params = parse_qsl(parsed.query)
        filtered_params = [
            (k, v) for k, v in query_params 
            if k.lower() not in tracking_params
        ]
        # Sort for consistency
        filtered_params.sort(key=lambda x: x[0])
        query = urlencode(filtered_params)
        
        # Reconstruct URL without fragment
        normalized = urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            query,
            ""  # No fragment
        ))
        
        return normalized
    except Exception:
        return url


def generate_dedup_key(
    url: Optional[str] = None,
    title: Optional[str] = None,
    published_at: Optional[datetime] = None,
    source: Optional[str] = None
) -> str:
    """
    Generate a deduplication key for a news article.
    
    Strategy:
    1. If URL exists, use normalized URL hash
    2. Otherwise, use hash of (canonical_title + date + source)
    """
    if url:
        normalized = normalize_url(url)
        if normalized:
            return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    # Fallback to content-based hash
    parts = []
    if title:
        parts.append(canonical_title(title))
    if published_at:
        parts.append(published_at.strftime("%Y-%m-%d"))
    if source:
        parts.append(source.lower().strip())
    
    if not parts:
        # Last resort: random key (shouldn't happen)
        import uuid
        return uuid.uuid4().hex[:32]
    
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length - 3].rsplit(" ", 1)[0] + "..."


def safe_parse_date(
    date_str: str,
    formats: Optional[list[str]] = None
) -> Optional[datetime]:
    """
    Try to parse a date string using multiple formats.
    Returns None if parsing fails.
    """
    from dateutil import parser as dateutil_parser
    from dateutil.tz import UTC
    
    if not date_str:
        return None
    
    # Try dateutil first (most flexible)
    try:
        dt = dateutil_parser.parse(date_str)
        # Ensure timezone (default to UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except Exception:
        pass
    
    # Try explicit formats
    formats = formats or [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    
    return None

