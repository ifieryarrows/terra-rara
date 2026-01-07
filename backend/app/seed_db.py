"""
CSV seed importer for cold start scenarios.

Usage:
    python -m app.seed_db [--dir /data/seed] [--mapping title=headline,published_at=date]
    
CSV files should contain columns like:
    title, published_at, url, source, description, content
    
Column mapping allows flexibility:
    --mapping title=headline,published_at=pub_date
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.db import get_engine, SessionLocal, init_db
from app.data_manager import get_upsert_stmt
from app.models import NewsArticle
from app.settings import get_settings
from app.utils import (
    clean_text,
    canonical_title,
    normalize_url,
    generate_dedup_key,
    truncate_text,
    safe_parse_date,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Default column mapping
DEFAULT_MAPPING = {
    "title": "title",
    "published_at": "published_at",
    "url": "url",
    "source": "source",
    "description": "description",
    "content": "content",
}


def parse_column_mapping(mapping_str: Optional[str]) -> dict[str, str]:
    """Parse comma-separated column mapping string."""
    mapping = DEFAULT_MAPPING.copy()
    
    if not mapping_str:
        return mapping
    
    for pair in mapping_str.split(","):
        if "=" in pair:
            our_col, csv_col = pair.split("=", 1)
            our_col = our_col.strip()
            csv_col = csv_col.strip()
            if our_col in mapping:
                mapping[our_col] = csv_col
    
    return mapping


def get_csv_value(
    row: dict,
    mapping: dict[str, str],
    field: str
) -> Optional[str]:
    """Get a value from CSV row using column mapping."""
    csv_col = mapping.get(field, field)
    
    # Try exact match
    if csv_col in row:
        return row[csv_col]
    
    # Try case-insensitive match
    csv_col_lower = csv_col.lower()
    for key in row:
        if key.lower() == csv_col_lower:
            return row[key]
    
    return None


def import_csv_file(
    session: Session,
    file_path: Path,
    mapping: dict[str, str],
    dry_run: bool = False
) -> tuple[int, int, int]:
    """
    Import a single CSV file into the database.
    
    Returns: (imported, skipped_dedup, errors)
    """
    imported = 0
    skipped_dedup = 0
    errors = 0
    
    logger.info(f"Processing: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # Detect delimiter
            sample = f.read(4096)
            f.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            
            reader = csv.DictReader(f, dialect=dialect)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is 1)
                try:
                    # Extract fields
                    title = get_csv_value(row, mapping, "title")
                    published_str = get_csv_value(row, mapping, "published_at")
                    url = get_csv_value(row, mapping, "url")
                    source = get_csv_value(row, mapping, "source")
                    description = get_csv_value(row, mapping, "description")
                    content = get_csv_value(row, mapping, "content")
                    
                    # Validate required fields
                    if not title:
                        logger.debug(f"Row {row_num}: Missing title, skipping")
                        errors += 1
                        continue
                    
                    # Parse date
                    published_at = safe_parse_date(published_str) if published_str else None
                    if not published_at:
                        # Default to now if no date
                        published_at = datetime.now(timezone.utc)
                    
                    # Clean and normalize
                    title = clean_text(title)
                    description = clean_text(description) if description else None
                    content = clean_text(content) if content else None
                    url = normalize_url(url) if url else None
                    source = source.strip() if source else "CSV Import"
                    
                    # Generate dedup key
                    dedup_key = generate_dedup_key(
                        url=url,
                        title=title,
                        published_at=published_at,
                        source=source
                    )
                    
                    # Generate canonical title for fuzzy dedup
                    canon_title = canonical_title(title)
                    
                    if dry_run:
                        logger.debug(f"[DRY RUN] Would import: {title[:50]}...")
                        imported += 1
                        continue
                    
                    # Upsert (insert or ignore if exists)
                    stmt = get_upsert_stmt(
                        NewsArticle,
                        values={
                            "dedup_key": dedup_key,
                            "title": truncate_text(title, 500),
                            "canonical_title": truncate_text(canon_title, 500),
                            "description": truncate_text(description, 2000) if description else None,
                            "content": truncate_text(content, 10000) if content else None,
                            "url": url,
                            "source": source,
                            "published_at": published_at,
                            "fetched_at": datetime.now(timezone.utc),
                            "language": "en",
                        },
                        index_elements=["dedup_key"]
                    )
                    
                    result = session.execute(stmt)
                    
                    if result.rowcount > 0:
                        imported += 1
                    else:
                        skipped_dedup += 1
                        
                except Exception as e:
                    logger.warning(f"Row {row_num}: Error processing - {e}")
                    errors += 1
                    continue
            
            # Commit batch
            if not dry_run:
                session.commit()
                
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        if not dry_run:
            session.rollback()
        raise
    
    return imported, skipped_dedup, errors


def main():
    parser = argparse.ArgumentParser(
        description="Import CSV files into the news database"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/data/seed",
        help="Directory containing CSV files (default: /data/seed)"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Column mapping (e.g., title=headline,published_at=date)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing to database"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Parse column mapping
    mapping = parse_column_mapping(args.mapping)
    logger.info(f"Column mapping: {mapping}")
    
    # Find CSV files
    seed_dir = Path(args.dir)
    if not seed_dir.exists():
        logger.warning(f"Seed directory does not exist: {seed_dir}")
        logger.info("Creating directory...")
        seed_dir.mkdir(parents=True, exist_ok=True)
        logger.info("No CSV files to import. Place CSV files in /data/seed/ and run again.")
        return
    
    csv_files = list(seed_dir.glob("*.csv"))
    if not csv_files:
        logger.info(f"No CSV files found in {seed_dir}")
        logger.info("Place CSV files in /data/seed/ and run again.")
        return
    
    logger.info(f"Found {len(csv_files)} CSV file(s)")
    
    # Process each CSV file
    total_imported = 0
    total_skipped = 0
    total_errors = 0
    
    with SessionLocal() as session:
        for csv_file in csv_files:
            try:
                imported, skipped, errors = import_csv_file(
                    session,
                    csv_file,
                    mapping,
                    dry_run=args.dry_run
                )
                total_imported += imported
                total_skipped += skipped
                total_errors += errors
                logger.info(
                    f"  → {csv_file.name}: "
                    f"{imported} imported, {skipped} duplicates, {errors} errors"
                )
            except Exception as e:
                logger.error(f"Failed to process {csv_file}: {e}")
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"TOTAL: {total_imported} imported, {total_skipped} duplicates, {total_errors} errors")
    
    if args.dry_run:
        logger.info("(Dry run - no changes were made)")


if __name__ == "__main__":
    main()

