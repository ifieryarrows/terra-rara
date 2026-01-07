"""
Quick database inspection utility.

Usage:
    python -m app.inspect_db
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, text

from app.db import SessionLocal, init_db, get_db_type
from app.models import NewsArticle, PriceBar, NewsSentiment, DailySentiment, AnalysisSnapshot
from app.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def inspect_database(verbose: bool = False):
    """Print database statistics."""
    
    init_db()
    
    print("\n" + "=" * 60)
    print("COPPERMIND DATABASE INSPECTION")
    print("=" * 60)
    
    settings = get_settings()
    
    # Database info
    print(f"\n📁 Database: {settings.database_url.split('?')[0]}")
    print(f"   Type: {get_db_type()}")
    
    with SessionLocal() as session:
        # News articles
        total_news = session.query(func.count(NewsArticle.id)).scalar()
        news_7d = session.query(func.count(NewsArticle.id)).filter(
            NewsArticle.published_at >= datetime.now(timezone.utc) - timedelta(days=7)
        ).scalar()
        
        print(f"\n📰 News Articles: {total_news}")
        print(f"   Last 7 days: {news_7d}")
        
        if verbose and total_news > 0:
            # Source breakdown
            sources = session.query(
                NewsArticle.source,
                func.count(NewsArticle.id)
            ).group_by(NewsArticle.source).order_by(func.count(NewsArticle.id).desc()).limit(5).all()
            
            print("   Top sources:")
            for source, count in sources:
                print(f"     - {source or 'Unknown'}: {count}")
            
            # Date range
            earliest = session.query(func.min(NewsArticle.published_at)).scalar()
            latest = session.query(func.max(NewsArticle.published_at)).scalar()
            print(f"   Date range: {earliest} to {latest}")
        
        # Sentiment scores
        scored_news = session.query(func.count(NewsSentiment.id)).scalar()
        unscored = total_news - scored_news
        
        print(f"\n🧠 Sentiment Scores: {scored_news}")
        print(f"   Unscored articles: {unscored}")
        
        if verbose and scored_news > 0:
            avg_score = session.query(func.avg(NewsSentiment.score)).scalar()
            print(f"   Average score: {avg_score:.3f}" if avg_score else "   Average score: N/A")
        
        # Daily sentiment
        daily_count = session.query(func.count(DailySentiment.id)).scalar()
        print(f"\n📊 Daily Sentiment Records: {daily_count}")
        
        # Price bars
        print("\n💰 Price Bars:")
        
        symbols = settings.symbols_list
        for symbol in symbols:
            count = session.query(func.count(PriceBar.id)).filter(
                PriceBar.symbol == symbol
            ).scalar()
            
            if count > 0:
                latest_bar = session.query(PriceBar).filter(
                    PriceBar.symbol == symbol
                ).order_by(PriceBar.date.desc()).first()
                
                latest_date = latest_bar.date.strftime("%Y-%m-%d") if latest_bar else "N/A"
                latest_price = f"${latest_bar.close:.4f}" if latest_bar else "N/A"
                
                print(f"   {symbol}: {count} bars (latest: {latest_date} @ {latest_price})")
            else:
                print(f"   {symbol}: 0 bars")
        
        # Analysis snapshots
        snapshot_count = session.query(func.count(AnalysisSnapshot.id)).scalar()
        print(f"\n📸 Analysis Snapshots: {snapshot_count}")
        
        if verbose and snapshot_count > 0:
            latest_snapshot = session.query(AnalysisSnapshot).order_by(
                AnalysisSnapshot.generated_at.desc()
            ).first()
            
            if latest_snapshot:
                print(f"   Latest: {latest_snapshot.symbol} @ {latest_snapshot.generated_at}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect CopperMind database")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed statistics"
    )
    args = parser.parse_args()
    
    inspect_database(verbose=args.verbose)


if __name__ == "__main__":
    main()

