import logging
import json
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from sqlalchemy.orm import Session
import pandas as pd

from app.models import HeatmapCache

logger = logging.getLogger(__name__)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _build_hierarchy(data: list) -> dict:
    """
    Build squarified treemap hierarchy:
    sector -> industry -> stock
    """
    # Group by sector
    sectors = {}
    for item in data:
        sec = item.get("sector", "Other")
        ind = item.get("industry", "Other")
        
        if sec not in sectors:
            sectors[sec] = {}
        if ind not in sectors[sec]:
            sectors[sec][ind] = []
            
        sectors[sec][ind].append(item)
        
    # Convert to D3 hierarchy format
    root = {"name": "Market", "children": []}
    for sec_name, inds in sectors.items():
        sec_node = {"name": sec_name, "children": []}
        for ind_name, stocks in inds.items():
            ind_node = {"name": ind_name, "children": stocks}
            sec_node["children"].append(ind_node)
        root["children"].append(sec_node)
        
    return root


def refresh_market_heatmap() -> None:
    """
    Background task to refresh market heatmap data.
    Uses yfinance to get live quotes and profiles for the broad universe.
    """
    from app.db import SessionLocal
    
    with SessionLocal() as session:
        try:
            # Mark as refreshing
            cache = session.query(HeatmapCache).first()
            if not cache:
                cache = HeatmapCache(payload_json={})
                session.add(cache)
            
            cache.refresh_started_at = _utcnow()
            cache.refresh_error = None
            session.commit()
            
            # 1. Load universe
            universe_path = Path("config/seeds/broad_universe.csv")
            if not universe_path.exists():
                universe_path = Path(__file__).parent.parent / "config" / "seeds" / "broad_universe.csv"
                
            symbols_to_fetch = []
            symbol_categories = {}
            
            if universe_path.exists():
                df = pd.read_csv(universe_path)
                symbols_to_fetch = df["ticker"].tolist()
                for _, row in df.iterrows():
                    symbol_categories[row["ticker"]] = row.get("category", "Other")
            else:
                symbols_to_fetch = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "HG=F", "SCCO", "FCX"]
            
            if not symbols_to_fetch:
                symbols_to_fetch = ["HG=F", "SCCO", "FCX"]
                
            # 2. Fetch Yahoo data
            import yfinance as yf
            logger.info(f"Fetching heatmap data for {len(symbols_to_fetch)} symbols")
            
            tickers = yf.Tickers(" ".join(symbols_to_fetch))
            
            stocks_data = []
            for symbol in symbols_to_fetch:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if not ticker:
                        continue
                        
                    info = ticker.info
                    if not info:
                        continue
                        
                    price = info.get("regularMarketPrice") or info.get("currentPrice")
                    change = info.get("regularMarketChangePercent")
                    market_cap = info.get("marketCap")
                    
                    if price is None or market_cap is None:
                        continue
                        
                    sector = info.get("sector")
                    industry = info.get("industry")
                    
                    if not sector:
                        cat = symbol_categories.get(symbol, "")
                        if "miner" in cat or symbol in ["HG=F", "SCCO", "FCX", "COPX"]:
                            sector = "Basic Materials"
                            industry = "Copper"
                        elif "macro" in cat:
                            sector = "Macro"
                            industry = "Index"
                        elif "etf" in cat:
                            sector = "Financial"
                            industry = "ETF"
                        else:
                            sector = "Other"
                            industry = "Other"
                            
                    if not industry:
                        industry = sector
                    
                    # Mock indices
                    is_sp500 = True if market_cap > 50e9 else False
                    is_nasdaq100 = True if market_cap > 100e9 and sector == "Technology" else False
                    
                    stocks_data.append({
                        "name": symbol,
                        "shortName": info.get("shortName", symbol),
                        "price": price,
                        "changePercent": change if change else 0.0,
                        "marketCap": market_cap,
                        "sector": sector,
                        "industry": industry,
                        "isSP500": is_sp500,
                        "isNasdaq100": is_nasdaq100,
                    })
                except Exception as e:
                    logger.debug(f"Heatmap fetch error for {symbol}: {e}")
                    
            # 3. Build hierarchy
            root = _build_hierarchy(stocks_data)
            
            # 4. Save to DB
            now = _utcnow()
            cache.payload_json = root
            cache.cached_at = now
            cache.expires_at = now + timedelta(minutes=15)
            cache.refresh_started_at = None
            cache.refresh_error = None
            
            session.commit()
            logger.info("Heatmap cache refreshed successfully.")
            
        except Exception as e:
            logger.error(f"Error in background heatmap refresh: {e}")
            try:
                cache = session.query(HeatmapCache).first()
                if cache:
                    cache.refresh_started_at = None
                    cache.refresh_error = str(e)
                    session.commit()
            except:
                pass

