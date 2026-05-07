ALTER TABLE news_sentiments_v2
ADD COLUMN IF NOT EXISTS market_date DATE,
ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS cutoff_version TEXT DEFAULT 'market_close_v1';

ALTER TABLE daily_sentiments_v2
ADD COLUMN IF NOT EXISTS market_date DATE,
ADD COLUMN IF NOT EXISTS material_news_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS after_close_news_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS stale_sentiment_flag BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS days_since_last_material_news INTEGER DEFAULT 999,
ADD COLUMN IF NOT EXISTS cutoff_version TEXT DEFAULT 'market_close_v1';

CREATE INDEX IF NOT EXISTS ix_news_sentiments_v2_market_date
ON news_sentiments_v2(market_date);

CREATE INDEX IF NOT EXISTS ix_daily_sentiments_v2_market_date
ON daily_sentiments_v2(market_date);
