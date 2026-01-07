-- =============================================================================
-- CopperMind: Supabase PostgreSQL Schema
-- =============================================================================

-- 1. News Articles
CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    dedup_key VARCHAR(64) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    canonical_title VARCHAR(500),
    description TEXT,
    content TEXT,
    url VARCHAR(2000),
    source VARCHAR(200),
    author VARCHAR(200),
    language VARCHAR(10) DEFAULT 'en',
    published_at TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_news_dedup_key ON news_articles(dedup_key);
CREATE INDEX idx_news_published_at ON news_articles(published_at);
CREATE INDEX idx_news_canonical_title ON news_articles(canonical_title);

-- 2. Price Bars (OHLCV)
CREATE TABLE IF NOT EXISTS price_bars (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION,
    adj_close DOUBLE PRECISION,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE INDEX idx_price_symbol ON price_bars(symbol);
CREATE INDEX idx_price_date ON price_bars(date);
CREATE INDEX idx_price_symbol_date ON price_bars(symbol, date);

-- 3. News Sentiment (FinBERT scores)
CREATE TABLE IF NOT EXISTS news_sentiments (
    id BIGSERIAL PRIMARY KEY,
    news_article_id BIGINT UNIQUE NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    prob_positive DOUBLE PRECISION NOT NULL,
    prob_neutral DOUBLE PRECISION NOT NULL,
    prob_negative DOUBLE PRECISION NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    model_name VARCHAR(100) DEFAULT 'ProsusAI/finbert',
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sentiment_article ON news_sentiments(news_article_id);
CREATE INDEX idx_sentiment_score ON news_sentiments(score);

-- 4. Daily Sentiment (Aggregated)
CREATE TABLE IF NOT EXISTS daily_sentiments (
    id BIGSERIAL PRIMARY KEY,
    date TIMESTAMPTZ UNIQUE NOT NULL,
    sentiment_index DOUBLE PRECISION NOT NULL,
    news_count INTEGER NOT NULL DEFAULT 0,
    avg_positive DOUBLE PRECISION,
    avg_neutral DOUBLE PRECISION,
    avg_negative DOUBLE PRECISION,
    weighting_method VARCHAR(50) DEFAULT 'recency_exponential',
    aggregated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_daily_sentiment_date ON daily_sentiments(date);

-- 5. Analysis Snapshots (Cached reports)
CREATE TABLE IF NOT EXISTS analysis_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    as_of_date TIMESTAMPTZ NOT NULL,
    report_json JSONB NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version VARCHAR(100),
    UNIQUE(symbol, as_of_date)
);

CREATE INDEX idx_snapshot_symbol ON analysis_snapshots(symbol);
CREATE INDEX idx_snapshot_generated ON analysis_snapshots(symbol, generated_at);

-- =============================================================================
-- Row Level Security (RLS) - Opsiyonel, şimdilik kapalı
-- =============================================================================
-- ALTER TABLE news_articles ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE price_bars ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE news_sentiments ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE daily_sentiments ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE analysis_snapshots ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- Verification
-- =============================================================================
SELECT 'Tables created successfully!' as status;

