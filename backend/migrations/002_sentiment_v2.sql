-- Migration: 002_sentiment_v2.sql
-- Commodity-aware sentiment pipeline (V2)
--
-- Adds:
--   1. news_sentiments_v2 (article-level scores on news_processed)
--   2. daily_sentiments_v2 (daily aggregate from V2 scores)
--   3. Stage-2 V2 run metrics columns on pipeline_run_metrics

BEGIN;

CREATE TABLE IF NOT EXISTS news_sentiments_v2 (
    id BIGSERIAL PRIMARY KEY,
    news_processed_id BIGINT NOT NULL REFERENCES news_processed(id) ON DELETE CASCADE,
    horizon_days INTEGER NOT NULL DEFAULT 5,

    label VARCHAR(20) NOT NULL,
    impact_score_llm FLOAT NOT NULL,
    confidence_llm FLOAT NOT NULL,
    confidence_calibrated FLOAT NOT NULL,
    relevance_score FLOAT NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    rule_sign INTEGER NOT NULL,
    final_score FLOAT NOT NULL,

    finbert_pos FLOAT NOT NULL,
    finbert_neu FLOAT NOT NULL,
    finbert_neg FLOAT NOT NULL,

    reasoning_json TEXT,
    model_fast VARCHAR(100),
    model_reliable VARCHAR(100),
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_news_sentiments_v2_processed_horizon UNIQUE (news_processed_id, horizon_days)
);

CREATE INDEX IF NOT EXISTS ix_news_sentiments_v2_processed_scored
ON news_sentiments_v2 (news_processed_id, scored_at);

CREATE INDEX IF NOT EXISTS ix_news_sentiments_v2_final_score
ON news_sentiments_v2 (final_score);

CREATE INDEX IF NOT EXISTS ix_news_sentiments_v2_label
ON news_sentiments_v2 (label);

CREATE TABLE IF NOT EXISTS daily_sentiments_v2 (
    id BIGSERIAL PRIMARY KEY,
    date TIMESTAMPTZ NOT NULL UNIQUE,
    sentiment_index FLOAT NOT NULL,
    news_count INTEGER NOT NULL DEFAULT 0,
    avg_confidence FLOAT,
    avg_relevance FLOAT,
    source_version VARCHAR(20) NOT NULL DEFAULT 'v2',
    aggregated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_daily_sentiments_v2_date
ON daily_sentiments_v2 (date);

CREATE INDEX IF NOT EXISTS ix_daily_sentiments_v2_index
ON daily_sentiments_v2 (sentiment_index);

ALTER TABLE pipeline_run_metrics
ADD COLUMN IF NOT EXISTS articles_scored_v2 INTEGER;

ALTER TABLE pipeline_run_metrics
ADD COLUMN IF NOT EXISTS llm_parse_fail_count INTEGER;

ALTER TABLE pipeline_run_metrics
ADD COLUMN IF NOT EXISTS escalation_count INTEGER;

ALTER TABLE pipeline_run_metrics
ADD COLUMN IF NOT EXISTS fallback_count INTEGER;

COMMIT;

