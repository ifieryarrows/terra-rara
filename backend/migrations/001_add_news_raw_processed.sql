-- Migration: 001_add_news_raw_processed.sql
-- Faz 2: Ham/İşlenmiş haber tabloları + reproducible news pipeline
-- 
-- Run on: Supabase PostgreSQL
-- Date: 2026-01-28
-- 
-- IMPORTANT: Run this migration BEFORE deploying Faz 2 pipeline code.

-- =============================================================================
-- 1. news_raw - Ham haber verisi (golden source)
-- =============================================================================

CREATE TABLE IF NOT EXISTS news_raw (
    id BIGSERIAL PRIMARY KEY,
    
    -- URL (nullable - RSS'te eksik olabilir)
    url VARCHAR(2000),
    url_hash VARCHAR(64),  -- sha256, nullable for partial unique
    
    -- Content
    title VARCHAR(500) NOT NULL,
    description TEXT,
    
    -- Metadata
    source VARCHAR(200),  -- "google_news", "newsapi", etc.
    source_feed VARCHAR(500),  -- Exact RSS URL or API query
    published_at TIMESTAMPTZ NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Pipeline run tracking
    run_id UUID,
    
    -- Raw payload for debugging
    raw_payload JSONB
);

-- Basic indexes
CREATE INDEX IF NOT EXISTS ix_news_raw_published ON news_raw(published_at);
CREATE INDEX IF NOT EXISTS ix_news_raw_run ON news_raw(run_id);
CREATE INDEX IF NOT EXISTS ix_news_raw_url_hash ON news_raw(url_hash);

-- PARTIAL UNIQUE INDEX: url_hash must be unique IF it exists
-- This allows NULL url_hash (for articles without URL) while preventing duplicates
CREATE UNIQUE INDEX IF NOT EXISTS ux_news_raw_url_hash
ON news_raw(url_hash)
WHERE url_hash IS NOT NULL;


-- =============================================================================
-- 2. news_processed - İşlenmiş haber (dedup authority)
-- =============================================================================

CREATE TABLE IF NOT EXISTS news_processed (
    id BIGSERIAL PRIMARY KEY,
    
    -- FK to raw (RESTRICT - don't allow deleting raw if processed exists)
    raw_id BIGINT NOT NULL REFERENCES news_raw(id) ON DELETE RESTRICT,
    
    -- Canonical content
    canonical_title VARCHAR(500) NOT NULL,
    canonical_title_hash VARCHAR(64) NOT NULL,  -- sha256
    cleaned_text TEXT,  -- title + description, cleaned
    
    -- Dedup key - THE AUTHORITY
    -- Priority: url_hash if available, else sha256(source + canonical_title_hash)
    dedup_key VARCHAR(64) NOT NULL UNIQUE,
    
    -- Language
    language VARCHAR(10) DEFAULT 'en',
    language_confidence FLOAT,
    
    -- Processing metadata
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID
);

-- Indexes
CREATE INDEX IF NOT EXISTS ix_news_processed_raw_id ON news_processed(raw_id);
CREATE INDEX IF NOT EXISTS ix_news_processed_run ON news_processed(run_id);
CREATE INDEX IF NOT EXISTS ix_news_processed_title_hash ON news_processed(canonical_title_hash);


-- =============================================================================
-- 3. Add Faz 2 columns to pipeline_run_metrics
-- =============================================================================

-- Cut-off time
ALTER TABLE pipeline_run_metrics 
ADD COLUMN IF NOT EXISTS news_cutoff_time TIMESTAMPTZ;

-- Raw stats
ALTER TABLE pipeline_run_metrics 
ADD COLUMN IF NOT EXISTS news_raw_inserted INTEGER;

ALTER TABLE pipeline_run_metrics 
ADD COLUMN IF NOT EXISTS news_raw_duplicates INTEGER;

-- Processed stats
ALTER TABLE pipeline_run_metrics 
ADD COLUMN IF NOT EXISTS news_processed_inserted INTEGER;

ALTER TABLE pipeline_run_metrics 
ADD COLUMN IF NOT EXISTS news_processed_duplicates INTEGER;

-- Quality state for degraded runs
ALTER TABLE pipeline_run_metrics 
ADD COLUMN IF NOT EXISTS quality_state VARCHAR(20) DEFAULT 'ok';


-- =============================================================================
-- Verification queries (run after migration to verify)
-- =============================================================================

-- Check tables exist:
-- SELECT table_name FROM information_schema.tables WHERE table_name IN ('news_raw', 'news_processed');

-- Check partial unique index:
-- SELECT indexname FROM pg_indexes WHERE tablename = 'news_raw' AND indexname = 'ux_news_raw_url_hash';

-- Check FK constraint:
-- SELECT conname FROM pg_constraint WHERE conrelid = 'news_processed'::regclass AND confrelid = 'news_raw'::regclass;
