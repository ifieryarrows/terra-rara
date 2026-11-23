-- Migration 005: Create long_term_outlook_snapshots table for 20D/60D structural forecasts.
-- Isolated from 5D TFT-ASRO checkpoints and snapshots.

CREATE TABLE IF NOT EXISTS long_term_outlook_snapshots (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL DEFAULT 'HG=F',
    as_of_date VARCHAR(10) NOT NULL,
    payload_json JSONB NOT NULL,
    validation_json JSONB,
    model_version VARCHAR(80) NOT NULL DEFAULT 'lt_structural_v1',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(80)
);

CREATE INDEX IF NOT EXISTS ix_lt_outlook_symbol_generated
    ON long_term_outlook_snapshots (symbol, generated_at);

CREATE INDEX IF NOT EXISTS ix_lt_outlook_symbol_asof
    ON long_term_outlook_snapshots (symbol, as_of_date);

CREATE INDEX IF NOT EXISTS ix_lt_outlook_run_id
    ON long_term_outlook_snapshots (run_id);
