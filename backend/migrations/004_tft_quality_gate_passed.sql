-- Migration 004: Add quality_gate_passed to tft_model_metadata
-- Tracks whether the single active model passed the deployment quality gate.
-- NULL means the model was persisted before this column existed (legacy rows).
-- Runtime startup and CI promotion also apply this migration idempotently.

ALTER TABLE tft_model_metadata
ADD COLUMN IF NOT EXISTS quality_gate_passed BOOLEAN;
