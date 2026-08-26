-- Migration 004: Add quality_gate_passed to tft_model_metadata
-- Tracks whether the most recent training run passed the deployment quality gate.
-- NULL means the model was persisted before this column existed (legacy rows).

ALTER TABLE tft_model_metadata
ADD COLUMN IF NOT EXISTS quality_gate_passed BOOLEAN;
