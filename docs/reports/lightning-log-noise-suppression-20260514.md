# Lightning Log Noise Suppression - 2026-05-14

## Context

Weekly TFT hyperopt/training logs included non-actionable Lightning promotional tips and a recurrent warning caused by a fixed `log_every_n_steps=20` value when fold batch count was smaller (for example 9).

## Changes Applied

1. Dynamic `log_every_n_steps` in hyperopt folds

- File: `backend/deep_learning/training/hyperopt.py`
- Change: replaced fixed `log_every_n_steps=20` with:
  - `log_steps = max(1, min(5, len(fold_train_dl)))`
  - `log_every_n_steps=log_steps`
- Also set `logger=False` for fold-level hyperopt Trainer instances.

2. Shared Lightning noise suppression helper

- New file: `backend/deep_learning/logging_utils.py`
- Added:
  - `suppress_lightning_noise()`
  - `configure_cli_logging()`
  - `LightningNoiseFilter`
- Behavior:
  - Sets `DISABLE_LIGHTNING_ADVICE=1` (if not already set).
  - Suppresses known `litlogger` / `litmodels` promotional messages.
  - Raises third-party logger thresholds (`lightning`, `lightning.pytorch`, `lightning.pytorch.utilities.rank_zero`, `pytorch_lightning`) to `WARNING`.

3. Apply suppression before Lightning import

- Files:
  - `backend/deep_learning/training/hyperopt.py`
  - `backend/deep_learning/training/trainer.py`
- Change: call `suppress_lightning_noise()` immediately before Lightning import blocks.

4. Cleaner CLI log configuration

- Files:
  - `backend/deep_learning/training/hyperopt.py`
  - `backend/deep_learning/training/trainer.py`
- Change: replaced local `logging.basicConfig(...)` calls with `configure_cli_logging(logging.INFO)` to keep project INFO logs while reducing third-party INFO noise.

## Validation Evidence

1. Compile checks:

- `py -m compileall backend/deep_learning/logging_utils.py backend/deep_learning/training/hyperopt.py backend/deep_learning/training/trainer.py`
- Result: passed.

2. Targeted regression tests:

- `py -m pytest backend/tests/deep_learning/test_hyperopt.py backend/tests/deep_learning/test_config.py backend/tests/deep_learning/test_trainer_weekly_evaluation.py -q -m "not online"`
- Result: `25 passed` (with existing environment warnings unrelated to this change).

## Expected Operational Impact

- Lightning promotional tip lines for `litlogger`/`litmodels` are suppressed.
- Hyperopt fold warnings due to `log_every_n_steps > train batches` are eliminated by dynamic log step sizing.
- Real warnings/errors remain visible.
