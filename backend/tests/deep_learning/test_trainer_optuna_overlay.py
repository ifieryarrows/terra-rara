import json
from dataclasses import replace

from deep_learning.config import TFTASROConfig, TrainingConfig, get_tft_config
from deep_learning.training import trainer as trainer_module


def test_apply_optuna_results_falls_back_on_structural_failure(tmp_path, monkeypatch):
    model_root = tmp_path / "tft"
    model_root.mkdir()
    results_path = model_root / "optuna_results.json"
    results_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "best_trial": 9,
                "best_value": 6.004787,
                "best_params": {
                    "lambda_magnitude": 0.25,
                    "lambda_bias": 0.0,
                    "lambda_directional": 0.0,
                },
                "structural_invalidity_report": {
                    "verdict": "STRUCTURAL_FAILURE",
                    "next_action": "Do not run additional hyperopt.",
                },
            }
        )
    )
    fallback = dict(trainer_module.KNOWN_GOOD_CONFIG)
    fallback["lambda_magnitude"] = 0.61
    fallback["lambda_bias"] = 0.19
    monkeypatch.setattr(trainer_module, "KNOWN_GOOD_CONFIG", fallback)

    cfg = get_tft_config()
    cfg = replace(
        cfg,
        training=TrainingConfig(
            checkpoint_dir=str(model_root / "checkpoints"),
            best_model_path=str(model_root / "best_tft_asro.ckpt"),
        ),
    )

    resolved = trainer_module._apply_optuna_results(cfg)

    assert resolved.weekly_loss.lambda_magnitude == 0.61
    assert resolved.weekly_loss.lambda_bias == 0.19


def test_apply_optuna_results_preserves_controlled_weekly_search_bounds(tmp_path):
    model_root = tmp_path / "tft"
    model_root.mkdir()
    results_path = model_root / "optuna_results.json"
    results_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "best_trial": 2,
                "best_value": 0.5,
                "best_params": {
                    "lambda_magnitude": 0.50,
                    "lambda_naive": 0.35,
                    "lambda_bias": 0.14,
                    "lambda_directional": 0.05,
                },
                "structural_invalidity_report": {
                    "verdict": "ACCEPTABLE",
                },
                "best_trial_preflight": {
                    "preflight_passed": True,
                },
            }
        )
    )

    cfg = get_tft_config()
    cfg = replace(
        cfg,
        training=TrainingConfig(
            checkpoint_dir=str(model_root / "checkpoints"),
            best_model_path=str(model_root / "best_tft_asro.ckpt"),
        ),
    )

    resolved = trainer_module._apply_optuna_results(cfg)

    assert resolved.weekly_loss.lambda_magnitude == 0.50
    assert resolved.weekly_loss.lambda_naive == 0.35
    assert resolved.weekly_loss.lambda_bias == 0.14
    assert resolved.weekly_loss.lambda_directional == 0.05
