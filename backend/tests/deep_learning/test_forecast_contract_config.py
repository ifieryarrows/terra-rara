from deep_learning.config import get_tft_config
from deep_learning.contract import FORECAST_CONTRACT_VERSION, TARGET_RETURN_TYPE, log_to_simple_return


def test_forecast_contract_defaults_are_weekly():
    cfg = get_tft_config()
    assert cfg.model.max_prediction_length == 5
    assert cfg.forecast.primary_horizon_days == 5
    assert cfg.forecast.primary_target_col == "target_5d_log_return"
    assert cfg.forecast.model_daily_target_col == "target"
    assert cfg.forecast.target_return_type == "log_return"
    assert cfg.weekly_loss.lambda_weekly_quantile == 0.70
    assert cfg.weekly_loss.lambda_t1_quantile == 0.20
    assert cfg.weekly_loss.lambda_dispersion == 0.35
    assert cfg.weekly_loss.lambda_magnitude == 0.55
    assert cfg.weekly_loss.lambda_naive == 0.40
    assert cfg.weekly_loss.lambda_bias == 0.10
    assert cfg.weekly_loss.lambda_directional == 0.08
    assert TARGET_RETURN_TYPE == "log_return"
    assert FORECAST_CONTRACT_VERSION == "weekly_log_v1"


def test_log_to_simple_return_contract():
    assert log_to_simple_return(0.0) == 0.0
    assert round(log_to_simple_return(0.05), 8) == round(0.05127109637602412, 8)
