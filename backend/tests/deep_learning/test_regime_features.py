import pandas as pd

from deep_learning.data.regime_features import REGIME_FEATURES, build_regime_event_features


def test_regime_features_are_complete_and_finite():
    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    df = pd.DataFrame(
        {
            "sentiment_index": [0.1] * 120,
            "news_count": [2] * 120,
            "evt_supply_disruption_count": [0] * 119 + [1],
            "evt_inventory_draw_count": [0] * 120,
            "target": [0.001] * 120,
        },
        index=idx,
    )
    out = build_regime_event_features(df)
    assert set(REGIME_FEATURES).issubset(out.columns)
    assert out.notna().all().all()
    assert out["regime_supply_shock"].iloc[-1] == 1.0


def test_regime_volatility_is_independent_from_forward_target():
    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    observed = pd.Series([0.001, -0.002, 0.003, -0.001] * 30, index=idx)
    base = pd.DataFrame({"observed_return_1d": observed, "target": 0.0}, index=idx)
    altered_target = base.copy()
    altered_target["target"] = [0.20, -0.20] * 60

    base_features = build_regime_event_features(base)
    altered_features = build_regime_event_features(altered_target)

    pd.testing.assert_frame_equal(base_features, altered_features)
