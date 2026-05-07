import pandas as pd

from deep_learning.config import get_tft_config
from deep_learning.data.future_frame import build_future_decoder_rows


def test_future_decoder_rows_do_not_forward_fill_news_or_returns():
    cfg = get_tft_config()
    idx = pd.date_range("2026-05-01", periods=3, freq="B")
    history = pd.DataFrame(
        {
            "time_idx": [0, 1, 2],
            "group_id": ["copper"] * 3,
            "target": [0.01, -0.02, 0.03],
            "target_1d_log_return": [0.01, -0.02, 0.03],
            "target_5d_log_return": [0.05, 0.04, 0.03],
            "realized_vol_20d": [0.01, 0.01, 0.01],
            "material_move_5d": [0.0, 1.0, 1.0],
            "sentiment_index": [0.5, 0.5, 0.5],
            "news_count": [4.0, 4.0, 4.0],
            "event_shock_score": [2.0, 2.0, 2.0],
            "emb_pca_0": [1.0, 1.0, 1.0],
            "HG_F_ret1": [0.2, 0.2, 0.2],
            "regime_usd_pressure": [0.0, 1.0, 1.0],
            "day_of_week": [0.0, 0.0, 0.0],
        },
        index=idx,
    )
    future = build_future_decoder_rows(history, 5, cfg)
    assert len(future) == 5
    assert future["sentiment_index"].eq(0.0).all()
    assert future["news_count"].eq(0.0).all()
    assert future["event_shock_score"].eq(0.0).all()
    assert future["emb_pca_0"].eq(0.0).all()
    assert future["HG_F_ret1"].eq(0.0).all()
    assert future["regime_usd_pressure"].eq(1.0).all()
    assert not future["day_of_week"].eq(0.0).all()
