"""Pin the positional alignment between the encoder window and the explain payload.

The riskiest join in the XAI feature: ``predictor._latest_window_explain`` must
label attention weights with the *same* sessions that produced the forward
pass — ``history.index`` for the encoder window — and read recent deltas from
the matching columns.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from deep_learning.inference.predictor import TFTPredictor


class _StubModel:
    """Text-minimal stand-in for a TFT exposing the interpretation contract."""

    encoder_variables = ["HG=F_close", "HG=F_sentiment_mean"]
    decoder_variables = []
    static_variables = []
    hparams = SimpleNamespace(max_encoder_length=3)

    def __call__(self, x):
        return {"encoder_lengths": torch.tensor([3]), "decoder_lengths": torch.tensor([2])}

    def interpret_output(self, out, reduction="none", attention_prediction_horizon=0):
        return {
            "encoder_variables": torch.tensor([[0.7, 0.3]]),
            "decoder_variables": torch.zeros((1, 0)),
            "static_variables": torch.zeros((1, 0)),
            "attention": torch.tensor([[0.2, 0.3, 0.5]]),
        }


class _OneSampleLoader:
    def __init__(self, x):
        self._items = [(x, None)]

    def __iter__(self):
        return iter(self._items)


@pytest.fixture
def predictor_stub():
    instance = TFTPredictor.__new__(TFTPredictor)
    instance._checkpoint_path = "artifacts/tft/best_tft_asro.ckpt"
    instance.get_model_metadata = lambda session: {"trained_at": "2026-10-01T00:00:00+00:00"}
    instance._date_label = TFTPredictor._date_label
    return instance


def test_attention_series_dates_match_the_encoder_window(predictor_stub):
    dates = ["2026-10-09", "2026-10-12", "2026-10-13"]
    history = pd.DataFrame(
        {
            "HG=F_close": [6.0, 6.1, 6.06],
            "HG=F_sentiment_mean": [0.1, 0.2, 0.15],
        },
        index=pd.Index(dates, name="date"),
    )

    payload = predictor_stub._latest_window_explain(
        session=None,
        model=_StubModel(),
        dataloader=_OneSampleLoader({"encoder_lengths": torch.tensor([3])}),
        history=history,
        symbol="HG=F",
        reference_price_date="2026-10-13",
    )

    assert payload is not None
    series_dates = [point["date"] for point in payload["attention"]["series"]]
    assert series_dates == dates  # attention day i <- encoder day i
    assert payload["attention"]["window"] == {
        "start": dates[0],
        "end": dates[-1],
        "days": len(dates),
    }
    assert payload["symbol"] == "HG=F"
    assert payload["reference_price_date"] == "2026-10-13"
    assert payload["model"]["checkpoint"] == "artifacts/tft/best_tft_asro.ckpt"
    assert payload["model"]["trained_at"] == "2026-10-01T00:00:00+00:00"

    # Recent deltas must come from the history frame columns, in payload order.
    top = payload["variable_selection"]["encoder"][0]
    assert top["feature"] == "HG=F_close"
    assert top["recent"]["last_value"] == 6.06
    assert top["recent"]["as_of"] == dates[-1]


def test_latest_window_explain_is_fail_open(predictor_stub):
    class _BrokenModel(_StubModel):
        def interpret_output(self, *args, **kwargs):
            raise RuntimeError("interpretation exploded")

    payload = predictor_stub._latest_window_explain(
        session=None,
        model=_BrokenModel(),
        dataloader=_OneSampleLoader({"encoder_lengths": torch.tensor([3])}),
        history=pd.DataFrame({"HG=F_close": [1.0, 2.0, 3.0]}, index=["a", "b", "c"]),
        symbol="HG=F",
        reference_price_date="2026-10-13",
    )

    assert payload is None  # an explanation must never fail a prediction


def test_latest_window_explain_accepts_bare_dict_batch(predictor_stub):
    dates = ["2026-10-09", "2026-10-12", "2026-10-13"]
    history = pd.DataFrame(
        {
            "HG=F_close": [6.0, 6.1, 6.06],
            "HG=F_sentiment_mean": [0.1, 0.2, 0.15],
        },
        index=pd.Index(dates, name="date"),
    )

    payload = predictor_stub._latest_window_explain(
        session=None,
        model=_StubModel(),
        dataloader=[{"encoder_lengths": torch.tensor([3])}],
        history=history,
        symbol="HG=F",
        reference_price_date="2026-10-13",
    )

    assert payload is not None
    assert [point["date"] for point in payload["attention"]["series"]] == dates
