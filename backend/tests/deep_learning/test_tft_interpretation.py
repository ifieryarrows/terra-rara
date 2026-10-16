"""Tests for the TFT interpretation core (interpret_batch) and variable importance."""

import math
from types import SimpleNamespace

import pytest
import torch

from deep_learning.models.tft_copper import get_variable_importance, interpret_batch


class _StubModel:
    """Stand-in for TemporalFusionTransformer covering the interpretation contract."""

    encoder_variables = ["HG=F_close_sma_5", "HG=F_sentiment_mean"]
    decoder_variables = []
    static_variables = ["symbol_id"]
    hparams = SimpleNamespace(max_encoder_length=3)

    def __init__(
        self,
        encoder_weights=(0.7, 0.3),
        attention=(0.2, 0.3, 0.5),
        encoder_length: int = 3,
    ):
        self._encoder_weights = list(encoder_weights)
        self._attention = list(attention)
        self._encoder_length = encoder_length
        self.interpret_calls = []
        self.grad_enabled_during_call = None

    def __call__(self, x):
        self.grad_enabled_during_call = torch.is_grad_enabled()
        return {
            "encoder_lengths": torch.tensor([self._encoder_length]),
            "decoder_lengths": torch.tensor([5]),
        }

    def interpret_output(self, out, reduction="none", attention_prediction_horizon=0):
        self.interpret_calls.append(
            {"reduction": reduction, "attention_prediction_horizon": attention_prediction_horizon}
        )
        return {
            "encoder_variables": torch.tensor([self._encoder_weights]),
            "decoder_variables": torch.zeros((1, 0)),
            "static_variables": torch.tensor([[1.0]]),
            "attention": torch.tensor([self._attention]),
        }


class _PerBatchWeightsModel(_StubModel):
    """Stub returning a different encoder weight vector for each interpreted batch."""

    encoder_variables = ["B", "A"]

    def __init__(self, per_batch_weights):
        super().__init__()
        self._per_batch_weights = [list(weights) for weights in per_batch_weights]
        self._batch_index = 0

    def interpret_output(self, out, reduction="none", attention_prediction_horizon=0):
        weights = self._per_batch_weights[self._batch_index]
        self._batch_index += 1
        return {
            "encoder_variables": torch.tensor([weights]),
            "decoder_variables": torch.zeros((1, 0)),
            "static_variables": torch.tensor([[1.0]]),
            "attention": torch.tensor([[0.2, 0.3, 0.5]]),
        }


class _BatchLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)


def test_interpret_batch_maps_group_names_attention_and_encoder_length():
    model = _StubModel()

    result = interpret_batch(model, {"encoder_lengths": torch.tensor([3])})

    assert result["encoder"] == [
        {"feature": "HG=F_close_sma_5", "weight": pytest.approx(0.7)},
        {"feature": "HG=F_sentiment_mean", "weight": pytest.approx(0.3)},
    ]
    assert result["decoder"] == []
    assert result["static"] == [{"feature": "symbol_id", "weight": pytest.approx(1.0)}]
    assert result["attention"] == pytest.approx([0.2, 0.3, 0.5])
    assert result["encoder_length"] == 3
    assert model.interpret_calls == [{"reduction": "none", "attention_prediction_horizon": 0}]
    assert model.grad_enabled_during_call is False


def test_interpret_batch_clips_attention_to_encoder_window_and_forwards_horizon():
    model = _StubModel(attention=(0.1, 0.2, 0.3, 0.25, 0.15))

    result = interpret_batch(
        model, {"encoder_lengths": torch.tensor([3])}, attention_prediction_horizon=2
    )

    assert result["attention"] == pytest.approx([0.1, 0.2, 0.3])
    assert model.interpret_calls == [{"reduction": "none", "attention_prediction_horizon": 2}]


def test_interpret_batch_raises_on_names_weights_mismatch():
    model = _StubModel(encoder_weights=(0.5, 0.3, 0.2))

    with pytest.raises(ValueError, match="names/weights mismatch"):
        interpret_batch(model, {"encoder_lengths": torch.tensor([3])})


def test_get_variable_importance_averages_normalizes_and_sorts():
    # Two batches (already shaped like real (x, y) dataloader items), names in
    # reversed order so a descending result cannot be dict insertion order.
    model = _PerBatchWeightsModel(per_batch_weights=[(0.6, 0.9), (0.4, 0.6)])
    loader = _BatchLoader(
        [
            ({"encoder_lengths": torch.tensor([3])}, None),
            ({"encoder_lengths": torch.tensor([3])}, None),
        ]
    )

    result = get_variable_importance(model, val_dataloader=loader)

    # Means: B=0.5, A=0.75 => normalised to A=0.6, B=0.4.
    assert result == pytest.approx({"A": 0.6, "B": 0.4})
    assert list(result) == ["A", "B"]
    assert all(isinstance(weight, float) for weight in result.values())
    assert sum(result.values()) == pytest.approx(1.0)
    assert model._batch_index == 2


def test_get_variable_importance_empty_without_loader_or_batches():
    model = _PerBatchWeightsModel(per_batch_weights=[(0.6, 0.9)])

    assert get_variable_importance(model) == {}
    assert get_variable_importance(model, val_dataloader=_BatchLoader([])) == {}
    assert model._batch_index == 0


def test_get_variable_importance_returns_empty_when_interpretation_fails(monkeypatch):
    # Importance is best-effort promotion metadata: an interpretation failure
    # (e.g. on a real checkpoint never exercised offline) must never propagate.
    def _raise(*args, **kwargs):
        raise RuntimeError("interpretation exploded")

    monkeypatch.setattr("deep_learning.models.tft_copper._run_interpretation", _raise)
    model = _PerBatchWeightsModel(per_batch_weights=[(0.6, 0.9)])
    loader = _BatchLoader([({"encoder_lengths": torch.tensor([3])}, None)])

    assert get_variable_importance(model, val_dataloader=loader) == {}


class _MultiSampleModel:
    """Stub whose interpret_output returns one weight row per batch sample."""

    encoder_variables = ["A", "B"]
    decoder_variables = []
    static_variables = []
    hparams = SimpleNamespace(max_encoder_length=3)

    def __init__(self, batches):
        self._batches = batches
        self._index = 0

    def __call__(self, x):
        return {"encoder_lengths": torch.tensor([3])}

    def interpret_output(self, out, reduction="none", attention_prediction_horizon=0):
        rows = self._batches[self._index]
        self._index += 1
        return {
            "encoder_variables": torch.tensor(rows),
            "decoder_variables": torch.zeros((1, 0)),
            "static_variables": torch.zeros((1, 0)),
            "attention": torch.tensor([[0.2, 0.3, 0.5]]) * len(rows),
        }


def test_get_variable_importance_uses_every_sample_of_each_batch():
    # Two samples per batch: the mean must pool all four samples, not just the
    # first sample of each batch. A: mean(0.1, 0.3, 0.5, 0.7) = 0.4,
    # B: mean(0.9, 0.7, 0.5, 0.3) = 0.6 -> normalized {B: 0.6, A: 0.4}.
    model = _MultiSampleModel(
        batches=[
            [[0.1, 0.9], [0.3, 0.7]],
            [[0.5, 0.5], [0.7, 0.3]],
        ]
    )
    loader = _BatchLoader([({"encoder_lengths": torch.tensor([3])}, None)] * 2)

    result = get_variable_importance(model, val_dataloader=loader)

    assert result == pytest.approx({"B": 0.6, "A": 0.4})
    assert sum(result.values()) == pytest.approx(1.0)


def test_get_variable_importance_ignores_non_finite_weights_instead_of_poisoning():
    # A degenerate batch with NaN weights (cf. exploding checkpoints) must not
    # turn the whole persisted map into NaN; finite contributions survive.
    model = _MultiSampleModel(
        batches=[
            [[0.2, 0.8], [float("nan"), 0.9]],
        ]
    )
    loader = _BatchLoader([({"encoder_lengths": torch.tensor([3])}, None)])

    result = get_variable_importance(model, val_dataloader=loader)

    # A: only the finite sample contributes (avg 0.2); B: (0.8 + 0.9) / 2 = 0.85.
    # Normalized over the finite total 1.05: B = 0.809523..., A = 0.190476...
    assert result["B"] == pytest.approx(0.85 / 1.05)
    assert result["A"] == pytest.approx(0.2 / 1.05)
    assert all(math.isfinite(value) for value in result.values())


def test_get_variable_importance_empty_when_every_weight_is_non_finite():
    model = _MultiSampleModel(
        batches=[
            [[float("nan"), float("inf")]],
        ]
    )
    loader = _BatchLoader([({"encoder_lengths": torch.tensor([3])}, None)])

    assert get_variable_importance(model, val_dataloader=loader) == {}
