from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from deep_learning.training.trainer import (
    _build_uniform_checkpoint_soup,
    _validation_ranked_checkpoint_paths,
)


def _write_checkpoint(path, *, weight: float, counter: int, marker: str) -> None:
    torch.save(
        {
            "state_dict": {
                "layer.weight": torch.tensor([weight], dtype=torch.float32),
                "step_counter": torch.tensor(counter, dtype=torch.int64),
            },
            "hyper_parameters": {"marker": marker},
        },
        path,
    )


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_validation_checkpoint_ranking_uses_monitor_score_then_path(tmp_path):
    later = tmp_path / "epoch-20.ckpt"
    earlier = tmp_path / "epoch-10.ckpt"
    third = tmp_path / "epoch-30.ckpt"
    callback = SimpleNamespace(
        best_k_models={
            str(later): torch.tensor(2.0),
            str(third): torch.tensor(3.0),
            str(earlier): torch.tensor(2.0),
        },
        best_model_path=str(later),
        best_model_score=torch.tensor(2.0),
    )

    ranked = _validation_ranked_checkpoint_paths(callback, max_count=2)

    assert ranked == [(earlier, 2.0), (later, 2.0)]


def test_uniform_checkpoint_soup_averages_only_floating_state(tmp_path):
    best = tmp_path / "best.ckpt"
    second = tmp_path / "second.ckpt"
    output = tmp_path / "promoted.ckpt"
    _write_checkpoint(best, weight=1.0, counter=10, marker="best")
    _write_checkpoint(second, weight=3.0, counter=20, marker="second")

    _build_uniform_checkpoint_soup([best, second], output)

    payload = _load_checkpoint(output)
    assert payload["state_dict"]["layer.weight"].item() == pytest.approx(2.0)
    assert payload["state_dict"]["step_counter"].item() == 10
    assert payload["hyper_parameters"]["marker"] == "best"


def test_uniform_checkpoint_soup_rejects_incompatible_state(tmp_path):
    best = tmp_path / "best.ckpt"
    incompatible = tmp_path / "incompatible.ckpt"
    output = tmp_path / "promoted.ckpt"
    _write_checkpoint(best, weight=1.0, counter=10, marker="best")
    torch.save({"state_dict": {"other.weight": torch.tensor([3.0])}}, incompatible)

    with pytest.raises(RuntimeError, match="state keys differ"):
        _build_uniform_checkpoint_soup([best, incompatible], output)
