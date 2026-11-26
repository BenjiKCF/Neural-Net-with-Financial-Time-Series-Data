"""Bounded adapter-runner boundaries using a tiny backend and in-memory Qlib data."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from financial_ts import loop_experiment
from financial_ts.workflow import resolve_segments


class TinyBackend(nn.Module):
    instances = []
    reverse_validation = False
    unfreeze_base = False

    def __init__(self, checkpoint, device, **kwargs):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(0.03), requires_grad=self.unfreeze_base)
        self.gate = nn.Linear(1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.parameter_count = 3
        self.trainable_parameter_count = 2
        self.context_calls = []
        self.to(device)
        self.instances.append(self)

    def base_parameters(self):
        return iter([self.base])

    def adapter_state_dict(self):
        return {
            name: value for name, value in self.state_dict().items() if name.startswith("gate.")
        }

    def load_adapter_state_dict(self, state):
        self.load_state_dict(state, strict=False)

    def baseline_forecast(self, contexts):
        return torch.stack((contexts[:, -1], contexts[:, -1] * (1 + self.base)), dim=-1)

    def forecast(self, contexts):
        self.context_calls.extend(contexts.detach().cpu().numpy().copy())
        correction = self.gate(torch.ones(len(contexts), 1, device=contexts.device)).squeeze(-1)
        if self.reverse_validation and not self.training:
            correction = -correction
        return torch.stack(
            (contexts[:, -1], contexts[:, -1] * (1 + self.base + correction)), dim=-1
        )

    def baseline_return_scores(self, contexts):
        forecast = self.baseline_forecast(contexts).double()
        return forecast[:, 1] / forecast[:, 0] - 1

    def return_scores(self, contexts):
        forecast = self.forecast(contexts).double()
        return forecast[:, 1] / forecast[:, 0] - 1


@pytest.fixture
def market(tmp_path, monkeypatch):
    import qlib
    import qlib.data

    calendar = pd.bdate_range("2023-01-03", periods=180)
    segments = resolve_segments(calendar)
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars/day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")))
    (provider / "metadata.json").write_text(json.dumps({"synthetic": True}))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"fake checkpoint; never deserialized")
    position = np.arange(len(calendar))
    frame = pd.DataFrame(
        {
            "$close": np.concatenate(
                [
                    100 + 100 * number + 0.2 * position + 0.1 * np.sin(position / 3)
                    for number in range(2)
                ]
            ).astype(np.float32)
        },
        index=pd.MultiIndex.from_product(
            [["AAA", "BBB"], calendar], names=["instrument", "datetime"]
        ),
    )
    requests = []

    def features(instruments, fields, start_time, end_time, freq):
        assert fields == ["$close"]
        assert pd.Timestamp(end_time) < pd.Timestamp(segments["test"][0])
        requests.append((pd.Timestamp(start_time), pd.Timestamp(end_time)))
        dates = frame.index.get_level_values("datetime")
        return frame.loc[(dates >= start_time) & (dates <= end_time)].copy()

    monkeypatch.setattr(qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        qlib.data,
        "D",
        SimpleNamespace(
            instruments=lambda name: name,
            list_instruments=lambda *args, **kwargs: ["AAA", "BBB"],
            features=features,
        ),
    )
    TinyBackend.instances = []
    monkeypatch.setattr(
        loop_experiment, "_make_model", lambda checkpoint, device: TinyBackend(checkpoint, device)
    )
    return SimpleNamespace(
        provider=provider,
        checkpoint=checkpoint,
        calendar=calendar,
        segments=segments,
        frame=frame,
        requests=requests,
    )


def _run(market, output, **kwargs):
    settings = dict(epochs=2, train_samples=12, valid_samples=8, context_length=16, batch_size=4)
    settings.update(kwargs)
    return loop_experiment.run_loop_experiment(
        market.provider, market.checkpoint, output, **settings
    )


def test_chronological_samples_training_and_safe_adapter_replay(market, tmp_path):
    output = tmp_path / "run"
    result = _run(market, output)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    model = TinyBackend.instances[-1]
    metrics, manifest = result["metrics"], result["run"]
    assert manifest["status"] == "completed" and manifest["synthetic"] is True
    assert manifest["segments"] == {name: list(bounds) for name, bounds in market.segments.items()}
    assert metrics["test_evaluated"] is False and metrics["pretraining_overlap"] == "unknown"
    assert metrics["optimizer_updates"] == 6 and metrics["training_examples_seen"] == 24
    assert metrics["trainable_parameter_count"] == 2
    assert model.base.item() == pytest.approx(0.03) and model.base.grad is None
    assert manifest["parameters"]["optimizer_names"] == ["gate.weight", "gate.bias"]
    assert metrics["last_epoch_gate_change_l2"] > 0
    assert metrics["adapter_replay_max_score_error"] == 0
    assert manifest["baseline_parity"]["max_forecast_absolute_error"] == 0
    assert metrics["best_epoch"] == int(
        np.argmin([row["validation_rmse"] for row in result["learning_curves"]])
    )
    samples = json.loads((output / "samples.json").read_text())
    assert set(samples) == {"train", "valid"}
    allowed_contexts = []
    for partition, rows in samples.items():
        assert {row["instrument"] for row in rows} == {"AAA", "BBB"}
        counts = pd.Series([row["instrument"] for row in rows]).value_counts()
        assert counts.max() - counts.min() <= 1
        next_start = market.segments["valid" if partition == "train" else "test"][0]
        for row in rows:
            assert market.segments[partition][0] <= row["datetime"] <= market.segments[partition][1]
            assert row["context_end"] == row["datetime"]
            assert (
                row["context_start"]
                < row["datetime"]
                < row["entry_date"]
                < row["exit_date"]
                < next_start
            )
            prices = market.frame.xs(row["instrument"])["$close"]
            context = prices.loc[row["context_start"] : row["context_end"]].to_numpy()
            assert len(context) == 16
            allowed_contexts.append(context)
            assert row["label"] == pytest.approx(
                float(prices.loc[row["exit_date"]] / prices.loc[row["entry_date"]] - 1)
            )
    assert all(
        any(np.array_equal(context, allowed) for allowed in allowed_contexts)
        for context in model.context_calls
    )
    labels = np.asarray([row["label"] for row in samples["train"]])
    normalization = manifest["target_normalization"]
    assert normalization["mean"] == pytest.approx(labels.mean())
    assert normalization["scale"] == pytest.approx(labels.std(ddof=0))
    assert len(market.requests) == 1
    assert market.requests[0][1] == pd.Timestamp(manifest["last_price_read"])
    state = torch.load(output / "adapter.pt", weights_only=True)
    assert set(state) == {"gate.weight", "gate.bias"}
    assert json.loads((output / "metrics.json").read_text()) == metrics
    assert (output / "validation_predictions.csv").is_file()
    before = (output / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        _run(market, output)
    assert (output / "metrics.json").read_bytes() == before


def test_epoch_zero_remains_eligible_when_adapter_updates_worsen_validation(
    market, tmp_path, monkeypatch
):
    class WorseningBackend(TinyBackend):
        reverse_validation = True

    monkeypatch.setattr(
        loop_experiment,
        "_make_model",
        lambda checkpoint, device: WorseningBackend(checkpoint, device),
    )
    result = _run(market, tmp_path / "worse")
    metrics = result["metrics"]
    assert metrics["best_epoch"] == 0
    assert metrics["optimizer_updates"] == 6
    assert metrics["last_epoch_gate_change_l2"] > 0
    assert metrics["selected_gate_change_l2"] == 0
    assert metrics["selected_validation_rmse"] == metrics["baseline_validation_rmse"]


def test_fixed_seed_repeats_selection_and_sample_hash(market, tmp_path):
    first = _run(market, tmp_path / "first", epochs=1)
    test_period = market.frame.index.get_level_values("datetime") >= pd.Timestamp(
        market.segments["test"][0]
    )
    market.frame.loc[test_period, "$close"] *= 1000
    second = _run(market, tmp_path / "second", epochs=1)
    assert first["learning_curves"] == second["learning_curves"]
    assert first["metrics"] == second["metrics"]
    assert first["run"]["samples_file"]["sha256"] == second["run"]["samples_file"]["sha256"]


def test_unfrozen_base_is_rejected_and_manifest_records_failure(market, tmp_path, monkeypatch):
    class UnfrozenBackend(TinyBackend):
        unfreeze_base = True

    monkeypatch.setattr(
        loop_experiment,
        "_make_model",
        lambda checkpoint, device: UnfrozenBackend(checkpoint, device),
    )
    output = tmp_path / "unfrozen"
    with pytest.raises(ValueError, match="frozen"):
        _run(market, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (output / "adapter.pt").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"train_samples": 0},
        {"context_length": 1},
        {"batch_size": 0},
        {"lr": float("nan")},
        {"seed": True},
        {"seed": -1},
    ],
)
def test_invalid_budget_is_rejected_before_output_creation(market, tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        _run(market, output, **kwargs)
    assert not output.exists()
