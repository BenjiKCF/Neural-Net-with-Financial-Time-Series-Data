"""Matched control and recovery boundaries without downloading model weights."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from financial_ts import timesfm_comparison, timesfm_loop
from financial_ts.loop_experiment import run_loop_experiment
from financial_ts.timesfm_comparison import GatedAdapterTimesFM3, run_timesfm_comparison
from financial_ts.timesfm_loop import LoopTimesFM3
from financial_ts.workflow import resolve_segments


class TinyLayer(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.calls = 0
        self.metadata = None

    def forward(
        self,
        hidden,
        patch_mask,
        segment_ids=None,
        segment_pos=None,
        decode_cache=None,
        var_segment_pos=None,
    ):
        self.calls += 1
        self.metadata = (patch_mask, segment_ids, segment_pos, var_segment_pos)
        clean = hidden.masked_fill(patch_mask.unsqueeze(-1), 0)
        result = clean + 0.02 * self.projection(torch.tanh(clean))
        return result, None, patch_mask


class TinyStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer() for _ in range(20)])

    def forward(
        self,
        hidden,
        patch_mask,
        segment_ids=None,
        segment_pos=None,
        decode_cache=None,
        var_segment_pos=None,
    ):
        masks = []
        for layer in self.layers:
            hidden, _, mask = layer(
                hidden, patch_mask, segment_ids, segment_pos, None, var_segment_pos
            )
            masks.append(mask)
        return hidden, [None] * len(self.layers), masks


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_config = SimpleNamespace(transformer=SimpleNamespace(model_dims=8))
        self.input_projection = nn.Linear(1, 8)
        self.transformer_stack = TinyStack()
        self.output_head = nn.Linear(8, 2)

    @torch.no_grad()
    def decode(self, target, horizon):
        assert horizon == 2
        tokens = self.input_projection(target.unsqueeze(-1) / 100)
        hidden, _, _ = self.transformer_stack(
            tokens, torch.zeros(tokens.shape[:-1], dtype=torch.bool, device=target.device)
        )
        median = target[..., -1:] + self.output_head(hidden[:, :, -1])
        return torch.stack((median - 0.1, median, median + 0.1), dim=-1)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"fake; never deserialized")
    torch.manual_seed(73)
    base = TinyModel()
    monkeypatch.setattr(
        timesfm_loop,
        "_load_official_model",
        lambda path, device: (copy.deepcopy(base).to(device), 1),
    )
    return checkpoint


def contexts():
    return torch.tensor([[100.0, 101.0, 99.0, 102.0], [120.0, 118.0, 121.0, 122.0]])


def test_control_matches_initial_parameters_and_forecasts_without_replaying(checkpoint):
    torch.manual_seed(42)
    loop = LoopTimesFM3(checkpoint)
    torch.manual_seed(42)
    control = GatedAdapterTimesFM3(checkpoint)
    assert loop.trainable_parameter_count == control.trainable_parameter_count
    assert loop.parameter_count == control.parameter_count
    for name, value in loop.adapter_state_dict().items():
        assert torch.equal(value, control.adapter_state_dict()[name])
    torch.testing.assert_close(loop(contexts()), control(contexts()), rtol=0, atol=0)
    assert [layer.calls for layer in loop.loop_stack.base.layers] == [1] * 8 + [2] * 4 + [1] * 8
    assert [layer.calls for layer in control.loop_stack.base.layers] == [1] * 20
    torch.testing.assert_close(
        control(contexts()), control.baseline_forecast(contexts()), rtol=0, atol=0
    )


def test_control_trains_gates_and_adapters_and_freezes_base(checkpoint):
    model = GatedAdapterTimesFM3(checkpoint).train()
    base = list(model.base_parameters())
    before = [p.detach().clone() for p in base]
    initial = model(contexts()).detach().clone()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=0.02)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            ((model.return_scores(contexts()) - torch.tensor([0.02, -0.01])) / 0.01).square().mean()
        )
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
        optimizer.step()
    refinement = model.loop_stack.refinements[0]
    for parameter in refinement.parameters():
        assert parameter.grad.abs().sum() > 0
    for original, parameter in zip(before, base):
        assert torch.equal(original, parameter) and parameter.grad is None
    assert not torch.equal(initial, model(contexts()))
    assert torch.equal(initial, model.baseline_forecast(contexts()))


def test_control_preserves_metadata_and_rejects_populated_caches(checkpoint):
    model = GatedAdapterTimesFM3(checkpoint)
    stack = model.loop_stack
    hidden = torch.randn(2, 1, 7, 8)
    mask = torch.zeros(2, 1, 7, dtype=torch.bool)
    ids = torch.zeros(2, 7, dtype=torch.long)
    positions = torch.arange(7).repeat(2, 1)
    variates = torch.zeros(14, 1, dtype=torch.long)
    stack(hidden, mask, ids, positions, None, variates)
    for layer in stack.base.layers:
        assert all(
            actual is expected
            for actual, expected in zip(layer.metadata, (mask, ids, positions, variates))
        )
    with pytest.raises(ValueError, match="KV caches"):
        stack(hidden, mask, decode_cache=[object()] * 20)
    with pytest.raises(ValueError, match="boolean"):
        stack(hidden, mask.float())


@pytest.fixture
def market(tmp_path, monkeypatch, checkpoint):
    import qlib
    import qlib.data

    calendar = pd.bdate_range("2023-01-03", periods=180)
    segments = resolve_segments(calendar)
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars/day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")))
    (provider / "metadata.json").write_text(json.dumps({"synthetic": True}))
    position = np.arange(len(calendar))
    frame = pd.DataFrame(
        {"$close": (100 + 0.2 * position + 0.1 * np.sin(position / 3)).astype(np.float32)},
        index=pd.MultiIndex.from_product([["SPX"], calendar], names=["instrument", "datetime"]),
    )
    requests = []

    def features(instruments, fields, start_time, end_time, freq):
        requests.append(pd.Timestamp(end_time))
        dates = frame.index.get_level_values("datetime")
        return frame.loc[(dates >= start_time) & (dates <= end_time)].copy()

    monkeypatch.setattr(qlib, "init", lambda **kwargs: None)
    monkeypatch.setattr(
        qlib.data,
        "D",
        SimpleNamespace(
            instruments=lambda name: name,
            list_instruments=lambda *args, **kwargs: ["SPX"],
            features=features,
        ),
    )
    evidence = tmp_path / "original_evidence"
    run_loop_experiment(
        provider,
        checkpoint,
        evidence,
        epochs=2,
        train_samples=8,
        valid_samples=6,
        context_length=16,
        batch_size=2,
    )
    requests.clear()
    return SimpleNamespace(
        provider=provider,
        checkpoint=checkpoint,
        evidence=evidence,
        calendar=calendar,
        segments=segments,
        frame=frame,
        requests=requests,
    )


def test_comparison_recovers_then_freezes_selection_before_test_reads(
    market, tmp_path, monkeypatch
):
    output = tmp_path / "comparison"
    original_reader = timesfm_comparison._read_evaluation_closes

    def checked_reader(provider, calendar, segments):
        report = json.loads((output / "comparison.json").read_text())
        assert report["selection_frozen_at_utc"]
        assert (output / "recovered_loop/adapter.pt").is_file()
        assert (output / "gated_adapter_only/adapter.pt").is_file()
        assert all(date < pd.Timestamp(segments["test"][0]) for date in market.requests)
        return original_reader(provider, calendar, segments)

    monkeypatch.setattr(timesfm_comparison, "_read_evaluation_closes", checked_reader)
    report = run_timesfm_comparison(market.provider, market.checkpoint, output, market.evidence)
    assert report["status"] == "completed"
    assert report["recovery"]["max_curve_absolute_error"] == 0
    assert report["recovery"]["max_score_absolute_errors"]["selected_score"] == 0
    assert report["recovery"]["validation_rows_checked"] == 6
    assert report["test_used_for_training_or_selection"] is False
    assert report["full_validation_used_for_selection"] is False
    assert report["control"]["parameter_matched"] is True
    assert report["control"]["compute_matched"] is False
    assert report["control_training"]["optimizer_updates"] == 8
    assert report["control_training"]["adapter_replay_max_score_error"] == 0
    original_samples = json.loads((market.evidence / "samples.json").read_text())
    control_samples = json.loads((output / "gated_adapter_only/samples.json").read_text())
    assert original_samples == control_samples
    frame = pd.read_csv(output / "all_period_predictions.csv")
    assert set(frame["segment"]) == {"valid", "test"}
    columns = ["timesfm3_frozen_128", "timesfm3_loop_128", "timesfm3_adapter_only_128"]
    assert np.isfinite(frame[columns].to_numpy()).all()
    for segment, group in frame.groupby("segment"):
        expected_count = len(
            market.calendar[
                (market.calendar >= market.segments[segment][0])
                & (market.calendar <= market.segments[segment][1])
            ]
        )
        assert len(group) == report["evaluations"][segment]["count"] == expected_count
        assert (group["context_end"] == group["datetime"]).all()
        assert (group["datetime"] < group["entry_date"]).all()
        assert (group["entry_date"] < group["exit_date"]).all()
        for name in columns:
            rmse = np.sqrt(np.mean((group[name] - group["label"]) ** 2))
            assert rmse == pytest.approx(report["evaluations"][segment]["models"][name]["rmse"])
    with pytest.raises(ValueError, match="new or empty"):
        run_timesfm_comparison(market.provider, market.checkpoint, output, market.evidence)


def test_recovery_mismatch_aborts_before_control_training_or_test_read(
    market, tmp_path, monkeypatch
):
    curves_path = market.evidence / "learning_curves.json"
    curves = json.loads(curves_path.read_text())
    curves[1]["validation_rmse"] += 0.1
    curves_path.write_text(json.dumps(curves))
    monkeypatch.setattr(
        timesfm_comparison,
        "_train_control",
        lambda *args: pytest.fail("must not train after failed recovery"),
    )
    monkeypatch.setattr(
        timesfm_comparison,
        "_read_evaluation_closes",
        lambda *args: pytest.fail("must not evaluate after failed recovery"),
    )
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="curve"):
        run_timesfm_comparison(market.provider, market.checkpoint, output, market.evidence)
    assert json.loads((output / "comparison.json").read_text())["status"] == "failed"
    assert all(date < pd.Timestamp(market.segments["test"][0]) for date in market.requests)


def test_control_selection_is_unchanged_when_test_prices_are_perturbed(market, tmp_path):
    first = run_timesfm_comparison(
        market.provider, market.checkpoint, tmp_path / "first", market.evidence
    )
    dates = market.frame.index.get_level_values("datetime")
    market.frame.loc[dates >= market.segments["test"][0], "$close"] *= 1000
    second = run_timesfm_comparison(
        market.provider, market.checkpoint, tmp_path / "second", market.evidence
    )
    for key in (
        "best_epoch",
        "selected_validation_rmse",
        "selected_gate_change_l2",
        "selected_adapter_change_l2",
    ):
        assert first["control_training"][key] == second["control_training"][key]
    assert first["loop_training"] == second["loop_training"]


@pytest.mark.parametrize("changed_source", ["provider", "checkpoint"])
def test_changed_source_is_rejected_before_any_recovery_training(
    market, tmp_path, monkeypatch, changed_source
):
    monkeypatch.setattr(
        timesfm_comparison,
        "run_loop_experiment",
        lambda *args, **kwargs: pytest.fail("must reject changed source before training"),
    )
    if changed_source == "provider":
        calendar = market.provider / "calendars/day.txt"
        calendar.write_text(calendar.read_text() + "\n2099-01-01")
    else:
        (market.checkpoint / "config.json").write_text('{"changed":true}')
    output = tmp_path / "changed_source"
    with pytest.raises(ValueError, match="original .*bytes"):
        run_timesfm_comparison(market.provider, market.checkpoint, output, market.evidence)
    assert not output.exists()


def test_explicit_numerical_replication_resumes_without_retraining(market, tmp_path, monkeypatch):
    original_loader = timesfm_loop._load_official_model

    def perturbed_numeric_backend(checkpoint, device):
        model, quantile = original_loader(checkpoint, device)
        with torch.no_grad():
            model.output_head.bias[1].add_(0.0001)
        return model, quantile

    monkeypatch.setattr(timesfm_loop, "_load_official_model", perturbed_numeric_backend)
    output = tmp_path / "replication"
    with pytest.raises(ValueError, match="Strict numerical recovery failed"):
        run_timesfm_comparison(market.provider, market.checkpoint, output, market.evidence)
    assert not (output / "gated_adapter_only").exists()
    assert all(date < pd.Timestamp(market.segments["test"][0]) for date in market.requests)
    monkeypatch.setattr(
        timesfm_comparison,
        "run_loop_experiment",
        lambda *args, **kwargs: pytest.fail("explicit resume must reuse the completed retrain"),
    )
    report = run_timesfm_comparison(
        market.provider,
        market.checkpoint,
        output,
        market.evidence,
        resume=True,
        allow_numerical_replication=True,
    )
    assert report["status"] == "completed"
    assert report["resumed_without_retraining"] is True
    assert report["recovery"]["status"] == "numerical_replication_only"
    assert report["recovery"]["strict_failures"]
    assert report["recovery"]["max_score_absolute_errors"]["selected_score"] > 0
    assert report["execution_environment"]["torch_cpu_capability"]
    assert json.loads((output / "strict_recovery_attempt.json").read_text())["status"] == "failed"
