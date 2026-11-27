"""Chronological boundaries and validation-only selection in the paper screen."""

import copy
import json
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from financial_ts import paper_research as screen
from financial_ts.research import paired_block_interval


def test_failure_diagnosis_volatility_uses_only_returns_realized_by_signal_time():
    dates = pd.bdate_range("2000-01-03", periods=100)
    labels = pd.Series(np.linspace(-0.02, 0.02, 100), index=dates)
    positions, original = screen._fit_positions(labels, 100)
    assert positions[0] == 21
    assert original[0] == pytest.approx(labels.iloc[:20].std(ddof=0))
    changed = labels.copy()
    # Label 20 is a close-21-to-close-22 return: it is unavailable at close 21.
    changed.iloc[20:] += 0.5
    changed_positions, revised = screen._fit_positions(changed, 100)
    np.testing.assert_array_equal(changed_positions, positions)
    assert revised[0] == original[0]
    assert revised[1] != original[1]
    assert len(positions) == len(set(positions))
    assert positions[-1] < len(labels)


class _EditMember(nn.Module):
    def __init__(self):
        super().__init__()
        projection = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(2))
        layer = nn.Module()
        layer.ff = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity(), projection)
        self.layers = nn.ModuleList([nn.Identity(), layer])

    def forward(self, features, padding):
        return self.layers[1].ff(features[:, 0, :2]).sum(-1)


@pytest.mark.parametrize("target_scale,expected_alpha", [(1.0, 0.0), (0.95, 0.05)])
def test_edit_strength_is_chosen_on_validation_and_can_reject_all_edits(
    target_scale, expected_alpha
):
    model = screen._MiniEnsemble([_EditMember()])
    initial = copy.deepcopy(model.state_dict())
    features = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]])
    padding = torch.zeros(2, 1, dtype=torch.bool)
    selected, selection = screen._select_edit(
        model,
        [torch.tensor([1.0, 0.0])],
        (features, padding),
        np.array([1.0, 2.0]) * target_scale,
    )
    assert selection["selected_alpha"] == expected_alpha
    assert [candidate["alpha"] for candidate in selection["candidates"]] == list(screen.ALPHAS)
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())
    np.testing.assert_allclose(
        screen._normalized_scores(selected, features, padding),
        np.array([1.0, 2.0]) * target_scale,
        atol=1e-7,
    )


class _ToyForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, features, padding):
        return features[:, 0, 0].double() + self.offset


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Mock expensive Qlib/model fitting while retaining the orchestration and files."""
    provider, base_runs = tmp_path / "provider", tmp_path / "base"
    provider.mkdir()
    base_runs.mkdir()
    train_dates = pd.bdate_range("2006-01-03", periods=80)
    valid_dates = pd.bdate_range("2012-01-03", periods=20)
    test_dates = pd.DatetimeIndex(
        np.concatenate([pd.bdate_range(f"{year}-01-03", periods=20) for year in (2013, 2015, 2017)])
    )

    def index(dates):
        return pd.MultiIndex.from_product([dates, ["SPX"]], names=["datetime", "instrument"])

    labels = {
        "train": pd.Series(np.tile([-0.01, 0.01], 40), index=index(train_dates), name="label"),
        "valid": pd.Series(0.003, index=index(valid_dates), name="label"),
        "test": pd.Series(np.linspace(-0.02, 0.02, 60), index=index(test_dates), name="label"),
    }
    # Qlib returns tuples; the archived JSON necessarily round-trips them as lists.
    segments = {
        name: (str(values.index[0][0].date()), str(values.index[-1][0].date()))
        for name, values in labels.items()
    }
    archived_selection = {
        "selected_by_validation": "timesfm3_mini__ensemble",
        "segments": {key: list(value) for key, value in segments.items()},
    }
    (base_runs / "selection.json").write_text(json.dumps(archived_selection))
    (base_runs / "protocol.json").write_text(json.dumps({"status": "completed"}))
    spy = tmp_path / "spy.csv"
    pd.DataFrame({"date": test_dates, "symbol": "SPY", "adj_close": 100.0}).to_csv(spy, index=False)
    case = SimpleNamespace(
        provider=provider,
        base_runs=base_runs,
        spy=spy,
        labels=labels,
        output=tmp_path / "run-0",
        selections=[],
        searches=[],
        test_reads=[],
        misalign_train=False,
        tamper_selection=False,
    )

    def refresh_archive():
        for segment, filename in (
            ("valid", "validation_predictions.csv"),
            ("test", "test_predictions.csv"),
        ):
            frame = labels[segment].to_frame()
            frame["timesfm3_mini__ensemble"] = 0.0
            frame.to_csv(base_runs / filename)

    refresh_archive()
    case.refresh_archive = refresh_archive
    monkeypatch.setattr(screen, "_provider_fingerprint", lambda _: {"sha256": "fixture"})
    monkeypatch.setattr(screen, "_read_calendar", lambda _: train_dates)
    monkeypatch.setattr(screen, "_build_dataset", lambda *_: (labels, segments))
    monkeypatch.setattr(
        screen,
        "_load_base",
        lambda *_: (_ToyForecast(), 0.0, 0.01, {}, archived_selection, []),
    )

    def read_labels(_, segment):
        if segment == "test":
            protocol = json.loads((case.output / "protocol.json").read_text())
            selection = json.loads((case.output / "selection.json").read_text())
            assert protocol["status"] == "registered"
            assert selection["test_predictions_started"] is False
            assert set(selection["states_sha256"]) == set(protocol["roster"])
            assert len(list((case.output / "checkpoints").glob("*.pt"))) == len(protocol["roster"])
            case.test_reads.append(segment)
        return labels[segment]

    def windows(_, segment, indices=None):
        identities = labels[segment].index
        if indices is not None:
            identities = identities[indices]
        if segment == "train" and case.misalign_train:
            identities = identities[::-1]
        if segment == "test" and case.tamper_selection:
            path = case.output / "selection.json"
            selected = json.loads(path.read_text())
            selected["selected_by_validation"] = "es_seed_7"
            path.write_text(json.dumps(selected))
        return (
            torch.zeros(len(identities), 1, 1),
            torch.zeros(len(identities), 1, dtype=torch.bool),
            identities,
        )

    def select_edit(base, directions, data, targets):
        candidate = copy.deepcopy(base)
        with torch.no_grad():
            candidate.offset.fill_(0.1)
        return candidate, {"selected_alpha": 0.1, "candidates": []}

    def fake_es(model, fit_reward, validation_loss, *, config):
        assert not (case.output / "selection.json").exists()
        assert json.loads((case.output / "protocol.json").read_text())["status"] == "registered"
        assert not case.test_reads
        assert np.isfinite(fit_reward(model))
        assert np.isfinite(validation_loss(model))
        case.searches.append(config.seed)
        with torch.no_grad():
            model.offset.fill_({7: 0.3, 42: 0.2, 2026: 0.1}[config.seed])
        return SimpleNamespace(
            selected_generation=1,
            selection_loss=validation_loss(model),
            candidate_evaluations=config.population * config.generations,
            history=[],
        )

    monkeypatch.setattr(screen, "_labels", read_labels)
    monkeypatch.setattr(screen, "_windows", windows)
    monkeypatch.setattr(screen, "_diagnose", lambda *_: ([torch.ones(1)], [torch.ones(1)], []))
    monkeypatch.setattr(screen, "_select_edit", select_edit)
    monkeypatch.setattr(screen, "evolution_strategy", fake_es)
    monkeypatch.setattr(screen, "paired_block_interval", partial(paired_block_interval, draws=100))
    # The real cost engine is independently covered by test_backtest.py. Here
    # preserving the orchestration boundary is the purpose of the small fixture.
    monkeypatch.setattr(
        screen,
        "run_backtest",
        lambda predictions, prices: {
            "metrics": pd.DataFrame({"model": predictions.columns, "cost_bps": 5.0}),
            "ledger": pd.DataFrame({"model": predictions.columns, "cost_bps": 5.0}),
            "settings": {},
        },
    )
    return case


def _run(case):
    return screen.run_paper_research(
        case.provider,
        case.base_runs,
        case.spy,
        case.output,
        fit_samples=32,
        population=2,
        generations=1,
    )


def test_selection_is_locked_before_test_and_test_labels_cannot_choose_model_or_seed(campaign):
    selections, summaries = [], []
    for trial in range(2):
        campaign.output = campaign.output.with_name(f"run-{trial}")
        campaign.searches.clear()
        campaign.test_reads.clear()
        if trial:
            campaign.labels["test"] *= -3
            campaign.refresh_archive()
        summaries.append(_run(campaign))
        selections.append(json.loads((campaign.output / "selection.json").read_text()))
        assert campaign.searches == [7, 7, 42, 42, 2026, 2026]
        assert campaign.test_reads == ["test"]
    for selection in selections:
        assert selection["selected_by_validation"] == "es_seed_42"
        # Seed 7 is deliberately better on validation; it still cannot replace 42.
        assert (
            selection["validation_rmse"]["es_seed_7"] < selection["validation_rmse"]["es_seed_42"]
        )
        assert selection["primary_es_seed"] == 42
    for key in ("states_sha256", "validation_rmse", "primary_roster", "selected_by_validation"):
        assert selections[0][key] == selections[1][key]
    assert summaries[0]["metrics"] != summaries[1]["metrics"]
    for summary in summaries:
        assert summary["selection_sha256_before_test"] == summary["selection_sha256_after_test"]
        assert summary["validation_replay_max_error"] == summary["test_replay_max_error"] == 0.0


def test_mismatched_incumbent_validation_scores_stop_before_search_or_test(campaign):
    path = campaign.base_runs / "validation_predictions.csv"
    frame = pd.read_csv(path)
    frame["timesfm3_mini__ensemble"] = 0.01
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="validation replay differs"):
        _run(campaign)
    assert not campaign.searches and not campaign.test_reads
    assert not (campaign.output / "selection.json").exists()
    assert json.loads((campaign.output / "protocol.json").read_text())["status"] == "failed"


def test_training_feature_label_misalignment_is_rejected_before_diagnosis(campaign):
    campaign.misalign_train = True
    with pytest.raises(ValueError, match="[Tt]rain.*rows differ"):
        _run(campaign)
    assert not campaign.searches and not campaign.test_reads


def test_changing_selection_during_test_cannot_produce_completed_report(campaign):
    campaign.tamper_selection = True
    with pytest.raises(RuntimeError, match="Selection changed after test scoring"):
        _run(campaign)
    assert not (campaign.output / "summary.json").exists()
    assert json.loads((campaign.output / "protocol.json").read_text())["status"] == "failed"


def test_existing_empty_output_is_accepted_but_nonempty_work_is_protected(campaign):
    campaign.output.mkdir()
    _run(campaign)
    selection = (campaign.output / "selection.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        _run(campaign)
    assert (campaign.output / "selection.json").read_bytes() == selection
