"""Common-date forecast evidence and saved SPY execution replay, without fitting."""

import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from financial_ts import comparison
from financial_ts.cli import main
from financial_ts.research import RESEARCH_MODELS, paired_block_interval


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    # Bootstrap behavior has its own tests; retain real resampling with fewer draws here.
    monkeypatch.setattr(
        comparison, "paired_block_interval", partial(paired_block_interval, draws=100)
    )
    historical_path = tmp_path / "archived_predictions.csv"
    archive_path = tmp_path / "archive.json"
    scratch = tmp_path / "scratch"
    official = tmp_path / "official"
    scratch.mkdir()
    official.mkdir()
    dates = pd.bdate_range("2020-01-06", periods=6)
    valid_dates = pd.bdate_range("2019-12-02", periods=4)
    labels = np.array([0.02, -0.03, 0.01, -0.02])
    valid_labels = np.array([-0.02, 0.01, -0.01, 0.03])
    seeds = (7, 42, 2026)

    def identities(calendar, targets):
        return pd.DataFrame({"datetime": calendar, "instrument": "SPX", "label": targets})

    historical = identities(dates[:4], labels)
    archived_runs = []
    for name in comparison.ARCHIVED_MODELS:
        for seed, offset in zip(seeds, (-0.03, 0, 0.03)):
            # Opposing seed errors cancel in forecast space, but not when averaging RMSEs.
            historical[f"{name}__seed_{seed}"] = labels + offset
            archived_runs.append(
                {
                    "config": {"model": {"name": name}, "seed": seed},
                    "metrics": {"model_details": {"parameter_count": 10}},
                }
            )
    historical["timesfm3_pretrained"] = 0.005
    historical.to_csv(historical_path, index=False)
    write_json(
        archive_path,
        {"status": "completed", "scratch": {"runs": archived_runs}, "protocol": {"seeds": seeds}},
    )

    def research_predictions(calendar, targets):
        frame = identities(calendar, targets)
        frame["zero_return"], frame["training_mean"] = 0.0, 0.001
        for number, name in enumerate(RESEARCH_MODELS):
            values = np.array([-0.004, -0.003, -0.002, -0.001]) + number * 0.004
            for seed, offset in zip(seeds, (-0.001, 0, 0.001)):
                frame[f"{name}__seed_{seed}"] = values + offset
            frame[f"{name}__ensemble"] = values
            frame[f"{name}__ensemble__shrunk"] = 0.001 + 0.5 * (values - 0.001)
        frame["three_architecture_blend"] = frame[
            [f"{name}__ensemble" for name in RESEARCH_MODELS]
        ].mean(axis=1)
        frame["three_architecture_blend__shrunk"] = 0.001 + 0.5 * (
            frame["three_architecture_blend"] - 0.001
        )
        return frame

    test = research_predictions(dates[:4], labels)
    valid = research_predictions(valid_dates, valid_labels)
    test.to_csv(scratch / "test_predictions.csv", index=False)
    valid.to_csv(scratch / "validation_predictions.csv", index=False)
    score_columns = [
        column
        for column in valid
        if column not in {"datetime", "instrument", "label"} and "__seed_" not in column
    ]
    selection = {
        "selected_by_validation": "timesfm3_mini__ensemble",
        "test_predictions_started": False,
        "validation_rmse": {
            column: float(np.sqrt(np.mean((valid[column] - valid_labels) ** 2)))
            for column in score_columns
        },
    }
    write_json(scratch / "selection.json", selection)
    write_json(
        scratch / "metrics.json",
        {
            "status": "completed",
            "trials": [
                {"model": name, "seed": seed, "parameter_count": 20}
                for name in RESEARCH_MODELS
                for seed in seeds
            ],
        },
    )
    write_json(scratch / "protocol.json", {"status": "completed", "seeds": seeds})
    frames = []
    for segment, calendar, targets in (
        ("valid", valid_dates, valid_labels),
        ("test", dates[:4], labels),
    ):
        frame = identities(calendar, targets)
        frame["segment"] = segment
        for number, name in enumerate(comparison.OFFICIAL_MODELS):
            frame[name] = targets * (number + 1) / 4
        frames.append(frame)
    pd.concat(frames).to_csv(official / "all_period_predictions.csv", index=False)
    recovery = {"strict_numerical_match": False, "status": "same_protocol_retrain"}
    write_json(official / "comparison.json", {"status": "completed", "recovery": recovery})
    spy_path = tmp_path / "spy.csv"
    prices = pd.DataFrame(
        {
            "date": dates,
            "symbol": "SPY",
            "adj_close": [90.0, 100.0, 110.0, 88.0, 101.2, 106.26],
        }
    )
    prices.to_csv(spy_path, index=False)
    return SimpleNamespace(
        args=(historical_path, archive_path, scratch, official, spy_path),
        historical=historical_path,
        scratch=scratch,
        official=official,
        spy=spy_path,
        output=tmp_path / "comparison",
        labels=labels,
        valid_labels=valid_labels,
        dates=dates,
        recovery=recovery,
    )


def test_averages_forecasts_before_scoring_and_keeps_archived_and_retrained_separate(evidence):
    result = comparison.build_financial_comparison(*evidence.args, evidence.output)
    scores = read_csv(evidence.output / "predictions.csv")
    metrics = read_csv(evidence.output / "forecast_metrics.csv").set_index("model")
    np.testing.assert_allclose(scores["timesfm3_mini"], evidence.labels, atol=1e-17)
    np.testing.assert_allclose(scores["timesfm3_mini_retrained"], [-0.004, -0.003, -0.002, -0.001])
    assert metrics.loc["timesfm3_mini", "test_rmse"] < 1e-16
    assert np.mean([0.03, 0, 0.03]) == pytest.approx(0.02)
    assert metrics.loc["timesfm3_mini_retrained", "test_rmse"] > 0.01
    assert metrics.loc["timesfm3_mini", "category"] == "archived_seed_ensemble"
    assert metrics.loc["timesfm3_mini_retrained", "category"] == "new_seed_ensemble"
    assert metrics.loc["timesfm3_mini", "parameter_count"] == 30
    assert metrics.loc["timesfm3_mini_retrained", "parameter_count"] == 60
    assert metrics.loc["three_architecture_blend", "ensemble_members"] == 9
    assert pd.isna(metrics.loc["timesfm3_mini", "validation_rmse"])
    assert result["settings"]["selected_by_validation"] == "timesfm3_mini_retrained"
    assert result["settings"]["timesfm_replication"] == evidence.recovery
    assert result["settings"]["period_scale"]["test_zero_rmse"] == pytest.approx(
        np.sqrt(np.mean(evidence.labels**2))
    )
    assert result["settings"]["period_scale"]["full_validation_zero_rmse"] == pytest.approx(
        np.sqrt(np.mean(evidence.valid_labels**2))
    )


@pytest.mark.parametrize("source", ["scratch_test", "official_test", "official_valid"])
@pytest.mark.parametrize("corruption", ["missing_row", "wrong_instrument", "wrong_label"])
def test_rejects_partial_identity_matches_and_different_targets(evidence, source, corruption):
    if source == "scratch_test":
        path = evidence.scratch / "test_predictions.csv"
        frame = read_csv(path)
        row = frame.index[0]
    else:
        path = evidence.official / "all_period_predictions.csv"
        frame = read_csv(path)
        segment = "test" if source == "official_test" else "valid"
        row = frame.index[frame["segment"] == segment][0]
    if corruption == "missing_row":
        frame = frame.drop(index=row)
    elif corruption == "wrong_instrument":
        frame["instrument"] = "OTHER"
    else:
        frame.loc[row, "label"] += 0.001
    frame.to_csv(path, index=False)
    match = "target values or units" if corruption == "wrong_label" else "exact common"
    with pytest.raises(ValueError, match=match):
        comparison.build_financial_comparison(*evidence.args, evidence.output)
    assert not evidence.output.exists()


def test_rejects_duplicate_prediction_identity(evidence):
    path = evidence.scratch / "test_predictions.csv"
    frame = read_csv(path)
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique rows"):
        comparison.build_financial_comparison(*evidence.args, evidence.output)


@pytest.mark.parametrize("invalid_score", [np.nan, np.inf])
def test_rejects_missing_archived_seed_instead_of_silently_changing_ensemble(
    evidence, invalid_score
):
    frame = read_csv(evidence.historical)
    frame.loc[0, "timesfm3_mini__seed_7"] = invalid_score
    frame.to_csv(evidence.historical, index=False)
    with pytest.raises(ValueError, match="finite"):
        comparison.build_financial_comparison(*evidence.args, evidence.output)


@pytest.mark.parametrize("source", ["scratch", "official"])
def test_rejects_unfinished_evidence(evidence, source):
    path = (
        evidence.scratch / "metrics.json"
        if source == "scratch"
        else evidence.official / "comparison.json"
    )
    report = json.loads(path.read_text())
    report["status"] = "failed"
    write_json(path, report)
    with pytest.raises(ValueError, match="completed"):
        comparison.build_financial_comparison(*evidence.args, evidence.output)


def test_spy_returns_cost_controls_and_saved_execution_are_independent_of_spx_labels(evidence):
    result = comparison.build_financial_comparison(*evidence.args, evidence.output)
    metrics = result["metrics"].set_index(["model", "one_way_cost_bps"])
    for name in metrics.index.get_level_values("model").unique():
        assert set(metrics.loc[name].index) == {0, 5, 10, 20}
    # Archived scores are positive on periods 1 and 3. Their SPX labels are +2%, +1%,
    # whereas the actual delayed-entry SPY returns for those periods are +10%, +15%.
    gross_equity = 1.10 * 1.15
    assert metrics.loc[("timesfm3_mini", 0), "ending_equity"] == pytest.approx(gross_equity)
    assert metrics.loc[("timesfm3_mini", 0), "ending_equity"] != pytest.approx(1.02 * 1.01)
    for bps in (0, 5, 10, 20):
        factor = (1 - bps / 10000) / (1 + bps / 10000)
        assert metrics.loc[("timesfm3_mini", bps), "ending_equity"] == pytest.approx(
            gross_equity * factor**2
        )
        assert metrics.loc[("buy_and_hold", bps), "ending_equity"] == pytest.approx(
            106.26 / 100 * factor
        )
        assert metrics.loc[("cash", bps), "ending_equity"] == 1
        assert metrics.loc[("cash", bps), "trade_count"] == 0
    assert metrics.loc[("timesfm3_mini", 5), "trade_count"] == 4
    assert metrics.loc[("buy_and_hold", 5), "trade_count"] == 2
    execution = read_csv(evidence.output / "execution.csv")
    assert list(execution["entry_date"]) == list(evidence.dates[1:5].strftime("%Y-%m-%d"))
    assert list(execution["exit_date"]) == list(evidence.dates[2:6].strftime("%Y-%m-%d"))
    np.testing.assert_allclose(execution["asset_return"], [0.10, -0.20, 0.15, 0.05])
    np.testing.assert_allclose(execution["entry_adjusted_close"], [100, 110, 88, 101.2])
    equity = read_csv(evidence.output / "equity.csv")
    assert len(equity) == 5
    assert equity.iloc[0, 0] == str(evidence.dates[1].date())
    assert (equity.iloc[0, 1:] == 1).all()
    assert equity.iloc[-1]["timesfm3_mini"] == pytest.approx(
        metrics.loc[("timesfm3_mini", 5), "ending_equity"]
    )
    for source in result["settings"]["sources"]:
        data = Path(source["path"]).read_bytes()
        assert source["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("occupied", ["file", "directory"])
def test_comparison_and_replay_refuse_to_overwrite_existing_work(evidence, occupied):
    if occupied == "file":
        evidence.output.write_text("keep this")
        sentinel = evidence.output
    else:
        evidence.output.mkdir()
        sentinel = evidence.output / "important.txt"
        sentinel.write_text("keep this")
    with pytest.raises(ValueError, match="new or empty"):
        comparison.build_financial_comparison(*evidence.args, evidence.output)
    with pytest.raises(ValueError, match="new or empty"):
        comparison.run_saved_backtest(evidence.historical, evidence.spy, evidence.output)
    assert sentinel.read_text() == "keep this"


def test_saved_backtest_cli_replays_all_comparison_portfolio_metrics(evidence, tmp_path, capsys):
    built = comparison.build_financial_comparison(*evidence.args, evidence.output)
    replay = tmp_path / "replay"
    assert (
        main(
            [
                "backtest",
                "--predictions",
                str(evidence.output / "predictions.csv"),
                "--spy-csv",
                str(evidence.spy),
                "--output",
                str(replay),
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["output_dir"] == str(replay.resolve())
    replay_metrics = read_csv(replay / "metrics.csv").sort_values(["model", "cost_bps"])
    expected = built["metrics"].rename(columns={"one_way_cost_bps": "cost_bps"})
    expected = expected[replay_metrics.columns].sort_values(["model", "cost_bps"])
    # CSV represents undefined ratios as NaN; the API uses JSON-safe None.
    expected = expected.where(expected.notna(), np.nan)
    pd.testing.assert_frame_equal(
        replay_metrics.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
    ledger = read_csv(replay / "ledger.csv")
    assert set(ledger["cost_bps"]) == {0, 5, 10, 20}
    assert {"cash", "buy_and_hold"}.issubset(ledger["model"])
    assert len(ledger) == len(replay_metrics) * len(evidence.labels)
    settings = json.loads((replay / "settings.json").read_text())
    assert len(settings["sources"]) == 2
    assert all(len(source["sha256"]) == 64 for source in settings["sources"])


def test_saved_replay_requires_explicit_spy_execution_data(evidence, tmp_path):
    prices = read_csv(evidence.spy)
    prices["symbol"] = "SPX"
    prices.to_csv(evidence.spy, index=False)
    with pytest.raises(ValueError, match="explicitly identify SPY"):
        comparison.run_saved_backtest(evidence.historical, evidence.spy, tmp_path / "replay")
