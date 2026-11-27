"""Temporal boundaries, complete trials, selection and whole-date uncertainty."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_ts import rolling
from financial_ts.data import create_demo_csv, import_csv


def test_observed_session_folds_mature_before_decision_and_preserve_warmup():
    calendar = pd.bdate_range("2020-01-01", periods=240).difference(
        pd.DatetimeIndex(["2020-01-20", "2020-02-17", "2020-04-10"])
    )
    folds = rolling.make_rolling_folds(
        calendar,
        train_sessions=60,
        valid_sessions=20,
        test_sessions=10,
        warmup_sessions=10,
        max_folds=3,
    )
    for fold in folds:
        segments = fold["segments"]
        assert calendar.get_loc(segments["train"][1]) + 2 < calendar.get_loc(segments["valid"][0])
        assert calendar.get_loc(segments["valid"][1]) + 2 < calendar.get_loc(segments["test"][0])
        assert pd.Timestamp(fold["validation_label_endpoint"]) <= pd.Timestamp(fold["retrain_asof"])
        assert (
            calendar.get_loc(segments["train"][0]) - calendar.get_loc(fold["data_start_time"]) == 10
        )
        assert calendar.get_loc(fold["data_end_time"]) == calendar.get_loc(segments["test"][1]) + 3
    assert folds[0]["segments"]["test"][1] < folds[1]["segments"]["test"][0]
    extended = calendar.append(pd.bdate_range(calendar[-1] + pd.Timedelta(days=1), periods=20))
    assert folds == rolling.make_rolling_folds(
        extended,
        train_sessions=60,
        valid_sessions=20,
        test_sessions=10,
        warmup_sessions=10,
        max_folds=3,
    )


def test_fold_recipe_rejects_overlap_and_insufficient_history():
    cal = pd.bdate_range("2020-01-01", periods=100)
    with pytest.raises(ValueError, match="overlap"):
        rolling.make_rolling_folds(cal, test_sessions=20, step_sessions=10)
    with pytest.raises(ValueError, match="No complete"):
        rolling.make_rolling_folds(cal)
    with pytest.raises(ValueError, match="warmup"):
        rolling.make_rolling_folds(cal, start="2020-01-01")
    with pytest.raises(ValueError, match="Calendar"):
        rolling.make_rolling_folds(cal[::-1])


@pytest.fixture
def source(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    dates = pd.bdate_range("2020-01-01", periods=160)
    (provider / "calendars/day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")))
    (provider / "metadata.json").write_text(
        json.dumps({"synthetic": True, "target_trade_date": str(dates[-1].date())})
    )
    return provider


def _spec():
    return {
        "region": "us",
        "instruments": "all",
        "models": ["zero", "ridge"],
        "seeds": [7, 42],
        "epochs": 1,
        "warmup_sessions": 10,
        "rolling": {"train_sessions": 20, "valid_sessions": 10, "test_sessions": 5, "max_folds": 2},
        "backtest": {"enabled": False},
        "export_inference_bundle": False,
        "bootstrap": {"block_lengths": [2, 20], "draws": 100, "seed": 11},
    }


def test_registration_is_immutable_and_rejects_future_label_fit(source, tmp_path):
    spec = _spec()
    protocol = rolling.prepare_rolling_campaign(source, tmp_path / "campaign", spec)
    assert len(protocol["trials"]) == 8
    assert rolling.prepare_rolling_campaign(source, tmp_path / "campaign", spec) == protocol
    changed = copy.deepcopy(spec)
    changed["seeds"] = [7]
    with pytest.raises(ValueError, match="differs"):
        rolling.prepare_rolling_campaign(source, tmp_path / "campaign", changed)
    changed = copy.deepcopy(spec)
    changed.pop("rolling")
    changed["folds"] = protocol["folds"]
    changed["folds"][0]["retrain_asof"] = changed["folds"][0]["segments"]["valid"][1]
    with pytest.raises(ValueError, match="mature"):
        rolling.prepare_rolling_campaign(source, tmp_path / "bad", changed)


def test_registered_provider_and_manifest_mutations_fail_before_fit(source, tmp_path, monkeypatch):
    protocol = rolling.prepare_rolling_campaign(source, tmp_path / "campaign", _spec())
    monkeypatch.setattr(rolling, "run_workflow", lambda config: pytest.fail("must not fit"))
    (source / "metadata.json").write_text('{"synthetic": false}')
    with pytest.raises(ValueError, match="manifest changed"):
        rolling.run_rolling_campaign(protocol)
    (source / "calendars/day.txt").write_text("2020-01-01")
    with pytest.raises(ValueError, match="Provider bytes changed"):
        rolling.run_rolling_campaign(protocol)


def test_protocol_tampering_is_detected(source, tmp_path):
    protocol = rolling.prepare_rolling_campaign(source, tmp_path / "campaign", _spec())
    path = Path(protocol["output_dir"]) / "protocol.json"
    changed = json.loads(path.read_text())
    changed["seeds"] = [999]
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="integrity"):
        rolling.summarize_rolling_campaign(protocol["output_dir"])


def _mock_workflow(config):
    path = Path(config["output_dir"])
    path.mkdir(parents=True)
    metrics = {}
    for segment, filename in (("valid", "validation_predictions.csv"), ("test", "predictions.csv")):
        dates = pd.bdate_range(*config["segments"][segment])
        index = pd.MultiIndex.from_product(
            [dates, ["A", "B", "C"]], names=["datetime", "instrument"]
        )
        label = np.tile([-0.01, 0.0, 0.02], len(dates))
        score = np.zeros(len(index)) if config["model"]["name"] == "zero" else label + 0.001
        frame = pd.DataFrame({"score": score, "label": label}, index=index)
        frame.to_csv(path / filename)
        stats, daily = rolling._prediction_statistics(frame)
        daily.to_csv(path / ("validation_daily_ic.csv" if segment == "valid" else "daily_ic.csv"))
        metrics[segment] = stats
    evidence = {
        **metrics["test"],
        "validation": metrics["valid"],
        "model_details": {"parameter_count": 2, "fit_seconds": 0.001},
    }
    run = {"status": "completed"}
    rolling._save(path / "metrics.json", evidence)
    rolling._save(path / "run.json", run)
    rolling._save(path / "resolved_config.json", config)
    return {"metrics": evidence, "run": run}


def test_all_failures_retained_retries_are_new_attempts_and_test_cannot_select(
    source, tmp_path, monkeypatch
):
    monkeypatch.setattr(rolling, "_code_identity", lambda: "registered-test-code")
    protocol = rolling.prepare_rolling_campaign(source, tmp_path / "campaign", _spec())
    calls = []
    fail = True

    def workflow(config):
        calls.append(config["output_dir"])
        if fail and config["model"]["name"] == "ridge" and config["seed"] == 42:
            raise RuntimeError("registered trial failure")
        return _mock_workflow(config)

    monkeypatch.setattr(rolling, "run_workflow", workflow)
    first = rolling.run_rolling_campaign(protocol)
    assert first["status"] == "failed" and first["completed_trials"] == 6
    assert len(calls) == 8 and first["failed_attempt_count"] == 2
    assert all(row["selected_candidate"] is None for row in first["selections"])
    fail = False
    second = rolling.run_rolling_campaign(protocol, retry_failed=True)
    assert second["status"] == "completed" and second["completed_trials"] == 8
    assert second["failed_attempt_count"] == 2
    assert len(calls) == 10
    assert all(row["selected_candidate"] == "ridge" for row in second["selections"])
    assert rolling.run_rolling_campaign(protocol)["completed_trials"] == 8
    assert len(calls) == 10
    history = json.loads((Path(protocol["output_dir"]) / "trial_history.json").read_text())
    assert len(history["attempts"]) == 10
    # Even absurd evaluation metrics cannot alter the registered selector.
    for trial in protocol["trials"]:
        attempt = rolling._attempts(Path(protocol["output_dir"]), trial)[-1]
        path = attempt / "workflow" / "metrics.json"
        evidence = json.loads(path.read_text())
        evidence["rmse"] = 1e12 if trial["candidate"] == "ridge" else 0
        evidence["rank_ic"] = -1 if trial["candidate"] == "ridge" else 1
        rolling._save(path, evidence)
        receipt_path = attempt / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["artifacts"] = rolling._artifact_manifest(attempt / "workflow")
        rolling._save(receipt_path, receipt)
    third = rolling.summarize_rolling_campaign(protocol)
    assert third["selections"] == second["selections"]


def test_completed_artifact_corruption_is_not_silently_reused(source, tmp_path, monkeypatch):
    monkeypatch.setattr(rolling, "_code_identity", lambda: "registered-test-code")
    monkeypatch.setattr(rolling, "run_workflow", _mock_workflow)
    protocol = rolling.prepare_rolling_campaign(source, tmp_path / "campaign", _spec())
    rolling.run_rolling_campaign(protocol)
    trial = protocol["trials"][0]
    path = (
        rolling._attempts(Path(protocol["output_dir"]), trial)[-1] / "workflow" / "predictions.csv"
    )
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="artifact integrity"):
        rolling.run_rolling_campaign(protocol)


def _panel():
    dates = pd.bdate_range("2024-01-01", periods=12)
    tuples = [
        (date, instrument)
        for day, date in enumerate(dates)
        for instrument in ["A", "B", "C"][: 2 + day % 2]
    ]
    index = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
    y = np.arange(len(index), dtype=float) * 0.01
    return pd.DataFrame({"score": y + 0.1, "label": y}, index=index)


def test_date_block_pairing_is_row_permutation_invariant_and_keeps_day_counts():
    candidate = _panel()
    reference = candidate.copy()
    reference["score"] = reference["label"] + 0.2
    a = rolling.paired_date_block_intervals(candidate, reference, block_length=3, draws=100)
    b = rolling.paired_date_block_intervals(
        candidate.sample(frac=1, random_state=7),
        reference.sample(frac=1, random_state=42),
        block_length=3,
        draws=100,
    )
    assert a == b
    assert a["date_count"] == 12 and a["row_count"] == len(candidate)
    assert a["metrics"]["rmse"]["difference"] == pytest.approx(-0.1)
    assert a["metrics"]["rmse"]["lower_95"] == pytest.approx(-0.1)
    assert a["metrics"]["rmse"]["upper_95"] == pytest.approx(-0.1)


def test_pairing_rejects_panel_or_label_mismatch_and_reports_undefined_ic():
    candidate = _panel()
    reference = candidate.copy()
    reference["score"] = 0
    result = rolling.paired_date_block_intervals(candidate, reference, block_length=3, draws=100)
    assert result["metrics"]["ic"]["difference"] is None
    assert result["metrics"]["ic"]["paired_usable_dates"] == 0
    with pytest.raises(ValueError, match="matching keys"):
        rolling.paired_date_block_intervals(candidate, reference.iloc[:-1], draws=100)
    reference.iloc[0, reference.columns.get_loc("label")] += 0.1
    with pytest.raises(ValueError, match="labels differ"):
        rolling.paired_date_block_intervals(candidate, reference, draws=100)


@pytest.mark.parametrize("block_length", [12, 13])
def test_forecast_blocks_reject_full_period_or_longer(block_length):
    candidate = _panel()
    reference = candidate.copy()
    reference["score"] = 0.0
    with pytest.raises(ValueError, match="must be smaller.*paired dates"):
        rolling.paired_date_block_intervals(
            candidate, reference, block_length=block_length, draws=100
        )


@pytest.mark.parametrize("block_length", [12, 13])
def test_return_blocks_reject_full_period_or_longer(block_length):
    dates = pd.bdate_range("2024-01-01", periods=12)
    candidate = pd.Series(np.linspace(-0.01, 0.02, len(dates)), index=dates)
    reference = pd.Series(np.linspace(0.01, -0.01, len(dates)), index=dates)
    with pytest.raises(ValueError, match="must be smaller.*paired portfolio dates"):
        rolling.paired_return_block_intervals(
            candidate, reference, block_length=block_length, draws=100
        )


def test_near_full_period_blocks_report_limited_information():
    candidate = _panel()
    reference = candidate.copy()
    reference["score"] += np.arange(len(reference)) * 0.001
    forecast = rolling.paired_date_block_intervals(candidate, reference, block_length=11, draws=100)
    dates = pd.bdate_range("2024-01-01", periods=12)
    returns = rolling.paired_return_block_intervals(
        pd.Series(np.linspace(-0.01, 0.02, len(dates)), index=dates),
        pd.Series(0.0, index=dates),
        block_length=11,
        draws=100,
    )
    for result in (forecast, returns):
        assert result["effective_blocks"] == pytest.approx(12 / 11)
        assert "fewer than two full block lengths" in result["uncertainty_qualification"]


def test_separate_analysis_preserves_originals_and_marks_full_period_noninformative(
    source, tmp_path, monkeypatch
):
    monkeypatch.setattr(rolling, "_code_identity", lambda: "registered-test-code")
    monkeypatch.setattr(rolling, "run_workflow", _mock_workflow)

    def scenarios(config, protocol, workflow_root):
        dates = pd.bdate_range(*config["segments"]["test"])
        for scenario in protocol["cost_scenarios"]:
            target = workflow_root / "scenarios" / scenario["id"]
            target.mkdir(parents=True)
            pd.DataFrame(
                {
                    "return": np.linspace(-0.01, 0.02, len(dates)),
                    "cost": 0.0001,
                    "bench": 0.0,
                    "turnover": 0.2,
                },
                index=dates,
            ).to_csv(target / "portfolio.csv")
        return {"status": "completed"}

    monkeypatch.setattr(rolling, "_replay_scenarios", scenarios)
    spec = _spec()
    spec["seeds"] = [7]
    spec["rolling"]["max_folds"] = 1
    spec["bootstrap"]["block_lengths"] = [2, 5, 6]
    protocol = rolling.prepare_rolling_campaign(source, tmp_path / "campaign", spec)
    original = rolling.run_rolling_campaign(protocol)
    assert original["status"] == "completed"
    for name in ("paired_comparisons", "portfolio_paired_comparisons"):
        for row in original[name]:
            if row["block_length"] >= 5:
                assert row["status"] == "insufficient_dates"
                assert row["date_count"] == 5
                assert "no resampling variation" in row["reason"]
                assert "result" not in row
            else:
                assert row["status"] == "completed"
    campaign = Path(protocol["output_dir"])
    before = {
        str(p.relative_to(campaign)): p.read_bytes() for p in campaign.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(rolling, "_code_identity", lambda: "reviewed-analysis-code")
    audit = tmp_path / "analysis-audit"
    assert rolling.summarize_rolling_campaign(protocol, summary_output_dir=audit) == original
    after = {
        str(p.relative_to(campaign)): p.read_bytes() for p in campaign.rglob("*") if p.is_file()
    }
    assert before == after
    receipt = json.loads((audit / "analysis_receipt.json").read_text())
    assert receipt["registered_source_code_sha256"] == "registered-test-code"
    assert receipt["analysis_source_code_sha256"] == "reviewed-analysis-code"
    assert receipt["original_comparison"]["sha256"] == rolling._sha256(campaign / "comparison.json")
    assert len(receipt["analysis_module"]["sha256"]) == 64
    assert (audit / "summary" / "fold-000" / "ridge" / "ensemble_predictions.csv").exists()
    for unsafe in (campaign, campaign / "summary" / "audit", campaign / "trials" / "audit"):
        with pytest.raises(ValueError, match="preserve the original"):
            rolling.summarize_rolling_campaign(protocol, summary_output_dir=unsafe)


@pytest.mark.integration
def test_native_qlib_two_fold_two_seed_baseline_campaign(tmp_path):
    provider = tmp_path / "provider"
    source = create_demo_csv(tmp_path / "prices.csv", n_instruments=3, n_days=150)
    metadata = import_csv(source, provider)
    metadata["synthetic"] = True
    (provider / "metadata.json").write_text(json.dumps(metadata))
    spec = _spec()
    spec["warmup_sessions"] = 60
    spec["rolling"].update(train_sessions=30, valid_sessions=10, test_sessions=5)
    protocol = rolling.prepare_rolling_campaign(provider, tmp_path / "native", spec)
    result = rolling.run_rolling_campaign(protocol)
    assert result["status"] == "completed", result["failed"]
    assert result["completed_trials"] == 8
    assert len(result["seed_ensembles"]) == 4
    assert all(row["test_rows"] == 15 for row in result["runs"])
    assert len(result["selections"]) == 2
    assert result["promotion"]["eligible"] is False


def test_paired_native_returns_keep_calendar_and_seed_pairing():
    dates = pd.bdate_range("2024-01-01", periods=12)
    c = pd.Series(0.01, index=dates)
    r = pd.Series(0.005, index=dates)
    result = rolling.paired_return_block_intervals(c, r, block_length=3, draws=100)
    metric = result["metrics"]["compounded_net_return"]
    assert metric["difference"] == pytest.approx(1.01**12 - 1.005**12)
    assert metric["lower_95"] == pytest.approx(metric["difference"])
    assert metric["upper_95"] == pytest.approx(metric["difference"])
    with pytest.raises(ValueError, match="matching dates"):
        rolling.paired_return_block_intervals(c, r.iloc[:-1], block_length=3, draws=100)


@pytest.mark.integration
def test_native_scenarios_replay_positions_and_pair_real_net_returns(tmp_path):
    provider = tmp_path / "provider"
    source = create_demo_csv(tmp_path / "prices.csv", n_instruments=3, n_days=140)
    metadata = import_csv(source, provider)
    metadata["synthetic"] = True
    (provider / "metadata.json").write_text(json.dumps(metadata))
    spec = _spec()
    spec.update(seeds=[7], warmup_sessions=60)
    spec["rolling"].update(train_sessions=30, valid_sessions=10, test_sessions=10, max_folds=1)
    spec["backtest"] = {"enabled": True, "topk": 2, "n_drop": 1, "account": 100000}
    spec["cost_scenarios"] = [
        {"id": "base", "fee_multiplier": 1},
        {"id": "fees_2x", "fee_multiplier": 2},
    ]
    protocol = rolling.prepare_rolling_campaign(provider, tmp_path / "scenarios", spec)
    result = rolling.run_rolling_campaign(protocol)
    assert result["status"] == "completed", result["failed"]
    assert len(result["portfolio_paired_comparisons"]) == 4
    comparisons = [
        row for row in result["portfolio_paired_comparisons"] if row["status"] == "completed"
    ]
    assert {row["scenario"] for row in comparisons} == {"base", "fees_2x"}
    for trial in protocol["trials"]:
        root = rolling._attempts(Path(protocol["output_dir"]), trial)[-1] / "workflow"
        scenario = json.loads((root / "cost_scenarios.json").read_text())
        assert scenario["status"] == "completed"
        for row in scenario["scenarios"]:
            assert row["trading_days"] == 11
            assert row["total_final_stale_weight"] == 0
            assert 0 <= row["mean_gross_exposure"] <= 1
            assert (root / "scenarios" / row["id"] / "positions.json.gz").exists()
