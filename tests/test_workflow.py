"""End-to-end checks for timing, causal preprocessing, and cost accounting."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from financial_ts.data import create_demo_csv, import_csv
from financial_ts.workflow import LABEL, build_task, run_workflow

pytestmark = pytest.mark.integration


def _init_qlib(provider):
    import qlib
    from qlib.data.cache import H

    H.clear()
    qlib.init(
        provider_uri=str(provider),
        region="us",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )


@pytest.fixture(scope="module")
def market_data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("workflow-market")
    source = create_demo_csv(folder / "market.csv", n_instruments=4, n_days=180)
    provider = folder / "provider"
    metadata = import_csv(source, provider)
    # Record the fixture's actual origin, as the demo CLI does. An experiment
    # name alone cannot establish whether arbitrary imported prices are synthetic.
    metadata["synthetic"] = True
    (provider / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return source, provider


def _config(provider, output):
    return {
        "provider_uri": str(provider),
        "output_dir": str(output),
        "region": "us",
        "experiment_name": "synthetic_demo_integration",
        "seed": 42,
        "warmup_sessions": 60,
        "model": {
            "name": "lightgbm",
            "params": {
                "num_boost_round": 8,
                "early_stopping_rounds": 3,
                "num_leaves": 7,
                "min_data_in_leaf": 10,
                "num_threads": 1,
                "verbosity": -1,
            },
        },
        "backtest": {
            "enabled": True,
            "topk": 2,
            "n_drop": 1,
            "open_cost": 0.001,
            "close_cost": 0.001,
            "min_cost": 0.0,
        },
    }


@pytest.fixture(scope="module")
def completed_run(market_data, tmp_path_factory):
    _, provider = market_data
    output = tmp_path_factory.mktemp("workflow-output") / "run"
    config = _config(provider, output)
    result = run_workflow(config)
    return config, result


def test_cpu_workflow_exports_finite_aligned_predictions_and_costed_portfolio(completed_run):
    from pathlib import Path

    config, result = completed_run
    output = Path(result["output_dir"])
    predictions = pd.read_csv(output / "predictions.csv", parse_dates=["datetime"])
    predictions = predictions.set_index(["datetime", "instrument"]).sort_index()
    assert predictions.index.is_unique
    assert len(predictions) > 0
    assert np.isfinite(predictions[["score", "label"]].to_numpy()).all()
    assert result["metrics"]["prediction_count"] == len(predictions)
    assert result["metrics"]["evaluated_count"] == len(predictions)
    assert predictions.index.get_level_values("instrument").nunique() == 4

    # Check the complete expected instrument x session index, not just row counts.
    calendar = pd.DatetimeIndex(
        pd.to_datetime(
            (Path(config["provider_uri"]) / "calendars/day.txt").read_text().splitlines()
        )
    )
    start, end = result["run"]["segments"]["test"]
    dates = calendar[(calendar >= start) & (calendar <= end)]
    expected_index = pd.MultiIndex.from_product(
        [dates, [f"DEMO{number:03d}" for number in range(4)]], names=["datetime", "instrument"]
    )
    pd.testing.assert_index_equal(predictions.index, expected_index)

    portfolio = pd.read_csv(output / "portfolio.csv", index_col=0, parse_dates=True)
    assert np.isfinite(portfolio[["return", "cost", "bench"]].to_numpy()).all()
    assert (portfolio["cost"] >= 0).all()
    assert portfolio["cost"].sum() > 0
    stats = result["metrics"]["portfolio"]
    assert stats["sum_daily_cost_ratios"] > 0
    assert stats["benchmark"] == "cash"
    assert stats["benchmark_cumulative_return"] == 0
    np.testing.assert_array_equal(portfolio["bench"].to_numpy(), 0)
    assert stats["net_cumulative_return"] <= stats["gross_cumulative_return"]
    assert stats["net_cumulative_return"] == pytest.approx(
        (1 + portfolio["return"] - portfolio["cost"]).prod() - 1
    )
    assert stats["gross_cumulative_return"] == pytest.approx((1 + portfolio["return"]).prod() - 1)
    assert portfolio.index[0] == calendar[calendar.get_loc(pd.Timestamp(start)) + 1]
    assert portfolio.index[-1] == calendar[calendar.get_loc(pd.Timestamp(end)) + 2]

    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["feature_count"] == 157
    assert metadata["label"] == LABEL
    assert metadata["synthetic"] is True
    assert len(metadata["provider"]["sha256"]) == 64
    assert metadata["recorder_id"]
    assert (output / "mlruns").is_dir()
    assert json.loads((output / "metrics.json").read_text()) == result["metrics"]
    assert json.loads((output / "resolved_config.json").read_text())["segments"]
    task = yaml.safe_load((output / "qlib_task.yaml").read_text())["task"]
    assert {"SignalRecord", "SigAnaRecord"}.issubset({record["class"] for record in task["record"]})
    portfolio_records = [
        record
        for record in task["record"]
        if "backtest" in record.get("kwargs", {}).get("config", {})
    ]
    assert len(portfolio_records) == 1
    assert portfolio_records[0]["kwargs"]["config"]["backtest"]["benchmark"] is None


def test_exported_labels_are_raw_delayed_returns(completed_run):
    from pathlib import Path

    from qlib.data import D

    config, result = completed_run
    _init_qlib(config["provider_uri"])
    prices = D.features([f"DEMO{number:03d}" for number in range(4)], ["$close"])["$close"]
    future_close = prices.groupby(level="instrument").shift(-2)
    entry_close = prices.groupby(level="instrument").shift(-1)
    expected = (future_close / entry_close - 1).rename("label")
    actual = pd.read_csv(Path(result["output_dir"]) / "predictions.csv", parse_dates=["datetime"])
    actual = actual.set_index(["instrument", "datetime"])["label"].sort_index()
    pd.testing.assert_series_equal(
        actual, expected.reindex(actual.index), check_dtype=False, rtol=1e-6, atol=1e-8
    )


def test_completed_run_cannot_be_silently_overwritten(completed_run):
    from pathlib import Path

    config, result = completed_run
    artifact = Path(result["output_dir"]) / "run.json"
    before = artifact.read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        run_workflow(config)
    assert artifact.read_bytes() == before


def test_single_instrument_uses_raw_labels_and_skips_cross_sectional_statistics(
    market_data, tmp_path
):
    _, provider = market_data
    config = _config(provider, tmp_path / "single-index-run")
    config["instruments"] = ["DEMO000"]
    config["backtest"]["enabled"] = False
    result = run_workflow(config)
    output = tmp_path / "single-index-run"
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions["instrument"].unique().tolist() == ["DEMO000"]
    assert np.isfinite(predictions[["score", "label"]].to_numpy()).all()
    assert predictions["label"].std() > 0
    assert result["metrics"]["evaluated_count"] == len(predictions)
    task = yaml.safe_load((output / "qlib_task.yaml").read_text())["task"]
    assert [record["class"] for record in task["record"]] == [
        "ValidationSignalRecord",
        "SignalRecord",
    ]
    processors = task["dataset"]["kwargs"]["handler"]["kwargs"]["learn_processors"]
    assert not any(processor["class"] == "CSRankNorm" for processor in processors)
    assert not (output / "portfolio.csv").exists()


def test_future_perturbations_leave_training_features_and_fitted_scaler_unchanged(
    market_data, tmp_path
):
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.processor import RobustZScoreNorm
    from qlib.utils import init_instance_by_config

    source, original_provider = market_data
    raw = pd.read_csv(source, parse_dates=["date"])
    calendar = pd.DatetimeIndex(raw["date"].unique()).sort_values()
    task, segments = build_task(_config(original_provider, tmp_path / "unused"), calendar)
    train_start, train_end = segments["train"]
    handler_config = task["dataset"]["kwargs"]["handler"]

    def training_features(provider):
        _init_qlib(provider)
        handler = init_instance_by_config(copy.deepcopy(handler_config))
        features = handler.fetch(
            slice(train_start, train_end), col_set="feature", data_key=DataHandlerLP.DK_I
        ).copy(deep=True)
        scaler = next(
            processor
            for processor in handler.infer_processors
            if isinstance(processor, RobustZScoreNorm)
        )
        return features, scaler.mean_train.copy(), scaler.std_train.copy()

    original, original_median, original_scale = training_features(original_provider)
    future = raw["date"] > pd.Timestamp(train_end)
    multiplier = np.linspace(2, 20, int(future.sum()))
    raw.loc[future, ["open", "high", "low", "close"]] *= multiplier[:, None]
    raw["volume"] = raw["volume"].astype(float)
    raw.loc[future, "volume"] = raw.loc[future, "volume"] / multiplier
    changed_csv = tmp_path / "changed-future.csv"
    raw.to_csv(changed_csv, index=False)
    changed_provider = tmp_path / "changed-provider"
    import_csv(changed_csv, changed_provider)
    changed, changed_median, changed_scale = training_features(changed_provider)

    assert original.shape[1] == 157
    assert original.index.get_level_values("datetime").max() == pd.Timestamp(train_end)
    pd.testing.assert_frame_equal(original, changed, check_exact=True)
    np.testing.assert_array_equal(original_median, changed_median)
    np.testing.assert_array_equal(original_scale, changed_scale)
