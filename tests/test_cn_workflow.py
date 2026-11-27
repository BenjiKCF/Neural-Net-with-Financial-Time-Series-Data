"""CN rules, observed-date boundaries, and cache/validation data integrity."""

import json

import numpy as np
import pandas as pd
import pytest

from financial_ts.data import create_demo_csv, import_csv
from financial_ts.workflow import (
    _portfolio_config,
    _prediction_statistics,
    build_task,
    run_workflow,
)


@pytest.fixture
def calendar():
    return pd.bdate_range("2023-01-02", periods=180)


def test_cn_defaults_and_delayed_execution(calendar):
    config = {"region": "cn", "backtest": {"enabled": True}}
    _, segments = build_task(config, calendar)
    result = _portfolio_config(config, segments, calendar, 300)
    strategy, bt = result["strategy"]["kwargs"], result["backtest"]
    assert (strategy["topk"], strategy["n_drop"]) == (50, 5)
    assert bt["account"] == 100000000
    assert bt["benchmark"] == "SH000300"
    assert bt["exchange_kwargs"] == {
        "freq": "day",
        "deal_price": "close",
        "limit_threshold": 0.095,
        "trade_unit": 100,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5.0,
    }
    first, last = map(pd.Timestamp, segments["test"])
    assert pd.Timestamp(bt["start_time"]) == calendar[calendar.get_loc(first) + 1]
    assert pd.Timestamp(bt["end_time"]) == calendar[calendar.get_loc(last) + 2]
    assert pd.Timestamp(bt["end_time"]) < calendar[-1]


def test_us_default_execution_is_preserved(calendar):
    config = {"region": "us"}
    _, segments = build_task(config, calendar)
    result = _portfolio_config(config, segments, calendar, 10)
    assert result["backtest"]["benchmark"] is None
    assert result["backtest"]["account"] == 100000
    assert result["backtest"]["exchange_kwargs"]["trade_unit"] is None
    assert result["backtest"]["exchange_kwargs"]["limit_threshold"] is None
    assert result["strategy"]["kwargs"]["topk"] == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("limit_threshold", 0),
        ("limit_threshold", 1),
        ("limit_threshold", float("nan")),
        ("trade_unit", 0),
        ("trade_unit", 1.5),
        ("trade_unit", True),
        ("open_cost", -0.1),
        ("close_cost", float("inf")),
        ("account", 0),
        ("risk_degree", float("nan")),
        ("topk", True),
        ("n_drop", -1),
    ],
)
def test_invalid_cn_execution_inputs_rejected(calendar, field, value):
    config = {"region": "cn", "backtest": {field: value}}
    _, segments = build_task(config, calendar)
    with pytest.raises(ValueError):
        _portfolio_config(config, segments, calendar, 300)


def test_window_bounds_limit_feature_computation_and_split(calendar):
    config = {"data_start_time": "2023-02-04", "data_end_time": "2023-08-31", "warmup_sessions": 10}
    task, segments = build_task(config, calendar)
    handler = task["dataset"]["kwargs"]["handler"]["kwargs"]
    assert handler["start_time"] == "2023-02-06"
    assert handler["end_time"] == "2023-08-31"
    assert segments["test"][1] == "2023-08-29"


def test_cross_sectional_ic_averages_days_not_pooled_panel():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"score": [1, 2, 3, 1, 2, 3], "label": [1, 2, 3, 3, 2, 1]}, index=index)
    stats, daily = _prediction_statistics(frame)
    assert daily["ic"].tolist() == pytest.approx([1.0, -1.0])
    assert stats["ic"] == pytest.approx(0)
    assert stats["rank_ic"] == pytest.approx(0)
    assert stats["ic_days"] == 2


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    root = tmp_path_factory.mktemp("cn-workflow-data")
    source = create_demo_csv(root / "prices.csv", n_instruments=4, n_days=160)
    result = root / "provider"
    metadata = import_csv(source, result)
    metadata["synthetic"] = True
    (result / "metadata.json").write_text(json.dumps(metadata))
    return result


def quick_config(provider, output):
    return {
        "provider_uri": str(provider),
        "output_dir": str(output),
        "region": "cn",
        "data_start_time": "2020-01-01",
        "model": {
            "name": "lightgbm",
            "params": {
                "num_boost_round": 3,
                "early_stopping_rounds": 2,
                "min_data_in_leaf": 10,
                "num_leaves": 4,
                "num_threads": 1,
                "verbosity": -1,
            },
        },
        "backtest": {"enabled": True, "topk": 2, "n_drop": 1, "benchmark": None},
    }


@pytest.mark.integration
def test_cn_native_run_validation_and_cache_reuse(provider, tmp_path):
    actual_calendar = pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
    config = quick_config(provider, tmp_path / "first")
    config["data_start_time"] = str(actual_calendar[0].date())
    config["handler_cache_path"] = str(tmp_path / "handler.pkl")
    first = run_workflow(config)
    assert first["run"]["handler_cache"]["reused"] is False
    assert first["metrics"]["portfolio"]["sum_daily_cost_ratios"] > 0
    assert first["metrics"]["portfolio"]["exchange_kwargs"]["trade_unit"] == 100
    validation = pd.read_csv(tmp_path / "first/validation_predictions.csv")
    assert np.isfinite(validation[["score", "label"]]).all().all()
    expected_rmse = np.sqrt(((validation["score"] - validation["label"]) ** 2).mean())
    assert first["metrics"]["validation"]["rmse"] == pytest.approx(expected_rmse)
    dates = pd.to_datetime(validation["datetime"])
    start, end = map(pd.Timestamp, first["run"]["segments"]["valid"])
    assert dates.min() == start and dates.max() == end
    assert (tmp_path / "first/validation_daily_ic.csv").is_file()
    config["output_dir"] = str(tmp_path / "second")
    second = run_workflow(config)
    assert second["run"]["handler_cache"]["reused"] is True
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "first/predictions.csv"),
        pd.read_csv(tmp_path / "second/predictions.csv"),
    )
    config["instruments"] = ["DEMO000", "DEMO001"]
    config["output_dir"] = str(tmp_path / "changed-universe")
    with pytest.raises(ValueError, match="does not match"):
        run_workflow(config)
    config.pop("instruments")
    config["output_dir"] = str(tmp_path / "tampered-cache")
    with (tmp_path / "handler.pkl").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        run_workflow(config)


@pytest.mark.integration
def test_manifest_observed_end_prevents_future_calendar_scoring(provider, tmp_path):
    config = quick_config(provider, tmp_path / "future")
    calendar = pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
    config["data_start_time"] = str(calendar[0].date())
    # Copy only metadata/calendar here: the invalid window is rejected before any data read.
    copy_provider = tmp_path / "provider"
    (copy_provider / "calendars").mkdir(parents=True)
    (copy_provider / "calendars/day.txt").write_text((provider / "calendars/day.txt").read_text())
    observed = str(calendar[-10].date())
    (copy_provider / "metadata.json").write_text(json.dumps({"target_trade_date": observed}))
    config["provider_uri"] = str(copy_provider)
    config["data_end_time"] = str(calendar[-1].date())
    with pytest.raises(ValueError, match="observed target_trade_date"):
        run_workflow(config)
