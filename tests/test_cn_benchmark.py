import copy
import json

import numpy as np
import pandas as pd
import pytest

from financial_ts.cn_benchmark import (
    cn_segments,
    prepare_cn_benchmark,
    run_cn_models,
    summarize_cn_benchmark,
)


def test_fixed_cn_boundaries_reserve_all_future_sessions():
    calendar = pd.bdate_range("2019-01-01", "2026-11-26")
    segments = cn_segments(calendar)
    assert segments["train"][0] == "2020-01-01"
    assert segments["valid"][0] == "2024-01-01"
    assert segments["test"][0] == "2025-01-01"
    for left, right in (("train", "valid"), ("valid", "test")):
        assert calendar.get_loc(segments[left][1]) + 2 < calendar.get_loc(segments[right][0])
    assert calendar.get_loc(segments["test"][1]) + 3 == len(calendar) - 1


def _source(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    dates = pd.bdate_range("2019-01-01", "2026-11-26")
    (provider / "calendars" / "day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")))
    (provider / "metadata.json").write_text(
        json.dumps(
            {
                "source": "chenditc/investment_data",
                "synthetic": False,
                "target_trade_date": "2026-11-26",
                "release_tag": "2026-11-26",
            }
        )
    )
    return provider


def test_protocol_recovery_rejects_changed_training_budget(tmp_path):
    provider = _source(tmp_path)
    root = tmp_path / "run"
    first = prepare_cn_benchmark(provider, root)
    assert prepare_cn_benchmark(provider, root) == first
    assert len(first["models"]) == 8
    assert all(c["instruments"] == "csi300" for c in first["configs"].values())
    with pytest.raises(ValueError, match="differs"):
        prepare_cn_benchmark(provider, root, epochs=7)


def test_protocol_rejects_future_calendar(tmp_path):
    provider = _source(tmp_path)
    with (provider / "calendars" / "day.txt").open("a") as stream:
        stream.write("\n2026-11-27")
    with pytest.raises(ValueError, match="cutoff"):
        prepare_cn_benchmark(provider, tmp_path / "run")


def test_provider_mutation_is_rejected_before_completed_result_reuse(tmp_path):
    provider = _source(tmp_path)
    protocol = prepare_cn_benchmark(provider, tmp_path / "run")
    binary = provider / "features" / "sh600000" / "close.day.bin"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"changed binary contents under unchanged metadata")
    with pytest.raises(ValueError, match="Provider bytes changed"):
        run_cn_models(protocol, models=["lightgbm"])
    with pytest.raises(ValueError, match="differs"):
        prepare_cn_benchmark(provider, tmp_path / "run")


def test_report_selects_from_validation_not_test(tmp_path):
    protocol = {"models": ["lightgbm", "transformer"]}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    metric = {
        "validation": {"rmse": 0.02, "ic": 0.02, "rank_ic": 0.03},
        "rmse": 0.024,
        "zero_return_rmse": 0.024,
        "ic": 0.01,
        "rank_ic": 0.02,
        "evaluated_count": 100,
        "model_details": {},
    }
    for name, valid, returns in (
        ("lightgbm", 0.03, [-0.05, 0.01]),
        ("transformer", 0.01, [0.2, 0.2]),
    ):
        path = tmp_path / name
        path.mkdir()
        evidence = copy.deepcopy(metric)
        evidence["validation"]["rank_ic"] = valid
        (path / "metrics.json").write_text(json.dumps(evidence))
        (path / "run.json").write_text(json.dumps({"status": "completed"}))
        pd.DataFrame(
            {"return": returns, "cost": [0, 0], "bench": [0, 0], "turnover": [0.1, 0.1]},
            index=pd.bdate_range("2025-01-02", periods=2),
        ).to_csv(path / "portfolio.csv")
    result = summarize_cn_benchmark(tmp_path)
    assert result["primary_model"] == "lightgbm"
    assert result["status"] == "completed"
    assert result["models"][0]["max_drawdown"] == pytest.approx(-0.05)
    assert result["models"][0]["net_return"] == pytest.approx(0.95 * 1.01 - 1)
    assert np.isfinite(result["models"][1]["annualized_return_252"])
