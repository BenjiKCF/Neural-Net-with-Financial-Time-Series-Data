"""Trade-only ties, native strategy equivalence, and independent CSV replay."""

import copy
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from financial_ts.cn_portfolio import (
    TRADE_TIE_CONVENTION,
    DeterministicRankSignal,
    deterministic_trade_signal,
    numeric_positions,
    replay_cn_portfolio,
)
from financial_ts.data import create_demo_csv, import_csv
from financial_ts.workflow import _portfolio_config, _provider_fingerprint


def test_trade_ranks_break_exact_ties_only_and_never_change_raw_forecasts():
    date = pd.Timestamp("2025-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, name) for name in ["ZZZ", "AAA", "NA", "BBB", "CCC"]],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        {"score": [0.1, 0.1, np.nextafter(0.1, np.inf), -0.0, 0.0], "label": [1, 2, 3, 4, 5]},
        index=index,
    )
    original = frame.copy(deep=True)
    ranks = deterministic_trade_signal(frame[["score"]])
    assert ranks.sort_values(ascending=False).index.get_level_values("instrument").tolist() == [
        "NA",
        "AAA",
        "ZZZ",
        "BBB",
        "CCC",
    ]
    pd.testing.assert_series_equal(
        ranks, deterministic_trade_signal(frame.sample(frac=1, random_state=27)[["score"]])
    )
    pd.testing.assert_frame_equal(frame, original)
    assert ranks.nunique() == len(frame)
    assert (ranks.to_numpy() < 0).all()
    for left in index:
        for right in index:
            if frame.loc[left, "score"] > frame.loc[right, "score"]:
                assert ranks.loc[left] > ranks.loc[right]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_trade_ranking_rejects_nonfinite_scores(invalid):
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-02"), "A")], names=["datetime", "instrument"]
    )
    with pytest.raises(ValueError, match="finite"):
        deterministic_trade_signal(pd.Series([invalid], index=index))


@pytest.fixture
def native_inputs(tmp_path):
    provider = tmp_path / "provider"
    import_csv(create_demo_csv(tmp_path / "prices.csv", n_instruments=4, n_days=100), provider)
    calendar = pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
    dates = calendar[80:88]
    index = pd.MultiIndex.from_product(
        [dates, ["DEMO000", "DEMO001", "DEMO002", "DEMO003"]], names=["datetime", "instrument"]
    )
    scores = pd.DataFrame(
        {"score": np.tile([0.1, 0.2, -0.1, -0.2], len(dates)), "label": 0.01}, index=index
    )
    config = {
        "region": "cn",
        "segments": {"test": [str(dates[0].date()), str(dates[-1].date())]},
        "backtest": {"topk": 2, "n_drop": 1, "benchmark": None},
    }
    return provider, calendar, scores, config


@pytest.mark.integration
def test_native_topk_without_ties_is_unchanged(native_inputs):
    import qlib
    from qlib.backtest import backtest

    provider, calendar, scores, config = native_inputs
    qlib.init(
        provider_uri=str(provider),
        region="cn",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    portfolio = _portfolio_config(config, config["segments"], calendar, 4)
    portfolio["backtest"]["benchmark"] = pd.Series(0.0, index=calendar)
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    reports = []
    positions = []
    for signal in [scores[["score"]], DeterministicRankSignal(scores[["score"]])]:
        strategy = copy.deepcopy(portfolio["strategy"])
        strategy["kwargs"]["signal"] = signal
        result, _ = backtest(executor=executor, strategy=strategy, **portfolio["backtest"])
        report, position = result["1day"]
        reports.append(report)
        positions.append(numeric_positions(position))
    pd.testing.assert_frame_equal(reports[0], reports[1], check_exact=True)
    assert positions[0] == positions[1]


@pytest.mark.integration
def test_raw_and_gzip_replay_are_identical_with_tied_shuffled_scores(native_inputs, tmp_path):
    provider, _, scores, config = native_inputs
    scores["score"] = 0.1
    outputs = []
    for name, compressed in [("raw", False), ("compressed", True)]:
        source = tmp_path / name
        source.mkdir()
        input_frame = scores if not compressed else scores.sample(frac=1, random_state=17)
        input_frame.to_csv(source / ("predictions.csv.gz" if compressed else "predictions.csv"))
        (source / "resolved_config.json").write_text(json.dumps(config))
        (source / "run.json").write_text(
            json.dumps(
                {
                    "status": "forecasts_completed" if compressed else "completed",
                    "provider": _provider_fingerprint(provider),
                }
            )
        )
        (source / "metrics.json").write_text(
            json.dumps(
                {
                    "rmse": 0.09,
                    "qlib_metrics": {
                        "IC": 0.2,
                        "1day.excess_return_with_cost.annualized_return": 999,
                    },
                    "portfolio": {"old": True},
                }
            )
        )
        output = tmp_path / f"{name}-replay"
        manifest = replay_cn_portfolio(source, provider, output)
        assert manifest["exact_score_ties"] == {"dates": 8, "rows": 32, "largest_group": 4}
        assert manifest["trade_signal"] == TRADE_TIE_CONVENTION
        assert not list(output.glob("*.pkl"))
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["rmse"] == 0.09 and metrics["qlib_metrics"]["IC"] == 0.2
        assert metrics["qlib_metrics"]["1day.excess_return_with_cost.annualized_return"] != 999
        assert "old" not in metrics["portfolio"]
        with gzip.open(output / "positions.json.gz", "rt") as stream:
            positions = json.load(stream)
        assert [stock["instrument"] for stock in positions["positions"][0]["holdings"]] == [
            "DEMO000",
            "DEMO001",
        ]
        outputs.append(output)
    for filename in [
        "portfolio.csv",
        "positions.json.gz",
        "metrics.json",
        "source_gap_holdings.json",
    ]:
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
