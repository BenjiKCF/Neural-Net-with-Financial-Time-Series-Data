"""Economic accounting, observed-session timing, and explicit data acquisition."""

import csv
import hashlib
import importlib.util
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_ts.backtest import run_backtest


@pytest.fixture
def market():
    # Friday Jan 12 is followed by Tuesday Jan 16, after a weekend and holiday.
    dates = pd.DatetimeIndex(
        ["2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"]
    )
    prices = pd.Series([100.0, 100.0, 110.0, 99.0, 108.9, 119.79], index=dates)
    predictions = pd.DataFrame({"model": [1.0, 1.0, -1.0, 1.0]}, index=dates[:4])
    return predictions, prices


def model_ledger(result, name="model", cost=5):
    frame = result["ledger"]
    return frame.loc[(frame["model"] == name) & (frame["cost_bps"] == cost)].reset_index(drop=True)


def model_metrics(result, name="model", cost=5):
    frame = result["metrics"]
    return frame.loc[(frame["model"] == name) & (frame["cost_bps"] == cost)].iloc[0]


def test_calendar_alignment_uses_two_future_sessions_and_no_same_close_return(market):
    predictions, prices = market
    result = run_backtest(predictions, prices)
    ledger = model_ledger(result)
    assert ledger["signal_date"].tolist() == prices.index[:4].tolist()
    assert ledger["entry_date"].tolist() == prices.index[1:5].tolist()
    assert ledger["exit_date"].tolist() == prices.index[2:6].tolist()
    assert ledger.iloc[0]["entry_date"] == pd.Timestamp("2024-01-12")
    assert ledger.iloc[0]["exit_date"] == pd.Timestamp("2024-01-16")
    np.testing.assert_allclose(ledger["asset_return"], [0.1, -0.1, 0.1, 0.1])
    assert set(result["metrics"]["cost_bps"]) == {0, 5, 10, 20}


def test_exact_self_financing_fees_and_terminal_liquidation(market):
    predictions, prices = market
    result = run_backtest(predictions, prices, costs_bps=(100,))
    ledger = model_ledger(result, cost=100)
    fee = 0.01
    expected = (1.1 * 0.9 * 1.1) * (1 - fee) ** 2 / (1 + fee) ** 2
    assert ledger["net_equity"].iloc[-1] == pytest.approx(expected)
    np.testing.assert_allclose(ledger["cost_factor"], [1 / 1.01, 1, 0.99, 1 / 1.01])
    np.testing.assert_allclose(ledger["liquidation_factor"], [1, 1, 1, 0.99])
    np.testing.assert_allclose(ledger["position"], [1, 1, 0, 1])
    assert ledger["entry_trade"].tolist() == [1, 0, 0, 1]
    assert ledger["exit_trade"].tolist() == [0, 0, 1, 0]
    assert ledger["liquidation_trade"].tolist() == [0, 0, 0, 1]
    assert ledger["trade_cost"].iloc[0] == pytest.approx(fee / (1 + fee))
    assert ledger["traded_notional"].iloc[0] + ledger["trade_cost"].iloc[0] == pytest.approx(1)
    assert ledger["trade_cost"].iloc[1] == 0
    assert ledger["gross_equity"].iloc[-1] == pytest.approx(1.089)
    metrics = model_metrics(result, cost=100)
    assert metrics["entries"] == metrics["exits"] == metrics["completed_round_trips"] == 2
    assert metrics["trade_count"] == metrics["exposure_turnover"] == 4
    assert metrics["mean_exposure"] == 0.75
    assert metrics["elapsed_days"] == 7
    assert metrics["cagr"] == pytest.approx(expected ** (365.25 / 7) - 1)


def test_positive_forecasts_equal_buy_hold_and_do_not_retrade_daily(market):
    predictions, prices = market
    predictions["model"] = 0.01
    result = run_backtest(predictions, prices)
    for cost in (0, 5, 10, 20):
        ledger = model_ledger(result, cost=cost)
        control = model_ledger(result, "buy_and_hold", cost)
        pd.testing.assert_frame_equal(
            ledger.drop(columns=["model", "score"]), control.drop(columns=["model", "score"])
        )
        rate = cost / 10000
        expected = prices.iloc[-1] / prices.iloc[1] * (1 - rate) / (1 + rate)
        assert ledger["net_equity"].iloc[-1] == pytest.approx(expected)
        assert ledger["entry_trade"].sum() == 1
        assert ledger["exit_trade"].sum() == 0
        assert ledger["liquidation_trade"].sum() == 1
        assert model_metrics(result, cost=cost)["trade_count"] == 2


def test_zero_and_negative_scores_stay_in_cash_and_have_undefined_sharpe(market):
    predictions, prices = market
    predictions["model"] = [0, -1, 0, -0.5]
    result = run_backtest(predictions, prices)
    ledger = model_ledger(result)
    assert (ledger["position"] == 0).all()
    assert (ledger["net_equity"] == 1).all()
    assert (ledger["trade_cost"] == 0).all()
    metrics = model_metrics(result)
    assert metrics["total_return"] == metrics["max_drawdown"] == metrics["cagr"] == 0
    assert metrics["annualized_volatility"] == 0
    assert metrics["sharpe_zero_cash"] is None
    assert metrics["trade_count"] == 0
    # The output can be converted to strict JSON without NaN placeholders.
    json.dumps(result["metrics"].to_dict(orient="records"), allow_nan=False)


def test_drawdown_includes_starting_cash_nav_and_compounds_daily_returns(market):
    predictions, prices = market
    predictions["model"] = 1.0
    prices.iloc[2:] = [90, 81, 81, 90]
    result = run_backtest(predictions, prices, costs_bps=(0,))
    ledger = model_ledger(result, cost=0)
    np.testing.assert_allclose(ledger["net_equity"], [0.9, 0.81, 0.81, 0.9])
    np.testing.assert_allclose(ledger["drawdown"], [-0.1, -0.19, -0.19, -0.1])
    np.testing.assert_allclose(np.cumprod(1 + ledger["net_return"]), ledger["net_equity"])
    assert model_metrics(result, cost=0)["max_drawdown"] == pytest.approx(-0.19)


def test_future_prices_or_signals_do_not_change_earlier_positions_or_pnl(market):
    predictions, prices = market
    baseline = model_ledger(run_backtest(predictions, prices))
    changed_prices = prices.copy()
    changed_prices.iloc[-1] *= 3
    changed = model_ledger(run_backtest(predictions, changed_prices))
    pd.testing.assert_frame_equal(baseline.iloc[:-1], changed.iloc[:-1])
    assert baseline["position"].equals(changed["position"])
    changed_signals = predictions.copy()
    changed_signals.iloc[-1, 0] = -100
    changed = model_ledger(run_backtest(changed_signals, prices))
    pd.testing.assert_frame_equal(baseline.iloc[:-1], changed.iloc[:-1])


def test_single_instrument_multiindex_is_normalized_without_changing_results(market):
    predictions, prices = market
    baseline = run_backtest(predictions, prices)
    predictions.index = pd.MultiIndex.from_arrays(
        [predictions.index, ["SPX"] * len(predictions)], names=["datetime", "instrument"]
    )
    result = run_backtest(predictions, prices)
    pd.testing.assert_frame_equal(result["ledger"], baseline["ledger"])
    assert result["settings"]["signal_instrument"] == "SPX"


def test_one_observation_pays_both_fees_without_inventing_volatility(market):
    predictions, prices = market
    result = run_backtest(predictions.iloc[:1], prices, costs_bps=(100,))
    metrics = model_metrics(result, cost=100)
    assert metrics["ending_equity"] == pytest.approx(1.1 * 0.99 / 1.01)
    assert metrics["trade_count"] == 2
    assert metrics["annualized_volatility"] is None
    assert metrics["sharpe_zero_cash"] is None


@pytest.mark.parametrize(
    "costs", [(), (-1,), (float("nan"),), (float("inf"),), (10000,), (True,), (5, 5), ("5",)]
)
def test_invalid_cost_scenarios_are_rejected(market, costs):
    with pytest.raises(ValueError):
        run_backtest(*market, costs_bps=costs)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_signal",
        "missing_future",
        "duplicate_signal",
        "unsorted_signal",
        "duplicate_price",
        "nan_price",
        "zero_price",
        "nan_signal",
        "timezone",
        "intraday",
        "reserved_name",
        "many_instruments",
    ],
)
def test_invalid_or_incomplete_inputs_are_rejected(market, problem):
    predictions, prices = market
    if problem == "missing_signal":
        predictions = predictions.drop(predictions.index[1])
    elif problem == "missing_future":
        prices = prices.iloc[:-1]
    elif problem == "duplicate_signal":
        predictions.index = [predictions.index[0], *predictions.index[:-1]]
    elif problem == "unsorted_signal":
        predictions = predictions.iloc[::-1]
    elif problem == "duplicate_price":
        prices.index = [prices.index[0], *prices.index[:-1]]
    elif problem == "nan_price":
        prices.iloc[2] = np.nan
    elif problem == "zero_price":
        prices.iloc[2] = 0
    elif problem == "nan_signal":
        predictions.iloc[2, 0] = np.nan
    elif problem == "timezone":
        predictions.index = predictions.index.tz_localize("UTC")
    elif problem == "intraday":
        predictions.index = predictions.index + pd.Timedelta(hours=1)
    elif problem == "reserved_name":
        predictions.columns = ["buy_and_hold"]
    elif problem == "many_instruments":
        predictions.index = pd.MultiIndex.from_arrays(
            [predictions.index, ["SPX", "SPX", "QQQ", "SPX"]], names=["datetime", "instrument"]
        )
    with pytest.raises(ValueError):
        run_backtest(predictions, prices)


def _downloader():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_spy.py"
    spec = importlib.util.spec_from_file_location("download_spy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_payload():
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": "SPY", "currency": "USD", "instrumentType": "ETF"},
                    "timestamp": [
                        int(datetime(2013, 1, day, 21, tzinfo=timezone.utc).timestamp())
                        for day in (2, 3)
                    ],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 102.0],
                                "high": [101.0, 104.0],
                                "low": [98.0, 100.0],
                                "close": [100.0, 102.0],
                                "volume": [1000, 2000],
                            }
                        ],
                        "adjclose": [{"adjclose": [50.0, 51.0]}],
                    },
                    "events": {"dividends": {"1357246800": {"amount": 0.5}}},
                }
            ],
        }
    }


def test_offline_snapshot_preserves_adjustments_raw_bytes_and_hashes(tmp_path, monkeypatch):
    downloader = _downloader()
    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("Unexpected network"),
    )
    raw = json.dumps(_source_payload()).encode()
    output = tmp_path / "snapshot"
    manifest = downloader.write_snapshot(
        raw,
        output,
        start=date(2013, 1, 1),
        end=date(2013, 1, 4),
        received_at_utc="2026-11-27T00:00:00+00:00",
    )
    assert (output / "spy_raw.json").read_bytes() == raw
    with (output / "spy_daily.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["date"] for row in rows] == ["2013-01-02", "2013-01-03"]
    assert [float(row["adj_close"]) for row in rows] == [50, 51]
    assert manifest["dividend_events"] == 1
    assert manifest["split_events"] == 0
    for record in manifest["files"]:
        assert (
            hashlib.sha256((output / record["name"]).read_bytes()).hexdigest() == record["sha256"]
        )
    with pytest.raises(ValueError, match="new or empty"):
        downloader.write_snapshot(
            raw, output, start=date(2013, 1, 1), end=date(2013, 1, 4), received_at_utc="now"
        )


def test_snapshot_rejects_missing_prices_before_writing(tmp_path):
    payload = _source_payload()
    payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"][0] = None
    with pytest.raises(ValueError, match="finite and positive"):
        _downloader().write_snapshot(
            json.dumps(payload).encode(),
            tmp_path / "bad",
            start=date(2013, 1, 1),
            end=date(2013, 1, 4),
            received_at_utc="now",
        )
    assert not (tmp_path / "bad").exists()


def test_costs_reduce_wealth_and_volatility_uses_daily_net_returns(market):
    result = run_backtest(*market)
    model = result["metrics"].query("model == 'model'").sort_values("cost_bps")
    assert (np.diff(model["ending_equity"]) < 0).all()
    ledger = model_ledger(result)
    metrics = model_metrics(result)
    deviation = ledger["net_return"].std(ddof=1)
    assert metrics["annualized_volatility"] == pytest.approx(deviation * math.sqrt(252))
    assert metrics["sharpe_zero_cash"] == pytest.approx(
        ledger["net_return"].mean() / deviation * math.sqrt(252)
    )
