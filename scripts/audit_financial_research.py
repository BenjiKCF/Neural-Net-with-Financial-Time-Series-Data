#!/usr/bin/env python3
"""Independently audit published forecasts and SPY portfolios, entirely offline.

Requires only NumPy and pandas; imports no project accounting or model code.
Run from any directory:

    python /path/to/repo/scripts/audit_financial_research.py

The default verifies published source hashes, exact forecast identities/labels,
all 88 portfolio rows, and primary-cost equity/positions using cash/share
accounting. The full original SPY download is not distributed; its hash check
is explicitly skipped unless supplied with --raw-json. Published price hashes
and every simulated return are still checked. No files are changed unless a
new --output JSON path is given; the completed independent_audit.json is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COSTS = (0.0, 5.0, 10.0, 20.0)
ARCHIVED_MODELS = (
    "lightgbm",
    "transformer",
    "deepseek_v41_mini",
    "glm54_mini",
    "qwen4_mini",
    "timesfm3_mini",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(path, dates=()):
    return pd.read_csv(path, parse_dates=list(dates), float_precision="round_trip")


def predictions(path):
    result = read_frame(path, ("datetime",)).set_index(["datetime", "instrument"]).sort_index()
    require(not result.empty and result.index.is_unique, f"Empty or duplicated identities: {path}")
    require(
        result.index.get_level_values("instrument").unique().tolist() == ["SPX"],
        f"Expected one SPX signal instrument: {path}",
    )
    require(np.isfinite(result.label).all(), f"Nonfinite target: {path}")
    return result


def check_hash(path, record):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == record["sha256"], f"SHA-256 mismatch: {path}")
    if "bytes" in record:
        require(len(data) == record["bytes"], f"Size mismatch: {path}")
    return {"path": str(path), "status": "verified", "sha256": record["sha256"]}


def source_checks(root, report, settings, raw_json):
    """Map original run locations to byte-identical published evidence copies."""
    mappings = {
        **{
            f"/reports/{name}": root / "reports" / name
            for name in ("architecture_predictions.csv", "architecture_benchmark.json")
        },
        **{
            f"/runs/model-research/{name}": root / "reports/model_research" / name
            for name in (
                "test_predictions.csv",
                "validation_predictions.csv",
                "selection.json",
                "metrics.json",
                "protocol.json",
            )
        },
        **{
            f"/runs/timesfm-comparison/{name}": root / "reports/timesfm_comparison" / name
            for name in ("all_period_predictions.csv", "comparison.json")
        },
    }
    checks = []
    for record in settings["sources"]:
        if record["path"].endswith("/.data/backtest/spy_daily.csv"):
            full_csv = raw_json.with_name("spy_daily.csv") if raw_json is not None else None
            if full_csv is not None and full_csv.is_file():
                checks.append(check_hash(full_csv, record))
            else:
                checks.append(
                    {
                        "path": record["path"],
                        "status": "skipped",
                        "reason": "Full 3,218-row download is not published; verify the published slice below.",
                    }
                )
            continue
        matches = [path for suffix, path in mappings.items() if record["path"].endswith(suffix)]
        require(len(matches) == 1, f"No unique published mapping for {record['path']}")
        checks.append(check_hash(matches[0], record))
    original_audit = load_json(report / "independent_audit.json")
    published = original_audit["published_only_offline_replay"]["files"]
    for record in published:
        checks.append(check_hash(report / Path(record["path"]).name, record))
    return checks


def raw_source_check(raw_json, settings, prices):
    if raw_json is None:
        return {
            "status": "skipped",
            "reason": "No --raw-json supplied; using hashed published prices.",
        }
    raw_record = next(r for r in settings["spy_provenance"]["files"] if r["name"] == "spy_raw.json")
    verified = check_hash(raw_json, raw_record)
    raw = load_json(raw_json)["chart"]["result"][0]
    require(raw["meta"]["symbol"] == "SPY", "Raw download does not identify SPY.")
    dates = (
        pd.to_datetime(raw["timestamp"], unit="s", utc=True)
        .tz_convert(raw["meta"]["exchangeTimezoneName"])
        .normalize()
        .tz_localize(None)
    )
    raw_prices = pd.Series(raw["indicators"]["adjclose"][0]["adjclose"], index=dates)
    require(
        np.array_equal(prices.to_numpy(), raw_prices.reindex(prices.index).to_numpy()),
        "Published adjusted closes differ from raw download.",
    )
    locations = dates.get_indexer(prices.index)
    require(
        (locations >= 0).all() and (np.diff(locations) == 1).all(),
        "Published slice omits raw observed sessions.",
    )
    return {**verified, "published_rows_matched": len(prices)}


def simulate(scores, bps, entry_prices, exit_prices, elapsed_days):
    """Buy or sell explicit shares at each boundary; retain holdings between trades."""
    allocation = np.asarray(scores) > 0
    cash, shares, fees = 1.0, 0.0, 0.0
    entries, exits, nav = 0, 0, []
    cost = bps / 10000
    for i, desired in enumerate(allocation):
        if desired and shares == 0:
            shares = cash / (entry_prices[i] * (1 + cost))
            fees += shares * entry_prices[i] * cost
            cash = 0.0
            entries += 1
        elif not desired and shares > 0:
            notional = shares * entry_prices[i]
            fees += notional * cost
            cash += notional * (1 - cost)
            shares = 0.0
            exits += 1
        if i == len(allocation) - 1 and shares > 0:
            notional = shares * exit_prices[i]
            fees += notional * cost
            cash += notional * (1 - cost)
            shares = 0.0
            exits += 1
        nav.append(cash + shares * exit_prices[i])
    nav = np.asarray(nav)
    full_path = np.concatenate(([1.0], nav))
    daily = full_path[1:] / full_path[:-1] - 1
    volatility = float(np.std(daily, ddof=1) * np.sqrt(252))
    gross = float(np.prod(np.where(allocation, exit_prices / entry_prices, 1)))
    result = {
        "periods": len(scores),
        "elapsed_days": elapsed_days,
        "ending_equity": float(nav[-1]),
        "total_return": float(nav[-1] - 1),
        "gross_total_return": gross - 1,
        "cagr": float(nav[-1] ** (365.25 / elapsed_days) - 1),
        "annualized_volatility": volatility,
        "sharpe_zero_cash": float(daily.mean() * 252 / volatility) if volatility else None,
        "max_drawdown": float((full_path / np.maximum.accumulate(full_path) - 1).min()),
        "mean_exposure": float(allocation.mean()),
        "entries": entries,
        "exits": exits,
        "trade_count": entries + exits,
        "completed_round_trips": entries,
        "exposure_turnover": float(entries + exits),
        "fees_paid_per_initial_equity": fees,
    }
    return result, nav, allocation


def audit(root, report, raw_json=None):
    pred = predictions(report / "predictions.csv")
    historical = predictions(root / "reports/architecture_predictions.csv")
    scratch = predictions(root / "reports/model_research/test_predictions.csv")
    valid = predictions(root / "reports/model_research/validation_predictions.csv")
    official = predictions(root / "reports/timesfm_comparison/all_period_predictions.csv")
    settings = load_json(report / "settings.json")
    require(
        settings["primary_cost_bps"] == 5 and tuple(settings["costs_bps"]) == COSTS,
        "Expected fixed 0/5/10/20 bps assumptions and 5 bps primary cost.",
    )
    require(
        settings["initial_equity"] == 1
        and settings["cash_return"] == 0
        and settings["terminal_liquidation"]
        and not settings["shorting"]
        and not settings["leverage"],
        "Unsupported portfolio assumptions.",
    )
    require(
        settings["threshold_selection"] == "none; threshold fixed at zero",
        "Expected the fixed zero threshold.",
    )
    hashes = source_checks(root, report, settings, raw_json)
    alignment = {}
    for name, reference, candidate in (
        ("archived_test", pred, historical),
        ("scratch_test", pred, scratch),
        ("official_test", pred, official[official.segment == "test"]),
        ("official_validation", valid, official[official.segment == "valid"]),
    ):
        require(reference.index.equals(candidate.index), f"Identity mismatch: {name}")
        delta = float(np.max(np.abs(reference.label.to_numpy() - candidate.label.to_numpy())))
        require(delta <= 1e-12, f"Target mismatch: {name}")
        alignment[name] = {"rows": len(reference), "maximum_absolute_label_difference": delta}
    require(np.isfinite(pred.to_numpy()).all(), "Nonfinite published forecast or target.")
    price_frame = read_frame(report / "spy_prices.csv", ("date",))
    require(set(price_frame.symbol) == {"SPY"}, "Execution prices must identify SPY.")
    prices = price_frame.set_index("date").adj_close
    require(
        prices.index.is_unique
        and prices.index.is_monotonic_increasing
        and not prices.index.hasnans
        and np.isfinite(prices).all()
        and (prices > 0).all(),
        "Published prices must have unique sorted dates and finite positive values.",
    )
    raw_check = raw_source_check(raw_json, settings, prices)
    signal_dates = pred.index.get_level_values("datetime")
    locations = prices.index.get_indexer(signal_dates)
    require(
        (locations >= 0).all()
        and (np.diff(locations) == 1).all()
        and locations[-1] + 2 < len(prices),
        "Missing signal or execution sessions.",
    )
    entry_dates, exit_dates = prices.index[locations + 1], prices.index[locations + 2]
    entry_prices, exit_prices = (
        prices.iloc[locations + 1].to_numpy(),
        prices.iloc[locations + 2].to_numpy(),
    )
    execution = read_frame(report / "execution.csv", ("signal_date", "entry_date", "exit_date"))
    for column, expected in (
        ("signal_date", signal_dates),
        ("entry_date", entry_dates),
        ("exit_date", exit_dates),
        ("signal_adjusted_close", prices.iloc[locations].to_numpy()),
        ("entry_adjusted_close", entry_prices),
        ("exit_adjusted_close", exit_prices),
    ):
        require(
            np.array_equal(execution[column].to_numpy(), np.asarray(expected)),
            f"Execution timing or price mismatch: {column}",
        )
    require(
        np.allclose(execution.asset_return, exit_prices / entry_prices - 1, rtol=0, atol=1e-16),
        "SPY asset returns differ from published prices.",
    )
    metrics = read_frame(report / "comparison.csv").set_index(["model", "one_way_cost_bps"])
    model_names = [name for name in pred if name != "label"]
    expected_rows = {
        (name, cost) for name in [*model_names, "cash", "buy_and_hold"] for cost in COSTS
    }
    require(
        metrics.index.is_unique and set(metrics.index) == expected_rows,
        "Comparison must cover every forecast and both controls at all four costs.",
    )
    equity = read_frame(report / "equity.csv", ("exit_date",)).set_index("exit_date")
    positions = read_frame(report / "positions.csv", ("signal_date",)).set_index("signal_date")
    require(
        equity.index[0] == entry_dates[0] and equity.index[1:].equals(exit_dates),
        "Equity dates do not preserve the initial NAV and each holding period.",
    )
    require(positions.index.equals(signal_dates), "Position dates differ from signal dates.")
    require(
        set(equity.columns) == set(positions.columns) == {*model_names, "cash", "buy_and_hold"},
        "Equity or positions omit strategies.",
    )
    elapsed_days = int((exit_dates[-1] - entry_dates[0]).days)
    date_metrics = {
        "signal_start": signal_dates[0],
        "signal_end": signal_dates[-1],
        "entry_start": entry_dates[0],
        "exit_end": exit_dates[-1],
    }
    maximum_metric_delta, maximum_equity_delta = 0.0, 0.0
    for (name, cost), reported in metrics.iterrows():
        scores = (
            np.ones(len(pred))
            if name == "buy_and_hold"
            else np.zeros(len(pred))
            if name == "cash"
            else pred[name].to_numpy()
        )
        computed, nav, allocation = simulate(scores, cost, entry_prices, exit_prices, elapsed_days)
        for key, value in computed.items():
            if value is None:
                require(pd.isna(reported[key]), f"Expected undefined {name}/{cost}/{key}")
            else:
                delta = abs(value - float(reported[key]))
                require(
                    np.isfinite(delta) and delta < 1e-10, f"Metric mismatch: {name}/{cost}/{key}"
                )
                maximum_metric_delta = max(maximum_metric_delta, delta)
        for key, date in date_metrics.items():
            require(pd.Timestamp(reported[key]) == date, f"Date mismatch: {name}/{cost}/{key}")
        if cost == 5:
            require(equity[name].iloc[0] == 1, f"Initial NAV is not one: {name}")
            delta = float(np.max(np.abs(equity[name].iloc[1:].to_numpy() - nav)))
            require(np.isfinite(delta) and delta < 1e-10, f"Equity path mismatch: {name}")
            maximum_equity_delta = max(maximum_equity_delta, delta)
            require(
                np.array_equal(positions[name], allocation.astype(int)),
                f"Position differs from fixed score > 0 policy: {name}",
            )
    forecasts = read_frame(report / "forecast_metrics.csv").set_index("model")
    require(
        forecasts.index.is_unique and set(forecasts.index) == set(model_names),
        "Forecast metrics omit or duplicate models.",
    )
    for name, row in forecasts.iterrows():
        value = float(np.sqrt(np.mean((pred[name].to_numpy() - pred.label.to_numpy()) ** 2)))
        require(abs(value - row.test_rmse) < 1e-16, f"Forecast RMSE mismatch: {name}")
    for name in ARCHIVED_MODELS:
        values = historical[[f"{name}__seed_{seed}" for seed in (7, 42, 2026)]].to_numpy()
        require(np.isfinite(values).all(), f"Missing archived seed value: {name}")
        require(
            np.array_equal(pred[name].to_numpy(), values.mean(axis=1)),
            f"Archived forecast ensemble mismatch: {name}",
        )
    selection = load_json(root / "reports/model_research/selection.json")
    require(
        settings["selected_by_validation"]
        == settings["strategy_model_mapping"][selection["selected_by_validation"]],
        "Published selected model differs from locked validation selection.",
    )
    official_report = load_json(root / "reports/timesfm_comparison/comparison.json")
    require(
        settings["timesfm_replication"] == official_report["recovery"],
        "Published official TimesFM replication label differs from source evidence.",
    )
    require(
        "(archived)" in settings["display_names"]["timesfm3_mini"]
        and "(retrained)" in settings["display_names"]["timesfm3_mini_retrained"],
        "Archived and retrained mini baselines must remain clearly labeled.",
    )
    if official_report["recovery"]["status"] == "numerical_replication_only":
        require(
            "(retrained)" in settings["display_names"]["timesfm3_loop_128"],
            "Numerically different loop retrain must be explicitly labeled.",
        )
    return {
        "status": "passed",
        "method": "Independent cash/share accounting; no project accounting or model imports",
        "forecast_models": len(model_names),
        "strategies_with_controls": len(model_names) + 2,
        "portfolio_scenarios_checked": len(metrics),
        "portfolio_metric_columns_checked": len(computed) + len(date_metrics),
        "primary_equity_points_checked": len(equity) * len(equity.columns),
        "alignment": alignment,
        "published_spy_price_rows": len(prices),
        "maximum_metric_absolute_difference": maximum_metric_delta,
        "maximum_primary_equity_absolute_difference": maximum_equity_delta,
        "source_hash_checks": hashes,
        "optional_raw_source_check": raw_check,
        "official_timesfm_replication_status": official_report["recovery"]["status"],
        "scope": "Verifies published arithmetic and recorded file integrity; does not retrain models, establish exchange-calendar completeness without raw data, or validate predictive performance on a fresh holdout.",
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Repository root (defaults to this script's parent)"
    )
    parser.add_argument(
        "--report", type=Path, help="Report directory (defaults to ROOT/reports/financial_research)"
    )
    parser.add_argument(
        "--raw-json", type=Path, help="Optional original Yahoo SPY JSON; no download occurs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional NEW JSON output path; existing files are never replaced",
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("--output must be a new file; completed audit evidence is never overwritten")
    root = args.root.resolve()
    report = (
        args.report.resolve() if args.report is not None else root / "reports/financial_research"
    )
    result = audit(root, report, args.raw_json)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
