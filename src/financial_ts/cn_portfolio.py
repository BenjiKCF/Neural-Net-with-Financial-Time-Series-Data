"""Deterministic trading ranks and native Qlib portfolio replay.

Only the signal delivered to TopkDropout is transformed. Forecast files,
labels, RMSE and IC always retain the model's original numerical predictions.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.backtest.signal import SignalWCache

TRADE_TIE_CONVENTION = {
    "order": ["raw score descending", "instrument ascending"],
    "scope": "Each prediction date; trading input only",
    "encoding": "Unique negative ordinal ranks: best=-1, second=-2, and so on",
    "strategy": "Native qlib.contrib.strategy.TopkDropoutStrategy",
    "forecast_metrics": "Original raw scores; no perturbation or ranking for RMSE/IC",
}


def deterministic_trade_signal(predictions: pd.Series | pd.DataFrame) -> pd.Series:
    """Return unique daily scores, preserving all strict raw-score comparisons.

    Native TopkDropout uses scores exclusively to sort candidates and holdings.
    A unique ordinal removes its dependence on pandas' unstable ordering of
    exact ties, including internal re-sorts after the held/new sets are joined.
    """
    scores = predictions.iloc[:, 0] if isinstance(predictions, pd.DataFrame) else predictions
    if not isinstance(scores, pd.Series) or not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("Trading scores require a datetime/instrument MultiIndex.")
    if set(scores.index.names) != {"datetime", "instrument"} or scores.index.nlevels != 2:
        raise ValueError("Trading score index levels must be datetime and instrument.")
    if scores.empty or not scores.index.is_unique:
        raise ValueError("Trading scores must be nonempty and unique per date/instrument.")
    scores = scores.reorder_levels(["datetime", "instrument"])
    dates = pd.DatetimeIndex(scores.index.get_level_values("datetime"))
    instruments = scores.index.get_level_values("instrument")
    if dates.hasnans or dates.tz is not None or not (dates == dates.normalize()).all():
        raise ValueError("Trading score dates must be finite, timezone-naive daily dates.")
    if any(not isinstance(value, str) or not value for value in instruments):
        raise ValueError("Trading instruments must be nonempty strings.")
    values = scores.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Trading scores must be finite.")
    ordered = pd.DataFrame(
        {"datetime": dates, "instrument": instruments, "score": values}
    ).sort_values(["datetime", "score", "instrument"], ascending=[True, False, True])
    ordered["trade_score"] = -(ordered.groupby("datetime", sort=False).cumcount() + 1)
    result = ordered.set_index(["datetime", "instrument"])["trade_score"].astype(np.float64)
    return result.sort_index()


class DeterministicRankSignal(SignalWCache):
    """A native Qlib signal wrapper for deterministic TopkDropout ties."""

    def __init__(self, signal):
        super().__init__(deterministic_trade_signal(signal))


def portfolio_metrics(report: pd.DataFrame, portfolio: dict, region: str) -> dict:
    """Shared native-report summaries for fitted runs and portfolio-only replay."""
    if not {"return", "cost", "bench"}.issubset(report.columns):
        raise ValueError("Backtest report is missing return, cost, or benchmark columns.")
    if report.empty or not np.isfinite(report[["return", "cost", "bench"]].to_numpy()).all():
        raise ValueError("Backtest report is empty or contains nonfinite returns/costs.")
    backtest = portfolio["backtest"]
    return {
        "gross_cumulative_return": (1 + report["return"]).prod() - 1,
        "net_cumulative_return": (1 + report["return"] - report["cost"]).prod() - 1,
        "benchmark_cumulative_return": (1 + report["bench"]).prod() - 1,
        "sum_daily_cost_ratios": report["cost"].sum(),
        "benchmark": backtest["benchmark"] or "cash",
        "assumptions": (
            "next-close execution; Qlib CN 100-share lots and 9.5% price-limit proxy "
            "unless explicitly overridden; suspensions blocked; explicit fees; "
            "no liquidity/slippage model; final positions marked to market"
            if region == "cn"
            else "next-close execution; explicit fees; fractional shares unless explicitly "
            "overridden; no liquidity/slippage model; final positions marked to market"
        ),
        "exchange_kwargs": backtest["exchange_kwargs"],
        "trade_signal": copy.deepcopy(TRADE_TIE_CONVENTION) if region == "cn" else "raw scores",
    }


def numeric_positions(positions: dict) -> dict:
    """Convert trusted in-memory Qlib positions to inspectable numerical JSON."""
    rows = []
    for date, obj in sorted(positions.items()):
        position = obj.position
        stocks = [
            {
                "instrument": instrument,
                **{
                    key: float(value[key])
                    for key in ["amount", "price", "weight", "count_day"]
                    if key in value
                },
            }
            for instrument, value in sorted(position.items())
            if isinstance(value, dict)
        ]
        rows.append(
            {
                "datetime": str(pd.Timestamp(date).date()),
                "cash": float(position["cash"]),
                "account": float(position["now_account_value"]),
                "holdings": stocks,
            }
        )
    return {
        "origin": "Native Qlib portfolio replay converted from in-memory positions",
        "quantity_units": "Qlib adjusted share amounts; original shares require the source factor",
        "positions": rows,
    }


def terminated_holdings_audit(positions: dict, provider: Path) -> dict:
    """Quantify positions held beyond a symbol's last observed source close."""
    calendar = pd.DatetimeIndex(
        pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
    )
    snapshots = numeric_positions(positions)["positions"]
    symbols = sorted({stock["instrument"] for row in snapshots for stock in row["holdings"]})
    final = pd.Timestamp(snapshots[-1]["datetime"])
    details = []
    for instrument in symbols:
        raw = np.fromfile(provider / "features" / instrument.lower() / "close.day.bin", dtype="<f4")
        offset = int(raw[0])
        valid = np.flatnonzero(np.isfinite(raw[1 : len(calendar) - offset + 1]))
        if not len(valid):
            continue
        end = calendar[offset + valid[-1]]
        if end >= final:
            continue
        held = [
            (pd.Timestamp(row["datetime"]), stock["weight"], stock["price"])
            for row in snapshots
            for stock in row["holdings"]
            if row["datetime"] > str(end.date()) and stock["instrument"] == instrument
        ]
        if held:
            details.append(
                {
                    "instrument": instrument,
                    "last_observed_close_date": str(end.date()),
                    "stale_holding_days": len(held),
                    "first_stale_holding_date": str(held[0][0].date()),
                    "last_stale_holding_date": str(held[-1][0].date()),
                    "first_stale_weight": held[0][1],
                    "maximum_stale_weight": max(row[1] for row in held),
                    "final_stale_weight": held[-1][1] if held[-1][0] == final else 0.0,
                    "marked_price": held[-1][2],
                    "last_price_constant": all(row[2] == held[0][2] for row in held),
                }
            )
    return {
        "status": "checked",
        "terminated_series_holdings": details,
        "total_final_stale_weight": sum(row["final_stale_weight"] for row in details),
        "interpretation": "Terminated-series corporate-action settlement remains unresolved. "
        "Deterministic tie handling changes trading reproducibility, not source valuation. "
        "Returns are Qlib simulations, not executable realized P&L.",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_cn_portfolio(run_dir, provider, output) -> dict:
    """Replay CSV/CSV.gz scores without loading a model or handler pickle.

    A fresh output directory preserves the original results. The recorded
    provider fingerprint must match the supplied local provider exactly.
    """
    import qlib
    from qlib.backtest import backtest
    from qlib.contrib.evaluate import indicator_analysis, risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data import D
    from qlib.utils import fill_placeholder, flatten_dict

    from financial_ts.workflow import (
        _portfolio_config,
        _provider_fingerprint,
        _read_calendar,
        _save_json,
    )

    run_dir, provider, output = (
        Path(p).expanduser().resolve() for p in (run_dir, provider, output)
    )
    if (
        output == run_dir
        or output == provider
        or output in run_dir.parents
        or provider in output.parents
    ):
        raise ValueError("Replay output must be separate from source files and the provider.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Replay output is not empty; choose a fresh output directory.")
    config_path, run_path = run_dir / "resolved_config.json", run_dir / "run.json"
    metrics_path = run_dir / "metrics.json"
    config, run, metrics = (
        json.loads(path.read_text()) for path in (config_path, run_path, metrics_path)
    )
    if config.get("region") != "cn" or run.get("status") not in {
        "completed",
        "forecasts_completed",
    }:
        raise ValueError("Replay requires a completed CN forecast run.")
    fingerprint = _provider_fingerprint(provider)
    if run.get("provider") != fingerprint:
        raise ValueError("Replay provider bytes differ from the original forecast run.")
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.is_file():
        predictions_path = run_dir / "predictions.csv.gz"
    predictions = pd.read_csv(
        predictions_path,
        parse_dates=["datetime"],
        dtype={"instrument": str},
        keep_default_na=False,
        na_values={"score": ["", "NaN", "nan"], "label": ["", "NaN", "nan"]},
        float_precision="round_trip",
    ).set_index(["datetime", "instrument"])
    scores = predictions[["score"]]
    deterministic_trade_signal(scores)  # Validate before starting native execution.
    calendar = _read_calendar(provider)
    segments = config["segments"]
    dates = scores.index.get_level_values("datetime")
    if dates.min() != pd.Timestamp(segments["test"][0]) or dates.max() != pd.Timestamp(
        segments["test"][1]
    ):
        raise ValueError("Replay prediction coverage differs from the declared test segment.")
    portfolio = _portfolio_config(
        config, segments, calendar, scores.index.get_level_values("instrument").nunique()
    )
    qlib.init(
        provider_uri=str(provider),
        region="cn",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    native = copy.deepcopy(portfolio)
    native["strategy"] = fill_placeholder(native["strategy"], {"<PRED>": scores})
    if native["backtest"]["benchmark"] is None:
        benchmark_dates = D.calendar(
            start_time=native["backtest"]["start_time"], end_time=native["backtest"]["end_time"]
        )
        native["backtest"]["benchmark"] = pd.Series(0.0, index=benchmark_dates)
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    reports, indicators = backtest(
        executor=executor, strategy=native["strategy"], **native["backtest"]
    )
    report, positions = reports["1day"]
    metrics["portfolio"] = portfolio_metrics(report, portfolio, "cn")
    risk = {
        "excess_return_without_cost": risk_analysis(report["return"] - report["bench"], freq="day"),
        "excess_return_with_cost": risk_analysis(
            report["return"] - report["bench"] - report["cost"], freq="day"
        ),
    }
    risk_frame = pd.concat(risk)
    native_metrics = flatten_dict(risk_frame["risk"].unstack().T.to_dict())
    native_metrics.update(indicator_analysis(indicators["1day"][0])["value"].to_dict())
    # Keep forecast-record metrics and replace every original daily portfolio metric.
    metrics["qlib_metrics"] = {
        k: v for k, v in metrics.get("qlib_metrics", {}).items() if not k.startswith("1day.")
    }
    metrics["qlib_metrics"].update({f"1day.{key}": value for key, value in native_metrics.items()})
    output.mkdir(parents=True, exist_ok=True)
    report.to_csv(output / "portfolio.csv")
    risk_frame.to_csv(output / "native_risk.csv")
    _save_json(output / "metrics.json", metrics)
    with (output / "positions.json.gz").open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            compressed.write(
                json.dumps(
                    numeric_positions(positions), allow_nan=False, separators=(",", ":")
                ).encode()
            )
    _save_json(output / "source_gap_holdings.json", terminated_holdings_audit(positions, provider))
    groups = predictions.reset_index().groupby(["datetime", "score"], sort=False).size()
    tied = groups[groups > 1]
    source_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=source_root, text=True
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "operation": "Native Qlib portfolio-only replay; no fitting or forecast modification",
        "trade_signal": TRADE_TIE_CONVENTION,
        "source_run": str(run_dir),
        "original_run": run,
        "input_files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in [predictions_path, config_path, run_path, metrics_path]
        },
        "provider": fingerprint,
        "portfolio_config": portfolio,
        "exact_score_ties": {
            "dates": int(tied.index.get_level_values("datetime").nunique()),
            "rows": int(tied.sum()),
            "largest_group": int(tied.max()) if len(tied) else 0,
        },
        "forecast_values": "CSV floats parsed with round_trip; original prediction and forecast-metric files retained",
        "native_risk_convention": "Qlib additive daily-return risk_analysis; its installed daily frequency uses 238 sessions. Portfolio cumulative returns are compounded.",
        "replay_source": {
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name) for name in ["pyqlib", "pandas", "numpy"]
            },
            "files": {
                path.name: _sha256(path)
                for path in [
                    Path(__file__),
                    Path(__file__).with_name("workflow.py"),
                    Path(inspect.getfile(TopkDropoutStrategy)),
                ]
            },
        },
        "outputs": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    _save_json(output / "replay_manifest.json", manifest)
    return manifest
