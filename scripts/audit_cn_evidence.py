"""Audit published numerical CSI300 evidence without reading model pickles."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_cn_benchmark import Source, _canonical_panel_labels, _check_predictions, _clean, _sha

from financial_ts.workflow import _provider_fingerprint


def _daily_index(values, name):
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssertionError(f"{name} contains malformed dates") from exc
    if (
        dates.empty
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or dates.tz is not None
        or not dates.equals(dates.normalize())
    ):
        raise AssertionError(f"{name} requires unique, increasing, finite daily dates")
    return dates


def _finite_values(values, name, *, positive=False):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssertionError(f"{name} must be numerical") from exc
    if not np.isfinite(array).all() or (positive and (array <= 0).any()):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise AssertionError(f"{name} must be {qualifier}")
    return array


def _check_segment_calendar(calendar, segments):
    calendar = _daily_index(calendar, "Provider calendar")
    locations = {}
    for name in ("train", "valid", "test"):
        bounds = segments[name]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise AssertionError(f"{name} requires start and end dates")
        endpoints = [_daily_index([value], f"{name} bound")[0] for value in bounds]
        if endpoints[0] > endpoints[1]:
            raise AssertionError(f"{name} dates are reversed")
        indices = calendar.get_indexer(endpoints)
        if (indices < 0).any():
            raise AssertionError(f"{name} bounds must be observed provider sessions")
        locations[name] = tuple(int(value) for value in indices)
    for left, right in (("train", "valid"), ("valid", "test")):
        if locations[left][1] + 2 >= locations[right][0]:
            raise AssertionError("Label horizon crosses a chronological split")
    # Qlib also needs t+3 as its executor reserve after t+1 execution and
    # the t+2 raw-return label endpoint. The reserve is not a forecast row.
    if locations["test"][1] + 3 >= len(calendar):
        raise AssertionError("Provider lacks three observed sessions after the final test date")
    return locations


def _check_portfolio_report(report, expected_dates, initial, recorded):
    dates = _daily_index(report.index, "Portfolio report")
    if not dates.equals(expected_dates):
        raise AssertionError("Portfolio does not cover all delayed execution dates")
    _finite_values(report[["return", "cost", "bench", "account"]], "Portfolio report values")
    _finite_values(report["account"], "Portfolio report accounts", positive=True)
    initial = float(_finite_values([initial], "Initial account", positive=True)[0])
    nav = initial * (1 + report["return"] - report["cost"]).cumprod()
    _finite_values(nav, "Recomputed portfolio accounts", positive=True)
    nav_errors = ((nav - report["account"]) / report["account"]).abs()
    _finite_values(nav_errors, "Portfolio account reconciliation errors")
    nav_error = float(nav_errors.max())
    if nav_error > 1e-10:
        raise AssertionError("Native account and compounded net returns disagree")
    expected = {
        "net_cumulative_return": float(nav.iloc[-1] / initial - 1),
        "gross_cumulative_return": float((1 + report["return"]).prod() - 1),
        "benchmark_cumulative_return": float((1 + report["bench"]).prod() - 1),
        "sum_daily_cost_ratios": float(report["cost"].sum()),
    }
    _finite_values(list(expected.values()), "Recomputed portfolio metrics")
    published = _finite_values([recorded[key] for key in expected], "Published portfolio metrics")
    if any(abs(actual - value) > 1e-11 for actual, value in zip(published, expected.values())):
        raise AssertionError("Published portfolio summary differs from saved daily evidence")
    return expected, nav_error


def check_numeric_positions(path, report, source):
    with gzip.open(path, "rt") as stream:
        document = json.load(stream)
    snapshots = document["positions"]
    dates = _daily_index([row["datetime"] for row in snapshots], "Numerical positions")
    report_dates = _daily_index(report.index, "Portfolio report")
    _finite_values(report["account"], "Portfolio report accounts", positive=True)
    if not dates.equals(report_dates):
        raise AssertionError("Numerical positions and portfolio dates disagree")
    max_account_error, max_valuation_error, max_weight_error = 0.0, 0.0, 0.0
    ended = {}
    for row, date in zip(snapshots, dates):
        account, cash = _finite_values([row["account"], row["cash"]], "Position account and cash")
        if account <= 0:
            raise AssertionError("Position account must be strictly positive")
        holdings = row["holdings"]
        if len({stock["instrument"] for stock in holdings}) != len(holdings):
            raise AssertionError("Duplicate instrument in numerical positions")
        value = cash
        for stock in holdings:
            symbol = stock["instrument"]
            quantity, price, weight = _finite_values(
                [stock["amount"], stock["price"], stock["weight"]], "Numerical holding"
            )
            if quantity < 0 or price <= 0:
                raise AssertionError("Invalid long-only numerical holding")
            value += quantity * price
            max_weight_error = max(max_weight_error, abs(weight - quantity * price / account))
            source.close(symbol)
            last = source.last_dates[symbol]
            if last is not None and date > last:
                ended.setdefault(symbol, []).append((date, float(weight), float(price)))
        max_account_error = max(max_account_error, abs(account / report.loc[date, "account"] - 1))
        max_valuation_error = max(max_valuation_error, abs(value / account - 1))
    errors = _finite_values(
        [max_account_error, max_valuation_error, max_weight_error],
        "Position reconciliation errors",
    )
    if errors.max() > 1e-10:
        raise AssertionError("Numerical positions fail independent account/weight reconciliation")
    gaps = []
    for symbol, rows in sorted(ended.items()):
        gaps.append(
            {
                "instrument": symbol,
                "last_observed_close_date": str(source.last_dates[symbol].date()),
                "stale_holding_days": len(rows),
                "first_stale_holding_date": str(rows[0][0].date()),
                "last_stale_holding_date": str(rows[-1][0].date()),
                "first_stale_weight": rows[0][1],
                "maximum_stale_weight": max(row[1] for row in rows),
                "final_stale_weight": rows[-1][1] if rows[-1][0] == dates[-1] else 0.0,
                "marked_price": rows[-1][2],
                "last_price_constant": all(row[2] == rows[0][2] for row in rows),
            }
        )
    return {
        "status": "checked",
        "sha256": _sha(path),
        "days": len(snapshots),
        "maximum_account_relative_error": max_account_error,
        "maximum_valuation_relative_error": max_valuation_error,
        "maximum_weight_absolute_error": max_weight_error,
        "terminated_series_holdings": gaps,
        "total_final_stale_weight": sum(row["final_stale_weight"] for row in gaps),
    }


def audit_evidence(root: Path, provider: Path, require_complete=True):
    protocol = json.loads((root / "protocol.json").read_text())
    fingerprint = _provider_fingerprint(provider)
    if protocol["provider_fingerprint"] != fingerprint:
        raise AssertionError("Published protocol and independently read provider differ")
    source = Source(provider)
    results, failures, pending, references = {}, [], [], {}
    segments = protocol["segments"]
    locations = _check_segment_calendar(source.calendar, segments)
    for name in protocol["models"]:
        directory = root / name
        if not (directory / "metrics.json").is_file():
            pending.append(name)
            continue
        try:
            metadata = json.loads((directory / "run.json").read_text())
            if metadata["status"] != "completed" or metadata["provider"] != fingerprint:
                raise AssertionError(
                    "Published model run is incomplete or uses different source bytes"
                )
            metrics = json.loads((directory / "metrics.json").read_text())
            item = {}
            for period, filename, values in (
                ("valid", "validation_predictions.csv.gz", metrics["validation"]),
                ("test", "predictions.csv.gz", metrics),
            ):
                checked, index, labels = _check_predictions(
                    directory / filename, values, source, segments[period]
                )
                # Native Qlib labels are float32. CSV representations may spell
                # those same values as short round-trip decimals or full doubles.
                # DatasetH and TSDatasetH may export different row orderings.
                canonical = _canonical_panel_labels(index, labels)
                if period in references:
                    if not canonical.equals(references[period]):
                        raise AssertionError(
                            "Model forecast indices or underlying float32 labels differ"
                        )
                else:
                    references[period] = canonical
                item[period] = checked
            report_path = directory / "portfolio.csv"
            report = pd.read_csv(
                report_path, index_col=0, parse_dates=True, float_precision="round_trip"
            )
            first, last = locations["test"]
            initial = protocol["configs"][name]["backtest"]["account"]
            expected, nav_error = _check_portfolio_report(
                report, source.calendar[first + 1 : last + 3], initial, metrics["portfolio"]
            )
            if metrics["portfolio"]["trade_signal"]["order"] != [
                "raw score descending",
                "instrument ascending",
            ]:
                raise AssertionError("Portfolio lacks the declared deterministic tie convention")
            positions = check_numeric_positions(directory / "positions.json.gz", report, source)
            item["portfolio"] = {
                "sha256": _sha(report_path),
                "days": len(report),
                "maximum_NAV_relative_error": nav_error,
                **expected,
                "positions": positions,
            }
            if (directory / "assembly.json").exists():
                assembly = json.loads((directory / "assembly.json").read_text())
                identities = [shard["identity"] for shard in assembly["shards"]]
                if not identities or any(
                    identity != assembly["identity"] for identity in identities
                ):
                    raise AssertionError("Official shard states or source identities differ")
                item["parallel_inference"] = {
                    "shards": len(identities),
                    "trained_state_sha256": identities[0]["trained_state_sha256"],
                    "exact_state_agreement": True,
                    "all_declared_rows_present": True,
                }
            results[name] = item
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
    return _clean(
        {
            "status": "passed"
            if not failures and (not require_complete or not pending)
            else "failed",
            "audited_models": list(results),
            "pending_models": pending,
            "failures": failures,
            "provider_fingerprint": fingerprint,
            "models": results,
            "scope": "Raw binary labels and membership, common complete forecast panels, recomputed metrics, chronological horizons, compounded NAV, numeric position valuation, exact official shard-state agreement, and terminated-series exposure. No pickle loading.",
            "qualification": "Arithmetic and provenance checks do not resolve source corporate-action settlement. Portfolio returns remain native Qlib simulations.",
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    result = audit_evidence(args.evidence, args.provider, not args.allow_pending)
    (args.evidence / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "audited_models", "pending_models", "failures")
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
