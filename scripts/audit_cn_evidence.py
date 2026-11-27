"""Audit published numerical CSI300 evidence without reading model pickles."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_cn_benchmark import Source, _check_predictions, _clean, _sha

from financial_ts.workflow import _provider_fingerprint


def check_numeric_positions(path, report, source):
    with gzip.open(path, "rt") as stream:
        document = json.load(stream)
    snapshots = document["positions"]
    dates = pd.DatetimeIndex(pd.to_datetime([row["datetime"] for row in snapshots]))
    if not dates.equals(report.index):
        raise AssertionError("Numerical positions and portfolio dates disagree")
    max_account_error, max_valuation_error, max_weight_error = 0.0, 0.0, 0.0
    ended = {}
    for row, date in zip(snapshots, dates):
        account, cash = float(row["account"]), float(row["cash"])
        holdings = row["holdings"]
        if len({stock["instrument"] for stock in holdings}) != len(holdings):
            raise AssertionError("Duplicate instrument in numerical positions")
        value = cash
        for stock in holdings:
            symbol, quantity, price = stock["instrument"], stock["amount"], stock["price"]
            if (
                not np.isfinite([quantity, price, stock["weight"]]).all()
                or quantity < 0
                or price <= 0
            ):
                raise AssertionError("Invalid long-only numerical holding")
            value += quantity * price
            max_weight_error = max(
                max_weight_error, abs(stock["weight"] - quantity * price / account)
            )
            source.close(symbol)
            last = source.last_dates[symbol]
            if last is not None and date > last:
                ended.setdefault(symbol, []).append((date, stock["weight"], price))
        max_account_error = max(max_account_error, abs(account / report.loc[date, "account"] - 1))
        max_valuation_error = max(max_valuation_error, abs(value / account - 1))
    if max(max_account_error, max_valuation_error, max_weight_error) > 1e-10:
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
    for left, right in (("train", "valid"), ("valid", "test")):
        if source.calendar.get_loc(pd.Timestamp(segments[left][1])) + 2 >= source.calendar.get_loc(
            pd.Timestamp(segments[right][0])
        ):
            raise AssertionError("Label horizon crosses a chronological split")
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
                canonical = labels.astype(np.float32)
                if period in references:
                    ref_index, ref_labels = references[period]
                    if not index.equals(ref_index) or not np.array_equal(
                        canonical, ref_labels, equal_nan=True
                    ):
                        raise AssertionError(
                            "Model forecast indices or underlying float32 labels differ"
                        )
                else:
                    references[period] = (index, canonical)
                item[period] = checked
            report_path = directory / "portfolio.csv"
            report = pd.read_csv(
                report_path, index_col=0, parse_dates=True, float_precision="round_trip"
            )
            first = source.calendar.get_loc(pd.Timestamp(segments["test"][0]))
            last = source.calendar.get_loc(pd.Timestamp(segments["test"][1]))
            if not report.index.equals(source.calendar[first + 1 : last + 3]):
                raise AssertionError("Portfolio does not cover all delayed execution dates")
            initial = protocol["configs"][name]["backtest"]["account"]
            nav = initial * (1 + report["return"] - report["cost"]).cumprod()
            nav_error = float(((nav - report["account"]) / report["account"]).abs().max())
            if nav_error > 1e-10:
                raise AssertionError("Native account and compounded net returns disagree")
            expected = {
                "net_cumulative_return": float(nav.iloc[-1] / initial - 1),
                "gross_cumulative_return": float((1 + report["return"]).prod() - 1),
                "benchmark_cumulative_return": float((1 + report["bench"]).prod() - 1),
                "sum_daily_cost_ratios": float(report["cost"].sum()),
            }
            if any(
                abs(metrics["portfolio"][key] - value) > 1e-11 for key, value in expected.items()
            ):
                raise AssertionError(
                    "Published portfolio summary differs from saved daily evidence"
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
