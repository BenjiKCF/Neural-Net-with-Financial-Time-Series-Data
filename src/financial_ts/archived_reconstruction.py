"""Recover an archived Qlib simulation's execution chain without inventing actions.

The native executor supplies the reproduced orders. An independent numerical
ledger must reconcile every archived daily share amount, cash balance, cost,
turnover and NAV before this is reported as a reproduction. Contemporaneous
quote factors explain raw-unit ambiguities; they are not broker allocations or
evidence that a corporate-action-aware raw account has been reconstructed.
"""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import shutil
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _read_json(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def _write_gzip_json(path, value):
    with Path(path).open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
            zipped.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _number(value, label, *, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return float(value)


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Sessions must be canonical ISO dates")
    return value


def _holdings(snapshot):
    result = {}
    for item in snapshot["holdings"]:
        symbol = item["instrument"]
        if not isinstance(symbol, str) or not symbol or symbol in result:
            raise ValueError("Duplicate or invalid instrument in daily holdings")
        result[symbol] = _number(item["amount"], "adjusted amount")
        _number(item["price"], "adjusted mark")
    return result


def reconcile_execution_chain(orders, positions, portfolio, initial_cash):
    """Independently replay recorded native fills against every archived session.

    All money comparisons use a fixed one-cent absolute tolerance, quantities
    use 1e-7 adjusted units. A missing order, excess sale, missing/duplicate date
    or non-finite value fails rather than being balanced with an invented flow.
    No corporate action, cash dividend, tax attribution or deposit is inferred.
    """
    money_tolerance, quantity_tolerance = 0.01, 1e-7
    cash = _number(initial_cash, "initial cash")
    snapshots = positions["positions"]
    if not snapshots or positions.get("quantity_units") != (
        "Qlib adjusted share amounts; original shares require the source factor"
    ):
        raise ValueError("Nonempty native adjusted-share positions are required")
    dates = [_day(row["datetime"]) for row in snapshots]
    if dates != sorted(set(dates)):
        raise ValueError("Position sessions must be unique and chronological")
    portfolio_dates = [_day(row["datetime"]) for row in portfolio]
    if portfolio_dates != dates:
        raise ValueError("Portfolio and position sessions differ")
    grouped = defaultdict(list)
    previous_date = None
    for index, order in enumerate(orders):
        session = _day(order["session"])
        if order.get("order_index") != index or session not in dates or (previous_date and session < previous_date):
            raise ValueError("Order identity, coverage or chronology is invalid")
        if order.get("side") not in {"buy", "sell"} or not isinstance(order.get("instrument"), str) or not order["instrument"]:
            raise ValueError("Order side/instrument is invalid")
        quantity = _number(order["adjusted_quantity"], "executed quantity")
        value = _number(order["trade_value"], "trade value")
        fee = _number(order["cost"], "trade cost")
        if quantity:
            price = _number(order["adjusted_price"], "executed price")
            factor = _number(order["factor"], "contemporaneous factor")
            if price <= 0 or factor <= 0:
                raise ValueError("Executed price and factor must be positive")
            raw_quantity = _number(order["raw_quote_quantity"], "raw quote quantity")
            raw_price = _number(order["raw_quote_price"], "raw quote price")
            if (abs(value - quantity * price) > money_tolerance
                    or abs(raw_quantity - quantity * factor) > quantity_tolerance
                    or abs(raw_price - price / factor) > 1e-10 * max(1.0, raw_price)):
                raise ValueError("Order price/value/factor identities do not reconcile")
        elif value or fee:
            raise ValueError("Unfilled order cannot move cash")
        grouped[session].append(order)
        previous_date = session
    holdings, cumulative_cost, cumulative_turnover, rows = {}, 0.0, 0.0, []
    for snapshot, report in zip(snapshots, portfolio, strict=True):
        session = snapshot["datetime"]
        for order in grouped[session]:
            quantity, value, fee = order["adjusted_quantity"], order["trade_value"], order["cost"]
            if not quantity:
                continue
            instrument, side = order["instrument"], order["side"]
            sign = 1 if side == "buy" else -1
            holdings[instrument] = holdings.get(instrument, 0.0) + sign * quantity
            if holdings[instrument] < -quantity_tolerance:
                raise ValueError("Reproduced sale exceeds the prior adjusted holding")
            if abs(holdings[instrument]) <= quantity_tolerance:
                holdings.pop(instrument)
            cash -= sign * value + fee
            cumulative_cost += fee
            cumulative_turnover += value
        expected = _holdings(snapshot)
        errors = [abs(holdings.get(symbol, 0.0) - expected.get(symbol, 0.0))
                  for symbol in set(holdings) | set(expected)]
        quantity_error = max(errors, default=0.0)
        reconstructed_value = sum(holdings.get(item["instrument"], 0.0) * item["price"] for item in snapshot["holdings"])
        nav = cash + reconstructed_value
        cash_error = max(abs(cash - _number(snapshot["cash"], "snapshot cash")),
                         abs(cash - _number(float(report["cash"]), "portfolio cash")))
        nav_error = max(abs(nav - _number(snapshot["account"], "snapshot NAV")),
                        abs(nav - _number(float(report["account"]), "portfolio NAV")))
        cost_error = abs(cumulative_cost - _number(float(report["total_cost"]), "cumulative costs"))
        turnover_error = abs(cumulative_turnover - _number(float(report["total_turnover"]), "cumulative turnover"))
        passed = quantity_error <= quantity_tolerance and max(cash_error, nav_error, cost_error, turnover_error) <= money_tolerance
        rows.append({"datetime": session, "orders": len(grouped[session]),
                     "filled_orders": sum(order["adjusted_quantity"] > 0 for order in grouped[session]),
                     "reconstructed_cash": cash, "reconstructed_nav": nav,
                     "maximum_adjusted_quantity_error": quantity_error,
                     "cash_error": cash_error, "nav_error": nav_error,
                     "cumulative_cost_error": cost_error, "cumulative_turnover_error": turnover_error,
                     "passed": passed})
    return {"all_sessions_reconciled": all(row["passed"] for row in rows),
            "session_count": len(rows), "order_count": len(orders),
            "filled_order_count": sum(order["adjusted_quantity"] > 0 for order in orders),
            "money_absolute_tolerance": money_tolerance, "adjusted_quantity_absolute_tolerance": quantity_tolerance,
            "maximum_cash_error": max(row["cash_error"] for row in rows),
            "maximum_nav_error": max(row["nav_error"] for row in rows),
            "maximum_adjusted_quantity_error": max(row["maximum_adjusted_quantity_error"] for row in rows),
            "cumulative_cost": cumulative_cost, "cumulative_turnover": cumulative_turnover,
            "rows": rows}


def event_execution_lineage(orders, positions, lifecycle_assessment):
    """Identify the native fills behind each qualified terminated position.

    Adding raw quote units across different adjustment factors is diagnostic
    only. Their discrepancy from a later factor-converted holding is retained;
    it never becomes a guessed dividend, split or registrar allocation.
    """
    snapshots = {row["datetime"]: row for row in positions["positions"]}
    result = []
    for event in lifecycle_assessment.get("events", []):
        symbol, session = event["instrument"], event["pre_event_snapshot"]
        if session not in snapshots:
            raise ValueError("Lifecycle snapshot missing from archived position history")
        expected = _holdings(snapshots[session]).get(symbol, 0.0)
        if abs(expected - event["adjusted_quantity"]) > 1e-7:
            raise ValueError("Lifecycle adjusted holding differs from the archived snapshot")
        selected = [order for order in orders if order["instrument"] == symbol
                    and order["session"] <= session and order["adjusted_quantity"] > 0]
        adjusted = sum((1 if row["side"] == "buy" else -1) * row["adjusted_quantity"] for row in selected)
        raw_sum = sum((1 if row["side"] == "buy" else -1) * row["raw_quote_quantity"] for row in selected)
        if abs(adjusted - expected) > 1e-7:
            raise ValueError("Event execution lineage does not reconcile adjusted holdings")
        result.append({"event_id": event["event"]["event_id"], "instrument": symbol,
                       "successor": event["event"].get("successor"), "pre_event_snapshot": session,
                       "executed_order_indices": [row["order_index"] for row in selected],
                       "executed_order_count": len(selected), "reconstructed_adjusted_quantity": adjusted,
                       "snapshot_adjusted_quantity": expected,
                       "signed_sum_of_contemporaneous_raw_quote_units": raw_sum,
                       "later_factor_implied_raw_quantity": event["raw_quantity"],
                       "raw_quote_unit_difference": event["raw_quantity"] - raw_sum,
                       "distinct_execution_factors": sorted({row["factor"] for row in selected}),
                       "raw_share_account_reconciled": False,
                       "remaining_blockers": list(event["unresolved_reasons"]),
                       "qualification": "Raw quote units across adjustment factors are not an action-aware raw account. "
                       "No issuer cashflow, tax or account allocation has been inferred."})
    return {"scope": "reproduced_simulation_execution_lineage", "events": result,
            "archived_backtest_repaired": False,
            "affected_final_weight": lifecycle_assessment.get("affected_final_weight")}


def _portfolio_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _native_parity(archived, reproduced):
    """Compare all numerical daily position fields, including marks and weights."""
    if len(archived["positions"]) != len(reproduced["positions"]):
        return {"passed": False, "reason": "session_count_differs"}
    maximum = 0.0
    for before, after in zip(archived["positions"], reproduced["positions"], strict=True):
        if before["datetime"] != after["datetime"]:
            return {"passed": False, "reason": "session_identity_differs"}
        left, right = {x["instrument"]: x for x in before["holdings"]}, {x["instrument"]: x for x in after["holdings"]}
        if left.keys() != right.keys():
            return {"passed": False, "reason": "holding_identity_differs"}
        for key in ("cash", "account"):
            difference = abs(_number(before[key], key) - _number(after[key], key))
            maximum = max(maximum, difference)
            if difference > 0.01:
                return {"passed": False, "reason": f"{key}_differs", "datetime": before["datetime"]}
        for symbol in left:
            if left[symbol].keys() != right[symbol].keys():
                return {"passed": False, "reason": "holding_fields_differ"}
            for key in left[symbol].keys() - {"instrument"}:
                difference = abs(_number(left[symbol][key], key) - _number(right[symbol][key], key))
                maximum = max(maximum, difference)
                if difference > 1e-7:
                    return {"passed": False, "reason": f"holding_{key}_differs", "datetime": before["datetime"]}
    return {"passed": True, "maximum_numeric_difference": maximum, "sessions": len(archived["positions"])}


def _portfolio_parity(archived, reproduced):
    if len(archived) != len(reproduced):
        return {"passed": False, "reason": "session_count_differs"}
    money = {"account", "total_turnover", "total_cost", "value", "cash"}
    maxima = {}
    for left, right in zip(archived, reproduced, strict=True):
        if left.keys() != right.keys() or left["datetime"] != right["datetime"]:
            return {"passed": False, "reason": "portfolio_schema_or_sessions_differ"}
        for key in left.keys() - {"datetime"}:
            a = _number(float(left[key]), key, nonnegative=False)
            b = _number(float(right[key]), key, nonnegative=False)
            maxima[key] = max(maxima.get(key, 0.0), abs(a - b))
            if abs(a - b) > (0.01 if key in money else 1e-10):
                return {"passed": False, "reason": f"{key}_differs", "datetime": left["datetime"]}
    return {"passed": True, "sessions": len(archived), "maximum_differences": maxima}


def reconstruct_archived_execution(run_dir, provider, lifecycle_assessment, output):
    """Reproduce the archived native simulation, capture fills, and retain proof."""
    import numpy as np
    import pandas as pd
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.data import D
    from qlib.utils import fill_placeholder

    from financial_ts.cn_portfolio import numeric_positions
    from financial_ts.workflow import _provider_fingerprint

    implementation_sha256 = _sha(__file__)
    run_dir, provider, lifecycle_assessment, output = [Path(x).resolve() for x in (run_dir, provider, lifecycle_assessment, output)]
    if any(output == source or source in output.parents or output in source.parents for source in (run_dir, provider)):
        raise ValueError("Reconstruction output must be separate from source and provider")
    if output.exists():
        raise ValueError("Choose a new reconstruction output directory")
    original = _read_json(run_dir / "replay_manifest.json")
    lifecycle = _read_json(lifecycle_assessment)
    if original.get("status") != "completed":
        raise ValueError("A completed native replay manifest is required")
    for name in ("positions.json.gz", "portfolio.csv"):
        if _sha(run_dir / name) != original["outputs"][name]["sha256"]:
            raise ValueError("Archived numerical evidence differs from its replay manifest")
    if lifecycle["source_positions_sha256"] != _sha(run_dir / "positions.json.gz"):
        raise ValueError("Lifecycle assessment belongs to different archived positions")
    fingerprint = _provider_fingerprint(provider)
    if fingerprint != original["provider"] or fingerprint != lifecycle["provider_fingerprint"]:
        raise ValueError("Provider differs from the exact archived data identity")
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.exists():
        predictions_path = run_dir / "predictions.csv.gz"
    if _sha(predictions_path) != original["input_files"][predictions_path.name]["sha256"]:
        raise ValueError("Prediction bytes differ from the archived replay input")
    frame = pd.read_csv(predictions_path, parse_dates=["datetime"], dtype={"instrument": str},
                        keep_default_na=False, na_values={"score": ["", "nan", "NaN"], "label": ["", "nan", "NaN"]},
                        float_precision="round_trip").set_index(["datetime", "instrument"])
    portfolio = copy.deepcopy(original["portfolio_config"])
    native = copy.deepcopy(portfolio)
    native["strategy"] = fill_placeholder(native["strategy"], {"<PRED>": frame[["score"]]})
    qlib.init(provider_uri=str(provider), region="cn", kernels=1, expression_cache=None, dataset_cache=None)
    if native["backtest"]["benchmark"] is None:
        native["backtest"]["benchmark"] = pd.Series(0.0, index=D.calendar(
            start_time=native["backtest"]["start_time"], end_time=native["backtest"]["end_time"]))
    captured = []

    class RecordingExecutor(SimulatorExecutor):
        def _collect_data(self, trade_decision, level=0):
            result, metadata = super()._collect_data(trade_decision, level)
            for order, value, cost, price in result:
                quantity = float(order.deal_amount)
                factor = float(order.factor) if order.factor is not None and np.isfinite(order.factor) else None
                price = float(price) if price is not None and np.isfinite(price) else None
                captured.append({"order_index": len(captured), "session": str(order.start_time.date()),
                                 "instrument": order.stock_id, "side": "buy" if order.direction == Order.BUY else "sell",
                                 "requested_adjusted_quantity": float(order.amount), "adjusted_quantity": quantity,
                                 "adjusted_price": price, "factor": factor,
                                 "raw_quote_quantity": quantity * factor if quantity and factor else None,
                                 "raw_quote_price": price / factor if quantity and price is not None and factor else None,
                                 "trade_value": float(value), "cost": float(cost)})
            return result, metadata

    executor = RecordingExecutor(time_per_step="day", generate_portfolio_metrics=True)
    reports, _ = backtest(executor=executor, strategy=native["strategy"], **native["backtest"])
    report, native_positions = reports["1day"]
    archived = _read_json(run_dir / "positions.json.gz")
    reproduced = numeric_positions(native_positions)
    parity = _native_parity(archived, reproduced)
    native_rows = report.reset_index().to_dict("records")
    for row in native_rows:
        row["datetime"] = str(pd.Timestamp(row["datetime"]).date())
    portfolio_parity = _portfolio_parity(_portfolio_rows(run_dir / "portfolio.csv"), native_rows)
    reconciliation = reconcile_execution_chain(captured, archived, _portfolio_rows(run_dir / "portfolio.csv"), portfolio["backtest"]["account"])
    lineage = event_execution_lineage(captured, archived, lifecycle)
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in ("positions.json.gz", "portfolio.csv", "replay_manifest.json"):
        shutil.copyfile(run_dir / name, inputs / name)
    shutil.copyfile(lifecycle_assessment, inputs / "lifecycle_assessment.json")
    _write_gzip_json(output / "native_positions.json.gz", reproduced)
    _write_gzip_json(output / "orders.json.gz", captured)
    report.to_csv(output / "native_portfolio.csv")
    _json(output / "native_parity.json", parity)
    _json(output / "event_lineage.json", lineage)
    with (output / "daily_reconciliation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reconciliation["rows"][0]))
        writer.writeheader()
        writer.writerows(reconciliation.pop("rows"))
    summary = {"schema_version": 1, "scope": "reproduced_qlib_simulation_execution",
               "status": "reproduced" if parity["passed"] and portfolio_parity["passed"] and reconciliation["all_sessions_reconciled"] else "failed",
               "recorded_at": datetime.now(timezone.utc).isoformat(),
               "native_position_parity": parity, "native_portfolio_parity": portfolio_parity,
               "independent_reconciliation": reconciliation,
               "raw_share_account_reconciled": False, "archived_backtest_repaired": False,
               "production_promoted": False, "orders_submitted": False,
               "affected_final_weight": lifecycle["affected_final_weight"],
               "remaining_requirements": ["complete_sourced_corporate_action_cash_tax_and_successor_history",
                                          "explicit_counterfactual_whole_share_allocations_or_bounds",
                                          "separate_action_aware_strategy_replay"],
               "qualification": "Native simulation orders reproduced from the frozen scores and provider, not observed broker fills. "
               "Contemporaneous factors yield raw quote units, not a reconciled raw-share account. "
               "Historical stale-mark returns and their original qualification remain unchanged."}
    _json(output / "assessment.json", summary)
    if _sha(__file__) != implementation_sha256:
        raise ValueError("Reconstruction implementation changed during execution; preserve this attempt and rerun")
    manifest = {"schema_version": 1, "operation": "capture_and_independently_reconcile_native_simulation",
                "provider": fingerprint, "initial_cash": portfolio["backtest"]["account"],
                "source_inputs": {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
                                  for path in (predictions_path, run_dir / "replay_manifest.json", lifecycle_assessment)},
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": {name: importlib.metadata.version(name) for name in ("pyqlib", "numpy", "pandas")},
                            "source_sha256": {"archived_reconstruction.py": implementation_sha256,
                                              "qlib_executor.py": _sha(inspect.getfile(SimulatorExecutor))}},
                "outputs": {str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
                            for path in sorted(output.rglob("*")) if path.is_file()}}
    _json(output / "manifest.json", manifest)
    return summary


def verify_archived_reconstruction(output):
    """Verify hashes and independently redo the numerical ledger and lineage."""
    output = Path(output).resolve()
    manifest = _read_json(output / "manifest.json")
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} - {"manifest.json"}
    if actual != set(manifest["outputs"]):
        raise ValueError("Reconstruction evidence inventory differs")
    for name, expected in manifest["outputs"].items():
        path = output / name
        if path.resolve().parent != output and output not in path.resolve().parents:
            raise ValueError("Unsafe reconstruction evidence path")
        if path.stat().st_size != expected["bytes"] or _sha(path) != expected["sha256"]:
            raise ValueError("Reconstruction evidence bytes differ")
    positions = _read_json(output / "inputs/positions.json.gz")
    orders = _read_json(output / "orders.json.gz")
    original = _read_json(output / "inputs/replay_manifest.json")
    if manifest["initial_cash"] != original["portfolio_config"]["backtest"]["account"]:
        raise ValueError("Opening cash differs from the archived simulation")
    for name in ("positions.json.gz", "portfolio.csv"):
        if _sha(output / "inputs" / name) != original["outputs"][name]["sha256"]:
            raise ValueError("Retained original evidence differs from original replay manifest")
    summary = _read_json(output / "assessment.json")
    result = reconcile_execution_chain(orders, positions, _portfolio_rows(output / "inputs/portfolio.csv"), manifest["initial_cash"])
    daily_rows = result.pop("rows")
    recorded_rows = _portfolio_rows(output / "daily_reconciliation.csv")
    if len(daily_rows) != len(recorded_rows):
        raise ValueError("Independent daily reconciliation differs")
    for expected, recorded in zip(daily_rows, recorded_rows, strict=True):
        if expected.keys() != recorded.keys():
            raise ValueError("Independent daily reconciliation differs")
        for key, value in expected.items():
            observed = recorded[key]
            if isinstance(value, bool):
                same = observed == str(value)
            elif isinstance(value, (int, float)):
                same = float(observed) == value
            else:
                same = observed == value
            if not same:
                raise ValueError("Independent daily reconciliation differs")
    parity = _native_parity(positions, _read_json(output / "native_positions.json.gz"))
    portfolio_parity = _portfolio_parity(_portfolio_rows(output / "inputs/portfolio.csv"),
                                       _portfolio_rows(output / "native_portfolio.csv"))
    lifecycle = _read_json(output / "inputs/lifecycle_assessment.json")
    if (lifecycle["source_positions_sha256"] != _sha(output / "inputs/positions.json.gz")
            or lifecycle["provider_fingerprint"] != original["provider"] or manifest["provider"] != original["provider"]):
        raise ValueError("Lifecycle or provider identity differs from archived source")
    lineage = event_execution_lineage(orders, positions, lifecycle)
    if (result != summary["independent_reconciliation"] or parity != summary["native_position_parity"]
            or portfolio_parity != summary["native_portfolio_parity"]
            or parity != _read_json(output / "native_parity.json") or lineage != _read_json(output / "event_lineage.json")):
        raise ValueError("Independent reconstruction recomputation differs")
    if (summary["archived_backtest_repaired"] or summary["raw_share_account_reconciled"]
            or summary["production_promoted"] or summary["orders_submitted"]):
        raise ValueError("Simulation reconstruction cannot close a raw-account or deployment gate")
    if summary["status"] != ("reproduced" if result["all_sessions_reconciled"] and parity["passed"] and portfolio_parity["passed"] else "failed"):
        raise ValueError("Reconstruction status differs from numerical evidence")
    return {"status": "verified", "execution_reproduced": summary["status"] == "reproduced",
            "files_verified": len(manifest["outputs"]), "session_count": result["session_count"],
            "filled_order_count": result["filled_order_count"], "archived_backtest_repaired": False}
