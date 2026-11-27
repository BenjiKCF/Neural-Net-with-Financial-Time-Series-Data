"""Execution-chain arithmetic, missing evidence and native archive reproduction."""

import copy
import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from financial_ts.archived_reconstruction import (
    _native_parity,
    event_execution_lineage,
    reconcile_execution_chain,
    reconstruct_archived_execution,
    verify_archived_reconstruction,
)


def order(index, session, side, quantity, price, factor, cost):
    return {"order_index": index, "session": session, "instrument": "A", "side": side,
            "adjusted_quantity": quantity, "adjusted_price": price, "factor": factor,
            "raw_quote_quantity": quantity * factor, "raw_quote_price": price / factor,
            "trade_value": quantity * price, "cost": cost}


@pytest.fixture
def ledger():
    orders = [order(0, "2025-01-02", "buy", 50, 20, 2, 1),
              order(1, "2025-01-03", "sell", 10, 25, 2.2, 2)]
    positions = {"quantity_units": "Qlib adjusted share amounts; original shares require the source factor",
                 "positions": [
                     {"datetime": "2025-01-02", "cash": 999, "account": 1999,
                      "holdings": [{"instrument": "A", "amount": 50, "price": 20}]},
                     {"datetime": "2025-01-03", "cash": 1247, "account": 2247,
                      "holdings": [{"instrument": "A", "amount": 40, "price": 25}]},
                     {"datetime": "2025-01-06", "cash": 1247, "account": 2287,
                      "holdings": [{"instrument": "A", "amount": 40, "price": 26}]},
                 ]}
    portfolio = [{"datetime": row["datetime"], "cash": row["cash"], "account": row["account"],
                  "total_cost": 1 if index == 0 else 3, "total_turnover": 1000 if index == 0 else 1250}
                 for index, row in enumerate(positions["positions"])]
    return orders, positions, portfolio


def test_hand_computed_execution_chain_reconciles_cash_shares_fees_and_nav(ledger):
    result = reconcile_execution_chain(*ledger, initial_cash=2000)
    assert result["all_sessions_reconciled"]
    assert result["filled_order_count"] == 2 and result["session_count"] == 3
    assert result["maximum_cash_error"] == result["maximum_nav_error"] == 0
    assert result["cumulative_cost"] == 3
    assert result["rows"][-1]["reconstructed_cash"] == 1247
    assert result["rows"][-1]["reconstructed_nav"] == 2287


def test_missing_order_is_not_balanced_using_archived_cash(ledger):
    orders, positions, portfolio = ledger
    result = reconcile_execution_chain(orders[:1], positions, portfolio, 2000)
    assert not result["all_sessions_reconciled"]
    assert result["maximum_adjusted_quantity_error"] == 10
    assert result["maximum_cash_error"] == 248


@pytest.mark.parametrize("change", ["duplicate_order", "foreign_session", "nan_fee", "bad_factor", "bad_side", "over_sell"])
def test_invalid_order_evidence_is_rejected(ledger, change):
    orders, positions, portfolio = copy.deepcopy(ledger)
    if change == "duplicate_order":
        orders[1]["order_index"] = 0
    elif change == "foreign_session":
        orders[1]["session"] = "2025-01-07"
    elif change == "nan_fee":
        orders[0]["cost"] = float("nan")
    elif change == "bad_factor":
        orders[0]["factor"] = 3
    elif change == "bad_side":
        orders[0]["side"] = "deposit"
    else:
        orders[1] = order(1, "2025-01-03", "sell", 100, 25, 2.2, 2)
    with pytest.raises(ValueError):
        reconcile_execution_chain(orders, positions, portfolio, 2000)


def test_daily_evidence_cannot_omit_dates_or_overwrite_holdings(ledger):
    orders, positions, portfolio = copy.deepcopy(ledger)
    with pytest.raises(ValueError, match="sessions differ"):
        reconcile_execution_chain(orders, positions, portfolio[:-1], 2000)
    positions["positions"][0]["holdings"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile_execution_chain(orders, positions, portfolio, 2000)


def test_parity_checks_intermediate_positions_and_marks_not_only_final_nav(ledger):
    positions = ledger[1]
    changed = copy.deepcopy(positions)
    changed["positions"][0]["holdings"][0]["price"] += 1
    assert not _native_parity(positions, changed)["passed"]
    assert _native_parity(positions, positions)["passed"]


def test_factor_change_is_diagnostic_and_does_not_create_a_raw_account(ledger):
    orders, positions, _ = ledger
    event = {"events": [{"event": {"event_id": "merge", "successor": "B"}, "instrument": "A",
                         "pre_event_snapshot": "2025-01-03", "adjusted_quantity": 40,
                         "raw_quantity": 88, "unresolved_reasons": ["actions_unverified"]}],
             "affected_final_weight": 0.05}
    report = event_execution_lineage(orders, positions, event)
    item = report["events"][0]
    assert item["signed_sum_of_contemporaneous_raw_quote_units"] == 78
    assert item["later_factor_implied_raw_quantity"] == 88
    assert item["raw_quote_unit_difference"] == 10
    assert not item["raw_share_account_reconciled"]
    assert not report["archived_backtest_repaired"]
    event["events"][0]["adjusted_quantity"] = 41
    with pytest.raises(ValueError, match="differs"):
        event_execution_lineage(orders, positions, event)


@pytest.mark.integration
def test_native_order_capture_matches_the_archived_account_and_detects_tampering(tmp_path):
    from financial_ts.cn_portfolio import replay_cn_portfolio
    from financial_ts.data import create_demo_csv, import_csv
    from financial_ts.workflow import _provider_fingerprint

    provider = tmp_path / "provider"
    import_csv(create_demo_csv(tmp_path / "prices.csv", n_instruments=4, n_days=100), provider)
    calendar = pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
    dates = calendar[80:88]
    source = tmp_path / "source"
    source.mkdir()
    index = pd.MultiIndex.from_product([dates, ["DEMO000", "DEMO001", "DEMO002", "DEMO003"]],
                                      names=["datetime", "instrument"])
    scores = pd.DataFrame({"score": np.tile([0.2, 0.1, -0.1, -0.2], len(dates)), "label": 0.01}, index=index)
    scores.to_csv(source / "predictions.csv")
    config = {"region": "cn", "segments": {"test": [str(dates[0].date()), str(dates[-1].date())]},
              "backtest": {"topk": 2, "n_drop": 1, "benchmark": None}}
    (source / "resolved_config.json").write_text(json.dumps(config))
    fingerprint = _provider_fingerprint(provider)
    (source / "run.json").write_text(json.dumps({"status": "completed", "provider": fingerprint}))
    (source / "metrics.json").write_text("{}")
    archive = tmp_path / "archived"
    replay_cn_portfolio(source, provider, archive)
    # A published numerical replay retains its prediction input alongside the outputs.
    (archive / "predictions.csv").write_bytes((source / "predictions.csv").read_bytes())
    assessment = tmp_path / "lifecycle.json"
    assessment.write_text(json.dumps({"events": [], "affected_final_weight": 0,
                                      "source_positions_sha256": hashlib.sha256((archive / "positions.json.gz").read_bytes()).hexdigest(),
                                      "provider_fingerprint": fingerprint}))
    output = tmp_path / "reconstruction"
    result = reconstruct_archived_execution(archive, provider, assessment, output)
    assert result["status"] == "reproduced"
    assert result["native_position_parity"]["passed"]
    assert result["independent_reconciliation"]["all_sessions_reconciled"]
    assert not result["archived_backtest_repaired"]
    verification = verify_archived_reconstruction(output)
    assert verification["execution_reproduced"] and verification["session_count"] == 9
    with pytest.raises(ValueError, match="new reconstruction"):
        reconstruct_archived_execution(archive, provider, assessment, output)
    # A rehashed daily diagnostic must still reproduce the ledger, not just its summary.
    daily_path = output / "daily_reconciliation.csv"
    before_daily, before_manifest = daily_path.read_bytes(), (output / "manifest.json").read_bytes()
    with daily_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[-1]["reconstructed_nav"] = str(float(rows[-1]["reconstructed_nav"]) + 1)
    with daily_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads(before_manifest)
    content = daily_path.read_bytes()
    manifest["outputs"]["daily_reconciliation.csv"] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="daily reconciliation"):
        verify_archived_reconstruction(output)
    daily_path.write_bytes(before_daily)
    (output / "manifest.json").write_bytes(before_manifest)
    # Updating a summary and its checksum cannot convert a simulation to real account evidence.
    summary = json.loads((output / "assessment.json").read_text())
    summary["raw_share_account_reconciled"] = True
    (output / "assessment.json").write_text(json.dumps(summary))
    manifest = json.loads((output / "manifest.json").read_text())
    content = (output / "assessment.json").read_bytes()
    manifest["outputs"]["assessment.json"] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cannot close"):
        verify_archived_reconstruction(output)
