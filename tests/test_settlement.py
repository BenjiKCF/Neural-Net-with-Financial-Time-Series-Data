"""Independent account statements, real-record requirements and late allocations."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from financial_ts.lifecycle import CorporateAction, EventLedger
from financial_ts.settlement import (
    reconcile_account_export,
    run_account_settlement,
    verify_account_settlement,
)

ACCOUNT = "test-account-001"
ASOF = "2025-01-08T18:00:00+08:00"
SOURCE = "https://example.org/synthetic-test-evidence"
SOURCES = {name: {"kind": kind} for name, kind in
           (("account", "broker"), ("issuer", "issuer"), ("market", "market_data"))}


def provenance(identifier, at, *, source="account", account=True):
    result = {"record_id": identifier, "source_id": source, "available_at": at}
    if account:
        result.update(account_id=ACCOUNT, currency="CNY")
    return result


def export_fixture():
    # These are synthetic test data, regardless of the format discriminator
    # used to exercise the actual-account success path. No test is deployment
    # evidence and no test source is used by the retained archived report.
    open_at, close_at = "2025-01-02T09:00:00+08:00", "2025-01-07T16:00:00+08:00"
    fill_at = "2025-01-02T10:00:00+08:00"
    event_at = "2025-01-03T09:00:00+08:00"
    allocation_at = "2025-01-07T09:00:00+08:00"
    cash_at, sell_at = "2025-01-07T10:00:00+08:00", "2025-01-07T15:00:00+08:00"
    event = {"event_id": "merger", "kind": "merger", "instrument": "A", "successor": "B",
             "effective_at": event_at, "known_at": "2025-01-02T18:00:00+08:00", "source": SOURCE,
             "ratio": 0.62, "cash_per_share": 0.8, "payment_at": "2025-01-06T09:00:00+08:00",
             "tradable_at": "2025-01-06T09:00:00+08:00"}
    operations = [
        {**provenance("buy", fill_at), "kind": "fill", "effective_at": fill_at,
         "instrument": "A", "side": "buy", "quantity": 100, "price": 10, "fees": 2, "tax": 0},
        {**provenance("action", event["known_at"], source="issuer"), "kind": "action",
         "effective_at": event_at, "event": event, "entitled_quantity": 101},
        {**provenance("allocation", allocation_at), "kind": "allocation", "effective_at": allocation_at,
         "event_id": "merger", "entitled_quantity": 101, "allocated_quantity": 63, "successor": "B",
         "tradable_at": event["tradable_at"], "tradability_source_id": "issuer"},
        {**provenance("cash", cash_at), "kind": "cash_credit", "effective_at": cash_at,
         "event_id": "merger", "gross_amount": 101, "tax_withheld": "20.2", "net_amount": "80.8"},
        {**provenance("sell", sell_at), "kind": "fill", "effective_at": sell_at,
         "instrument": "B", "side": "sell", "quantity": 3, "price": 10, "fees": 1, "tax": "0.03"},
    ]
    return {
        "schema_version": 1, "evidence_kind": "account_records", "quantity_units": "raw_shares",
        "account_id": ACCOUNT, "currency": "CNY",
        "opening": {**provenance("opening", open_at), "asof": open_at, "cash": 2000,
                    "positions": {"A": 1, "B": 10}, "receivables": {}},
        "closing": {**provenance("closing", "2025-01-07T18:00:00+08:00"), "asof": close_at,
                    "cash": "1107.77", "positions": {"B": 70}, "receivables": {}, "nav": "1807.77"},
        "coverage": {**provenance("coverage", "2025-01-07T18:00:00+08:00"), "from": open_at,
                     "through": close_at, "complete": True, "attested_by": "synthetic-test-operator",
                     "operation_ids": [row["record_id"] for row in operations], "event_ids": ["merger"]},
        "operations": operations,
        "marks": {"B": {**provenance("B-mark", close_at, source="market", account=False), "price": 10,
                          "observed_at": sell_at, "source": SOURCE, "price_convention": "raw"}},
        "calendar": {**provenance("calendar", open_at, source="market", account=False),
                     "sessions": ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]},
    }


def reconcile(value):
    return reconcile_account_export(value, SOURCES, account_id=ACCOUNT, currency="CNY", asof=ASOF)


def write_case(tmp_path, export=None):
    payload = export_fixture() if export is None else export
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload))
    refs = []
    for name, source in SOURCES.items():
        record = tmp_path / (name + ".source")
        record.write_text("Synthetic original source fixture: " + name)
        refs.append({"source_id": name, "kind": source["kind"], "path": record.name,
                     "sha256": hashlib.sha256(record.read_bytes()).hexdigest()})
    spec = {"schema_version": 1, "campaign_id": "synthetic-account-test", "account_id": ACCOUNT,
            "currency": "CNY", "asof": ASOF, "account_export": {
                "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "source_artifacts": refs, "archived_assessment": None, "event_catalog": None}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    return spec_path


def test_independent_account_replay_reconciles_shares_cash_withholding_and_nav():
    result = reconcile(export_fixture())
    assert result["status"] == "passed"
    assert result["reconciliation"]["all_matched"]
    assert result["reconstructed"] == {"cash": "1107.77", "positions": {"B": 70}, "receivables": {}, "nav": "1807.77"}
    assert result["fees"] == "3" and result["taxes"] == "20.23"
    assert not result["archived_backtest_repaired"] and not result["production_promoted"]
    assert result["scope"] == "supplied_account_window"


def test_synthetic_arithmetic_cannot_pass_actual_account_evidence_gate():
    value = export_fixture()
    value["evidence_kind"] = "synthetic_fixture"
    result = reconcile(value)
    assert result["reconciliation"]["all_matched"] and result["status"] == "blocked"
    assert "synthetic_records_are_not_account_evidence" in result["blockers"]


@pytest.mark.parametrize("field,new,reason", [
    ("cash", "1108.77", "closing_cash_does_not_reconcile"),
    ("positions", {"B": 71}, "closing_positions_does_not_reconcile"),
    ("receivables", {"phantom": 1}, "closing_receivables_does_not_reconcile"),
    ("nav", "1808.77", "closing_nav_does_not_reconcile"),
])
def test_independent_closing_statement_is_not_a_balancing_entry(field, new, reason):
    value = export_fixture()
    value["closing"][field] = new
    result = reconcile(value)
    assert result["status"] == "blocked" and reason in result["blockers"]


@pytest.mark.parametrize("section", ["opening", "closing", "coverage"])
def test_wrong_account_statement_is_rejected(section):
    value = export_fixture()
    value[section]["account_id"] = "different-account"
    with pytest.raises(ValueError, match="identity"):
        reconcile(value)


@pytest.mark.parametrize("fault,reason", [
    ("fractional_raw_shares", "whole raw shares"), ("adjusted_units", "raw_shares"),
    ("future_receipt", "unavailable"), ("naive_receipt", "timezone-aware"),
    ("foreign_currency", "currency"), ("wrong_source_kind", "suitable"),
    ("wrong_successor", "different successor"), ("impossible_allocation", "bounds"),
    ("wrong_entitlement", "entitlement"), ("cash_tax_mismatch", "cash/tax"),
    ("duplicate_cash", "no outstanding"), ("early_tradability", "precede"),
])
def test_account_evidence_failures_are_rejected(fault, reason):
    value = export_fixture()
    ops = value["operations"]
    if fault == "fractional_raw_shares":
        value["opening"]["positions"]["A"] = 1.1
    elif fault == "adjusted_units":
        value["quantity_units"] = "qlib_adjusted"
    elif fault == "future_receipt":
        ops[2]["available_at"] = "2025-01-09T00:00:00+08:00"
    elif fault == "naive_receipt":
        ops[2]["available_at"] = "2025-01-07T09:00:00"
    elif fault == "foreign_currency":
        ops[2]["currency"] = "USD"
    elif fault == "wrong_source_kind":
        ops[2]["source_id"] = "issuer"
    elif fault == "wrong_successor":
        ops[2]["successor"] = "C"
    elif fault == "impossible_allocation":
        ops[2]["allocated_quantity"] = 64
    elif fault == "wrong_entitlement":
        ops[2]["entitled_quantity"] = 100
    elif fault == "cash_tax_mismatch":
        ops[3]["tax_withheld"] = 20
    elif fault == "duplicate_cash":
        ops.insert(4, copy.deepcopy(ops[3]))
    elif fault == "early_tradability":
        ops[2]["tradable_at"] = "2025-01-02T09:00:00+08:00"
    with pytest.raises(ValueError, match=reason):
        reconcile(value)


def test_payment_date_does_not_manufacture_an_unobserved_cash_credit():
    value = export_fixture()
    value["operations"] = [row for row in value["operations"] if row["kind"] != "cash_credit"]
    value["coverage"]["operation_ids"] = [row["record_id"] for row in value["operations"]]
    value["closing"].update(cash="1026.97", receivables={"merger": "80.8"})
    result = reconcile(value)
    assert result["reconciliation"]["all_matched"]
    assert result["status"] == "blocked"
    assert "due_corporate_cash_credit_not_observed" in result["blockers"]


def test_future_account_transaction_cannot_claim_earlier_observation():
    value = export_fixture()
    value["operations"][2]["available_at"] = "2025-01-06T18:00:00+08:00"
    with pytest.raises(ValueError, match="available before its effective"):
        reconcile(value)


def test_opening_obligation_requires_sourced_terms_and_rejects_early_cash_credit():
    value = export_fixture()
    value["opening"]["receivables"] = {"old-dividend": 5}
    value["closing"]["receivables"] = {"old-dividend": 5}
    value["closing"]["nav"] = "1812.77"
    result = reconcile(value)
    assert result["reconciliation"]["all_matched"]
    assert "opening_receivable_terms_not_verified" in result["blockers"]
    value["opening"]["receivable_terms"] = {
        "old-dividend": {**provenance("old-dividend-terms", "2025-01-01T18:00:00+08:00"),
                         "effective_at": "2025-01-01T10:00:00+08:00",
                         "payment_at": "2025-01-09T10:00:00+08:00"}}
    assert reconcile(value)["status"] == "passed"
    at = "2025-01-07T15:30:00+08:00"
    value["operations"].append({**provenance("old-early-credit", at), "effective_at": at,
                                "kind": "cash_credit", "event_id": "old-dividend",
                                "gross_amount": 5, "tax_withheld": 0, "net_amount": 5})
    with pytest.raises(ValueError, match="precedes.*payment"):
        reconcile(value)


def test_unknown_successor_tradability_blocks_even_reconciled_holdings():
    value = export_fixture()
    value["operations"][1]["event"]["tradable_at"] = None
    value["operations"][2]["tradable_at"] = None
    value["operations"].pop()  # Cannot sell a still-restricted successor.
    value["coverage"]["operation_ids"] = [row["record_id"] for row in value["operations"]]
    value["closing"].update(cash="1078.8", positions={"B": 73}, nav="1808.8")
    result = reconcile(value)
    assert result["reconciliation"]["all_matched"]
    assert "held_instrument_tradability_unverified" in result["blockers"]


def test_completeness_cannot_be_inferred_from_matching_account_balances():
    value = export_fixture()
    value["coverage"]["complete"] = False
    result = reconcile(value)
    assert result["reconciliation"]["all_matched"]
    assert result["status"] == "blocked"


def test_completeness_attestation_cannot_predict_future_account_activity():
    value = export_fixture()
    value["coverage"]["available_at"] = value["opening"]["asof"]
    with pytest.raises(ValueError, match="completeness cannot be attested before"):
        reconcile(value)


def test_stale_mark_does_not_certify_account_nav():
    value = export_fixture()
    value["marks"]["B"]["observed_at"] = "2025-01-06T15:00:00+08:00"
    result = reconcile(value)
    assert result["status"] == "blocked" and result["reconstructed"]["nav"] is None
    assert "closing_raw_marks_missing_or_stale" in result["blockers"]


def test_integral_theoretical_conversion_still_requires_actual_account_record():
    value = export_fixture()
    value["operations"][1]["event"]["ratio"] = 1
    value["operations"] = value["operations"][:2]
    value["coverage"]["operation_ids"] = [row["record_id"] for row in value["operations"]]
    value["marks"]["A"] = copy.deepcopy(value["marks"]["B"])
    value["marks"]["A"]["record_id"] = "A-mark"
    result = reconcile(value)
    assert "actual_account_allocations_missing" in result["blockers"]
    assert result["reconstructed"]["positions"] == {"A": 101, "B": 10}


def test_durable_account_evidence_is_recomputed_and_identical_run_is_reused(tmp_path):
    spec = write_case(tmp_path)
    output = tmp_path / "result"
    receipt = run_account_settlement(spec, output)
    assert receipt["status"] == "passed"
    assert verify_account_settlement(output) == receipt
    before = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert run_account_settlement(spec, output) == receipt
    assert before == {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_interrupted_account_report_resumes_only_identical_registered_inputs(tmp_path, monkeypatch):
    import financial_ts.settlement as module

    spec, output = write_case(tmp_path), tmp_path / "interrupted"
    original = module._write_atomic

    def interrupt(path, raw):
        if path.name == "assessment.json":
            raise OSError("simulated filesystem interruption")
        return original(path, raw)

    monkeypatch.setattr(module, "_write_atomic", interrupt)
    with pytest.raises(OSError, match="interruption"):
        run_account_settlement(spec, output)
    assert (output / "protocol.json").is_file() and not (output / "completion.json").exists()
    monkeypatch.setattr(module, "_write_atomic", original)
    assert run_account_settlement(spec, output)["status"] == "passed"


@pytest.mark.parametrize("name", ["inputs/sources/account", "inputs/account.json", "assessment.json", "completion.json"])
def test_tampered_sources_or_passed_flags_are_not_accepted(tmp_path, name):
    spec = write_case(tmp_path)
    output = tmp_path / "result"
    run_account_settlement(spec, output)
    path = output / name
    if name.startswith("inputs"):
        path.write_bytes(path.read_bytes() + b" ")
    else:
        value = json.loads(path.read_text())
        value["status"] = "fabricated_pass"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify_account_settlement(output)


def test_original_source_hash_mismatch_fails_before_writing_a_report(tmp_path):
    spec = write_case(tmp_path)
    (tmp_path / "account.source").write_text("Changed evidence")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_account_settlement(spec, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_archived_registration_writes_explicit_missing_evidence_not_corrected_returns(tmp_path):
    spec = Path(__file__).parents[1] / "configs/account_settlement_archived.json"
    report = run_account_settlement(spec, tmp_path / "blocked")
    assert report["status"] == "blocked" and not report["reconciliation"]["all_matched"]
    assert len(report["archived_exposure"]["events"]) == 3
    assert report["archived_exposure"]["affected_final_weight"] == pytest.approx(0.06489845337614548)
    assert not report["archived_backtest_repaired"]
    assert report["archived_settlement_status"] == "blocked"


def test_late_allocation_preserves_economic_dates_and_cash_entitlement():
    event = CorporateAction("late", "merger", "A", "2025-01-03T09:00:00+08:00",
                            "2025-01-02T18:00:00+08:00", SOURCE, successor="B", ratio=.62,
                            cash_per_share=1, payment_at="2025-01-06T09:00:00+08:00",
                            tradable_at="2025-01-06T09:00:00+08:00")
    ledger = EventLedger(0, {"A": 101, "B": 10})
    ledger.apply(event, asof=event.effective_at)
    kwargs = {"allocated_quantity": 63, "source": SOURCE, "asof": "2025-01-07T09:00:00+08:00"}
    receipt = ledger.resolve_allocation("late", **kwargs)
    assert ledger.positions == {"B": 73} and ledger.cash == 0
    assert receipt["entitlement_effective_at"] == event.effective_at
    assert ledger.receivables["late"]["payment_at"] == event.payment_at
    assert ledger.resolve_allocation("late", **kwargs) == receipt
    assert ledger.positions == {"B": 73}
    ledger.settle_cash(asof=kwargs["asof"])
    assert ledger.cash == 101 and ledger.reconcile()["shares_reconciled"]
    with pytest.raises(ValueError, match="cannot change"):
        ledger.resolve_allocation("late", **{**kwargs, "allocated_quantity": 62})


def test_changed_frozen_entitlement_requires_replay():
    event = CorporateAction("split", "split", "A", "2025-01-03T09:00:00+08:00",
                            "2025-01-02T18:00:00+08:00", SOURCE, ratio=.1)
    ledger = EventLedger(0, {"A": 105})
    ledger.apply(event, asof=event.effective_at)
    ledger.positions["A"] = 200
    with pytest.raises(ValueError, match="Frozen entitlement"):
        ledger.resolve_allocation("split", allocated_quantity=20, source=SOURCE,
                                  asof="2025-01-07T09:00:00+08:00")


def test_late_allocation_does_not_clear_existing_successor_restriction():
    event = CorporateAction("late", "merger", "A", "2025-01-03T09:00:00+08:00",
                            "2025-01-02T18:00:00+08:00", SOURCE, successor="B", ratio=.62,
                            tradable_at="2025-01-06T09:00:00+08:00")
    ledger = EventLedger(0, {"A": 101, "B": 10})
    ledger.restrictions["B"] = None
    ledger.apply(event, asof=event.effective_at)
    ledger.resolve_allocation("late", allocated_quantity=63, source=SOURCE,
                              asof="2025-01-07T09:00:00+08:00")
    assert not ledger.can_trade("B", "2025-01-07T10:00:00+08:00")


def test_integral_merger_does_not_clear_existing_successor_suspension():
    event = CorporateAction("integral", "merger", "A", "2025-01-03T09:00:00+08:00",
                            "2025-01-02T18:00:00+08:00", SOURCE, successor="B", ratio=.62,
                            tradable_at="2025-01-06T09:00:00+08:00")
    ledger = EventLedger(0, {"A": 100, "B": 10})
    ledger.restrictions["B"] = None
    ledger.apply(event, asof=event.effective_at)
    assert ledger.positions == {"B": 72}
    assert not ledger.can_trade("B", "2025-01-07T10:00:00+08:00")
    assert ledger.reconcile()["shares_reconciled"]


@pytest.mark.parametrize("kind,successor,ratio,allocated", [
    ("merger", "B", .62, 63), ("split", None, .1, 10),
])
def test_cash_consideration_can_settle_before_fractional_share_allocation(kind, successor, ratio, allocated):
    event = CorporateAction("cash-before-shares", kind, "A", "2025-01-03T09:00:00+08:00",
                            "2025-01-02T18:00:00+08:00", SOURCE, successor=successor, ratio=ratio,
                            cash_per_share=.8, payment_at="2025-01-06T09:00:00+08:00",
                            tradable_at="2025-01-06T09:00:00+08:00")
    ledger = EventLedger(1000, {"A": 101})
    first = ledger.apply(event, asof=event.effective_at)
    assert first["status"] == "unresolved" and first["receivable_added"] == pytest.approx(80.8)
    assert ledger.positions == {"A": 101}
    ledger.settle_cash(asof=event.payment_at)
    assert ledger.cash == pytest.approx(1080.8) and not ledger.receivables
    args = dict(allocated_quantity=allocated, source=SOURCE, asof="2025-01-07T09:00:00+08:00")
    receipt = ledger.resolve_allocation(event.event_id, **args)
    assert receipt["receivable_added"] == 0 and not ledger.receivables
    assert ledger.resolve_allocation(event.event_id, **args) == receipt
    assert ledger.settle_cash(asof=args["asof"]) == []
    assert ledger.cash == pytest.approx(1080.8)
    assert ledger.positions == {(successor or "A"): allocated}
    assert ledger.reconcile()["shares_reconciled"]
