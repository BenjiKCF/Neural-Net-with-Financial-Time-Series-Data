"""Reconcile an evidenced raw-share account without inventing settlement.

Account exports are normalized, independently sourced records. Hashes establish
their integrity, not the authenticity of a broker or the correctness of an
extraction. The closing statement is a control input, never a balancing entry.
This retrospective accounting check is not a repaired strategy backtest or
publication-time/prospective evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from filelock import FileLock

from financial_ts.execution import _json_bytes, _write_atomic
from financial_ts.lifecycle import CorporateAction, EventLedger, PriceMark, _time

_CENT = Decimal("0.01")
_ACCOUNT_SOURCES = {"broker", "custodian", "registrar"}
_SOURCE_KINDS = _ACCOUNT_SOURCES | {"issuer", "market_data", "calendar"}


def _read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate settlement JSON key: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _money(value, name, *, signed=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not result.is_finite() or (not signed and result < 0):
        raise ValueError(f"{name} must be finite and {'signed' if signed else 'nonnegative'}")
    return result


def _quantity(value):
    number = _money(value, "raw share quantity")
    if number != number.to_integral_value():
        raise ValueError("Account quantities must be actual whole raw shares, not adjusted quote units")
    return int(number)


def _positions(value):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
        raise ValueError("Raw positions need nonempty instrument identifiers")
    return {symbol: qty for symbol, amount in value.items() if (qty := _quantity(amount))}


def _amounts(value):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not k for k in value):
        raise ValueError("Receivables need event identifiers")
    return {key: amount for key, v in value.items() if (amount := _money(v, "receivable"))}


def _plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _provenance(record, sources, asof, *, kinds=None, account_id=None, currency=None):
    if not isinstance(record, dict) or not isinstance(record.get("record_id"), str) or not record["record_id"]:
        raise ValueError("Every account record needs a stable record_id")
    source_id = record.get("source_id")
    if source_id not in sources or (kinds and sources[source_id]["kind"] not in kinds):
        raise ValueError("Record has no suitable hash-verified source artifact")
    if _time(record["available_at"]) > asof:
        raise ValueError("Account evidence is unavailable at the audit cutoff")
    if account_id is not None and record.get("account_id") != account_id:
        raise ValueError("Account record identity does not match the registered account")
    if currency is not None and record.get("currency") != currency:
        raise ValueError("Account record currency does not match the registered account")


def reconcile_account_export(export, sources, *, account_id, currency, asof):
    """Replay independently evidenced account records and compare a closing statement.

    All sources must already have passed byte-hash verification. Quantities use
    integer raw shares; money uses Decimal, with at most one cent reconciliation
    tolerance. Real allocations are required even for integral entitlements.
    Availability is checked against the audit cutoff; this is a retrospective
    reconciliation and does not assert information was available for trading.
    """
    asof = _time(asof)
    if export.get("schema_version") != 1 or export.get("quantity_units") != "raw_shares":
        raise ValueError("An account export requires schema_version 1 and raw_shares units")
    if export.get("account_id") != account_id or export.get("currency") != currency:
        raise ValueError("Export account identity/currency differs from the registration")
    kind = export.get("evidence_kind")
    if kind not in {"account_records", "synthetic_fixture"}:
        raise ValueError("Account export must declare its evidence_kind")
    opening, closing, coverage = (export[name] for name in ("opening", "closing", "coverage"))
    for statement in (opening, closing, coverage):
        _provenance(statement, sources, asof, kinds=_ACCOUNT_SOURCES,
                    account_id=account_id, currency=currency)
    start, end = _time(opening["asof"]), _time(closing["asof"])
    if not start < end <= asof:
        raise ValueError("Opening and closing statements must bound a past ordered account window")
    for statement, at in ((opening, start), (closing, end)):
        if _time(statement["available_at"]) < at:
            raise ValueError("An account statement cannot be available before its observation")
    if _time(coverage["available_at"]) < _time(coverage["through"]):
        raise ValueError("Account completeness cannot be attested before its covered through timestamp")
    cash = _money(opening["cash"], "opening cash")
    positions = _positions(opening["positions"])
    receivables = _amounts(opening.get("receivables", {}))
    opening_receivables = dict(receivables)
    pending, actions, payments, restrictions = {}, {}, {}, {}
    unknown_terminal = set()
    rows = []
    ids = [opening["record_id"], closing["record_id"], coverage["record_id"]]
    fees, taxes, external_cashflows = Decimal(0), Decimal(0), Decimal(0)
    operations = export.get("operations")
    if not isinstance(operations, list):
        raise ValueError("Account operations must be an ordered list")
    previous = start
    opening_terms = opening.get("receivable_terms", {})
    if opening_terms and (not isinstance(opening_terms, dict) or set(opening_terms) != set(receivables)):
        raise ValueError("Opening receivable terms must identify every and only outstanding obligation")
    for event_id, terms in opening_terms.items():
        _provenance(terms, sources, asof, kinds=_ACCOUNT_SOURCES,
                    account_id=account_id, currency=currency)
        if _time(terms["available_at"]) < _time(terms["effective_at"]):
            raise ValueError("An account obligation cannot be observed before it exists")
        if _time(terms["effective_at"]) > start:
            raise ValueError("Opening obligation does not exist at the opening statement")
        payments[event_id] = _time(terms["payment_at"])
        ids.append(terms["record_id"])
    for record in operations:
        op = record.get("kind")
        _provenance(record, sources, asof, kinds=_ACCOUNT_SOURCES if op != "action" else (_ACCOUNT_SOURCES | {"issuer"}),
                    account_id=account_id, currency=currency)
        ids.append(record["record_id"])
        at = _time(record["effective_at"])
        if not start < at <= end or at < previous:
            raise ValueError("Account operations must be chronological within the statement window")
        if op != "action" and _time(record["available_at"]) < at:
            raise ValueError("An observed account transaction cannot be available before its effective time")
        previous = at
        before_cash, before_positions = cash, dict(positions)
        if op == "fill":
            symbol, side = record["instrument"], record["side"]
            if side not in {"buy", "sell"}:
                raise ValueError("Fill side must be buy or sell")
            if any(item["event"].instrument == symbol for item in pending.values()) or symbol in unknown_terminal:
                raise ValueError("A fill cannot spend unresolved corporate-action shares")
            if symbol in restrictions and (restrictions[symbol] is None or at < restrictions[symbol]):
                raise ValueError("A fill precedes verified successor tradability or resumption")
            qty, price = _quantity(record["quantity"]), _money(record["price"], "raw fill price")
            fee, tax = _money(record["fees"], "fill fees"), _money(record.get("tax", 0), "fill tax")
            if not qty or not price or (side == "sell" and qty > positions.get(symbol, 0)):
                raise ValueError("Invalid raw executed shares or fill price")
            cash += (-qty * price if side == "buy" else qty * price) - fee - tax
            positions[symbol] = positions.get(symbol, 0) + (qty if side == "buy" else -qty)
            fees, taxes = fees + fee, taxes + tax
        elif op == "action":
            event = CorporateAction(**record["event"])
            if event.event_id in actions or _time(event.effective_at) != at:
                raise ValueError("Corporate-action IDs must be unique with matching effective times")
            if _time(event.known_at) > asof or _time(event.known_at) > _time(record["available_at"]):
                raise ValueError("Corporate-action knowledge is unavailable in its evidence record")
            if event.allocated_quantity is not None:
                raise ValueError("Actual account allocations must be separate registrar/account records")
            symbol, qty = event.instrument, positions.get(event.instrument, 0)
            if _quantity(record["entitled_quantity"]) != qty:
                raise ValueError("Corporate-action entitlement differs from reconstructed raw shares")
            if any(item["event"].instrument == symbol for item in pending.values()):
                raise ValueError("Overlapping unresolved actions require an explicit account reconstruction")
            actions[event.event_id] = event
            if event.kind in {"split", "merger", "ticker_change"} and qty:
                pending[event.event_id] = {"event": event, "quantity": qty}
            elif event.kind == "delisting" and qty:
                if event.cash_per_share is None or event.payment_at is None:
                    unknown_terminal.add(symbol)
                else:
                    positions.pop(symbol)
            elif event.kind == "suspension":
                restrictions[symbol] = _time(event.tradable_at) if event.tradable_at else None
            elif event.kind == "resumption":
                restrictions.pop(symbol, None)
            if qty and event.cash_per_share is not None:
                if event.event_id in receivables:
                    raise ValueError("Corporate-action receivable already exists in opening statement")
                receivables[event.event_id] = Decimal(qty) * Decimal(str(event.cash_per_share))
                payments[event.event_id] = _time(event.payment_at)
        elif op == "allocation":
            event_id = record["event_id"]
            if event_id not in pending:
                raise ValueError("Allocation has no unresolved account entitlement")
            item = pending[event_id]
            event, qty = item["event"], item["quantity"]
            if positions.get(event.instrument, 0) != qty or _quantity(record["entitled_quantity"]) != qty:
                raise ValueError("Allocation does not match the frozen account entitlement")
            allocation = _quantity(record["allocated_quantity"])
            exact = Decimal(qty) * Decimal(str(event.ratio))
            if allocation not in {math.floor(exact), math.ceil(exact)}:
                raise ValueError("Observed allocation is outside whole-share entitlement bounds")
            successor = event.instrument if event.kind == "split" else event.successor
            if record.get("successor") != successor:
                raise ValueError("Account allocation identifies a different successor")
            positions.pop(event.instrument)
            positions[successor] = positions.get(successor, 0) + allocation
            if event.kind == "merger":
                # A separate issuer/registrar source must support the release;
                # knowledge of a conversion ratio alone is insufficient.
                trade = record.get("tradable_at") or event.tradable_at
                source_id = record.get("tradability_source_id")
                if trade is not None:
                    if source_id not in sources or sources[source_id]["kind"] not in (_ACCOUNT_SOURCES | {"issuer"}):
                        raise ValueError("Successor tradability needs a hash-verified issuer/account source")
                    if _time(trade) < _time(event.effective_at):
                        raise ValueError("Successor tradability cannot precede the economic conversion")
                release = _time(trade) if trade else None
                if successor in restrictions:
                    old_release = restrictions[successor]
                    release = None if old_release is None or release is None else max(old_release, release)
                restrictions[successor] = release
            elif event.kind == "ticker_change" and event.instrument in restrictions:
                restrictions[successor] = restrictions.pop(event.instrument)
            pending.pop(event_id)
        elif op == "cash_credit":
            event_id = record["event_id"]
            if event_id not in receivables:
                raise ValueError("Cash credit has no outstanding corporate-action obligation")
            gross = _money(record["gross_amount"], "gross cash credit")
            tax = _money(record["tax_withheld"], "cash withholding tax")
            net = _money(record["net_amount"], "net cash credit")
            if gross - tax != net or abs(net - receivables[event_id]) > _CENT:
                raise ValueError("Observed net cash/tax does not reconcile with the entitlement")
            if event_id in payments and at < payments[event_id]:
                raise ValueError("Cash credit precedes its documented payment date")
            cash, taxes = cash + net, taxes + tax
            receivables.pop(event_id)
        elif op == "cash_transfer":
            amount = _money(record["amount"], "external cash transfer", signed=True)
            cash, external_cashflows = cash + amount, external_cashflows + amount
        else:
            raise ValueError(f"Unsupported account operation: {op}")
        if cash < 0:
            raise ValueError("Account operation would overdraw cash")
        positions = {symbol: qty for symbol, qty in positions.items() if qty}
        rows.append({"record_id": record["record_id"], "kind": op, "effective_at": at.isoformat(),
                     "cash_delta": cash - before_cash, "cash_after": cash,
                     "quantity_deltas": {symbol: positions.get(symbol, 0) - before_positions.get(symbol, 0)
                                         for symbol in sorted(set(positions) | set(before_positions))
                                         if positions.get(symbol, 0) != before_positions.get(symbol, 0)}})
    if len(ids) != len(set(ids)):
        raise ValueError("Account record IDs must be unique; duplicate settlement is rejected")
    blockers = []
    if kind != "account_records":
        blockers.append("synthetic_records_are_not_account_evidence")
    expected_ops = [row["record_id"] for row in operations]
    if (coverage.get("complete") is not True
            or coverage.get("operation_ids") != expected_ops
            or set(coverage.get("event_ids", [])) != set(actions)
            or _time(coverage["from"]) != start or _time(coverage["through"]) != end
            or not isinstance(coverage.get("attested_by"), str) or not coverage["attested_by"].strip()):
        blockers.append("complete_account_activity_and_corporate_action_coverage_not_attested")
    if pending:
        blockers.append("actual_account_allocations_missing")
    if unknown_terminal:
        blockers.append("terminal_cash_consideration_unknown")
    if any(symbol in positions and release is None for symbol, release in restrictions.items()):
        blockers.append("held_instrument_tradability_unverified")
    if any(key in receivables and date <= end for key, date in payments.items()):
        blockers.append("due_corporate_cash_credit_not_observed")
    if opening_receivables and set(opening_terms) != set(opening_receivables):
        blockers.append("opening_receivable_terms_not_verified")
    expected_cash = _money(closing["cash"], "closing cash")
    expected_positions = _positions(closing["positions"])
    expected_receivables = _amounts(closing.get("receivables", {}))
    marks = {}
    for symbol, record in export.get("marks", {}).items():
        _provenance(record, sources, asof, kinds={"market_data", "broker", "custodian"})
        if _time(record["available_at"]) < _time(record["observed_at"]):
            raise ValueError("A closing mark cannot be available before its observation")
        ids.append(record["record_id"])
        marks[symbol] = PriceMark(float(_money(record["price"], "raw closing mark")),
                                  record["observed_at"], record["source"], record.get("price_convention", "raw"))
    calendar = export["calendar"]
    _provenance(calendar, sources, asof, kinds={"calendar", "market_data", "broker", "custodian"})
    ids.append(calendar["record_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Statement, operation, mark and calendar record IDs must be unique")
    marking = EventLedger(float(cash), positions).mark(
        marks, asof=end, observed_calendar=pd.to_datetime(calendar["sessions"]),
        maximum_staleness_sessions=0,
    )
    nav = None if marking["nav"] is None else cash + sum(receivables.values(), Decimal(0)) + sum(
        (Decimal(qty) * _money(export["marks"][symbol]["price"], "raw closing mark")
         for symbol, qty in positions.items()), Decimal(0)
    )
    expected_nav = _money(closing["nav"], "independent closing NAV")
    statement_nav = None if any(symbol not in marks for symbol in expected_positions) else (
        expected_cash + sum(expected_receivables.values(), Decimal(0))
        + sum((Decimal(qty) * _money(export["marks"][symbol]["price"], "raw closing mark")
               for symbol, qty in expected_positions.items()), Decimal(0)))
    checks = {
        "cash": abs(cash - expected_cash) <= _CENT,
        "positions": positions == expected_positions,
        "receivables": (set(receivables) == set(expected_receivables)
                        and all(abs(value - expected_receivables[key]) <= _CENT for key, value in receivables.items())),
        "nav": nav is not None and abs(nav - expected_nav) <= _CENT,
        "statement_nav": statement_nav is not None and abs(statement_nav - expected_nav) <= _CENT,
    }
    checks["all_matched"] = all(checks.values())
    blockers.extend(f"closing_{name}_does_not_reconcile" for name, passed in checks.items()
                    if name != "all_matched" and not passed)
    if marking["unresolved"]:
        blockers.append("closing_raw_marks_missing_or_stale")
    return _plain({"status": "blocked" if blockers else "passed", "scope": "supplied_account_window",
                   "account_id": account_id, "currency": currency, "evidence_kind": kind,
                   "window": {"from": start.isoformat(), "through": end.isoformat()},
                   "blockers": blockers, "reconciliation": checks,
                   "reconstructed": {"cash": cash, "positions": positions, "receivables": receivables, "nav": nav},
                   "independent_closing": {"cash": expected_cash, "positions": expected_positions,
                                           "receivables": expected_receivables, "nav": expected_nav},
                   "fees": fees, "taxes": taxes, "external_cashflows": external_cashflows,
                   "operations": rows, "pending_allocation_event_ids": sorted(pending),
                   "unpriced_terminal_instruments": sorted(unknown_terminal),
                   "source_authenticity": "external account/provider attestation; byte hashes verify integrity only",
                   "historical_information_availability_certified": False,
                   "archived_backtest_repaired": False, "production_promoted": False})


def _validate_spec(spec):
    required = {"schema_version", "campaign_id", "account_id", "currency", "asof", "account_export",
                "source_artifacts", "archived_assessment", "event_catalog"}
    if set(spec) != required or spec["schema_version"] != 1:
        raise ValueError("Settlement spec requires exactly the version-1 registered fields")
    if any(not isinstance(spec[key], str) or not spec[key] for key in ("campaign_id", "account_id", "currency")):
        raise ValueError("Campaign, account and currency identifiers are required")
    _time(spec["asof"])
    if not isinstance(spec["source_artifacts"], list):
        raise ValueError("Source artifacts must be a list")
    identifiers = set()
    for item in spec["source_artifacts"]:
        identifier = item.get("source_id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier)
                or identifier in identifiers or item.get("kind") not in _SOURCE_KINDS):
            raise ValueError("Source artifacts require unique safe IDs and supported source kinds")
        identifiers.add(identifier)


def _calculate(spec, files):
    sources = {item["source_id"]: item for item in spec["source_artifacts"]}
    if spec["account_export"] is None:
        result = {"status": "blocked", "scope": "supplied_account_window", "account_id": spec["account_id"],
                  "currency": spec["currency"], "evidence_kind": "missing",
                  "blockers": ["account_export_not_supplied", "complete_account_action_coverage_not_supplied",
                               "independent_closing_statement_not_supplied"],
                  "reconciliation": {"cash": False, "positions": False, "receivables": False,
                                     "nav": False, "all_matched": False},
                  "archived_backtest_repaired": False, "production_promoted": False}
    else:
        result = reconcile_account_export(files["account.json"], sources, account_id=spec["account_id"],
                                          currency=spec["currency"], asof=spec["asof"])
    assessment = files.get("archived_assessment.json")
    if assessment is not None:
        events = {item["event_id"]: item for item in files.get("event_catalog.json", {}).get("events", [])}
        result["archived_exposure"] = {
            "source_positions_sha256": assessment.get("source_positions_sha256"),
            "affected_final_weight": assessment.get("affected_final_weight"),
            "archived_final_date": assessment.get("archived_final_date"),
            "events": [{"event_id": row["event"]["event_id"], "instrument": row["instrument"],
                        "successor": row["event"].get("successor"),
                        "theoretical_successor_quantity": row.get("theoretical_successor_quantity"),
                        "implied_raw_quantity": row.get("raw_quantity"),
                        "missing_evidence": row.get("unresolved_reasons", []),
                        "catalog_identity_matches": events.get(row["event"]["event_id"], {}).get("instrument") == row["instrument"]}
                       for row in assessment.get("events", []) if "event" in row],
            "qualification": "The archived account is a hypothetical Qlib simulation, with no implied real broker statement. It needs a separate raw-share/action-aware reconstructed replay and explicit counterfactual allocation assumptions or bounds; real account exports are a different evidence path.",
        }
        result["archived_settlement_status"] = "blocked"
        result["archived_settlement_blockers"] = [
            "original_raw_fill_and_account_history_not_reconstructed",
            "all_archived_corporate_actions_and_allocations_not_verified",
            "separate_corrected_portfolio_replay_not_executed",
        ]
    result.update(schema_version=1, campaign_id=spec["campaign_id"], audit_asof=spec["asof"])
    return result


def run_account_settlement(spec_path, output):
    """Hash-bind source files, run reconciliation, and retain a verifiable report.

    A null account_export writes a useful blocked evidence inventory. Malformed,
    tampered, wrong-account, future or adjusted-unit records raise before any
    success receipt. Repeated calls verify and reuse the identical immutable run.
    """
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    spec_raw, spec = spec_path.read_bytes(), _read(spec_path)
    _validate_spec(spec)
    blobs = {"inputs/spec.json": spec_raw}
    files = {}
    refs = [("account.json", spec["account_export"]), ("archived_assessment.json", spec["archived_assessment"]),
            ("event_catalog.json", spec["event_catalog"])]
    seen = set()
    for item in spec["source_artifacts"]:
        identifier = item.get("source_id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier)
                or identifier in seen or item.get("kind") not in _SOURCE_KINDS):
            raise ValueError("Source artifacts require unique safe IDs and supported source kinds")
        seen.add(identifier)
        refs.append(("sources/" + identifier, item))
    for name, ref in refs:
        if ref is None:
            continue
        if not isinstance(ref, dict) or not isinstance(ref.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", ref["sha256"]):
            raise ValueError("Every input artifact needs an expected SHA256")
        path = (spec_path.parent / ref["path"]).resolve()
        if output == path or output in path.parents:
            raise ValueError("Settlement output cannot contain or replace an input artifact")
        raw = path.read_bytes()
        if _digest(raw) != ref["sha256"]:
            raise ValueError("Settlement input source hash mismatch")
        blobs["inputs/" + name] = raw
        if not name.startswith("sources/"):
            files[name] = _read(path)
    result = _calculate(spec, files)
    protocol = {"schema_version": 1, "source_spec_sha256": _digest(spec_raw),
                "inputs": {name: _digest(raw) for name, raw in blobs.items()},
                "implementation_sha256": _digest(Path(__file__).read_bytes())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / ("." + output.name + ".settlement.lock")), timeout=30):
        if output.exists():
            if not (output / "protocol.json").exists() or _read(output / "protocol.json") != protocol:
                raise ValueError("Settlement output belongs to different inputs or implementation")
            if (output / "completion.json").exists():
                return verify_account_settlement(output)
        else:
            output.mkdir()
            _write_atomic(output / "protocol.json", _json_bytes(protocol))
        for name, raw in blobs.items():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            if (output / name).exists():
                if (output / name).read_bytes() != raw:
                    raise ValueError("Interrupted settlement input differs from its registration")
            else:
                _write_atomic(output / name, raw)
        result_raw = _json_bytes(result)
        if (output / "assessment.json").exists():
            if (output / "assessment.json").read_bytes() != result_raw:
                raise ValueError("Interrupted settlement report differs from independent recomputation")
        else:
            _write_atomic(output / "assessment.json", result_raw)
        receipt = {**result, "artifacts": {"protocol.json": _digest(_json_bytes(protocol)),
                                          "assessment.json": _digest(_json_bytes(result))}}
        _write_atomic(output / "completion.json", _json_bytes(receipt))
    return verify_account_settlement(output)


def verify_account_settlement(output):
    """Verify every retained byte and recompute accounting before returning status."""
    output = Path(output).resolve()
    protocol, receipt = _read(output / "protocol.json"), _read(output / "completion.json")
    if protocol.get("implementation_sha256") != _digest(Path(__file__).read_bytes()):
        raise ValueError("Settlement verification requires the registered implementation")
    inventory = protocol.get("inputs", {})
    if "inputs/spec.json" not in inventory:
        raise ValueError("Settlement protocol is missing its source specification")
    for name, digest in inventory.items():
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output) or _digest(path.read_bytes()) != digest:
            raise ValueError("Retained settlement input failed integrity verification")
    spec = _read(output / "inputs/spec.json")
    _validate_spec(spec)
    if protocol["source_spec_sha256"] != inventory["inputs/spec.json"]:
        raise ValueError("Settlement specification identity mismatch")
    expected = {"inputs/spec.json"}
    files = {}
    for field, name in (("account_export", "account.json"), ("archived_assessment", "archived_assessment.json"),
                        ("event_catalog", "event_catalog.json")):
        if spec[field] is not None:
            expected.add("inputs/" + name)
            if inventory.get("inputs/" + name) != spec[field]["sha256"]:
                raise ValueError("Settlement input is not the registered source artifact")
            files[name] = _read(output / "inputs" / name)
    for source in spec["source_artifacts"]:
        name = "inputs/sources/" + source["source_id"]
        expected.add(name)
        if inventory.get(name) != source["sha256"]:
            raise ValueError("Settlement source evidence is not the registered artifact")
    if set(inventory) != expected:
        raise ValueError("Settlement input inventory does not match its registration")
    recomputed = _calculate(spec, files)
    if recomputed != _read(output / "assessment.json"):
        raise ValueError("Settlement report does not match independent recomputation")
    expected_receipt = {**recomputed, "artifacts": {
        name: _digest((output / name).read_bytes()) for name in ("protocol.json", "assessment.json")}}
    if receipt != expected_receipt:
        raise ValueError("Settlement completion does not match its verified artifacts")
    return receipt
