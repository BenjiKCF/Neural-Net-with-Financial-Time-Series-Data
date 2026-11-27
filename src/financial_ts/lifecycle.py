"""Independent raw-share corporate-action and cash ledger.

This ledger does not reinterpret Qlib's adjusted positions as raw shares. The
adapter below requires the source factor at the conversion date. Missing share
allocations, payment amounts, or terminal marks remain unresolved; there is no
synthetic sale at the last quote and no retrospective deletion of securities.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


def _time(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is None:
        raise ValueError("Event times must be finite timezone-aware timestamps")
    return result


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _source(value):
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError("A source HTTPS URL is required")


@dataclass(frozen=True)
class CorporateAction:
    event_id: str
    kind: str
    instrument: str
    effective_at: str
    known_at: str
    source: str
    successor: str | None = None
    ratio: float = 1.0
    cash_per_share: float | None = None
    payment_at: str | None = None
    tradable_at: str | None = None
    allocated_quantity: float | None = None
    allocation_source: str | None = None

    def __post_init__(self):
        if not self.event_id or not self.instrument:
            raise ValueError("Event and instrument identifiers are required")
        if self.kind not in {"split", "cash_dividend", "merger", "ticker_change", "delisting", "suspension", "resumption"}:
            raise ValueError("Unsupported corporate action")
        _time(self.effective_at)
        _time(self.known_at)
        _source(self.source)
        _number(self.ratio, "ratio", positive=True)
        if self.kind in {"merger", "ticker_change"} and (not self.successor or self.successor == self.instrument):
            raise ValueError("Conversion requires a different successor instrument")
        if self.kind == "ticker_change" and self.ratio != 1:
            raise ValueError("Ticker changes preserve share count")
        if self.cash_per_share is not None:
            _number(self.cash_per_share, "cash_per_share")
        if self.kind == "cash_dividend" and (self.cash_per_share is None or self.payment_at is None):
            raise ValueError("Dividends require net cash per share and a payment time")
        if self.kind in {"split", "merger", "ticker_change"} and self.cash_per_share is not None and self.payment_at is None:
            raise ValueError("Cash consideration requires a payment time")
        for value in (self.payment_at, self.tradable_at):
            if value is not None and _time(value) < _time(self.effective_at):
                raise ValueError("Settlement/tradability cannot precede the effective event")
        if self.allocated_quantity is not None:
            _number(self.allocated_quantity, "allocated_quantity")
            _source(self.allocation_source)


@dataclass(frozen=True)
class PriceMark:
    price: float
    observed_at: str
    source: str
    price_convention: str = "raw"

    def __post_init__(self):
        _number(self.price, "price")
        _time(self.observed_at)
        _source(self.source)
        if self.price_convention != "raw":
            raise ValueError("Corporate-action ledger accepts raw price marks only")


def qlib_to_raw_position(adjusted_quantity: float, adjusted_price: float, factor: float) -> dict:
    """Convert one contemporaneous Qlib position without changing its value.

    Qlib 0.9.7 rounds adjusted quantity using ``raw_qty = adjusted_qty * factor``;
    its raw price is ``adjusted_price / factor``. Never use a later factor for a
    historical position, and do not apply raw corporate actions to adjusted marks.
    """
    _number(adjusted_quantity, "adjusted_quantity")
    _number(adjusted_price, "adjusted_price")
    _number(factor, "factor", positive=True)
    return {"raw_quantity": adjusted_quantity * factor, "raw_price": adjusted_price / factor,
            "source_factor": factor, "value": adjusted_quantity * adjusted_price}


class EventLedger:
    """Long-only raw-share ledger with explicit receivables and unresolved events."""

    def __init__(self, cash: float, positions: dict[str, float] | None = None, currency="CNY"):
        self.cash = _number(cash, "cash")
        self.positions = {str(k): _number(v, "quantity") for k, v in (positions or {}).items()}
        if any(not k for k in self.positions) or not currency:
            raise ValueError("Instrument identifiers and currency are required")
        self.currency = currency
        self.receivables: dict[str, dict] = {}
        self.restrictions: dict[str, str | None] = {}
        self.unresolved: dict[str, dict] = {}
        self.receipts: list[dict] = []
        self._actions: dict[str, CorporateAction] = {}
        self._fills: dict[str, dict] = {}
        self._last_at: pd.Timestamp | None = None
        self.initial_cash = self.cash
        self.initial_positions = self.positions.copy()

    def _advance(self, at):
        at = _time(at)
        if self._last_at is not None and at < self._last_at:
            raise ValueError("Ledger updates must be chronological")
        return at

    def _stamp(self, at):
        self._last_at = _time(at)

    def can_trade(self, instrument: str, at) -> bool:
        at = _time(at)
        if any(x["instrument"] == instrument for x in self.unresolved.values()):
            return False
        if instrument not in self.restrictions:
            return True
        release = self.restrictions[instrument]
        return release is not None and at >= _time(release)

    def apply(self, event: CorporateAction, *, asof) -> dict:
        """Apply an available effective event exactly once at a chronological time.

        A fractional merger entitlement needs an observed account allocation.
        Exchange-wide largest-remainder/random allocation cannot be recreated by
        rounding an isolated account. Until supplied, the old holding is frozen.
        """
        if event.event_id in self._actions:
            if self._actions[event.event_id] != event:
                raise ValueError("An existing event ID cannot change contents")
            return next(copy.deepcopy(x) for x in self.receipts if x.get("event_id") == event.event_id)
        asof = self._advance(asof)
        if asof < max(_time(event.effective_at), _time(event.known_at)):
            raise ValueError("Event is not yet effective or known at this cutoff")
        if self._last_at is not None and _time(event.effective_at) < self._last_at:
            raise ValueError("Retroactive event requires a replay of historical entitlements")
        quantity = self.positions.get(event.instrument, 0.0)
        positions_before = self.positions.copy()
        receipt = {"event_id": event.event_id, "event": asdict(event), "applied_at": asof.isoformat(),
                   "instrument": event.instrument, "quantity_before": quantity, "cash_delta": 0.0,
                   "status": "applied"}
        if event.kind == "split":
            exact = Decimal(str(quantity)) * Decimal(str(event.ratio))
            if event.allocated_quantity is None and exact != exact.to_integral_value():
                receipt.update(status="unresolved", reason="account_specific_fractional_allocation_required")
                self.unresolved[event.event_id] = {"instrument": event.instrument, "reason": receipt["reason"],
                                                   "theoretical_successor_quantity": float(exact)}
            else:
                allocation = float(exact) if event.allocated_quantity is None else event.allocated_quantity
                if event.allocated_quantity is not None and allocation not in {math.floor(exact), math.ceil(exact)}:
                    raise ValueError("Observed allocation must match the whole-share entitlement bounds")
                self.positions[event.instrument] = allocation
                if event.cash_per_share is not None and quantity:
                    self.receivables[event.event_id] = {"amount": quantity * event.cash_per_share,
                                                       "payment_at": event.payment_at, "instrument": event.instrument}
        elif event.kind in {"merger", "ticker_change"}:
            exact = Decimal(str(quantity)) * Decimal(str(event.ratio))
            if event.allocated_quantity is None and exact != exact.to_integral_value():
                receipt.update(status="unresolved", reason="account_specific_fractional_allocation_required")
                self.unresolved[event.event_id] = {"instrument": event.instrument, "reason": receipt["reason"],
                                                   "theoretical_successor_quantity": float(exact)}
            else:
                allocation = float(exact) if event.allocated_quantity is None else event.allocated_quantity
                if event.allocated_quantity is not None and allocation not in {math.floor(exact), math.ceil(exact)}:
                    raise ValueError("Observed allocation must match the whole-share entitlement bounds")
                if quantity:
                    self.positions.pop(event.instrument, None)
                    self.positions[event.successor] = self.positions.get(event.successor, 0.0) + allocation
                    restricted = event.instrument in self.restrictions
                    original_release = self.restrictions.get(event.instrument)
                    self.restrictions.pop(event.instrument, None)
                    if event.kind == "merger":
                        self.restrictions[event.successor] = event.tradable_at
                        if event.tradable_at is None:
                            self.unresolved[event.event_id] = {
                                "instrument": event.successor,
                                "reason": "successor_tradability_unverified",
                            }
                    elif restricted:
                        self.restrictions[event.successor] = original_release
                receipt["successor_quantity"] = allocation
                if event.cash_per_share is not None and quantity:
                    if event.payment_at is None:
                        raise ValueError("Cash consideration requires a payment time")
                    self.receivables[event.event_id] = {"amount": quantity * event.cash_per_share,
                                                       "payment_at": event.payment_at, "instrument": event.instrument}
        elif event.kind == "cash_dividend":
            # Caller applies this at entitlement fixation with the actual entitled
            # holdings; payment is a later operation, independent of later sales.
            self.receivables[event.event_id] = {"amount": quantity * event.cash_per_share,
                                               "payment_at": event.payment_at, "instrument": event.instrument}
        elif event.kind == "delisting":
            if quantity and (event.cash_per_share is None or event.payment_at is None):
                self.unresolved[event.event_id] = {"instrument": event.instrument, "reason": "terminal_value_unknown"}
                receipt.update(status="unresolved", reason="terminal_value_unknown")
            elif quantity:
                self.positions.pop(event.instrument)
                self.receivables[event.event_id] = {"amount": quantity * event.cash_per_share,
                                                   "payment_at": event.payment_at, "instrument": event.instrument}
            self.restrictions[event.instrument] = None
        elif event.kind == "suspension":
            self.restrictions[event.instrument] = event.tradable_at
        elif event.kind == "resumption":
            self.restrictions.pop(event.instrument, None)
            for identifier, item in list(self.unresolved.items()):
                if item["instrument"] == event.instrument and item["reason"] == "successor_tradability_unverified":
                    self.unresolved.pop(identifier)
        receipt["quantity_after"] = self.positions.get(event.instrument, 0.0)
        receipt["quantity_deltas"] = {
            symbol: self.positions.get(symbol, 0.0) - positions_before.get(symbol, 0.0)
            for symbol in sorted(set(self.positions) | set(positions_before))
            if self.positions.get(symbol, 0.0) != positions_before.get(symbol, 0.0)
        }
        receipt["receivable_added"] = self.receivables.get(event.event_id, {}).get("amount", 0.0)
        self._actions[event.event_id] = event
        self.receipts.append(receipt)
        self._stamp(asof)
        return copy.deepcopy(receipt)

    def resolve_allocation(self, event_id: str, *, allocated_quantity: float, source: str, asof, tradable_at=None) -> dict:
        """Resolve a preserved fractional entitlement with actual allocation evidence."""
        if event_id not in self.unresolved or self.unresolved[event_id]["reason"] != "account_specific_fractional_allocation_required":
            raise ValueError("No unresolved fractional entitlement for this event")
        event = self._actions[event_id]
        resolved = replace(event, event_id=event_id + ":allocation", known_at=_time(asof).isoformat(),
                           effective_at=_time(asof).isoformat(),
                           allocated_quantity=allocated_quantity, allocation_source=source,
                           tradable_at=tradable_at or event.tradable_at)
        receipt = self.apply(resolved, asof=asof)
        self.unresolved.pop(event_id)
        return receipt

    def settle_cash(self, *, asof) -> list[dict]:
        asof = self._advance(asof)
        receipts = []
        for event_id, item in list(self.receivables.items()):
            if _time(item["payment_at"]) <= asof:
                self.cash += item["amount"]
                receipt = {"settlement_for": event_id, "applied_at": asof.isoformat(),
                           "cash_delta": item["amount"], "instrument": item["instrument"]}
                self.receipts.append(receipt)
                receipts.append(copy.deepcopy(receipt))
                del self.receivables[event_id]
        self._stamp(asof)
        return receipts

    def record_fill(self, fill_id, instrument, side, quantity, price, fees, *, asof) -> dict:
        if not fill_id or not instrument or side not in {"buy", "sell"}:
            raise ValueError("Fill identity, instrument and side are required")
        quantity = _number(quantity, "quantity", positive=True)
        price = _number(price, "price", positive=True)
        fees = _number(fees, "fees")
        asof = _time(asof)
        fill = {"fill_id": fill_id, "instrument": instrument, "side": side, "quantity": quantity,
                "price": price, "fees": fees, "applied_at": asof.isoformat()}
        if fill_id in self._fills:
            if fill != self._fills[fill_id]:
                raise ValueError("Existing fill ID has different contents")
            return copy.deepcopy(fill)
        self._advance(asof)
        if not self.can_trade(instrument, asof):
            raise ValueError("Instrument is restricted or has an unresolved event")
        held = self.positions.get(instrument, 0.0)
        delta = -(quantity * price + fees) if side == "buy" else quantity * price - fees
        if side == "sell" and quantity > held + 1e-10:
            raise ValueError("Cannot sell more than the raw shares held")
        if self.cash + delta < -1e-8:
            raise ValueError("Fill would overdraw cash")
        self.cash = max(0.0, self.cash + delta)
        self.positions[instrument] = held + (quantity if side == "buy" else -quantity)
        if abs(self.positions[instrument]) < 1e-10:
            self.positions.pop(instrument)
        self._fills[fill_id] = fill
        self.receipts.append({**fill, "cash_delta": delta,
                              "quantity_deltas": {instrument: quantity if side == "buy" else -quantity}})
        self._stamp(asof)
        return copy.deepcopy(fill)

    def mark(self, marks: dict[str, PriceMark], *, asof, observed_calendar, maximum_staleness_sessions=0) -> dict:
        """Return valid NAV only if every holding has a fresh raw-price mark.

        The session calendar must cover the exchange-local cutoff date, extending
        to the following session for nontrading dates. A truncated calendar cannot
        establish mark freshness beyond its last known session.
        """
        from financial_ts.data_contracts import _calendar

        asof = self._advance(asof)
        if isinstance(maximum_staleness_sessions, bool) or not isinstance(maximum_staleness_sessions, int) or maximum_staleness_sessions < 0:
            raise ValueError("maximum_staleness_sessions must be a nonnegative integer")
        calendar = _calendar(observed_calendar)
        date = asof.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
        if calendar[-1] < date:
            raise ValueError("Observed calendar does not cover the valuation cutoff date")
        now = calendar.searchsorted(date, side="right") - 1
        if now < 0:
            raise ValueError("Valuation cutoff precedes observed calendar")
        total = self.cash + sum(x["amount"] for x in self.receivables.values())
        unresolved = copy.deepcopy(list(self.unresolved.values()))
        holdings = []
        missing = False
        for instrument, quantity in sorted(self.positions.items()):
            if not quantity:
                continue
            mark = marks.get(instrument)
            if mark is None:
                unresolved.append({"instrument": instrument, "reason": "missing_raw_mark"})
                missing = True
                continue
            if not isinstance(mark, PriceMark):
                raise ValueError("Marks must be raw PriceMark values")
            observed = _time(mark.observed_at)
            if observed > asof:
                raise ValueError("Price mark is from after the valuation cutoff")
            observed_date = observed.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            position = calendar.get_indexer([observed_date])[0]
            if position < 0:
                raise ValueError("Mark date is outside the observed calendar")
            age = int(now - position)
            value = quantity * mark.price
            total += value
            holdings.append({"instrument": instrument, "quantity": quantity, "raw_price": mark.price,
                             "value": value, "staleness_sessions": age, "tradable": self.can_trade(instrument, asof)})
            if age > maximum_staleness_sessions:
                unresolved.append({"instrument": instrument, "reason": "stale_mark", "indicative_value": value,
                                   "staleness_sessions": age})
        return {"asof": asof.isoformat(), "currency": self.currency, "quantity_units": "raw shares",
                "cash": self.cash, "receivables": sum(x["amount"] for x in self.receivables.values()),
                "nav": total if not unresolved else None, "indicative_nav": None if missing else total,
                "promotion_ready": not unresolved, "unresolved": unresolved, "holdings": holdings}

    def reconcile(self) -> dict:
        """Recompute cash and shares from the immutable event deltas."""
        expected_cash = self.initial_cash + sum(float(x.get("cash_delta", 0)) for x in self.receipts)
        if not math.isclose(expected_cash, self.cash, rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError("Cash ledger does not reconcile")
        expected_positions = self.initial_positions.copy()
        for receipt in self.receipts:
            for instrument, change in receipt.get("quantity_deltas", {}).items():
                expected_positions[instrument] = expected_positions.get(instrument, 0.0) + change
        for instrument in set(expected_positions) | set(self.positions):
            if not math.isclose(expected_positions.get(instrument, 0.0), self.positions.get(instrument, 0.0), rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("Share ledger does not reconcile")
        state = {"cash": self.cash, "positions": self.positions, "receivables": self.receivables,
                 "unresolved": self.unresolved, "receipts": self.receipts}
        digest = hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {"cash_reconciled": True, "shares_reconciled": True,
                "expected_cash": expected_cash, "cash": self.cash,
                "state_sha256": digest, "unresolved_events": len(self.unresolved), "receipts": len(self.receipts)}


def load_corporate_actions(path: str | Path) -> list[CorporateAction]:
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != 1 or not isinstance(payload.get("events"), list):
        raise ValueError("Unsupported corporate-action catalog")
    actions = [CorporateAction(**row) for row in payload["events"]]
    if len({event.event_id for event in actions}) != len(actions):
        raise ValueError("Corporate-action event IDs must be unique")
    return sorted(actions, key=lambda event: (_time(event.effective_at), event.event_id))


def assess_archived_lifecycles(run_dir, provider, event_catalog, output) -> dict:
    """Assess a retained Qlib ledger against sourced successor entitlements.

    The source positions and benchmark return are never rewritten. This is a
    separate P0 qualification receipt, not a corrected strategy replay. A real
    replay needs dated tradability, complete corporate-action coverage, and
    account-level settlement/rounding evidence before successor orders are run.
    """
    from financial_ts.workflow import _provider_fingerprint

    run_dir, provider, output = (Path(p).resolve() for p in (run_dir, provider, output))
    if output.exists():
        raise ValueError("Lifecycle assessment output must be a new directory")
    if output == run_dir or run_dir in output.parents or provider in output.parents:
        raise ValueError("Assessment must not be written inside an archived run or provider")
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("provider") != _provider_fingerprint(provider):
        raise ValueError("Assessment provider differs from the archived run")
    positions_path = run_dir / "positions.json.gz"
    with gzip.open(positions_path, "rt") as stream:
        archive = json.load(stream)
    snapshots = archive["positions"]
    if not snapshots:
        raise ValueError("Archived positions are empty")
    calendar = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines()))
    actions = load_corporate_actions(event_catalog)
    final = snapshots[-1]
    final_positions = {x["instrument"]: x for x in final["holdings"]}
    entries = []

    def binary_at(instrument, field, at):
        values = np.fromfile(provider / "features" / instrument.lower() / f"{field}.day.bin", dtype="<f4")
        if len(values) < 2 or not np.isfinite(values[0]) or values[0] != int(values[0]):
            raise ValueError("Malformed provider binary")
        offset = int(values[0])
        end = int(calendar.searchsorted(pd.Timestamp(at), side="right")) - offset
        candidates = values[1:1 + max(0, end)]
        valid = np.flatnonzero(np.isfinite(candidates))
        if not len(valid):
            raise ValueError("No observed source factor/price before the entitlement date")
        location = int(valid[-1])
        return float(candidates[location]), str(calendar[offset + location].date())

    for event in actions:
        symbol = event.instrument
        if symbol not in final_positions or _time(event.effective_at).date() > pd.Timestamp(final["datetime"]).date():
            continue
        before = [row for row in snapshots if pd.Timestamp(row["datetime"]).date() < _time(event.effective_at).date()]
        if not before:
            entries.append({"instrument": symbol, "unresolved_reasons": ["no_pre_event_position_snapshot"]})
            continue
        snapshot = before[-1]
        holding = next((x for x in snapshot["holdings"] if x["instrument"] == symbol), None)
        if holding is None:
            entries.append({"instrument": symbol, "unresolved_reasons": ["not_held_before_event_but_held_in_archived_final_positions"]})
            continue
        factor, factor_date = binary_at(symbol, "factor", snapshot["datetime"])
        price, price_date = binary_at(symbol, "close", snapshot["datetime"])
        converted = qlib_to_raw_position(holding["amount"], price, factor)
        qty = converted["raw_quantity"]
        # Merely display numerical closeness, never silently change the quantity
        # used by the archived account or invent a registrar allocation.
        entitlement = qty * event.ratio
        reasons = ["complete_corporate_action_replay_not_executed", "account_allocation_not_verified"]
        if abs(qty - round(qty)) > 1e-6:
            reasons.append("implied_raw_quantity_is_not_an_integer")
        if event.tradable_at is None:
            reasons.append("successor_tradability_not_registered")
        entries.append({
            "instrument": symbol, "event": asdict(event), "pre_event_snapshot": snapshot["datetime"],
            "source_factor_date": factor_date, "source_close_date": price_date,
            "adjusted_quantity": holding["amount"], **converted,
            "quantity_qualification": "Implied contemporaneous Qlib quote units; original executed raw shares and intervening action cashflows are not reconstructed by this multiplication.",
            "raw_quantity_distance_to_integer": abs(qty - round(qty)),
            "theoretical_successor_quantity": entitlement,
            "whole_share_entitlement_bounds": [math.floor(entitlement), math.ceil(entitlement)],
            "archived_final_stale_value": final_positions[symbol]["amount"] * final_positions[symbol]["price"],
            "archived_final_stale_weight": final_positions[symbol]["weight"],
            "unresolved_reasons": reasons,
        })
    result = {
        "campaign_kind": "lifecycle_source_assessment_v1", "status": "research_only",
        "promotion_ready": False, "corrected_returns_available": False,
        "archived_final_date": final["datetime"], "archived_final_nav": final["account"],
        "source_positions_sha256": hashlib.sha256(positions_path.read_bytes()).hexdigest(),
        "source_event_catalog_sha256": hashlib.sha256(Path(event_catalog).read_bytes()).hexdigest(),
        "provider_fingerprint": run["provider"], "events": entries,
        "affected_final_weight": sum(row.get("archived_final_stale_weight", 0) for row in entries),
        "interpretation": "Successor entitlements are sourced; the archived stale marks are not executable settlement values. No historical universe exclusion or last-price liquidation has been applied.",
    }
    raw = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True)
    (output / "assessment.json").write_text(raw)
    return result
