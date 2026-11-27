"""Dated instrument rules and an explicit daily-bar paper execution model.

Actual broker orders are never submitted. Liquidity and impact are assumptions
registered in a scenario, not inferred capacity claims. Missing dated rules
reject an order; the engine does not infer a board or price limit from a ticker.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from filelock import FileLock

from financial_ts.lifecycle import CorporateAction, EventLedger, PriceMark, _number, _source, _time


@dataclass(frozen=True)
class TradingRule:
    instrument: str
    valid_from: str
    valid_until: str
    known_at: str
    source: str
    buy_minimum: int
    quantity_step: int
    maximum_order: int
    price_tick: float
    limit_up: float | None
    limit_down: float | None
    allow_odd_lot_liquidation: bool = True
    sell_delay_sessions: int = 1

    def __post_init__(self):
        _source(self.source)
        _time(self.known_at)
        if not self.instrument or pd.Timestamp(self.valid_from) > pd.Timestamp(self.valid_until):
            raise ValueError("Dated instrument rule has an invalid interval")
        if any(str(pd.Timestamp(value).date()) != value for value in (self.valid_from, self.valid_until)):
            raise ValueError("Rule dates must use canonical YYYY-MM-DD")
        for name in ("buy_minimum", "quantity_step", "maximum_order"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_order < self.buy_minimum:
            raise ValueError("maximum_order is below the minimum")
        if self.buy_minimum % self.quantity_step:
            raise ValueError("Minimum quantity must align to the declared step")
        _number(self.price_tick, "price_tick", positive=True)
        for value in (self.limit_up, self.limit_down):
            if value is not None and (not math.isfinite(value) or not 0 < value < 1):
                raise ValueError("Price limits must be in (0, 1) or explicitly disabled")
        if self.sell_delay_sessions not in {0, 1}:
            raise ValueError("Daily engine supports explicit T+0 or T+1 settlement only")


@dataclass(frozen=True)
class ExecutionScenario:
    name: str
    fee_source: str
    commission_rate: float
    minimum_commission: float
    sell_tax_rate: float
    participation_limit: float
    slippage_bps: float = 0.0
    impact_bps_at_full_participation: float = 0.0
    fee_multiplier: float = 1.0
    slippage_multiplier: float = 1.0

    def __post_init__(self):
        if not self.name or not self.fee_source:
            raise ValueError("Scenario identity and sourced/explicitly assumed fees are required")
        for name in ("commission_rate", "minimum_commission", "sell_tax_rate", "slippage_bps",
                     "impact_bps_at_full_participation", "fee_multiplier", "slippage_multiplier"):
            _number(getattr(self, name), name)
        if not math.isfinite(self.participation_limit) or not 0 < self.participation_limit <= 1:
            raise ValueError("participation_limit must be in (0, 1]")


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    instrument: str
    side: str
    quantity: int
    submitted_at: str

    def __post_init__(self):
        if not self.order_id or not self.instrument or self.side not in {"buy", "sell"}:
            raise ValueError("Order identifier, instrument, and valid side are required")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("Requested raw share quantity must be a positive integer")
        _time(self.submitted_at)


class PaperExecutor:
    """Execute predetermined orders against observed raw daily bars and one ledger."""

    def __init__(self, ledger: EventLedger, rules: list[TradingRule], scenario: ExecutionScenario):
        self.ledger, self.rules, self.scenario = ledger, list(rules), scenario
        self._volume: dict[tuple[str, str], int] = {}
        self._bought: dict[tuple[str, str], int] = {}
        self._orders: dict[str, dict] = {}
        self._bars: dict[tuple[str, str], tuple] = {}

    def execute(self, order: PaperOrder, *, at, raw_price: float, raw_volume: int,
                reference_price: float, suspended=False, quote_source: str) -> dict:
        """Return a reproducible fill/rejection and update cash only for actual fills.

        ``reference_price`` is the exchange's corporate-action-adjusted limit
        reference in raw-price units, not blindly yesterday's close. Limit-lock
        rejection is conservative daily-bar behavior; intraday queue fills are
        unavailable. Same-day bought shares are excluded from T+1 sales.
        """
        at = _time(at)
        _source(quote_source)
        _number(raw_price, "raw_price", positive=True)
        _number(reference_price, "reference_price", positive=True)
        if isinstance(raw_volume, bool) or not isinstance(raw_volume, int) or raw_volume < 0:
            raise ValueError("Observed raw-share volume must be a nonnegative integer")
        if at <= _time(order.submitted_at):
            raise ValueError("Execution must follow order submission")
        day = str(at.tz_convert("Asia/Shanghai").date())
        key = (day, order.instrument)
        bar_identity = (raw_price, raw_volume, reference_price, suspended, quote_source)
        if key in self._bars and self._bars[key] != bar_identity:
            raise ValueError("A daily bar cannot change between same-session orders")
        request = {"order": asdict(order), "at": at.isoformat(), "raw_price": raw_price,
                   "raw_volume": raw_volume, "reference_price": reference_price,
                   "suspended": suspended, "quote_source": quote_source}
        if order.order_id in self._orders:
            prior = self._orders[order.order_id]
            if prior["request"] != request:
                raise ValueError("Order ID was reused with different execution inputs")
            return copy.deepcopy(prior)
        matching = [r for r in self.rules if r.instrument == order.instrument
                    and r.valid_from <= day <= r.valid_until and _time(r.known_at) <= at]
        receipt = {"request": request, "scenario": asdict(self.scenario), "status": "rejected",
                   "filled_quantity": 0, "requested_quantity": order.quantity,
                   "fees": 0.0, "cash_delta": 0.0}

        def reject(reason):
            receipt["reason"] = reason
            self._bars[key] = bar_identity
            self._orders[order.order_id] = copy.deepcopy(receipt)
            return receipt

        if len(matching) != 1:
            return reject("missing_or_ambiguous_dated_rule")
        rule = matching[0]
        receipt["rule"] = asdict(rule)
        if suspended or not raw_volume or not self.ledger.can_trade(order.instrument, at):
            return reject("suspended_untradeable_or_no_volume")
        if order.quantity > rule.maximum_order:
            return reject("order_exceeds_exchange_maximum")
        direction = 1 if order.side == "buy" else -1
        # Round exchange limits to a price tick using positive half-up rounding.
        upper = None if rule.limit_up is None else math.floor(reference_price * (1 + rule.limit_up) / rule.price_tick + 0.5) * rule.price_tick
        lower = None if rule.limit_down is None else math.floor(reference_price * (1 - rule.limit_down) / rule.price_tick + 0.5) * rule.price_tick
        if order.side == "buy" and upper is not None and raw_price >= upper - 1e-10:
            return reject("buy_at_upper_limit")
        if order.side == "sell" and lower is not None and raw_price <= lower + 1e-10:
            return reject("sell_at_lower_limit")
        held = self.ledger.positions.get(order.instrument, 0)
        settled = held - (self._bought.get(key, 0) if rule.sell_delay_sessions else 0)
        if order.side == "sell" and order.quantity > settled + 1e-10:
            return reject("insufficient_settled_raw_shares")
        odd_liquidation = order.side == "sell" and rule.allow_odd_lot_liquidation and order.quantity == held
        if not odd_liquidation and (order.quantity < rule.buy_minimum or (order.quantity - rule.buy_minimum) % rule.quantity_step):
            return reject("invalid_order_lot")
        remaining = max(0, math.floor(raw_volume * self.scenario.participation_limit) - self._volume.get(key, 0))
        quantity = min(order.quantity, remaining)
        if not (odd_liquidation and quantity == order.quantity):
            quantity = (quantity // rule.quantity_step) * rule.quantity_step
            if quantity < rule.buy_minimum:
                return reject("participation_below_minimum_lot")

        def economics(qty):
            participation = qty / raw_volume
            impact = self.scenario.impact_bps_at_full_participation * math.sqrt(participation)
            bps = self.scenario.slippage_multiplier * self.scenario.slippage_bps + impact
            price = raw_price * (1 + direction * bps / 10_000)
            ticks = price / rule.price_tick
            price = (math.ceil(ticks - 1e-10) if direction == 1 else math.floor(ticks + 1e-10)) * rule.price_tick
            if not math.isfinite(price) or price <= 0:
                raise ValueError("Scenario produced an invalid fill price")
            amount = qty * price
            commission = max(self.scenario.minimum_commission, amount * self.scenario.commission_rate)
            tax = amount * self.scenario.sell_tax_rate if order.side == "sell" else 0
            return price, self.scenario.fee_multiplier * (commission + tax)

        price, fees = economics(quantity)
        if order.side == "buy":
            # Find the largest affordable permitted lot; impact and minimum fees
            # are recomputed at each size, rather than overdraw then repair cash.
            low, high = 0, quantity // rule.quantity_step
            while low < high:
                middle = (low + high + 1) // 2
                candidate = middle * rule.quantity_step
                p, f = economics(candidate)
                if candidate * p + f <= self.ledger.cash + 1e-10:
                    low = middle
                else:
                    high = middle - 1
            quantity = low * rule.quantity_step
            if quantity < rule.buy_minimum:
                return reject("insufficient_cash_for_minimum_lot")
            price, fees = economics(quantity)
        if (upper is not None and price > upper + 1e-10) or (lower is not None and price < lower - 1e-10):
            return reject("scenario_fill_outside_exchange_limits")
        before = self.ledger.cash
        self.ledger.record_fill(order.order_id, order.instrument, order.side, quantity, price, fees, asof=at)
        self._volume[key] = self._volume.get(key, 0) + quantity
        if order.side == "buy":
            self._bought[key] = self._bought.get(key, 0) + quantity
        receipt.update(status="filled" if quantity == order.quantity else "partial", reason=None,
                       filled_quantity=quantity, fill_price=price, fees=fees,
                       cash_delta=self.ledger.cash - before, participation=quantity / raw_volume,
                       cumulative_daily_participation=self._volume[key] / raw_volume)
        self._orders[order.order_id] = copy.deepcopy(receipt)
        self._bars[key] = bar_identity
        return receipt


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_atomic(path: Path, raw: bytes):
    descriptor, temporary = tempfile.mkstemp(prefix=".paper-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_paper_session(spec_path: str | Path, output: str | Path) -> dict:
    """Replay a registered paper session with durable, idempotent receipts.

    Version-1 specs contain ``initial``, ``rules``, ``scenario``,
    ``observed_calendar``, and chronological ``operations``. An operation is a
    corporate_action (event/asof), order (order/execution), settle_cash (asof),
    or valuation (marks/asof). Every replay starts from the registered initial
    account; an interruption cannot double-submit external orders because this
    adapter never has any external execution side effect. A completed receipt
    is returned only after all artifacts are atomically written and hashed.
    """
    from financial_ts.data_contracts import _calendar

    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate paper specification key: {key}")
            result[key] = value
        return result

    spec = json.loads(spec_path.read_text(), object_pairs_hook=unique)
    required = {"version", "campaign_id", "initial", "rules", "scenario", "observed_calendar", "operations"}
    if set(spec) != required or spec["version"] != 1 or not isinstance(spec["campaign_id"], str) or not spec["campaign_id"]:
        raise ValueError("Paper spec requires exactly the version-1 registered fields")
    calendar = _calendar(pd.to_datetime(spec["observed_calendar"]))
    if not isinstance(spec["operations"], list) or not spec["operations"]:
        raise ValueError("Paper session operations must be a nonempty ordered list")
    rules = [TradingRule(**rule) for rule in spec["rules"]]
    scenario = ExecutionScenario(**spec["scenario"])
    EventLedger(**spec["initial"])
    prior = None
    # Validate each operation before creating the registration artifact. The
    # exact declared ordering is significant when operations share a timestamp.
    for operation in spec["operations"]:
        kind = operation.get("type")
        if kind == "order":
            if set(operation) != {"type", "order", "execution"}:
                raise ValueError("Malformed paper order operation")
            PaperOrder(**operation["order"])
            at = _time(operation["execution"]["at"])
        elif kind == "corporate_action":
            if set(operation) != {"type", "event", "asof"}:
                raise ValueError("Malformed corporate-action operation")
            CorporateAction(**operation["event"])
            at = _time(operation["asof"])
        elif kind == "settle_cash":
            if set(operation) != {"type", "asof"}:
                raise ValueError("Malformed cash-settlement operation")
            at = _time(operation["asof"])
        elif kind == "resolve_allocation":
            required_allocation = {"type", "event_id", "allocated_quantity", "source", "asof"}
            if (
                not required_allocation.issubset(operation)
                or set(operation) - required_allocation - {"tradable_at"}
                or not isinstance(operation["event_id"], str)
                or not operation["event_id"]
            ):
                raise ValueError("Malformed account-allocation operation")
            _number(operation["allocated_quantity"], "allocated_quantity")
            _source(operation["source"])
            at = _time(operation["asof"])
            if operation.get("tradable_at") is not None:
                _time(operation["tradable_at"])
        elif kind == "valuation":
            if set(operation) - {"type", "asof", "marks", "maximum_staleness_sessions"} or "marks" not in operation:
                raise ValueError("Malformed valuation operation")
            for mark in operation["marks"].values():
                PriceMark(**mark)
            at = _time(operation["asof"])
        else:
            raise ValueError("Unsupported paper operation type")
        if prior is not None and at < prior:
            raise ValueError("Registered operations must be chronological")
        prior = at
    if spec["operations"][-1]["type"] != "valuation":
        raise ValueError("A paper session must end with an explicit valuation")
    raw_spec = _json_bytes(spec)
    identity = hashlib.sha256(raw_spec).hexdigest()
    code_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ("execution.py", "lifecycle.py", "data_contracts.py")}
    protocol = {"version": 1, "spec_sha256": identity, "spec": spec,
                "implementation_sha256": code_hashes,
                "runtime": {"python": platform.python_version(), "pandas": pd.__version__}}
    raw_protocol = _json_bytes(protocol)
    if output == spec_path or output in spec_path.parents:
        raise ValueError("Paper output cannot contain the source specification")
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output.parent / ("." + output.name + ".paper.lock")), timeout=30):
        output.mkdir(exist_ok=True)
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if protocol_path.read_bytes() != raw_protocol:
                raise ValueError("Paper output already belongs to different spec, code or runtime")
        elif any(output.iterdir()):
            raise ValueError("Paper output contains unregistered files")
        else:
            _write_atomic(protocol_path, raw_protocol)
        complete = output / "completion.json"
        if complete.exists():
            receipt = json.loads(complete.read_text())
            if receipt.get("spec_sha256") != identity:
                raise ValueError("Paper completion spec identity mismatch")
            expected_artifacts = {"operations.json", "valuations.json", "ledger.json"}
            if set(receipt.get("artifacts", {})) != expected_artifacts:
                raise ValueError("Paper completion has an invalid artifact inventory")
            for name, digest in receipt["artifacts"].items():
                if not (output / name).is_file() or hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                    raise ValueError("A completed paper artifact failed its integrity check")
            if (receipt.get("protocol_sha256") != hashlib.sha256(raw_protocol).hexdigest()
                    or receipt.get("final_valuation") != json.loads((output / "valuations.json").read_text())[-1]
                    or receipt.get("reconciliation") != json.loads((output / "ledger.json").read_text())["reconciliation"]
                    or receipt.get("status") != "completed" or receipt.get("deployment_approved") is not False):
                raise ValueError("Paper completion metadata does not match its immutable artifacts")
            return receipt
        ledger = EventLedger(**spec["initial"])
        executor = PaperExecutor(ledger, rules, scenario)
        operation_receipts, valuations = [], []
        try:
            for operation in spec["operations"]:
                kind = operation["type"]
                if kind == "corporate_action":
                    receipt = ledger.apply(CorporateAction(**operation["event"]), asof=operation["asof"])
                elif kind == "order":
                    receipt = executor.execute(PaperOrder(**operation["order"]), **operation["execution"])
                elif kind == "settle_cash":
                    receipt = ledger.settle_cash(asof=operation["asof"])
                elif kind == "resolve_allocation":
                    receipt = ledger.resolve_allocation(
                        **{key: value for key, value in operation.items() if key != "type"}
                    )
                else:
                    receipt = ledger.mark({symbol: PriceMark(**mark) for symbol, mark in operation["marks"].items()},
                                          asof=operation["asof"], observed_calendar=calendar,
                                          maximum_staleness_sessions=operation.get("maximum_staleness_sessions", 0))
                    valuations.append(receipt)
                operation_receipts.append({"ordinal": len(operation_receipts), "type": kind, "receipt": receipt})
            reconciliation = ledger.reconcile()
        except Exception as exc:
            failures = output / "failures"
            failures.mkdir(exist_ok=True)
            number = len(list(failures.glob("attempt-*.json"))) + 1
            failure = {"status": "failed", "spec_sha256": identity,
                       "completed_operations": operation_receipts,
                       "error_type": type(exc).__name__, "error": str(exc)}
            _write_atomic(failures / f"attempt-{number:06d}.json", _json_bytes(failure))
            raise
        artifacts = {
            "operations.json": operation_receipts,
            "valuations.json": valuations,
            "ledger.json": {"currency": ledger.currency, "cash": ledger.cash,
                            "raw_positions": ledger.positions, "receivables": ledger.receivables,
                            "restrictions": ledger.restrictions, "unresolved": ledger.unresolved,
                            "receipts": ledger.receipts, "reconciliation": reconciliation},
        }
        hashes = {}
        for name, value in artifacts.items():
            raw = _json_bytes(value)
            _write_atomic(output / name, raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        fills = [x["receipt"] for x in operation_receipts if x["type"] == "order"]
        receipt = {"status": "completed", "mode": "paper_replay_only", "spec_sha256": identity,
                   "campaign_id": spec["campaign_id"], "artifacts": hashes,
                   "protocol_sha256": hashlib.sha256(raw_protocol).hexdigest(),
                   "orders": len(fills), "filled": sum(x["status"] == "filled" for x in fills),
                   "partial": sum(x["status"] == "partial" for x in fills),
                   "rejected": sum(x["status"] == "rejected" for x in fills),
                   "final_valuation": valuations[-1], "reconciliation": reconciliation,
                   "deployment_approved": False,
                   "qualification": "Registered paper simulation; economic edge, prospective arrival and deployment approval are not established."}
        _write_atomic(complete, _json_bytes(receipt))
        return receipt
