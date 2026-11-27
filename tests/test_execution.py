"""Partial fills, liquidity, lots, limits, settlement and independent cost stress."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from financial_ts.execution import (
    ExecutionScenario,
    PaperExecutor,
    PaperOrder,
    TradingRule,
    run_paper_session,
)
from financial_ts.lifecycle import EventLedger

SOURCE = "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250612_10824490.shtml"
QUOTE = "https://example.org/synthetic-raw-bar"
RULE = TradingRule("A", "2025-01-01", "2025-12-31", "2024-12-31T18:00:00+08:00", SOURCE,
                   100, 100, 1000000, 0.01, 0.1, 0.1)
SCENARIO = ExecutionScenario("fixture", "Explicit synthetic fee assumption", 0.001, 1, 0.001, 0.1)
SUBMITTED = "2025-01-02T16:00:00+08:00"
EXECUTION = "2025-01-03T15:00:00+08:00"


def order(identity="one", side="buy", quantity=1000):
    return PaperOrder(identity, "A", side, quantity, SUBMITTED)


def execute(engine, requested=None, **kwargs):
    values = dict(at=EXECUTION, raw_price=10., raw_volume=5000, reference_price=10., quote_source=QUOTE)
    values.update(kwargs)
    return engine.execute(requested or order(), **values)


def test_participation_partial_fill_is_cumulative_and_retry_does_not_double_spend():
    ledger = EventLedger(100000)
    engine = PaperExecutor(ledger, [RULE], SCENARIO)
    receipt = execute(engine)
    assert receipt["status"] == "partial" and receipt["filled_quantity"] == 500
    assert ledger.cash == 94995 and ledger.positions == {"A": 500}
    assert execute(engine) == receipt and ledger.cash == 94995
    assert execute(engine, order("two"))["reason"] == "participation_below_minimum_lot"
    assert ledger.reconcile()["cash_reconciled"]


def test_cash_affordability_preserves_minimum_commission():
    ledger = EventLedger(2001.5)
    engine = PaperExecutor(ledger, [RULE], SCENARIO)
    result = execute(engine)
    assert result["filled_quantity"] == 100
    assert ledger.cash == pytest.approx(1000.5)


def test_limit_lock_rejects_without_inventing_a_sale_or_cash():
    ledger = EventLedger(1000, {"A": 1000})
    engine = PaperExecutor(ledger, [RULE], SCENARIO)
    assert execute(engine, order(side="sell"), raw_price=9)["reason"] == "sell_at_lower_limit"
    assert ledger.cash == 1000 and ledger.positions == {"A": 1000}
    engine = PaperExecutor(EventLedger(100000), [RULE], SCENARIO)
    assert execute(engine, raw_price=11)["reason"] == "buy_at_upper_limit"


def test_odd_lot_liquidation_and_same_day_settlement():
    ledger = EventLedger(10000, {"A": 75})
    engine = PaperExecutor(ledger, [RULE], SCENARIO)
    result = execute(engine, order(side="sell", quantity=75))
    assert result["filled_quantity"] == 75 and not ledger.positions
    ledger = EventLedger(10000)
    engine = PaperExecutor(ledger, [RULE], SCENARIO)
    execute(engine, order(quantity=100))
    result = execute(engine, order("sell-bought", "sell", 100))
    assert result["reason"] == "insufficient_settled_raw_shares"


def test_unknown_or_ambiguous_dated_rules_fail_closed():
    for rules in ([], [RULE, RULE], [replace(RULE, valid_until="2025-01-02")],
                  [replace(RULE, known_at="2025-01-04T00:00:00+08:00")]):
        engine = PaperExecutor(EventLedger(10000), rules, SCENARIO)
        assert execute(engine)["reason"] == "missing_or_ambiguous_dated_rule"


def test_invalid_lots_and_order_ids_are_rejected():
    engine = PaperExecutor(EventLedger(10000), [RULE], SCENARIO)
    assert execute(engine, order(quantity=150))["reason"] == "invalid_order_lot"
    with pytest.raises(ValueError, match="Order ID"):
        execute(engine, order(quantity=200))
    with pytest.raises(ValueError, match="daily bar"):
        execute(engine, order("other"), raw_volume=6000)


def test_fee_and_slippage_stresses_are_independent_and_tick_aligned():
    results = []
    for scenario in (replace(SCENARIO, slippage_bps=5),
                     replace(SCENARIO, slippage_bps=5, fee_multiplier=2),
                     replace(SCENARIO, slippage_bps=5, slippage_multiplier=4)):
        engine = PaperExecutor(EventLedger(100000), [RULE], scenario)
        results.append(execute(engine))
    assert results[1]["fill_price"] == results[0]["fill_price"]
    assert results[1]["fees"] == 2 * results[0]["fees"]
    assert results[2]["fill_price"] > results[0]["fill_price"]
    assert results[0]["fill_price"] == pytest.approx(10.01)


def test_suspension_and_zero_volume_are_observable_rejections():
    for kwargs in ({"suspended": True}, {"raw_volume": 0}):
        engine = PaperExecutor(EventLedger(100000), [RULE], SCENARIO)
        assert execute(engine, **kwargs)["reason"] == "suspended_untradeable_or_no_volume"


def demo_spec():
    return Path(__file__).parents[1] / "configs/paper_session_demo.json"


def test_durable_session_reconciles_hand_cash_and_is_idempotent(tmp_path):
    output = tmp_path / "paper"
    receipt = run_paper_session(demo_spec(), output)
    assert receipt["partial"] == 1 and receipt["filled"] == 1
    # Buy 500 at 10.01 + 5 fee; sell 500 at 9.79 minus 5 commission
    # and 4.895 sell tax; add 500 * .2 dividend.
    assert receipt["final_valuation"]["cash"] == pytest.approx(9975.105)
    assert receipt["final_valuation"]["nav"] == pytest.approx(9975.105)
    assert receipt["reconciliation"]["shares_reconciled"]
    assert not receipt["deployment_approved"]
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert run_paper_session(demo_spec(), output) == receipt
    assert before == {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}


def test_durable_replay_resumes_after_artifact_interruption_without_double_fills(tmp_path, monkeypatch):
    import financial_ts.execution as module

    original = module._write_atomic
    def fail_on_ledger(path, raw):
        if path.name == "ledger.json":
            raise OSError("simulated storage interruption")
        return original(path, raw)
    output = tmp_path / "paper"
    monkeypatch.setattr(module, "_write_atomic", fail_on_ledger)
    with pytest.raises(OSError, match="interruption"):
        run_paper_session(demo_spec(), output)
    assert (output / "protocol.json").is_file() and not (output / "completion.json").exists()
    monkeypatch.setattr(module, "_write_atomic", original)
    receipt = run_paper_session(demo_spec(), output)
    assert receipt["final_valuation"]["cash"] == pytest.approx(9975.105)
    assert receipt["orders"] == 2


def test_durable_replay_rejects_conflicts_tamper_and_retains_failed_attempts(tmp_path):
    output = tmp_path / "paper"
    run_paper_session(demo_spec(), output)
    spec = json.loads(demo_spec().read_text())
    spec["initial"]["cash"] += 1
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="different spec"):
        run_paper_session(changed, output)
    (output / "ledger.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        run_paper_session(demo_spec(), output)
    spec["operations"][0]["execution"]["raw_price"] = -1
    changed.write_text(json.dumps(spec))
    failed = tmp_path / "failed"
    for attempt in range(2):
        with pytest.raises(ValueError):
            run_paper_session(changed, failed)
        assert len(list((failed / "failures").glob("*.json"))) == attempt + 1
    assert not (failed / "completion.json").exists()
