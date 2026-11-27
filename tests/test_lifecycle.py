"""Hand-computed raw-share fixtures, sourcing, cutoff and unresolved-state tests."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from financial_ts.lifecycle import (
    CorporateAction,
    EventLedger,
    PriceMark,
    load_corporate_actions,
    qlib_to_raw_position,
)

SOURCE = "https://example.org/synthetic-event-fixture"
DAY1 = "2025-01-02T16:00:00+08:00"
DAY2 = "2025-01-03T16:00:00+08:00"
DAY3 = "2025-01-06T16:00:00+08:00"
CALENDAR = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])


def action(kind, **kwargs):
    return CorporateAction("event", kind, "A", DAY2, DAY1, SOURCE, **kwargs)


def test_split_dividend_sale_settlement_hand_ledger_and_quantity_reconciliation():
    ledger = EventLedger(1000, {"A": 100})
    ledger.apply(action("split", ratio=2), asof=DAY2)
    dividend = replace(action("cash_dividend", cash_per_share=0.25, payment_at=DAY3), event_id="dividend")
    ledger.apply(dividend, asof=DAY2)
    assert ledger.cash == 1000 and ledger.positions == {"A": 200}
    assert ledger.receivables["dividend"]["amount"] == 50
    ledger.record_fill("sell", "A", "sell", 100, 6, 5, asof=DAY2)
    assert ledger.cash == 1595
    nav = ledger.mark({"A": PriceMark(6, DAY2, SOURCE)}, asof=DAY2, observed_calendar=CALENDAR)
    assert nav["nav"] == 2245  # 1595 cash + 600 stock + 50 entitlement
    assert ledger.settle_cash(asof=DAY2) == []
    ledger.settle_cash(asof=DAY3)
    assert ledger.cash == 1645 and not ledger.receivables
    assert ledger.reconcile()["shares_reconciled"]
    ledger.positions["A"] += 1
    with pytest.raises(ValueError, match="Share ledger"):
        ledger.reconcile()


def test_adjusted_factor_conversion_preserves_nav_and_rejects_double_adjustment():
    raw = qlib_to_raw_position(50, 20, 2)
    assert raw == {"raw_quantity": 100, "raw_price": 10, "source_factor": 2, "value": 1000}
    with pytest.raises(ValueError, match="raw price"):
        PriceMark(20, DAY1, SOURCE, price_convention="adjusted")
    with pytest.raises(ValueError):
        qlib_to_raw_position(50, 20, 0)


def test_sourced_merger_ratios_use_final_notice_not_earlier_proposal():
    catalog = Path(__file__).parents[1] / "configs/lifecycle_events.json"
    events = {x.instrument: x for x in load_corporate_actions(catalog)}
    assert events["SH601989"].ratio == 0.1339
    assert events["SH600837"].ratio == 0.62
    assert events["SH601059"].ratio == 0.521
    for instrument, event in events.items():
        ledger = EventLedger(0, {instrument: 10000})
        receipt = ledger.apply(event, asof=event.effective_at)
        assert receipt["successor_quantity"] == 10000 * event.ratio
        assert ledger.positions == {event.successor: receipt["successor_quantity"]}
        assert not ledger.can_trade(event.successor, event.effective_at)
        assert ledger.reconcile()["shares_reconciled"]


def test_fractional_conversion_preserves_unresolved_old_shares_until_actual_allocation():
    event = action("merger", successor="B", ratio=0.1339, tradable_at=DAY3)
    ledger = EventLedger(0, {"A": 100})
    receipt = ledger.apply(event, asof=DAY2)
    assert receipt["status"] == "unresolved" and ledger.positions == {"A": 100}
    marked = ledger.mark({"A": PriceMark(5, DAY2, SOURCE)}, asof=DAY2, observed_calendar=CALENDAR)
    assert marked["nav"] is None and marked["indicative_nav"] == 500
    with pytest.raises(ValueError, match="restricted"):
        ledger.record_fill("bad", "A", "sell", 100, 5, 0, asof=DAY2)
    ledger.resolve_allocation("event", allocated_quantity=13, source=SOURCE, asof=DAY3)
    assert ledger.positions == {"B": 13} and not ledger.unresolved
    assert ledger.can_trade("B", DAY3)


def test_unknown_delisting_is_not_written_off_or_sold_at_last_quote():
    ledger = EventLedger(50, {"A": 100})
    ledger.apply(action("delisting"), asof=DAY2)
    assert ledger.positions == {"A": 100} and ledger.cash == 50
    marked = ledger.mark({"A": PriceMark(5, DAY1, SOURCE)}, asof=DAY2, observed_calendar=CALENDAR)
    assert marked["nav"] is None and marked["indicative_nav"] == 550
    assert {x["reason"] for x in marked["unresolved"]} == {"terminal_value_unknown", "stale_mark"}


def test_cash_delisting_creates_receivable_then_settles_once():
    ledger = EventLedger(50, {"A": 100})
    event = action("delisting", cash_per_share=3, payment_at=DAY3)
    ledger.apply(event, asof=DAY2)
    ledger.apply(event, asof=DAY2)
    assert not ledger.positions and ledger.cash == 50
    assert ledger.mark({}, asof=DAY2, observed_calendar=CALENDAR)["nav"] == 350
    ledger.settle_cash(asof=DAY3)
    ledger.settle_cash(asof=DAY3)
    assert ledger.cash == 350 and ledger.reconcile()["cash_reconciled"]


def test_events_and_fills_are_causal_and_idempotent():
    ledger = EventLedger(1000, {"A": 100})
    event = action("split", ratio=2)
    with pytest.raises(ValueError, match="not yet"):
        ledger.apply(event, asof=DAY1)
    assert ledger.positions == {"A": 100}
    ledger.apply(event, asof=DAY2)
    with pytest.raises(ValueError, match="cannot change"):
        ledger.apply(replace(event, ratio=3), asof=DAY2)
    ledger.record_fill("buy", "A", "buy", 10, 5, 1, asof=DAY3)
    ledger.record_fill("buy", "A", "buy", 10, 5, 1, asof=DAY3)
    assert ledger.cash == 949
    with pytest.raises(ValueError, match="chronological"):
        ledger.record_fill("old", "A", "buy", 10, 5, 1, asof=DAY2)


def test_ticker_change_preserves_suspension_and_suspension_never_changes_nav():
    ledger = EventLedger(100, {"A": 100})
    ledger.apply(action("suspension"), asof=DAY2)
    rename = replace(action("ticker_change", successor="B"), event_id="rename")
    ledger.apply(rename, asof=DAY2)
    assert ledger.positions == {"B": 100} and not ledger.can_trade("B", DAY2)
    resumed = replace(action("resumption"), event_id="resume", instrument="B", effective_at=DAY3)
    ledger.apply(resumed, asof=DAY3)
    assert ledger.can_trade("B", DAY3)


def test_stale_and_future_marks_cannot_publish_a_usable_nav():
    ledger = EventLedger(100, {"A": 10})
    missing = ledger.mark({}, asof=DAY2, observed_calendar=CALENDAR)
    assert missing["nav"] is None and missing["indicative_nav"] is None
    with pytest.raises(ValueError, match="after the valuation"):
        ledger.mark({"A": PriceMark(5, DAY3, SOURCE)}, asof=DAY2, observed_calendar=CALENDAR)


@pytest.mark.parametrize("cutoff", [
    "2025-01-07T16:00:00+08:00",
    "2025-06-02T16:00:00+08:00",
    "2025-01-06T17:00:00Z",  # Already 7 January in the exchange timezone.
])
def test_valuation_rejects_calendar_ending_before_exchange_cutoff(cutoff):
    ledger = EventLedger(100, {"A": 10})
    with pytest.raises(ValueError, match="calendar does not cover the valuation cutoff"):
        ledger.mark(
            {"A": PriceMark(5, DAY3, SOURCE)},
            asof=cutoff,
            observed_calendar=CALENDAR,
        )


def test_weekend_valuation_requires_following_session_calendar_coverage():
    ledger = EventLedger(100, {"A": 10})
    marks = {"A": PriceMark(5, DAY2, SOURCE)}
    cutoff = "2025-01-05T16:00:00+08:00"
    # A calendar ending on Friday cannot establish that the next two days
    # contain no sessions. Monday's entry makes that interval explicit.
    with pytest.raises(ValueError, match="calendar does not cover the valuation cutoff"):
        ledger.mark(marks, asof=cutoff, observed_calendar=CALENDAR[:2])
    valuation = ledger.mark(marks, asof=cutoff, observed_calendar=CALENDAR)
    assert valuation["nav"] == 150
    assert valuation["promotion_ready"]
    assert valuation["holdings"][0]["staleness_sessions"] == 0


@pytest.mark.parametrize("cutoff", [
    DAY3,
    "2025-01-07T00:30:00+09:00",  # Still 6 January in the exchange timezone.
])
def test_valuation_accepts_calendar_ending_on_exchange_cutoff(cutoff):
    ledger = EventLedger(100, {"A": 10})
    valuation = ledger.mark(
        {"A": PriceMark(5, DAY3, SOURCE)}, asof=cutoff, observed_calendar=CALENDAR
    )
    assert valuation["nav"] == 150
    assert valuation["promotion_ready"]
    assert valuation["holdings"][0]["staleness_sessions"] == 0


def test_retroactive_entitlement_requires_replay_instead_of_using_current_holdings():
    ledger = EventLedger(1000, {"A": 100})
    ledger.record_fill("sell", "A", "sell", 50, 5, 0, asof=DAY3)
    with pytest.raises(ValueError, match="Retroactive"):
        ledger.apply(action("cash_dividend", cash_per_share=1, payment_at=DAY3), asof=DAY3)
    assert not ledger.receivables and ledger.positions == {"A": 50}


def test_reverse_split_requires_fractional_allocation_evidence():
    ledger = EventLedger(1000, {"A": 105})
    receipt = ledger.apply(action("split", ratio=0.1), asof=DAY2)
    assert receipt["status"] == "unresolved" and ledger.positions["A"] == 105
    ledger.resolve_allocation("event", allocated_quantity=10, source=SOURCE, asof=DAY3)
    assert ledger.positions == {"A": 10} and not ledger.unresolved
