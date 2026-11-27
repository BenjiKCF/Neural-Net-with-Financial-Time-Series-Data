"""Public accounting commands retain incomplete evidence and delayed settlements."""

import json
from pathlib import Path

import pytest

from financial_ts.cli import main
from financial_ts.execution import run_paper_session


def _allocation_spec():
    spec = json.loads(
        (Path(__file__).parents[1] / "configs/paper_session_demo.json").read_text()
    )
    spec["campaign_id"] = "synthetic-late-allocation-acceptance"
    spec["initial"] = {"cash": 100, "positions": {"OLD": 101}, "currency": "CNY"}
    spec["observed_calendar"].append("2025-01-07")
    allocation = {
        "type": "resolve_allocation",
        "event_id": "merger",
        "allocated_quantity": 63,
        "source": "https://example.org/synthetic-account-allocation",
        "asof": "2025-01-07T16:00:00+08:00",
    }
    spec["operations"] = [
        {
            "type": "corporate_action",
            "asof": "2025-01-03T09:00:00+08:00",
            "event": {
                "event_id": "merger",
                "kind": "merger",
                "instrument": "OLD",
                "successor": "NEW",
                "ratio": 0.62,
                "cash_per_share": 1,
                "known_at": "2025-01-02T18:00:00+08:00",
                "effective_at": "2025-01-03T09:00:00+08:00",
                "payment_at": "2025-01-06T12:00:00+08:00",
                "tradable_at": "2025-01-06T09:30:00+08:00",
                "source": "https://example.org/synthetic-merger-notice",
            },
        },
        allocation,
        dict(allocation),
        {"type": "settle_cash", "asof": "2025-01-07T16:00:00+08:00"},
        {
            "type": "valuation",
            "asof": "2025-01-07T17:00:00+08:00",
            "marks": {
                "NEW": {
                    "price": 10,
                    "observed_at": "2025-01-07T15:00:00+08:00",
                    "source": "https://example.org/synthetic-raw-price",
                }
            },
        },
    ]
    return spec


def test_paper_session_accepts_late_account_allocation_and_reuses_it(tmp_path):
    spec = tmp_path / "paper.json"
    spec.write_text(json.dumps(_allocation_spec()))
    output = tmp_path / "replay"
    receipt = run_paper_session(spec, output)
    assert receipt["final_valuation"]["nav"] == 831
    ledger = json.loads((output / "ledger.json").read_text())
    assert ledger["cash"] == 201
    assert ledger["raw_positions"] == {"NEW": 63}
    assert not ledger["receivables"] and not ledger["unresolved"]
    assert receipt["deployment_approved"] is False
    assert run_paper_session(spec, output) == receipt


@pytest.mark.parametrize("change", [{"allocated_quantity": -1}, {"unknown": True}])
def test_paper_allocation_input_fails_before_registration(tmp_path, change):
    value = _allocation_spec()
    value["operations"][1].update(change)
    spec = tmp_path / "paper.json"
    spec.write_text(json.dumps(value))
    output = tmp_path / "replay"
    with pytest.raises(ValueError):
        run_paper_session(spec, output)
    assert not output.exists()


def test_account_settlement_commands_fail_closed_for_missing_account_evidence(tmp_path, capsys):
    root = Path(__file__).parents[1]
    spec = root / "configs/account_settlement_archived.json"
    output = tmp_path / "settlement"
    assert main(["account-settlement", "--spec", str(spec), "--output", str(output)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert result["archived_backtest_repaired"] is False
    assert main(["audit-account-settlement", "--output", str(output)]) == 1
    audited = json.loads(capsys.readouterr().out)
    assert audited["status"] == "blocked"
