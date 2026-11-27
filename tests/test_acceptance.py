"""Incomplete, unrelated and corrupted evidence cannot close roadmap gates."""

from financial_ts.acceptance import assess_roadmap_acceptance


def test_no_evidence_reports_every_gate_open():
    result = assess_roadmap_acceptance()
    assert result["status"] == "open"
    assert len(result["gates"]) == 4
    assert all(gate["status"] == "not_started" for gate in result["gates"].values())
    assert result["production_ready"] is False
    assert result["orders_submitted"] is False


def test_missing_evidence_stores_do_not_hide_other_gates(tmp_path):
    result = assess_roadmap_acceptance(
        campaign=tmp_path / "missing-campaign",
        account_settlement=tmp_path / "missing-settlement",
        prospective=tmp_path / "missing-prospective",
    )
    assert result["status"] == "open"
    assert all(gate["status"] == "blocked" for gate in result["gates"].values())
    assert all(gate["reasons"][0].startswith("evidence_invalid:")
               for gate in result["gates"].values())


def test_passing_account_fixture_cannot_repair_archived_portfolio(tmp_path, monkeypatch):
    monkeypatch.setattr("financial_ts.settlement.verify_account_settlement", lambda _: {
        "status": "passed", "scope": "supplied_account_window",
        "evidence_kind": "synthetic_fixture", "blockers": [],
        "archived_backtest_repaired": False,
    })
    result = assess_roadmap_acceptance(account_settlement=tmp_path)
    gate = result["gates"]["account_level_settlement"]
    assert gate["supplied_account_window_status"] == "passed"
    assert gate["status"] == "blocked"
    assert "actual_account_records_not_verified" in gate["reasons"]
    assert "complete_linked_archived_account_replay_not_verified" in gate["reasons"]


def test_operational_pilot_cannot_close_quality_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr("financial_ts.prospective.evaluate_prospective", lambda _: {
        "candidate_quality": {"status": "research_only", "reasons": ["operations_pilot"]},
        "operational_acceptance": {"status": "passed", "reasons": []},
        "production_ready": False,
    })
    result = assess_roadmap_acceptance(prospective=tmp_path)
    assert result["gates"]["prospective_shadow"]["forecast_delivery_status"] == "passed"
    assert result["gates"]["prospective_shadow"]["status"] == "blocked"
    assert "deployment_fault_rehearsals_not_verified" in result["gates"]["prospective_shadow"]["reasons"]
    assert result["gates"]["uninspected_confirmation"]["status"] == "research_only"
    assert result["status"] == "open"
    assert result["production_ready"] is False
