"""Evidence-derived status of the four remaining roadmap acceptance gates.

The report rechecks each supplied evidence store. A completion checkbox, a
successful fixture, or an old JSON summary cannot close a gate on its own.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _missing(reason):
    return {"status": "not_started", "reasons": [reason]}


def _invalid(error):
    return {"status": "blocked", "reasons": [f"evidence_invalid: {type(error).__name__}: {error}"]}


def assess_roadmap_acceptance(*, campaign=None, account_settlement=None, prospective=None):
    """Read and independently revalidate available evidence; never promote a model.

    Account-window reconciliation is useful on its own. The archived portfolio
    gate additionally needs a complete linked replay, which an unrelated account
    statement or a synthetic fixture does not establish.
    """
    gates = {
        "five_seed_campaign": _missing("full_campaign_not_registered"),
        "uninspected_confirmation": _missing("confirmation_protocol_not_registered"),
        "account_level_settlement": _missing("account_settlement_evidence_not_supplied"),
        "prospective_shadow": _missing("prospective_shadow_not_registered"),
    }
    evidence = {}
    if campaign is not None:
        from financial_ts.campaign_control import campaign_status

        try:
            result = campaign_status(campaign, verify_artifacts=True)
            evidence["campaign"] = {key: value for key, value in result.items() if key != "trials"}
            reasons = []
            if not result["matches_full_campaign_recipe"]:
                reasons.append("campaign_does_not_match_the_registered_full_budget")
            if result["verified_completed_trials"] != result["expected_trials"]:
                reasons.append("registered_trials_remain_incomplete")
            gates["five_seed_campaign"] = {
                "status": result["full_campaign_execution_gate"],
                "reasons": reasons,
                "expected_trials": result["expected_trials"],
                "verified_completed_trials": result["verified_completed_trials"],
                "counts": result["counts"],
                "qualification": "Execution completion does not establish an economic advantage.",
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            gates["five_seed_campaign"] = _invalid(error)
    if account_settlement is not None:
        from financial_ts.settlement import verify_account_settlement

        try:
            result = verify_account_settlement(account_settlement)
            evidence["account_settlement"] = {
                key: value for key, value in result.items() if key != "operations"
            }
            reasons = list(result.get("blockers", []))
            if result.get("evidence_kind") != "account_records":
                reasons.append("actual_account_records_not_verified")
            if result.get("archived_backtest_repaired") is not True:
                reasons.append("complete_linked_archived_account_replay_not_verified")
            gates["account_level_settlement"] = {
                "status": "passed" if result["status"] == "passed" and not reasons else "blocked",
                "reasons": sorted(set(reasons)),
                "supplied_account_window_status": result["status"],
                "scope": result.get("scope"),
                "qualification": "An account-window check cannot silently repair the archived campaign.",
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            gates["account_level_settlement"] = _invalid(error)
    if prospective is not None:
        from financial_ts.prospective import evaluate_prospective

        try:
            result = evaluate_prospective(prospective)
            evidence["prospective"] = result
            quality, operations = result["candidate_quality"], result["operational_acceptance"]
            gates["uninspected_confirmation"] = dict(quality)
            # This store proves forecast capture. The roadmap also requires
            # deployment fault rehearsals and prospective portfolio/execution
            # reconciliation; historical fixtures cannot stand in for those.
            gates["prospective_shadow"] = {
                **operations,
                "forecast_delivery_status": operations["status"],
                "status": "pending" if operations["status"] == "pending" else "blocked",
                "reasons": sorted(set(operations.get("reasons", []) + [
                    "deployment_fault_rehearsals_not_verified",
                    "prospective_execution_reconciliation_not_verified",
                ])),
                "scope": "prospective_shadow_operation_including_execution_and_recovery",
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            gates["uninspected_confirmation"] = _invalid(error)
            gates["prospective_shadow"] = _invalid(error)
    return {
        "schema_version": 1,
        "assessed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "passed_for_review" if all(
            gate["status"] == "passed" for gate in gates.values()
        ) else "open",
        "gates": gates,
        "evidence": evidence,
        "stores": {
            name: str(Path(value).expanduser().resolve()) if value is not None else None
            for name, value in (
                ("campaign", campaign), ("account_settlement", account_settlement),
                ("prospective", prospective),
            )
        },
        "production_ready": False,
        "orders_submitted": False,
        "qualification": "A gate report is not the recorded deployment decision or usage-rights approval.",
    }
