"""Future registration, actual-time capture, lineage and first-vintage safeguards."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from test_bundles import metadata
from test_operations import FixtureBaseline

import financial_ts.prospective as prospective
from financial_ts.bundles import create_bundle
from financial_ts.operations import BatchPolicy, run_daily_batch


@pytest.fixture
def clock(monkeypatch):
    value = [pd.Timestamp("2026-11-27T09:00:00Z")]
    monkeypatch.setattr(prospective, "_now", lambda: value[0])
    return value


def spec():
    return {
        "name": "synthetic-prospective-contract-test", "evaluation_role": "confirmation",
        "timezone": "Asia/Shanghai",
        "calendar": {
            "source": "synthetic exchange calendar; test only",
            "sessions": ["2026-11-30", "2026-12-01", "2026-12-02", "2026-12-03"],
        },
        "first_session": "2026-11-30", "last_session": "2026-12-01",
        "decision_time": "18:00:00", "label_available_time": "16:30:00",
        "incident_owner": "Test operator", "selection_reason": "Fixed software comparator pair",
        "source_revision": "test-source", "data_contract": {
            "source": "synthetic", "adjustment_convention": "raw",
            "availability_policy": "observed receipt by cutoff",
        },
        "policy": {"min_cross_section": 3},
        "acceptance": {
            "minimum_sessions": 2, "maximum_publish_lag_minutes": 30,
            "minimum_success_fraction": 1.0, "maximum_failed_attempts": 0,
            "minimum_label_coverage": 1.0, "minimum_rank_ic": 0,
            "minimum_paired_rank_ic_gain": 0, "block_lengths": [1],
            "bootstrap_draws": 100, "bootstrap_seed": 913, "minimum_effective_blocks": 2,
        },
    }


def bundle(tmp_path, role, direction=1):
    model = FixtureBaseline()
    model.coefficient_ = np.array([0.1, -0.1]) * direction
    values = np.array([[1., 3.], [2., 2.], [3., 1.]])
    path = tmp_path / role
    create_bundle(
        model, path, feature_names=["x", "y"], preprocessing={"kind": "identity"},
        metadata=metadata(universe=["AAA", "BBB", "CCC"]), fixture_features=values,
        fixture_scores=values @ model.coefficient_,
    )
    return path


def setup(tmp_path, clock, recipe=None):
    candidate, reference = bundle(tmp_path, "candidate"), bundle(tmp_path, "reference", -1)
    store = tmp_path / "prospective"
    protocol = prospective.register_prospective(
        store, recipe or spec(), candidate_bundle=candidate, reference_bundle=reference,
    )
    return store, protocol


def forecasts(store, date, *, fingerprint="snapshot-A", extra_instrument=False):
    packet = {
        "decision_date": date, "decision_cutoff": f"{date}T18:00:00+08:00",
        "data_available_at": f"{date}T16:30:00+08:00", "observed_through": date,
        "expected_session": date, "instruments": ["AAA", "BBB", "CCC"],
        "expected_instruments": ["AAA", "BBB", "CCC"], "feature_names": ["x", "y"],
        "data_fingerprint": fingerprint, "source_revision": "prospective-arrival",
    }
    paths = {}
    for role in ("candidate", "reference"):
        inputs = copy.deepcopy(packet)
        if extra_instrument and role == "reference":
            inputs["instruments"][-1] = inputs["expected_instruments"][-1] = "DDD"
        run_daily_batch(
            store / "registry", role, store / "daily" / role, packet=inputs,
            features=np.array([[1., 3.], [2., 2.], [3., 1.]]),
            policy=BatchPolicy(min_cross_section=3),
        )
        paths[f"{role}_artifact"] = store / "daily" / role / date
    arrival = {
        "expected_session": date, "decision_cutoff": packet["decision_cutoff"],
        "data_available_at": packet["data_available_at"],
        "observed_at": f"{date}T16:35:00+08:00", "provider_fingerprint": fingerprint,
        "snapshot_audit_status": "admitted", "source": "synthetic arrival receipt",
    }
    return {**paths, "arrival_receipt": arrival}


def labels(date, *, missing=False):
    endpoints = {"2026-11-30": "2026-12-02", "2026-12-01": "2026-12-03"}
    return pd.DataFrame({
        "datetime": [date] * 3, "instrument": ["AAA", "BBB", "CCC"],
        "label": [-0.02, 0, np.nan if missing else 0.02],
        "available_at": [f"{endpoints[date]}T16:30:00+08:00"] * 3,
    })


def test_registration_is_future_immutable_and_bundle_pinned(tmp_path, clock):
    store, first = setup(tmp_path, clock)
    assert not first["production_ready"] and not first["reused"]
    second = prospective.register_prospective(
        store, spec(), candidate_bundle=tmp_path / "candidate", reference_bundle=tmp_path / "reference",
    )
    assert second["protocol_id"] == first["protocol_id"] and second["reused"]
    revised = spec()
    revised["acceptance"]["minimum_rank_ic"] = -0.1
    with pytest.raises(ValueError, match="cannot be changed"):
        prospective.register_prospective(
            store, revised, candidate_bundle=tmp_path / "candidate", reference_bundle=tmp_path / "reference",
        )
    status = prospective.evaluate_prospective(store)
    assert status["status"] == "pending" and len(status["future_sessions"]) == 2


def test_retrospective_registration_cannot_be_backdated(tmp_path, clock):
    clock[0] = pd.Timestamp("2026-11-30T10:01:00Z")
    with pytest.raises(ValueError, match="Registration must precede"):
        setup(tmp_path, clock)


@pytest.mark.parametrize("now", ["2026-11-30T09:59:00Z", "2026-11-30T10:31:00Z", "2026-12-01T10:10:00Z"])
def test_historical_or_early_capture_rejected_despite_receipt_timestamps(tmp_path, clock, now):
    store, _ = setup(tmp_path, clock)
    arguments = forecasts(store, "2026-11-30")
    clock[0] = pd.Timestamp(now)
    with pytest.raises(ValueError, match="outside the prospective"):
        prospective.record_observation(store, **arguments)
    assert not (store / "observations/2026-11-30").exists()
    assert len(list((store / "incidents").glob("*.json"))) == 1


def test_identical_retry_does_not_add_day_or_retimestamp(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    arguments = forecasts(store, "2026-11-30")
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    first = prospective.record_observation(store, **arguments)
    clock[0] = pd.Timestamp("2026-12-01T11:00:00Z")
    second = prospective.record_observation(store, **arguments)
    assert second["reused"] and first["receipt_id"] == second["receipt_id"]
    status = prospective.evaluate_prospective(store)
    assert status["operational_acceptance"]["captured_sessions"] == 1
    assert status["missing_sessions"] == ["2026-12-01"]
    assert status["operational_acceptance"]["status"] == "failed"


def test_paired_panels_and_arrival_must_match(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    arguments = forecasts(store, "2026-11-30", extra_instrument=True)
    with pytest.raises(ValueError, match="identical eligible panel"):
        prospective.record_observation(store, **arguments)


@pytest.mark.parametrize("fault", ["unadmitted", "future_observed", "wrong_provider", "wrong_cutoff"])
def test_bad_arrival_receipts_fail_closed(tmp_path, clock, fault):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    arguments = forecasts(store, "2026-11-30")
    arrival = arguments["arrival_receipt"]
    if fault == "unadmitted":
        arrival["snapshot_audit_status"] = "quarantined"
    elif fault == "future_observed":
        arrival["observed_at"] = "2026-11-30T18:20:00+08:00"
    elif fault == "wrong_provider":
        arrival["provider_fingerprint"] = "other-provider"
    else:
        arrival["decision_cutoff"] = "2026-11-30T18:01:00+08:00"
    with pytest.raises(ValueError):
        prospective.record_observation(store, **arguments)


def test_tampered_capture_or_protocol_cannot_supply_evidence(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    prospective.record_observation(store, **forecasts(store, "2026-11-30"))
    path = store / "observations/2026-11-30/candidate/predictions.csv"
    path.write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        prospective.evaluate_prospective(store)


def test_mature_labels_require_calendar_endpoint_and_freeze_first_vintage(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    prospective.record_observation(store, **forecasts(store, "2026-11-30"))
    false_availability = labels("2026-11-30")
    false_availability["available_at"] = "2026-11-30T16:30:00+08:00"
    with pytest.raises(ValueError, match="not matured"):
        prospective.record_mature_labels(store, session="2026-11-30", labels=false_availability, source="fixture")
    clock[0] = pd.Timestamp("2026-12-02T09:00:00Z")
    with pytest.raises(ValueError, match="not matured"):
        prospective.record_mature_labels(store, session="2026-11-30", labels=false_availability, source="fixture")
    first = prospective.record_mature_labels(store, session="2026-11-30", labels=labels("2026-11-30"), source="fixture")
    second = prospective.record_mature_labels(store, session="2026-11-30", labels=labels("2026-11-30"), source="fixture")
    assert second["reused"] and first["receipt_id"] == second["receipt_id"]
    revised = labels("2026-11-30")
    revised.loc[0, "label"] = 0.9
    with pytest.raises(ValueError, match="vintage is frozen"):
        prospective.record_mature_labels(store, session="2026-11-30", labels=revised, source="fixture")


def test_complete_fixed_window_can_pass_forecasts_but_never_deployment(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    for date in ("2026-11-30", "2026-12-01"):
        clock[0] = pd.Timestamp(f"{date}T10:05:00Z")
        prospective.record_observation(store, **forecasts(store, date))
    status = prospective.evaluate_prospective(store)
    assert status["status"] == "pending"
    clock[0] = pd.Timestamp("2026-12-03T09:00:00Z")
    for date in ("2026-11-30", "2026-12-01"):
        prospective.record_mature_labels(store, session=date, labels=labels(date), source="fixture")
    status = prospective.evaluate_prospective(store)
    assert status["operational_acceptance"]["status"] == "passed"
    assert status["candidate_quality"]["status"] == "passed"
    assert status["status"] == "passed_for_review"
    assert not status["production_ready"] and not status["orders_submitted"]


def test_missing_outcomes_are_not_dropped_from_label_coverage(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    for date in ("2026-11-30", "2026-12-01"):
        clock[0] = pd.Timestamp(f"{date}T10:05:00Z")
        prospective.record_observation(store, **forecasts(store, date))
    clock[0] = pd.Timestamp("2026-12-03T09:00:00Z")
    for date in ("2026-11-30", "2026-12-01"):
        prospective.record_mature_labels(store, session=date, labels=labels(date, missing=True), source="fixture")
    status = prospective.evaluate_prospective(store)
    assert status["candidate_quality"]["label_coverage"] == pytest.approx(2 / 3)
    assert status["candidate_quality"]["status"] == "failed"
    assert status["status"] == "failed"


def test_protocol_tampering_is_detected(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    path = store / "protocol.json"
    changed = json.loads(path.read_text())
    changed["specification"]["acceptance"]["minimum_rank_ic"] = -1
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="protocol integrity"):
        prospective.evaluate_prospective(store)


@pytest.mark.parametrize("change", [
    {"incident_owner": "REPLACE_WITH_OPERATOR"},
    {"last_session": "2026-12-03"},
    {"acceptance": {"minimum_sessions": 0}},
])
def test_incomplete_or_unusable_preregistration_rejected(tmp_path, clock, change):
    recipe = spec()
    recipe.update(change)
    with pytest.raises(ValueError):
        setup(tmp_path, clock, recipe)


def test_operations_pilot_cannot_be_promoted_to_confirmation(tmp_path, clock):
    recipe = spec()
    recipe["evaluation_role"] = "operations_pilot"
    store, _ = setup(tmp_path, clock, recipe)
    for date in ("2026-11-30", "2026-12-01"):
        clock[0] = pd.Timestamp(f"{date}T10:05:00Z")
        prospective.record_observation(store, **forecasts(store, date))
    clock[0] = pd.Timestamp("2026-12-03T09:00:00Z")
    for date in ("2026-11-30", "2026-12-01"):
        prospective.record_mature_labels(store, session=date, labels=labels(date), source="fixture")
    status = prospective.evaluate_prospective(store)
    assert status["status"] == "operations_pilot_complete"
    assert status["operational_acceptance"]["status"] == "passed"
    assert status["candidate_quality"]["status"] == "research_only"
    assert not status["production_ready"]


@pytest.mark.parametrize("key,value", [
    ("minimum_success_fraction", 0), ("minimum_label_coverage", 0),
    ("minimum_rank_ic", float("nan")), ("maximum_publish_lag_minutes", 0),
    ("minimum_sessions", True), ("minimum_effective_blocks", 0),
])
def test_acceptance_cannot_be_disabled_by_zero_nan_or_boolean(tmp_path, clock, key, value):
    recipe = spec()
    recipe["acceptance"][key] = value
    with pytest.raises((ValueError, TypeError)):
        setup(tmp_path, clock, recipe)


def test_close_label_cannot_be_declared_available_before_market_close(tmp_path, clock):
    recipe = spec()
    recipe["label_available_time"] = "00:00:00"
    with pytest.raises(ValueError, match="exchange close"):
        setup(tmp_path, clock, recipe)


def test_capture_crossing_deadline_is_rejected_after_copy(tmp_path, clock, monkeypatch):
    store, _ = setup(tmp_path, clock)
    arguments = forecasts(store, "2026-11-30")
    times = iter([
        pd.Timestamp("2026-11-30T10:29:59Z"),
        pd.Timestamp("2026-11-30T10:30:01Z"),
        pd.Timestamp("2026-11-30T10:30:02Z"),
    ])
    monkeypatch.setattr(prospective, "_now", lambda: next(times))
    with pytest.raises(ValueError, match="Durable capture exceeded"):
        prospective.record_observation(store, **arguments)
    assert not (store / "observations/2026-11-30").exists()


def test_nonfinite_telemetry_cannot_bypass_service_objectives(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    arguments = forecasts(store, "2026-11-30")
    path = arguments["candidate_artifact"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["runtime_seconds"] = float("nan")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="must be finite"):
        prospective.record_observation(store, **arguments)


@pytest.mark.parametrize("slow_operation", ["rename", "receipt"])
def test_slow_final_publication_cannot_cross_deadline_and_count(tmp_path, clock, monkeypatch, slow_operation):
    store, _ = setup(tmp_path, clock)
    arguments = forecasts(store, "2026-11-30")
    clock[0] = pd.Timestamp("2026-11-30T10:29:59Z")
    destination = store / "observations/2026-11-30"
    original_rename = prospective.os.rename
    original_atomic = prospective.atomic_json

    def slow_rename(source, target):
        result = original_rename(source, target)
        if slow_operation == "rename" and target == destination:
            clock[0] = pd.Timestamp("2026-11-30T10:30:01Z")
        return result

    def slow_receipt(path, value):
        result = original_atomic(path, value)
        if slow_operation == "receipt" and path == destination / "receipt.json":
            clock[0] = pd.Timestamp("2026-11-30T10:30:01Z")
        return result

    monkeypatch.setattr(prospective.os, "rename", slow_rename)
    monkeypatch.setattr(prospective, "atomic_json", slow_receipt)
    with pytest.raises(ValueError, match="publication completed after"):
        prospective.record_observation(store, **arguments)
    assert not destination.exists()
    rejected = list((store / "rejected_observations").iterdir())
    assert len(rejected) == 1
    receipt = json.loads((rejected[0] / "receipt.json").read_text())
    assert receipt["publication_status"] == "late"
    status = prospective.evaluate_prospective(store)
    assert status["operational_acceptance"]["captured_sessions"] == 0
    assert status["operational_acceptance"]["failed_attempts"] == 1
    assert status["missing_sessions"] == ["2026-11-30"]
