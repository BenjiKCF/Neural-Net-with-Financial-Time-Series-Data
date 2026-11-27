"""Failure, restart, alias pinning, maturity and operational acceptance."""

import json

import numpy as np
import pandas as pd
import pytest
from test_bundles import metadata

from financial_ts.bundles import create_bundle, file_hash
from financial_ts.operations import (
    BatchPolicy,
    evaluate_promotion,
    monitor_mature_labels,
    read_prediction_artifact,
    register_bundle,
    resolve_alias,
    run_daily_batch,
    set_alias,
)


class FixtureBaseline:
    kind = "ridge"
    fitted_ = True
    coefficient_ = np.array([0.1, -0.1])
    intercept_ = 0.0


def setup_registry(tmp_path, *, intercept=0.0, name="bundle"):
    model = FixtureBaseline()
    model.intercept_ = intercept
    path = tmp_path / name
    create_bundle(
        model,
        path,
        feature_names=["x", "y"],
        preprocessing={"kind": "identity"},
        metadata=metadata(),
        fixture_features=np.array([[1.0, 2.0], [3.0, 4.0]]),
        fixture_scores=np.array([-0.1, -0.1]) + intercept,
    )
    registered = register_bundle(tmp_path / "registry", path)
    set_alias(tmp_path / "registry", "shadow", registered["version"])
    return registered["version"]


def packet():
    return {
        "decision_date": "2024-01-05",
        "decision_cutoff": "2024-01-05T18:00:00+08:00",
        "data_available_at": "2024-01-05T16:30:00+08:00",
        "observed_through": "2024-01-05",
        "expected_session": "2024-01-05",
        "instruments": ["BBB", "AAA"],
        "expected_instruments": ["AAA", "BBB"],
        "feature_names": ["x", "y"],
        "data_fingerprint": "synthetic-5",
        "source_revision": "fixture-5",
    }


def run(tmp_path, **kwargs):
    return run_daily_batch(
        tmp_path / "registry",
        "shadow",
        tmp_path / "daily",
        packet=kwargs.pop("packet", packet()),
        features=kwargs.pop("features", np.array([[1.0, 2.0], [3.0, 4.0]])),
        **kwargs,
    )


def test_duplicate_run_is_idempotent_and_tamper_rejected(tmp_path):
    version = setup_registry(tmp_path)
    first, duplicate = run(tmp_path), run(tmp_path)
    assert first["bundle_id"] == version
    assert not first["reused"] and duplicate["reused"]
    frame, manifest = read_prediction_artifact(tmp_path / "daily/2024-01-05")
    assert frame.instrument.tolist() == ["AAA", "BBB"]
    assert not manifest["orders_submitted"]
    with pytest.raises(ValueError, match="conflicting"):
        run(tmp_path, features=np.zeros((2, 2)))
    assert len(list((tmp_path / "daily/failures").glob("*.json"))) == 1
    (tmp_path / "daily/2024-01-05/predictions.csv").write_text("tampered")
    with pytest.raises(ValueError, match="invalid"):
        run(tmp_path)


@pytest.mark.parametrize("fault", ["missing", "tampered", "legacy", "metadata"])
def test_missing_or_changed_input_lineage_rejects_reads_and_duplicate_reuse(tmp_path, fault):
    setup_registry(tmp_path)
    run(tmp_path)
    path = tmp_path / "daily/2024-01-05"
    _, manifest = read_prediction_artifact(path)
    assert manifest["schema_version"] == 2
    assert manifest["input_manifest_sha256"] == file_hash(path / "input_manifest.json")
    if fault == "missing":
        (path / "input_manifest.json").unlink()
    elif fault == "tampered":
        changed = packet()
        changed["source_revision"] = "unregistered-revision"
        (path / "input_manifest.json").write_text(json.dumps(changed))
    else:
        if fault == "legacy":
            manifest["schema_version"] = 1
            manifest.pop("input_manifest_sha256")
        else:
            manifest["source_revision"] = "wrong-published-revision"
        (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="input lineage"):
        read_prediction_artifact(path)
    with pytest.raises(ValueError, match="input lineage"):
        run(tmp_path)


def test_prediction_instruments_must_match_bound_input_lineage(tmp_path):
    setup_registry(tmp_path)
    run(tmp_path)
    path = tmp_path / "daily/2024-01-05"
    frame, manifest = read_prediction_artifact(path)
    frame.loc[0, "instrument"] = "UNEXPECTED"
    frame.to_csv(path / "predictions.csv", index=False)
    manifest["predictions_sha256"] = file_hash(path / "predictions.csv")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rows differ"):
        read_prediction_artifact(path)


@pytest.mark.parametrize(
    "update,reason",
    [
        ({"data_available_at": "2024-01-05T19:00:00+08:00"}, "after"),
        ({"observed_through": "2024-01-04"}, "Stale"),
        ({"decision_cutoff": "2024-01-05T18:00:00"}, "timezone"),
        ({"expected_instruments": ["AAA", "BBB", "CCC"]}, "coverage"),
        ({"feature_names": ["y", "x"]}, "order"),
        ({"instruments": ["AAA", "AAA"]}, "distinct"),
        ({"data_fingerprint": ""}, "identity"),
    ],
)
def test_invalid_inputs_fail_closed_and_retry_can_recover(tmp_path, update, reason):
    setup_registry(tmp_path)
    with pytest.raises(ValueError, match=reason):
        run(tmp_path, packet={**packet(), **update})
    assert not (tmp_path / "daily/2024-01-05").exists()
    result = run(tmp_path)
    assert result["status"] == "ready"


def test_missingness_score_and_runtime_limits(tmp_path):
    setup_registry(tmp_path)
    with pytest.raises(ValueError, match="missingness"):
        run(tmp_path, features=np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="magnitude"):
        run(tmp_path, policy=BatchPolicy(max_abs_score=0.01))
    with pytest.raises(ValueError, match="runtime"):
        run(tmp_path, policy=BatchPolicy(max_runtime_seconds=1e-12))
    assert not (tmp_path / "daily/2024-01-05").exists()


def test_alias_pinned_per_batch_and_rollback(tmp_path):
    first = setup_registry(tmp_path)
    run(tmp_path)
    second = setup_registry(tmp_path, intercept=0.03, name="second")
    assert first != second
    assert resolve_alias(tmp_path / "registry", "shadow")[1] == second
    with pytest.raises(ValueError, match="conflicting"):
        run(tmp_path)
    receipt = set_alias(tmp_path / "registry", "shadow", first, rollback=True)
    assert receipt["previous_version"] == second
    assert run(tmp_path)["bundle_id"] == first
    assert run(tmp_path)["reused"]


def test_production_promotion_requires_every_evidence_gate_and_usage_rights(tmp_path):
    version = setup_registry(tmp_path)
    assessment = evaluate_promotion(tmp_path / "bundle", {"bundle_id": version})
    assert assessment["status"] == "blocked"
    assert any("usage rights" in item for item in assessment["reasons"])
    with pytest.raises(ValueError, match="Promotion blocked"):
        set_alias(tmp_path / "registry", "champion", version)
    state = json.loads((tmp_path / "registry/registry.json").read_text())
    assert "champion" not in state["aliases"]


def test_only_mature_labels_enter_delayed_monitoring():
    predicted = pd.DataFrame(
        {"datetime": ["2024-01-05"] * 3, "instrument": ["A", "B", "C"], "score": [1.0, 2.0, 3.0]}
    )
    labels = predicted.drop(columns="score").assign(
        label=[1.0, 2.0, 100.0],
        available_at=["2024-01-08T16:00:00+08:00"] * 2 + ["2024-01-09T16:00:00+08:00"],
    )
    waiting = monitor_mature_labels(predicted, labels, cutoff="2024-01-06T18:00:00+08:00")
    assert waiting["status"] == "waiting_for_labels"
    result = monitor_mature_labels(
        predicted, labels, cutoff="2024-01-08T18:00:00+08:00", minimum_rank_ic=0.1
    )
    assert result["mature_count"] == 2 and result["deferred_labels"] == 1
    assert result["metrics"]["rank_ic"] == pytest.approx(1.0)
    assert not result["automatic_retraining"] and not result["automatic_promotion"]


def test_tabular_padding_cannot_hide_an_absent_decision_row(tmp_path):
    setup_registry(tmp_path)
    with pytest.raises(ValueError, match="cannot be padded"):
        run(
            tmp_path,
            features=np.array([[np.nan, np.nan], [3.0, 4.0]]),
            padding=np.array([True, False]),
        )


@pytest.mark.parametrize("missing_count", [1, 3])
def test_missing_mature_labels_reduce_monitoring_coverage(missing_count):
    predicted = pd.DataFrame(
        {
            "datetime": ["2024-01-05"] * 3,
            "instrument": ["A", "B", "C"],
            "score": [0.1, 0.2, 0.3],
        }
    )
    labels = predicted.drop(columns="score").assign(
        label=[0.1, 0.2, 0.3], available_at="2024-01-08T16:00:00-05:00"
    )
    labels.loc[: missing_count - 1, "label"] = np.nan
    result = monitor_mature_labels(predicted, labels, cutoff="2024-01-08T22:00:00Z")
    assert result["status"] == "review_required"
    assert result["evaluated_count"] == 3 - missing_count
    assert result["unavailable_mature_count"] == missing_count
    assert result["coverage"] == pytest.approx((3 - missing_count) / 3)
    assert "label_coverage" in result["alerts"]


def test_monitor_rejects_naive_maturity_and_respects_overseas_offset():
    predicted = pd.DataFrame({"datetime": ["2024-01-05"], "instrument": ["A"], "score": [0.1]})
    labels = predicted.drop(columns="score").assign(label=0.1, available_at="2024-01-08T16:00:00")
    with pytest.raises(ValueError, match="available_at requires an explicit timezone"):
        monitor_mature_labels(predicted, labels, cutoff="2024-01-08T18:00:00Z")
    labels["available_at"] = "2024-01-08T16:00:00-05:00"
    result = monitor_mature_labels(predicted, labels, cutoff="2024-01-08T18:00:00Z")
    assert result["status"] == "waiting_for_labels"


def test_immutable_model_pinning_and_memory_bound(tmp_path):
    first = setup_registry(tmp_path)
    setup_registry(tmp_path, intercept=0.01, name="new")
    with pytest.raises(ValueError, match="Registry changed"):
        run(tmp_path, pinned_bundle_id=first)
    with pytest.raises(ValueError, match="resident memory"):
        run(tmp_path, policy=BatchPolicy(max_memory_mb=0.001))


def test_shadow_approval_cannot_be_used_for_production(tmp_path):
    model = FixtureBaseline()
    created = create_bundle(
        model,
        tmp_path / "permitted",
        feature_names=["x", "y"],
        preprocessing={"kind": "identity"},
        metadata=metadata(
            usage_rights={"model": "approved_for_intended_use", "data": "approved_for_intended_use"}
        ),
        fixture_features=np.array([[1.0, 2.0]]),
        fixture_scores=np.array([-0.1]),
    )
    gates = {
        key: {"status": "passed", "evidence_uri": "fixture://receipt"}
        for key in (
            "causal_data",
            "lifecycle_accounting",
            "candidate_quality",
            "prospective_shadow",
            "operations",
            "execution_reconciliation",
        )
    }
    evidence = {
        "bundle_id": created["bundle_id"],
        "gates": gates,
        "service_objectives": {"runtime": 600},
        "incident_owner": "test",
        "approval": {"decision": "approved", "reviewer": "test", "intended_use": "shadow"},
    }
    assert evaluate_promotion(tmp_path / "permitted", evidence)["status"] == "blocked"
    evidence["approval"]["intended_use"] = "production"
    assert evaluate_promotion(tmp_path / "permitted", evidence)["status"] == "passed"
