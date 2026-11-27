"""Historical software acceptance using completed numerical inference bundles.

Arrival timestamps in this replay are explicitly hypothetical. This verifies
software behavior against archived forecasts, not publication-time data access
or prospective operation. It never submits orders or promotes a champion.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import financial_ts.bundles as bundle_module
import financial_ts.operations as operations_module
from financial_ts.bundles import NumericalPredictor, file_hash, verify_bundle
from financial_ts.operations import (
    BatchPolicy,
    build_provider_packet,
    evaluate_promotion,
    read_prediction_artifact,
    register_bundle,
    resolve_alias,
    run_daily_batch,
    set_alias,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def accept(provider, primary_run, rollback_run, output, work_dir, *, tolerance=1e-6):
    paths = [
        Path(value).expanduser().resolve()
        for value in (provider, primary_run, rollback_run, output, work_dir)
    ]
    provider, primary_run, rollback_run, output, work_dir = paths
    if any(path.exists() and any(path.iterdir()) for path in (output, work_dir)):
        raise ValueError("Acceptance output/work directories must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not 0 < tolerance < 1:
        raise ValueError("Declare a positive absolute prediction tolerance")
    bundles = [path / "inference_bundle" for path in (primary_run, rollback_run)]
    manifests = [verify_bundle(bundle) for bundle in bundles]
    if manifests[0]["bundle_id"] == manifests[1]["bundle_id"]:
        raise ValueError("Rollback acceptance needs two distinct immutable bundle versions")
    for path in (primary_run, rollback_run):
        if json.loads((path / "run.json").read_text()).get("status") != "completed":
            raise ValueError("Only completed workflow runs qualify for acceptance")
    policy = BatchPolicy(
        max_runtime_seconds=600.0,
        max_memory_mb=4096.0,
        minimum_coverage=1.0,
        max_missing_feature_fraction=0.25,
        max_abs_score=10.0,
        max_data_age_hours=96.0,
        min_cross_section=2,
    )
    expected = pd.read_csv(primary_run / "predictions.csv", dtype={"instrument": str})
    date = str(expected["datetime"].min())
    expected = expected.loc[expected["datetime"] == date].set_index("instrument")["score"]
    protocol = {
        "schema_version": 1,
        "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "qualification": "historical software replay; hypothetical arrival times; not prospective or publication-time evidence",
        "primary_run": str(primary_run),
        "rollback_run": str(rollback_run),
        "primary_bundle_id": manifests[0]["bundle_id"],
        "secondary_bundle_id": manifests[1]["bundle_id"],
        "decision_date": date,
        "absolute_score_tolerance": tolerance,
        "policy": asdict(policy),
        "hypothetical_available_at": f"{date}T16:30:00+08:00",
        "hypothetical_decision_cutoff": f"{date}T18:00:00+08:00",
        "checks": [
            "raw_provider_forecast_parity",
            "registry_idempotency",
            "atomic_publication",
            "duplicate_idempotency",
            "stale_data_rejection",
            "late_data_rejection",
            "failure_retry",
            "conflicting_request_rejection",
            "immutable_alias_pinning",
            "compatible_rollback",
            "production_promotion_blocked",
        ],
        "runtime_sources": {
            str(Path(module.__file__).resolve()): file_hash(module.__file__)
            for module in (bundle_module, operations_module)
        },
    }
    write(output / "protocol.json", protocol)
    write(output / "status.json", {"status": "running"})
    checks = {}
    try:
        packet, features, padding = build_provider_packet(
            provider,
            bundles[0],
            decision_date=date,
            decision_cutoff=protocol["hypothetical_decision_cutoff"],
            data_available_at=protocol["hypothetical_available_at"],
        )
        packet["availability_qualification"] = protocol["qualification"]
        if set(packet["instruments"]) != set(expected.index):
            raise AssertionError("Provider eligibility differs from the archived forecast panel")
        actual = NumericalPredictor(bundles[0]).predict(
            features, padding=padding, feature_names=packet["feature_names"]
        )
        reference = expected.reindex(packet["instruments"]).to_numpy(dtype=float)
        error = np.abs(actual - reference)
        if not np.isfinite(error).all() or error.max() > tolerance:
            raise AssertionError(f"Raw-provider parity error {error.max()} exceeds {tolerance}")
        parity = pd.DataFrame(
            {
                "instrument": packet["instruments"],
                "archived_score": reference,
                "replayed_score": actual,
                "absolute_error": error,
            }
        )
        parity.to_csv(output / "raw_provider_parity.csv", index=False, float_format="%.17g")
        checks["raw_provider_forecast_parity"] = {
            "status": "passed",
            "count": len(error),
            "maximum_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
        }
        registry = work_dir / "registry"
        primary = register_bundle(registry, bundles[0])["version"]
        duplicate_registration = register_bundle(registry, bundles[0])["version"]
        if primary != duplicate_registration:
            raise AssertionError("Duplicate registration changed the immutable version")
        checks["registry_idempotency"] = {"status": "passed", "bundle_id": primary}
        set_alias(registry, "shadow", primary)
        run_kwargs = {
            "packet": packet,
            "features": features,
            "padding": padding,
            "policy": policy,
            "pinned_bundle_id": primary,
        }
        first = run_daily_batch(registry, "shadow", work_dir / "daily", **run_kwargs)
        duplicate = run_daily_batch(registry, "shadow", work_dir / "daily", **run_kwargs)
        published, manifest = read_prediction_artifact(work_dir / "daily" / date)
        if (
            first["reused"]
            or not duplicate["reused"]
            or first["request_id"] != duplicate["request_id"]
        ):
            raise AssertionError("Repeated batch is not idempotent")
        if manifest["orders_submitted"] or manifest["mode"] != "shadow_predictions":
            raise AssertionError("Acceptance must remain shadow predictions only")
        checks["atomic_publication"] = {"status": "passed", "manifest": manifest}
        checks["duplicate_idempotency"] = {"status": "passed", "request_id": first["request_id"]}
        published.to_csv(output / "published_predictions.csv", index=False, float_format="%.17g")
        write(output / "input_manifest.json", packet)
        write(output / "publication_manifest.json", manifest)

        def rejected(name, target, **overrides):
            kwargs = {**run_kwargs, **overrides}
            already_present = (target / date).exists()
            try:
                run_daily_batch(registry, "shadow", target, **kwargs)
            except ValueError as error:
                checks[name] = {
                    "status": "passed",
                    "observed_error": str(error),
                    "existing_dated_artifact_before_rejection": already_present,
                    "new_dated_artifact_published": (target / date).exists()
                    and not already_present,
                }
                return
            raise AssertionError(f"{name} unexpectedly accepted an invalid request")

        stale = copy.deepcopy(packet)
        stale["observed_through"] = str((pd.Timestamp(date) - pd.Timedelta(days=1)).date())
        rejected("stale_data_rejection", work_dir / "stale", packet=stale)
        if (work_dir / "stale" / date).exists():
            raise AssertionError("Stale data published a dated artifact")
        late = copy.deepcopy(packet)
        late["data_available_at"] = f"{date}T19:00:00+08:00"
        rejected("late_data_rejection", work_dir / "late", packet=late)
        if (work_dir / "late" / date).exists():
            raise AssertionError("Late data published a dated artifact")
        recovered = run_daily_batch(registry, "shadow", work_dir / "stale", **run_kwargs)
        checks["failure_retry"] = {
            "status": "passed",
            "request_id": recovered["request_id"],
            "same_as_initial_request": recovered["request_id"] == first["request_id"],
        }
        changed = features.copy()
        changed.flat[int(np.flatnonzero(np.isfinite(changed))[0])] += 0.0001
        original_digest = file_hash(work_dir / "daily" / date / "predictions.csv")
        rejected("conflicting_request_rejection", work_dir / "daily", features=changed)
        checks["conflicting_request_rejection"]["existing_artifact_unchanged"] = (
            file_hash(work_dir / "daily" / date / "predictions.csv") == original_digest
        )
        secondary = register_bundle(registry, bundles[1])["version"]
        set_alias(registry, "shadow", secondary)
        rejected("immutable_alias_pinning", work_dir / "alias_changed")
        restored = set_alias(registry, "shadow", primary, rollback=True)
        if resolve_alias(registry, "shadow")[1] != primary:
            raise AssertionError("Rollback did not restore the original immutable version")
        restored_batch = run_daily_batch(registry, "shadow", work_dir / "daily", **run_kwargs)
        if not restored_batch["reused"] or restored_batch["request_id"] != first["request_id"]:
            raise AssertionError("Rollback did not reproduce the archived request")
        checks["compatible_rollback"] = {
            "status": "passed",
            "receipt": restored,
            "reused_original_artifact": True,
        }
        gate = evaluate_promotion(bundles[0], {"bundle_id": primary}, intended_use="production")
        if gate["status"] != "blocked":
            raise AssertionError("Unqualified candidate unexpectedly passed production promotion")
        checks["production_promotion_blocked"] = {"status": "passed", "assessment": gate}
        state = json.loads((registry / "registry.json").read_text())
        if "champion" in state["aliases"]:
            raise AssertionError("Acceptance unexpectedly created a production alias")
        write(output / "registry_history.json", state)
        bundle_refs = []
        for bundle, bundle_manifest in zip(bundles, manifests):
            bundle_refs.append(
                {
                    "path": str(bundle),
                    "bundle_id": bundle_manifest["bundle_id"],
                    "files": {
                        path.relative_to(bundle).as_posix(): file_hash(path)
                        for path in sorted(bundle.rglob("*"))
                        if path.is_file()
                    },
                }
            )
        write(output / "bundle_references.json", bundle_refs)
        result = {
            "status": "passed",
            "qualification": protocol["qualification"],
            "checks": checks,
            "orders_submitted": False,
            "prospective_evidence": False,
            "publication_time_certified": False,
            "production_promoted": False,
        }
        write(output / "results.json", result)
        write(output / "status.json", {"status": "completed"})
        write(
            output / "artifacts.json",
            {
                "files": {
                    path.name: file_hash(path)
                    for path in sorted(output.iterdir())
                    if path.is_file()
                }
            },
        )
        return result
    except Exception as error:
        write(
            output / "results.json",
            {
                "status": "failed",
                "checks_completed": checks,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        write(output / "status.json", {"status": "failed"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--primary-run", required=True, type=Path)
    parser.add_argument("--rollback-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    result = accept(
        args.provider,
        args.primary_run,
        args.rollback_run,
        args.output,
        args.work_dir,
        tolerance=args.tolerance,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
