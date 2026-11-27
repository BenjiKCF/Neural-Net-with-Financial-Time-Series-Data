"""Local model registry, atomic daily forecasts and auditable shadow operation.

The runner emits predictions only. Broker submission is intentionally a separate
integration: paper fills and deployment approval cannot be inferred from a model
score or a successful historical replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from financial_ts.bundles import (
    NumericalPredictor,
    atomic_json,
    canonical_json,
    file_hash,
    verify_bundle,
    write_json,
)


@dataclass(frozen=True)
class BatchPolicy:
    """Declared service objectives, evaluated before atomic publication."""

    max_runtime_seconds: float = 600.0
    max_memory_mb: float = 4096.0
    minimum_coverage: float = 1.0
    max_missing_feature_fraction: float = 0.25
    max_abs_score: float = 10.0
    max_data_age_hours: float = 96.0
    min_cross_section: int = 1

    def validate(self):
        for key in ("max_runtime_seconds", "max_memory_mb", "max_abs_score", "max_data_age_hours"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite.")
        for key in ("minimum_coverage", "max_missing_feature_fraction"):
            value = getattr(self, key)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be in [0, 1].")
        if (
            not 0 < self.minimum_coverage <= 1
            or type(self.min_cross_section) is not int
            or self.min_cross_section < 1
        ):
            raise ValueError("Coverage must be positive and min_cross_section a positive integer.")


def _aware(value, name):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} requires an explicit timezone.")
    return result.tz_convert("UTC")


def _symbol_list(values, name):
    values = list(values)
    if (
        not values
        or len(set(values)) != len(values)
        or any(not isinstance(x, str) or not x for x in values)
    ):
        raise ValueError(f"{name} must contain distinct nonempty instrument identifiers.")
    return values


def _registry_state(root):
    path = Path(root) / "registry.json"
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"schema_version": 1, "versions": {}, "aliases": {}, "history": []}
    )
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported registry schema.")
    return state


def register_bundle(registry, bundle) -> dict:
    """Copy a verified immutable bundle into a local, lock-protected registry."""
    root = Path(registry).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = Path(bundle).expanduser().resolve()
    manifest = verify_bundle(source)
    version = manifest["bundle_id"]
    with FileLock(str(root / ".registry.lock"), timeout=30):
        state = _registry_state(root)
        destination = root / "bundles" / version
        destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            if verify_bundle(destination)["bundle_id"] != version:
                raise ValueError("Existing registry version has changed.")
        else:
            temporary = Path(tempfile.mkdtemp(prefix=".register-", dir=destination.parent))
            try:
                shutil.copytree(source, temporary, dirs_exist_ok=True)
                verify_bundle(temporary)
                os.rename(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        if version not in state["versions"]:
            state["versions"][version] = {
                "path": f"bundles/{version}",
                "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            state["history"].append({"action": "register", "version": version})
            atomic_json(root / "registry.json", state)
    return {"version": version, "registry": str(root)}


def evaluate_promotion(bundle, evidence, *, intended_use="production") -> dict:
    manifest = verify_bundle(bundle)
    metadata = json.loads((Path(bundle) / "metadata.json").read_text())
    parity = json.loads((Path(bundle) / "parity.json").read_text())
    reasons = []
    if evidence.get("bundle_id") != manifest["bundle_id"]:
        reasons.append("evidence does not identify this immutable bundle")
    if parity.get("status") != "passed":
        reasons.append("reload parity has not passed")
    if intended_use == "production":
        rights = metadata.get("usage_rights", {})
        for family in ("model", "data"):
            if rights.get(family) != "approved_for_intended_use":
                reasons.append(f"{family} usage rights are unresolved or restricted")
        for gate in (
            "causal_data",
            "lifecycle_accounting",
            "candidate_quality",
            "prospective_shadow",
            "operations",
            "execution_reconciliation",
        ):
            receipt = evidence.get("gates", {}).get(gate, {})
            if receipt.get("status") != "passed" or not receipt.get("evidence_uri"):
                reasons.append(f"{gate} has no passed, referenced evidence")
        approval = evidence.get("approval", {})
        if (
            approval.get("decision") != "approved"
            or not approval.get("reviewer")
            or approval.get("intended_use") != intended_use
        ):
            reasons.append("recorded deployment decision is missing")
        if not evidence.get("service_objectives") or not evidence.get("incident_owner"):
            reasons.append("service objectives or incident ownership is missing")
    elif intended_use != "shadow":
        raise ValueError("intended_use must be shadow or production.")
    return {
        "bundle_id": manifest["bundle_id"],
        "intended_use": intended_use,
        "status": "passed" if not reasons else "blocked",
        "reasons": reasons,
    }


def set_alias(registry, alias, version, *, evidence=None, rollback=False) -> dict:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", alias):
        raise ValueError("Invalid registry alias.")
    if not isinstance(version, str) or not re.fullmatch(r"[a-f0-9]{64}", version):
        raise ValueError("Version must be an immutable bundle SHA256.")
    root = Path(registry).expanduser().resolve()
    with FileLock(str(root / ".registry.lock"), timeout=30):
        state = _registry_state(root)
        if version not in state["versions"]:
            raise ValueError("Version is not registered.")
        path = root / "bundles" / version
        verify_bundle(path)
        intended_use = "production" if alias == "champion" else "shadow"
        evidence = evidence or {"bundle_id": version}
        gate = evaluate_promotion(path, evidence, intended_use=intended_use)
        if gate["status"] != "passed":
            raise ValueError("Promotion blocked: " + "; ".join(gate["reasons"]))
        previous = state["aliases"].get(alias)
        if rollback:
            seen = {item.get("version") for item in state["history"] if item.get("alias") == alias}
            if version not in seen or previous is None:
                raise ValueError("Rollback target must be a retained prior version of this alias.")
            old_schema = json.loads((root / "bundles" / previous / "schema.json").read_text())
            new_schema = json.loads((path / "schema.json").read_text())
            if old_schema != new_schema:
                raise ValueError("Rollback target has an incompatible feature schema or context.")
        state["aliases"][alias] = version
        receipt = {
            "action": "rollback" if rollback else "alias",
            "alias": alias,
            "version": version,
            "previous_version": previous,
            "gate": gate,
            "evidence": evidence,
            "at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        state["history"].append(receipt)
        atomic_json(root / "registry.json", state)
    return receipt


def resolve_alias(registry, alias) -> tuple[Path, str]:
    root = Path(registry).expanduser().resolve()
    # Atomic registry-file replacement gives one consistent alias snapshot.
    state = _registry_state(root)
    version = state["aliases"].get(alias)
    if (
        version is None
        or version not in state["versions"]
        or not re.fullmatch(r"[a-f0-9]{64}", version)
    ):
        raise ValueError("Registry alias is missing or malformed.")
    path = root / "bundles" / version
    if verify_bundle(path)["bundle_id"] != version:
        raise ValueError("Resolved bundle identity differs from registry version.")
    return path, version


def _validate_packet(packet, policy):
    required = {
        "decision_date",
        "decision_cutoff",
        "data_available_at",
        "observed_through",
        "expected_session",
        "instruments",
        "expected_instruments",
        "feature_names",
        "data_fingerprint",
        "source_revision",
    }
    if not required.issubset(packet):
        raise ValueError(f"Input manifest is missing {sorted(required - set(packet))}.")
    cutoff = _aware(packet["decision_cutoff"], "decision_cutoff")
    available = _aware(packet["data_available_at"], "data_available_at")
    if available > cutoff:
        raise ValueError("Data was published after the registered decision cutoff.")
    if (cutoff - available).total_seconds() / 3600 > policy.max_data_age_hours:
        raise ValueError("Input data exceeds the registered freshness limit.")
    day = pd.Timestamp(packet["decision_date"])
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("decision_date must be a timezone-naive daily session.")
    if any(pd.Timestamp(packet[key]) != day for key in ("observed_through", "expected_session")):
        raise ValueError(
            "Stale or future data: observed session must equal the expected decision session."
        )
    local_day = cutoff.tz_convert(packet.get("timezone", "Asia/Shanghai")).date()
    if day.date() != local_day:
        raise ValueError("Decision cutoff and session disagree in the market timezone.")
    names = _symbol_list(packet["instruments"], "instruments")
    expected = _symbol_list(packet["expected_instruments"], "expected_instruments")
    unknown = sorted(set(names) - set(expected))
    if unknown:
        raise ValueError(f"Instruments outside the declared eligible universe: {unknown[:5]}")
    coverage = len(names) / len(expected)
    if coverage < policy.minimum_coverage or len(names) < policy.min_cross_section:
        raise ValueError("Eligible-universe coverage is below the declared objective.")
    if not packet["data_fingerprint"] or not packet["source_revision"]:
        raise ValueError("Input data identity is required.")
    return day.strftime("%Y-%m-%d"), cutoff, coverage


def read_prediction_artifact(path) -> tuple[pd.DataFrame, dict]:
    """Verify forecasts and their complete input lineage before consumption.

    Version 1 did not bind the input manifest. Those artifacts cannot establish
    intact lineage and must be regenerated at a fresh output path.
    """
    path = Path(path)
    expected_files = {"manifest.json", "input_manifest.json", "predictions.csv"}
    if (
        path.is_symlink()
        or not path.is_dir()
        or {item.name for item in path.iterdir()} != expected_files
        or any((path / name).is_symlink() or not (path / name).is_file()
               for name in expected_files)
    ):
        raise ValueError("Prediction artifact is incomplete or has unsafe input lineage files.")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 2 or not manifest.get("input_manifest_sha256"):
        raise ValueError(
            "Prediction artifact lacks verified input lineage; regenerate at a fresh output path."
        )
    if file_hash(path / "input_manifest.json") != manifest["input_manifest_sha256"]:
        raise ValueError("Prediction artifact input lineage failed its integrity check.")
    if manifest.get("status") != "ready" or file_hash(path / "predictions.csv") != manifest.get(
        "predictions_sha256"
    ):
        raise ValueError("Prediction artifact is invalid or incomplete.")
    packet = json.loads((path / "input_manifest.json").read_text())
    policy = BatchPolicy(**manifest["policy"])
    policy.validate()
    date, cutoff, coverage = _validate_packet(packet, policy)
    expected_metadata = {
        "decision_date": date,
        "decision_cutoff": cutoff.isoformat(),
        "data_fingerprint": packet["data_fingerprint"],
        "source_revision": packet["source_revision"],
        "coverage": coverage,
    }
    if any(manifest.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("Prediction publication metadata differs from its input lineage.")
    frame = pd.read_csv(path / "predictions.csv", dtype={"instrument": str}, keep_default_na=False)
    if list(frame.columns) != ["datetime", "instrument", "score"]:
        raise ValueError("Prediction artifact has an unexpected forecast schema.")
    if (
        frame.empty
        or frame["instrument"].duplicated().any()
        or not np.isfinite(frame["score"].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Prediction artifact contains malformed scores.")
    if (
        len(frame) != manifest["count"]
        or set(frame["datetime"].astype(str)) != {manifest["decision_date"]}
        or set(frame["instrument"]) != set(packet["instruments"])
    ):
        raise ValueError("Prediction rows differ from publication metadata.")
    return frame, manifest


def run_daily_batch(
    registry, alias, output, *, packet, features, padding=None, policy=None, pinned_bundle_id=None
):
    """Resolve one immutable model, validate inputs, publish once or fail closed.

    The output path names a store, with a subdirectory for each decision date.
    Concurrent duplicate calls are serialized; a conflicting revision cannot
    silently replace a published day. Failures get separate operator receipts.
    """
    started = time.perf_counter()
    policy = BatchPolicy() if policy is None else policy
    policy.validate()
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        date, cutoff, coverage = _validate_packet(packet, policy)
        # Resolve exactly once. Subsequent alias moves do not alter this batch.
        bundle, version = resolve_alias(registry, alias)
        if pinned_bundle_id is not None and version != pinned_bundle_id:
            raise ValueError(
                "Registry changed after preparation; retry with one immutable version."
            )
        predictor = NumericalPredictor(bundle)
        if packet["feature_names"] != predictor.schema["features"]:
            raise ValueError("Input feature order differs from bundle schema.")
        fit_end = pd.Timestamp(predictor.metadata["fit_end"])
        if pd.isna(fit_end):
            raise ValueError("Model fit cutoff must be a finite timestamp.")
        if fit_end.tzinfo is None:
            fit_end = fit_end.tz_localize(packet.get("timezone", "Asia/Shanghai"))
        if fit_end.tz_convert("UTC") >= cutoff:
            raise ValueError("Model fit cutoff is not earlier than the prediction cutoff.")
        values = np.asarray(features, dtype=np.float64)
        if values.ndim not in (2, 3) or len(values) != len(packet["instruments"]):
            raise ValueError("Feature rows and instrument identifiers do not align.")
        mask = (
            np.zeros(values.shape[:-1], dtype=bool)
            if padding is None
            else np.asarray(padding, dtype=bool)
        )
        if mask.shape != values.shape[:-1]:
            raise ValueError("Input padding does not align with feature rows.")
        if values.ndim == 2 and mask.any():
            raise ValueError("A tabular decision-date row cannot be padded or absent.")
        observed = values[~mask]
        missing = float((~np.isfinite(observed)).mean()) if observed.size else 1.0
        if missing > policy.max_missing_feature_fraction:
            raise ValueError("Feature missingness exceeds the registered objective.")
        if np.isinf(values).any():
            raise ValueError("Infinite raw features are invalid for production inference.")
        digest = hashlib.sha256(
            canonical_json({"packet": packet, "version": version, "policy": asdict(policy)})
        )
        digest.update(str(values.shape).encode())
        # Canonicalize NaN bits and endian order for repeatable request identity.
        numeric = values.astype("<f8", copy=True)
        numeric[np.isnan(numeric)] = np.nan
        digest.update(numeric.tobytes())
        digest.update(mask.tobytes())
        identity = digest.hexdigest()
        destination = root / date
        with FileLock(str(root / f".{date}.lock"), timeout=30):
            if destination.exists():
                _, existing = read_prediction_artifact(destination)
                if existing.get("request_id") != identity:
                    raise ValueError("A conflicting batch is already published for this session.")
                return {**existing, "reused": True, "path": str(destination)}
            scores = predictor.predict(values, padding=mask, feature_names=packet["feature_names"])
            if np.max(np.abs(scores)) > policy.max_abs_score:
                raise ValueError("Score magnitude exceeds the registered objective.")
            elapsed = time.perf_counter() - started
            if elapsed > policy.max_runtime_seconds:
                raise ValueError("Batch runtime exceeded the registered objective.")
            import psutil

            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            if memory_mb > policy.max_memory_mb:
                raise ValueError("Batch resident memory exceeded the registered objective.")
            temporary = Path(tempfile.mkdtemp(prefix=f".{date}-", dir=root))
            frame = pd.DataFrame(
                {"datetime": date, "instrument": packet["instruments"], "score": scores}
            )
            frame = frame.sort_values("instrument", kind="stable")
            frame.to_csv(temporary / "predictions.csv", index=False, float_format="%.17g")
            write_json(temporary / "input_manifest.json", packet)
            manifest = {
                "schema_version": 2,
                "status": "ready",
                "decision_date": date,
                "decision_cutoff": cutoff.isoformat(),
                "bundle_id": version,
                "alias_at_start": alias,
                "data_fingerprint": packet["data_fingerprint"],
                "source_revision": packet["source_revision"],
                "request_id": identity,
                "count": len(frame),
                "coverage": coverage,
                "missing_feature_fraction": missing,
                "runtime_seconds": elapsed,
                "resident_memory_mb": memory_mb,
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "predictions_sha256": file_hash(temporary / "predictions.csv"),
                "input_manifest_sha256": file_hash(temporary / "input_manifest.json"),
                "policy": asdict(policy),
                "mode": "production_predictions" if alias == "champion" else "shadow_predictions",
                "orders_submitted": False,
            }
            write_json(temporary / "manifest.json", manifest)
            read_prediction_artifact(temporary)
            os.rename(temporary, destination)
            temporary = None
            return {**manifest, "reused": False, "path": str(destination)}
    except Exception as error:
        failure = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "decision_date": str(packet.get("decision_date", "unknown")),
            "at": pd.Timestamp.now(tz="UTC").isoformat(),
            "published": False,
        }
        failure_id = hashlib.sha256(canonical_json(failure)).hexdigest()
        atomic_json(root / "failures" / f"{failure_id}.json", failure)
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def monitor_mature_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    cutoff,
    minimum_rank_ic=None,
    minimum_coverage=1.0,
) -> dict:
    """Only evaluate labels whose documented information endpoint has matured."""
    from financial_ts.workflow import _prediction_statistics

    asof = _aware(cutoff, "cutoff")
    if not np.isfinite(minimum_coverage) or not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be finite and in (0, 1].")
    keys = ["datetime", "instrument"]
    for frame, columns in (
        (predictions, [*keys, "score"]),
        (labels, [*keys, "label", "available_at"]),
    ):
        if not set(columns).issubset(frame) or frame.duplicated(keys).any():
            raise ValueError("Monitoring inputs require unique aligned date/instrument rows.")
    # utc=True alone interprets a naive local timestamp as UTC, admitting some
    # overseas labels too early. Require a documented offset on every row.
    available = labels["available_at"].map(lambda value: _aware(value, "available_at"))
    if not np.isfinite(predictions["score"].to_numpy(dtype=float)).all():
        raise ValueError("Monitoring predictions must be finite.")
    if np.isinf(labels["label"].to_numpy(dtype=float)).any():
        raise ValueError("Monitoring labels must not contain infinite returns.")
    mature = labels.loc[available <= asof].copy()
    delayed = len(labels) - len(mature)
    merged = predictions.merge(
        mature[[*keys, "label"]], on=keys, how="inner", validate="one_to_one"
    )
    if merged.empty:
        return {
            "status": "waiting_for_labels",
            "mature_count": 0,
            "deferred_labels": delayed,
            "action": "retain current model",
        }
    merged["datetime"] = pd.to_datetime(merged["datetime"])
    evaluated_count = int(merged["label"].notna().sum())
    unavailable_mature_count = len(merged) - evaluated_count
    if evaluated_count:
        stats, daily = _prediction_statistics(merged.set_index(keys))
    else:
        stats, daily = {"rank_ic": np.nan, "evaluated_count": 0}, pd.DataFrame()
    coverage = evaluated_count / len(predictions)
    alerts = []
    if coverage < minimum_coverage:
        alerts.append("label_coverage")
    rank_ic = stats["rank_ic"]
    if minimum_rank_ic is not None and (not np.isfinite(rank_ic) or rank_ic < minimum_rank_ic):
        alerts.append("rank_ic")
    cleaned = {
        key: (float(value) if np.isfinite(value) else None)
        for key, value in stats.items()
        if isinstance(value, (int, float, np.number))
    }
    return {
        "status": "review_required" if alerts else "observed",
        "metrics": cleaned,
        "mature_count": len(merged),
        "evaluated_count": evaluated_count,
        "unavailable_mature_count": unavailable_mature_count,
        "deferred_labels": delayed,
        "coverage": coverage,
        "daily_rows": len(daily),
        "alerts": alerts,
        "automatic_retraining": False,
        "automatic_promotion": False,
    }


def build_provider_packet(
    provider, bundle, *, decision_date, decision_cutoff, data_available_at, universe=None
):
    """Compute causal Alpha158 inputs from a pinned provider without labels.

    Only factor expressions are requested, ending at the decision session. The
    fitted normalization is applied later by NumericalPredictor, never refitted.
    """
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D
    from qlib.data.cache import H

    from financial_ts.workflow import _market_timezone, _provider_fingerprint, _read_calendar

    provider = Path(provider).expanduser().resolve()
    predictor = NumericalPredictor(bundle)
    calendar = _read_calendar(provider)
    date = pd.Timestamp(decision_date)
    if date not in calendar:
        raise ValueError("Decision date is absent from the provider calendar.")
    length = predictor.schema["sequence_length"]
    location = calendar.get_loc(date)
    if location < length - 1:
        raise ValueError("Provider has insufficient historical context.")
    universe = universe or predictor.metadata["universe"]
    H.clear()
    qlib.init(
        provider_uri=str(provider),
        region=predictor.metadata.get("region", "cn"),
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    market = D.instruments(universe) if isinstance(universe, str) else universe
    instruments = (
        sorted(D.list_instruments(market, start_time=date, end_time=date, as_list=True))
        if isinstance(market, dict)
        else sorted(market)
    )
    if not instruments:
        raise ValueError("No instruments are eligible on the decision date.")
    fields, names = Alpha158DL.get_feature_config(
        {"kbar": {}, "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]}, "rolling": {}}
    )
    if names != predictor.schema["features"]:
        raise ValueError("Provider factor recipe does not match the bundle.")
    dates = calendar[location - length + 1 : location + 1]
    # Membership holes must match the historical training sampler, including
    # names newly added to or re-entering the eligible universe.
    frame = D.features(market, fields, start_time=dates[0], end_time=date)
    frame.columns = names
    target = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])
    frame = frame.reorder_levels(["instrument", "datetime"]).reindex(target)
    array = frame.to_numpy(dtype=np.float64).reshape(len(instruments), length, len(names))
    padding = ~np.isfinite(array).any(axis=-1)
    # Absent current observations must fail rather than inventing a tradable row.
    present = ~padding[:, -1]
    selected = [instrument for instrument, exists in zip(instruments, present) if exists]
    array, padding = array[present], padding[present]
    kind = predictor.info["kind"]
    if kind not in {"sequence", "cross_sectional"}:
        array, padding = array[:, -1], None
    fingerprint = _provider_fingerprint(provider)
    source = (
        json.loads((provider / "metadata.json").read_text())
        if (provider / "metadata.json").exists()
        else {}
    )
    packet = {
        "decision_date": date.strftime("%Y-%m-%d"),
        "expected_session": date.strftime("%Y-%m-%d"),
        "observed_through": date.strftime("%Y-%m-%d"),
        "decision_cutoff": decision_cutoff,
        "data_available_at": data_available_at,
        "timezone": predictor.metadata.get("market_timezone")
        or _market_timezone(predictor.metadata.get("region", "cn")),
        "instruments": selected,
        "expected_instruments": instruments,
        "feature_names": names,
        "data_fingerprint": fingerprint,
        "source_revision": str(
            source.get("release_tag", source.get("release", fingerprint["sha256"]))
        ),
        "availability_qualification": "caller-supplied arrival receipt; historical publication-time accuracy is not certified",
    }
    return packet, array, padding


def export_to_mlflow(registry, alias, *, tracking_uri, model_name):
    """Optional explicit local MLflow lineage registration; resolves alias once."""
    from mlflow import MlflowClient

    bundle, version = resolve_alias(registry, alias)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    from mlflow.exceptions import MlflowException

    try:
        client.create_registered_model(model_name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    created = client.create_model_version(
        model_name,
        source=bundle.as_uri(),
        tags={
            "bundle_id": version,
            "format": "financial_ts_numerical_bundle",
            "loader": "financial_ts.bundles.NumericalPredictor",
        },
    )
    client.set_registered_model_alias(model_name, alias, created.version)
    return {
        "name": model_name,
        "version": str(created.version),
        "bundle_id": version,
        "alias": alias,
    }


def compare_daily_artifacts(
    current, previous, *, minimum_rank_stability=-1.0, maximum_mean_shift=5.0
):
    """Observable score/rank drift on common securities; alerts never retrain."""
    if (
        not -1 <= minimum_rank_stability <= 1
        or not np.isfinite(maximum_mean_shift)
        or maximum_mean_shift <= 0
    ):
        raise ValueError("Declare valid rank-stability and normalized mean-shift thresholds.")
    current_frame, current_meta = read_prediction_artifact(current)
    previous_frame, previous_meta = read_prediction_artifact(previous)
    if current_meta["decision_date"] <= previous_meta["decision_date"]:
        raise ValueError("Drift comparison must follow chronological session order.")
    joined = current_frame[["instrument", "score"]].merge(
        previous_frame[["instrument", "score"]],
        on="instrument",
        suffixes=("_current", "_previous"),
        validate="one_to_one",
    )
    scale = max(float(previous_frame.score.std(ddof=0)), 1e-8)
    mean_shift = float(abs(current_frame.score.mean() - previous_frame.score.mean()) / scale)
    stability = (
        joined.score_current.corr(joined.score_previous, method="spearman")
        if len(joined) >= 2
        and joined.score_current.nunique() > 1
        and joined.score_previous.nunique() > 1
        else np.nan
    )
    alerts = []
    if not np.isfinite(stability) or stability < minimum_rank_stability:
        alerts.append("rank_stability")
    if mean_shift > maximum_mean_shift:
        alerts.append("score_mean_shift")
    return {
        "status": "review_required" if alerts else "observed",
        "alerts": alerts,
        "common_instruments": len(joined),
        "mean_shift_in_previous_std": mean_shift,
        "rank_stability": float(stability) if np.isfinite(stability) else None,
        "previous_bundle": previous_meta["bundle_id"],
        "current_bundle": current_meta["bundle_id"],
        "automatic_retraining": False,
        "automatic_promotion": False,
    }


def run_scheduled_shadow(config, *, now=None):
    """One scheduled invocation, driven by a real upstream arrival receipt.

    A missing receipt is an outage, not permission to manufacture an arrival
    timestamp. Scheduling examples invoke this function; they do not start a
    daemon or install a system service on the user's machine.
    """
    config = dict(config)
    required = {"registry", "provider", "arrival_receipt", "output", "incident_owner", "policy"}
    if not required.issubset(config) or not config["incident_owner"]:
        raise ValueError(
            "Scheduled shadow requires paths, a declared policy and incident ownership."
        )
    if config.get("alias", "shadow") == "champion":
        raise ValueError("This scheduler is for shadow observations only.")
    current = pd.Timestamp.now(tz="UTC") if now is None else _aware(now, "now")
    receipt = json.loads(Path(config["arrival_receipt"]).read_text())
    local_date = current.tz_convert(config.get("timezone", "Asia/Hong_Kong")).strftime("%Y-%m-%d")
    if receipt.get("expected_session") != local_date:
        raise ValueError("Arrival receipt is stale for the current scheduled session.")
    if receipt.get("status") == "no_session":
        if not receipt.get("calendar_source"):
            raise ValueError("A nontrading-session receipt needs a calendar source.")
        return {
            "status": "no_session",
            "date": local_date,
            "calendar_source": receipt["calendar_source"],
        }
    cutoff = _aware(receipt["decision_cutoff"], "decision_cutoff")
    maximum_lag = float(config.get("max_publish_lag_minutes", 30))
    if not np.isfinite(maximum_lag) or maximum_lag <= 0:
        raise ValueError("Maximum publication lag must be positive and finite.")
    if current < cutoff or (current - cutoff).total_seconds() > maximum_lag * 60:
        raise ValueError("Scheduled invocation is outside its declared publication window.")
    if not receipt.get("provider_fingerprint"):
        raise ValueError("Arrival receipt must bind the admitted provider fingerprint.")
    bundle, version = resolve_alias(config["registry"], config.get("alias", "shadow"))
    packet, features, padding = build_provider_packet(
        config["provider"],
        bundle,
        decision_date=local_date,
        decision_cutoff=receipt["decision_cutoff"],
        data_available_at=receipt["data_available_at"],
    )
    if packet["data_fingerprint"] != receipt["provider_fingerprint"]:
        raise ValueError("Provider differs from the admitted arrival receipt.")
    if receipt.get("snapshot_audit_status") != "admitted":
        raise ValueError("Snapshot refresh has not passed its data-quality audit.")
    # Pin preparation and publication to the same alias generation.
    if resolve_alias(config["registry"], config.get("alias", "shadow"))[1] != version:
        raise ValueError("Registry changed during feature preparation; retry the batch.")
    result = run_daily_batch(
        config["registry"],
        config.get("alias", "shadow"),
        config["output"],
        packet=packet,
        features=features,
        padding=padding,
        policy=BatchPolicy(**config["policy"]),
        pinned_bundle_id=version,
    )
    atomic_json(Path(config["output"]) / "last_scheduled_run.json", result)
    return result
