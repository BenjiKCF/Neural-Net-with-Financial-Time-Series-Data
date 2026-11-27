"""Preregistered, real-clock prospective forecast and shadow acceptance.

A historical replay cannot supply observations to this store: the first durable
capture of each forecast must occur inside its future registered publication
window. These local receipts require a trusted host clock and filesystem; they
are not an independent timestamp authority or authorization to trade.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from financial_ts.bundles import (
    atomic_json,
    canonical_json,
    environment_versions,
    file_hash,
    verify_bundle,
    write_json,
)
from financial_ts.operations import (
    BatchPolicy,
    _aware,
    _validate_packet,
    build_provider_packet,
    read_prediction_artifact,
    register_bundle,
    run_daily_batch,
    set_alias,
)

_ROLES = ("candidate", "reference")
_TARGET = "Ref($close, -2) / Ref($close, -1) - 1"
_EQUIVALENT_TARGETS = {_TARGET, "close[t+2]/close[t+1]-1"}


def _now():
    # No public clock override: tests replace this private clock explicitly.
    return pd.Timestamp.now(tz="UTC")


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Evidence must be a regular non-symlinked file: {path.name}")
    return json.loads(path.read_text())


def _source_identity():
    root = Path(__file__).parent
    names = (
        "prospective.py", "operations.py", "bundles.py", "workflow.py", "rolling.py",
        "catalog.py", "model.py", "cross_sectional.py", "objectives.py",
    )
    paths = [root / name for name in names] + sorted((root / "architectures").rglob("*.py"))
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _finite(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its allowed range.")
    return value


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Sessions require ISO daily dates.")
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("Session dates must be finite.")
    return date


def _specification(spec):
    """Validate a complete recipe; no permissive inferred trading calendar."""
    spec = json.loads(canonical_json(spec))
    required = {
        "name", "evaluation_role", "timezone", "calendar", "first_session", "last_session",
        "decision_time", "label_available_time", "incident_owner", "selection_reason",
        "source_revision", "data_contract", "policy", "acceptance",
    }
    if not required.issubset(spec):
        raise ValueError(f"Prospective specification missing {sorted(required - set(spec))}.")
    for name in ("name", "incident_owner", "selection_reason", "source_revision"):
        if not isinstance(spec[name], str) or not spec[name].strip():
            raise ValueError(f"{name} must be declared.")
        if "REPLACE" in spec[name].upper() or "TODO" in spec[name].upper():
            raise ValueError(f"{name} still contains a template placeholder.")
    if spec["evaluation_role"] not in {"operations_pilot", "confirmation"}:
        raise ValueError("evaluation_role must distinguish operations_pilot from confirmation.")
    if spec["timezone"] not in {"Asia/Shanghai", "Asia/Hong_Kong"}:
        raise ValueError("This CSI300 evaluator requires the China exchange timezone.")
    calendar = spec["calendar"]
    if not isinstance(calendar, dict) or not calendar.get("source") or not calendar.get("sessions"):
        raise ValueError("Declare the exchange calendar sessions and their source.")
    days = [_date(value) for value in calendar["sessions"]]
    if days != sorted(set(days)):
        raise ValueError("Calendar sessions must be sorted and unique.")
    first, last = _date(spec["first_session"]), _date(spec["last_session"])
    if first not in days or last not in days or first > last:
        raise ValueError("Observation window must use ordered declared exchange sessions.")
    if days.index(last) + 2 >= len(days):
        raise ValueError("Calendar must include the two label endpoint sessions after the window.")
    for key in ("decision_time", "label_available_time"):
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", str(spec[key])):
            raise ValueError(f"{key} requires HH:MM:SS in the declared timezone.")
    if spec["decision_time"] < "15:00:00" or spec["label_available_time"] < "15:00:00":
        raise ValueError("Close-price features and labels cannot be available before the exchange close.")
    if not isinstance(spec["data_contract"], dict) or not all(
        spec["data_contract"].get(key)
        for key in ("source", "adjustment_convention", "availability_policy")
    ):
        raise ValueError("Declare source, adjustment convention and availability policy.")
    policy = BatchPolicy(**spec["policy"])
    policy.validate()
    spec["policy"] = asdict(policy)
    acceptance = spec["acceptance"]
    required_acceptance = {
        "minimum_sessions", "maximum_publish_lag_minutes", "minimum_success_fraction",
        "maximum_failed_attempts", "minimum_label_coverage", "minimum_rank_ic",
        "minimum_paired_rank_ic_gain", "block_lengths", "bootstrap_draws",
        "bootstrap_seed", "minimum_effective_blocks",
    }
    if not isinstance(acceptance, dict) or not required_acceptance.issubset(acceptance):
        raise ValueError("Declare every prospective acceptance threshold before registration.")
    _integer(acceptance["minimum_sessions"], "minimum_sessions", 2)
    _integer(acceptance["maximum_failed_attempts"], "maximum_failed_attempts", 0)
    _integer(acceptance["bootstrap_draws"], "bootstrap_draws", 100)
    _integer(acceptance["bootstrap_seed"], "bootstrap_seed", 0)
    _finite(acceptance["maximum_publish_lag_minutes"], "maximum_publish_lag_minutes", 0.01, 60)
    for key in ("minimum_success_fraction", "minimum_label_coverage"):
        _finite(acceptance[key], key, 0.000001, 1)
    for key in ("minimum_rank_ic", "minimum_paired_rank_ic_gain"):
        _finite(acceptance[key], key, -1, 1)
    _finite(acceptance["minimum_effective_blocks"], "minimum_effective_blocks", 2)
    blocks = acceptance["block_lengths"]
    if not isinstance(blocks, list) or not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("Declare distinct bootstrap block lengths.")
    for length in blocks:
        _integer(length, "block_length")
        if length * acceptance["minimum_effective_blocks"] > acceptance["minimum_sessions"]:
            raise ValueError("Minimum sessions cannot support the registered block lengths.")
    selected = days[days.index(first):days.index(last) + 1]
    if len(selected) < acceptance["minimum_sessions"]:
        raise ValueError("Observation window is shorter than minimum_sessions.")
    sessions = []
    for day in selected:
        # Localize explicitly; DST ambiguities and nonexistent times fail.
        cutoff = pd.Timestamp(f"{day.date()} {spec['decision_time']}").tz_localize(
            spec["timezone"], ambiguous="raise", nonexistent="raise"
        ).tz_convert("UTC")
        endpoint = days[days.index(day) + 2]
        mature = pd.Timestamp(f"{endpoint.date()} {spec['label_available_time']}").tz_localize(
            spec["timezone"], ambiguous="raise", nonexistent="raise"
        ).tz_convert("UTC")
        if mature <= cutoff:
            raise ValueError("Label endpoint must follow the forecast cutoff.")
        sessions.append({
            "date": str(day.date()), "cutoff": cutoff.isoformat(),
            "deadline": (cutoff + pd.Timedelta(
                minutes=acceptance["maximum_publish_lag_minutes"]
            )).isoformat(),
            "label_mature_at": mature.isoformat(),
        })
    return spec, sessions


def register_prospective(output, spec, *, candidate_bundle, reference_bundle):
    """Freeze a future protocol and numerical bundles with actual registration time."""
    root = Path(output).expanduser().absolute()
    if root.is_symlink():
        raise ValueError("Prospective store cannot be symlinked.")
    recipe, sessions = _specification(spec)
    roots = {"candidate": Path(candidate_bundle), "reference": Path(reference_bundle)}
    identities, metadata = {}, {}
    for role, path in roots.items():
        identities[role] = verify_bundle(path)["bundle_id"]
        metadata[role] = _json(path / "metadata.json")
        if _json(path / "parity.json").get("status") != "passed":
            raise ValueError("Every prospective bundle must have passed numerical reload parity.")
    if identities["candidate"] == identities["reference"]:
        raise ValueError("Candidate and reference must be distinct frozen bundles.")
    if any(metadata[role]["target"] not in _EQUIVALENT_TARGETS for role in _ROLES):
        raise ValueError("This evaluator supports the registered two-session close return only.")
    if metadata["candidate"]["universe"] != metadata["reference"]["universe"]:
        raise ValueError("Candidate and reference must use the same declared universe.")
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root.parent / f".{root.name}.register.lock"), timeout=30):
        if root.exists():
            existing = _load(root)
            if existing["specification"] != recipe or existing["bundle_ids"] != identities:
                raise ValueError("An existing prospective protocol cannot be changed.")
            return {**existing, "reused": True}
        now = _now()
        if now >= _aware(sessions[0]["cutoff"], "first cutoff"):
            raise ValueError("Registration must precede the first prospective decision cutoff.")
        for role in _ROLES:
            fit_end = pd.Timestamp(metadata[role]["fit_end"])
            if fit_end.tzinfo is None:
                fit_end = fit_end.tz_localize(recipe["timezone"])
            if fit_end >= now:
                raise ValueError("Model fitting must end before actual prospective registration.")
        body = {
            "schema_version": 1, "kind": "prospective_shadow_forecasts",
            "registered_at": now.isoformat(), "specification": recipe,
            "sessions": sessions, "bundle_ids": identities,
            "bundle_training_lineage": metadata, "source_files": _source_identity(),
            "environment": environment_versions(), "target": _TARGET,
            "clock_qualification": "trusted local wall clock; not an independent timestamp authority",
            "production_ready": False,
        }
        protocol = {**body, "protocol_id": _digest(body)}
        temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
        try:
            for role in _ROLES:
                register_bundle(temporary / "registry", roots[role])
                set_alias(temporary / "registry", role, identities[role])
            write_json(temporary / "protocol.json", protocol)
            (temporary / "observations").mkdir()
            (temporary / "labels").mkdir()
            (temporary / "incidents").mkdir()
            os.rename(temporary, root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {**protocol, "reused": False}


def _load(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Prospective store must be a real directory.")
    protocol = _json(root / "protocol.json")
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if protocol.get("schema_version") != 1 or _digest(body) != protocol.get("protocol_id"):
        raise ValueError("Prospective protocol integrity failed.")
    if protocol["source_files"] != _source_identity() or protocol["environment"] != environment_versions():
        raise ValueError("Prospective runtime changed; use the registered source and environment.")
    for role, identity in protocol["bundle_ids"].items():
        if verify_bundle(root / "registry" / "bundles" / identity)["bundle_id"] != identity:
            raise ValueError(f"Frozen {role} bundle changed.")
    return protocol


def _session(protocol, date):
    _date(date)
    result = next((item for item in protocol["sessions"] if item["date"] == date), None)
    if result is None:
        raise ValueError("Date is outside the registered prospective sessions.")
    return result


def _incident(root, protocol, operation, error, date=None):
    body = {
        "protocol_id": protocol["protocol_id"], "recorded_at": _now().isoformat(),
        "operation": operation, "session": date, "error": f"{type(error).__name__}: {error}",
    }
    receipt = {**body, "receipt_id": _digest(body)}
    atomic_json(Path(root) / "incidents" / f"{receipt['receipt_id']}.json", receipt)


def _files(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinked evidence is not allowed.")
        if path.is_file() and path.name != "receipt.json":
            result[str(path.relative_to(root))] = file_hash(path)
    return result


def _verify_receipt(directory, protocol):
    receipt = _json(Path(directory) / "receipt.json")
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if (
        _digest(body) != receipt.get("receipt_id")
        or receipt.get("protocol_id") != protocol["protocol_id"]
        or receipt.get("files") != _files(directory)
    ):
        raise ValueError("Prospective evidence integrity failed.")
    if receipt.get("mode") == "prospective_shadow_forecasts" and receipt.get(
        "publication_status"
    ) != "committed":
        raise ValueError("Prospective observation has no completed publication receipt.")
    return receipt


def _arrival(receipt, session, current):
    if receipt.get("expected_session") != session["date"]:
        raise ValueError("Arrival receipt belongs to a stale or different session.")
    if receipt.get("snapshot_audit_status") != "admitted":
        raise ValueError("Arrival snapshot must have passed its audit.")
    if not receipt.get("provider_fingerprint") or not receipt.get("source"):
        raise ValueError("Arrival receipt needs provider identity and a source reference.")
    cutoff = _aware(receipt["decision_cutoff"], "decision_cutoff")
    if cutoff != _aware(session["cutoff"], "cutoff"):
        raise ValueError("Arrival cutoff differs from preregistration.")
    available = _aware(receipt["data_available_at"], "data_available_at")
    observed = _aware(receipt["observed_at"], "observed_at")
    if available > observed or observed > current or observed > cutoff:
        raise ValueError("Arrival publication/observation must precede the registered cutoff.")


def record_observation(output, *, candidate_artifact, reference_artifact, arrival_receipt):
    """Capture paired forecasts now; first capture outside its deadline is rejected.

    Retrying an identical already captured day verifies and reuses its receipt.
    Caller-supplied historical timestamps cannot backdate this first capture.
    """
    root = Path(output)
    protocol = _load(root)
    arrival = _json(arrival_receipt) if isinstance(arrival_receipt, (str, Path)) else arrival_receipt
    date = arrival.get("expected_session")
    temporary = None
    try:
        session = _session(protocol, date)
        paths = {"candidate": Path(candidate_artifact), "reference": Path(reference_artifact)}
        identity = {role: _files(path) for role, path in paths.items()}
        request = _digest({"artifacts": identity, "arrival": arrival})
        destination = root / "observations" / date
        with FileLock(str(root / ".observations.lock"), timeout=30):
            if destination.exists():
                previous = _verify_receipt(destination, protocol)
                if previous["request_id"] != request:
                    raise ValueError("Conflicting paired forecasts for an already captured session.")
                return {**previous, "reused": True}
            current = _now()
            cutoff, deadline = (_aware(session[key], key) for key in ("cutoff", "deadline"))
            if current < cutoff or current > deadline:
                raise ValueError("First observation capture is outside the prospective publication window.")
            _arrival(arrival, session, current)
            policy = BatchPolicy(**protocol["specification"]["policy"])
            frames, manifests = {}, {}
            for role, path in paths.items():
                frames[role], manifest = read_prediction_artifact(path)
                packet = _json(path / "input_manifest.json")
                _validate_packet(packet, policy)
                if (
                    manifest["bundle_id"] != protocol["bundle_ids"][role]
                    or manifest["decision_date"] != date
                    or _aware(manifest["decision_cutoff"], "cutoff") != cutoff
                    or manifest["data_fingerprint"] != arrival["provider_fingerprint"]
                    or _aware(packet["data_available_at"], "available") != _aware(
                        arrival["data_available_at"], "available"
                    )
                    or manifest["mode"] != "shadow_predictions"
                    or manifest["orders_submitted"] is not False
                ):
                    raise ValueError("Forecast identity, source or decision differs from preregistration.")
                for metric in ("runtime_seconds", "resident_memory_mb", "missing_feature_fraction"):
                    _finite(manifest[metric], metric, 0)
                if manifest["runtime_seconds"] > policy.max_runtime_seconds:
                    raise ValueError("Runtime exceeds prospective service objective.")
                if manifest["resident_memory_mb"] > policy.max_memory_mb:
                    raise ValueError("Memory exceeds prospective service objective.")
                if manifest["missing_feature_fraction"] > policy.max_missing_feature_fraction:
                    raise ValueError("Feature missingness exceeds prospective service objective.")
                if frames[role]["score"].abs().max() > policy.max_abs_score:
                    raise ValueError("Score magnitude exceeds prospective service objective.")
                manifests[role] = manifest
            if frames["candidate"][["datetime", "instrument"]].to_dict("records") != frames[
                "reference"
            ][["datetime", "instrument"]].to_dict("records"):
                raise ValueError("Prospective models must predict the identical eligible panel.")
            temporary = Path(tempfile.mkdtemp(prefix=f".{date}-", dir=root / "observations"))
            for role in _ROLES:
                shutil.copytree(paths[role], temporary / role)
                read_prediction_artifact(temporary / role)
                if _files(temporary / role) != identity[role]:
                    raise ValueError("Prediction changed during prospective capture.")
            write_json(temporary / "arrival.json", arrival)
            captured = _now()
            if captured > deadline:
                raise ValueError("Durable capture exceeded the prospective publication deadline.")
            body = {
                "protocol_id": protocol["protocol_id"], "session": date,
                "captured_at": captured.isoformat(), "request_id": request,
                "files": _files(temporary), "count": len(frames["candidate"]),
                "bundle_ids": protocol["bundle_ids"], "mode": "prospective_shadow_forecasts",
                "publication_status": "prepared",
            }
            receipt = {**body, "receipt_id": _digest(body)}
            write_json(temporary / "receipt.json", receipt)
            os.rename(temporary, destination)
            temporary = None
            committed = _now()
            if committed <= deadline:
                body.update(captured_at=committed.isoformat(), publication_status="committed")
                receipt = {**body, "receipt_id": _digest(body)}
                atomic_json(destination / "receipt.json", receipt)
                committed = _now()
            if committed > deadline:
                # A slow rename or final receipt write cannot create on-time evidence.
                # Downgrade before moving it so even failed quarantine is fail-closed.
                body.update(captured_at=committed.isoformat(), publication_status="late")
                receipt = {**body, "receipt_id": _digest(body)}
                atomic_json(destination / "receipt.json", receipt)
                rejected = root / "rejected_observations"
                rejected.mkdir(exist_ok=True)
                os.rename(destination, rejected / f"{date}-{receipt['receipt_id']}")
                raise ValueError("Durable publication completed after its prospective deadline.")
            return {**receipt, "reused": False}
    except Exception as error:
        _incident(root, protocol, "record_observation", error, date)
        raise
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def run_prospective_day(output, *, provider, arrival_receipt):
    """Run the frozen pair on admitted arriving features and capture them on time."""
    root = Path(output)
    protocol = _load(root)
    arrival = _json(arrival_receipt)
    date = arrival.get("expected_session")
    try:
        session = _session(protocol, date)
        current = _now()
        if not _aware(session["cutoff"], "cutoff") <= current <= _aware(
            session["deadline"], "deadline"
        ):
            raise ValueError("Daily runner is outside its prospective publication window.")
        _arrival(arrival, session, current)
        artifacts = {}
        for role in _ROLES:
            version = protocol["bundle_ids"][role]
            bundle = root / "registry" / "bundles" / version
            packet, features, padding = build_provider_packet(
                provider, bundle, decision_date=date, decision_cutoff=session["cutoff"],
                data_available_at=arrival["data_available_at"],
            )
            if packet["data_fingerprint"] != arrival["provider_fingerprint"]:
                raise ValueError("Live provider differs from the admitted arrival identity.")
            run_daily_batch(
                root / "registry", role, root / "daily" / role, packet=packet,
                features=features, padding=padding,
                policy=BatchPolicy(**protocol["specification"]["policy"]),
                pinned_bundle_id=version,
            )
            artifacts[f"{role}_artifact"] = root / "daily" / role / date
    except Exception as error:
        _incident(root, protocol, "run_prospective_day", error, date)
        raise
    return record_observation(output, arrival_receipt=arrival, **artifacts)


def record_mature_labels(output, *, session, labels, source):
    """Freeze the first sourced label vintage after the registered endpoint."""
    root = Path(output)
    protocol = _load(root)
    item = _session(protocol, session)
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Mature labels need an inspectable source reference.")
    frame = pd.read_csv(labels, dtype={"instrument": str}) if isinstance(labels, (str, Path)) else labels.copy()
    expected_columns = ["datetime", "instrument", "label", "available_at"]
    if list(frame.columns) != expected_columns or frame.empty or frame.duplicated(
        ["datetime", "instrument"]
    ).any():
        raise ValueError("Labels require unique date/instrument rows and the declared columns.")
    if set(frame["datetime"].astype(str)) != {session}:
        raise ValueError("Label file must cover exactly one registered session.")
    if np.isinf(frame["label"].to_numpy(dtype=float)).any():
        raise ValueError("Infinite labels are invalid.")
    current = _now()
    maturity = _aware(item["label_mature_at"], "label maturity")
    available = frame["available_at"].map(lambda value: _aware(value, "label availability"))
    if current < maturity or (available < maturity).any() or (available > current).any():
        raise ValueError("Labels have not matured under the registered calendar and availability rule.")
    observation = root / "observations" / session
    observation_receipt = _verify_receipt(observation, protocol)
    predictions, _ = read_prediction_artifact(observation / "candidate")
    if set(frame["instrument"]) != set(predictions["instrument"]):
        raise ValueError("Labels must retain every predicted instrument, using null for unavailable outcomes.")
    frame = frame.sort_values("instrument", kind="stable")
    encoded = frame.to_csv(index=False, float_format="%.17g").encode()
    request = _digest({"labels_sha256": hashlib.sha256(encoded).hexdigest(), "source": source})
    destination = root / "labels" / session
    with FileLock(str(root / ".labels.lock"), timeout=30):
        if destination.exists():
            existing = _verify_receipt(destination, protocol)
            if existing["request_id"] != request:
                raise ValueError("First prospective label vintage is frozen; revisions require a separate study.")
            return {**existing, "reused": True}
        temporary = Path(tempfile.mkdtemp(prefix=f".{session}-", dir=root / "labels"))
        try:
            (temporary / "labels.csv").write_bytes(encoded)
            body = {
                "protocol_id": protocol["protocol_id"], "session": session,
                "captured_at": current.isoformat(), "source": source, "request_id": request,
                "observation_receipt_id": observation_receipt["receipt_id"],
                "files": _files(temporary),
            }
            receipt = {**body, "receipt_id": _digest(body)}
            write_json(temporary / "receipt.json", receipt)
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return {**receipt, "reused": False}


def evaluate_prospective(output):
    """Assess all expected sessions; a future or incomplete window stays pending."""
    from financial_ts.rolling import paired_date_block_intervals
    from financial_ts.workflow import _prediction_statistics

    root = Path(output)
    protocol = _load(root)
    now = _now()
    acceptance = protocol["specification"]["acceptance"]
    sessions = protocol["sessions"]
    registered = {item["date"] for item in sessions}
    for directory in (root / "observations", root / "labels"):
        entries = {path.name for path in directory.iterdir() if not path.name.startswith(".")}
        if not entries.issubset(registered):
            raise ValueError("Evidence contains an unregistered session.")
    captured, missing, future, unmatured, missing_labels = [], [], [], [], []
    paired = {role: [] for role in _ROLES}
    label_count, finite_count = 0, 0
    for session in sessions:
        date = session["date"]
        observation = root / "observations" / date
        if not observation.exists():
            (missing if now > _aware(session["deadline"], "deadline") else future).append(date)
            continue
        receipt = _verify_receipt(observation, protocol)
        captured_at = _aware(receipt["captured_at"], "captured_at")
        if not _aware(session["cutoff"], "cutoff") <= captured_at <= _aware(session["deadline"], "deadline"):
            raise ValueError("Retained observation was captured outside its publication window.")
        if captured_at > now or captured_at <= _aware(protocol["registered_at"], "registered_at"):
            raise ValueError("Retained observation has impossible prospective chronology.")
        captured.append(date)
        if now < _aware(session["label_mature_at"], "label maturity"):
            unmatured.append(date)
            continue
        label_directory = root / "labels" / date
        if not label_directory.exists():
            missing_labels.append(date)
            continue
        label_receipt = _verify_receipt(label_directory, protocol)
        if label_receipt["observation_receipt_id"] != receipt["receipt_id"]:
            raise ValueError("Labels refer to a different prospective forecast receipt.")
        label_capture = _aware(label_receipt["captured_at"], "label capture")
        if not _aware(session["label_mature_at"], "label maturity") <= label_capture <= now:
            raise ValueError("Label capture has impossible prospective chronology.")
        labels = pd.read_csv(label_directory / "labels.csv", dtype={"instrument": str})
        label_count += len(labels)
        finite_count += int(np.isfinite(labels["label"].to_numpy(dtype=float)).sum())
        for role in _ROLES:
            forecast, _ = read_prediction_artifact(observation / role)
            paired[role].append(forecast.merge(
                labels[["datetime", "instrument", "label"]],
                on=["datetime", "instrument"], how="left", validate="one_to_one",
            ))
    incidents = []
    for path in sorted((root / "incidents").glob("*.json")):
        receipt = _json(path)
        body = {key: value for key, value in receipt.items() if key != "receipt_id"}
        if _digest(body) != receipt.get("receipt_id") or receipt.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError("Incident receipt integrity failed.")
        incidents.append(receipt)
    closed = now > _aware(sessions[-1]["deadline"], "last deadline")
    success_fraction = len(captured) / len(sessions)
    operational_reasons = []
    if len(captured) < acceptance["minimum_sessions"]:
        operational_reasons.append("insufficient_observed_sessions")
    if success_fraction < acceptance["minimum_success_fraction"]:
        operational_reasons.append("missing_or_failed_sessions")
    if len(incidents) > acceptance["maximum_failed_attempts"]:
        operational_reasons.append("failed_attempt_budget_exceeded")
    operational_status = "pending" if not closed else ("failed" if operational_reasons else "passed")
    quality_reasons, comparisons, metrics = [], [], None
    quality_closed = now >= _aware(sessions[-1]["label_mature_at"], "last label maturity")
    usable_days = len(paired["candidate"])
    coverage = finite_count / label_count if label_count else 0.0
    if usable_days < acceptance["minimum_sessions"]:
        quality_reasons.append("insufficient_paired_mature_sessions")
    if missing_labels or unmatured:
        quality_reasons.append("missing_or_unmatured_labels")
    if coverage < acceptance["minimum_label_coverage"]:
        quality_reasons.append("incomplete_label_coverage")
    # Partial-period metrics remain monitoring; they never close the fixed window.
    if paired["candidate"] and finite_count:
        frames = {}
        for role in _ROLES:
            frames[role] = pd.concat(paired[role], ignore_index=True)
            frames[role]["datetime"] = pd.to_datetime(frames[role]["datetime"])
            frames[role] = frames[role].set_index(["datetime", "instrument"]).sort_index()
        stats, _ = _prediction_statistics(frames["candidate"])
        rank_ic = stats["rank_ic"]
        metrics = {"candidate_rank_ic": float(rank_ic) if np.isfinite(rank_ic) else None}
        if not np.isfinite(rank_ic) or rank_ic < acceptance["minimum_rank_ic"]:
            quality_reasons.append("candidate_rank_ic_below_threshold")
        finite_dates = frames["candidate"].loc[frames["candidate"]["label"].notna()].index.get_level_values("datetime").nunique()
        for length in acceptance["block_lengths"]:
            if finite_dates < length * acceptance["minimum_effective_blocks"]:
                quality_reasons.append(f"insufficient_blocks_{length}")
                continue
            comparison = paired_date_block_intervals(
                frames["candidate"], frames["reference"], block_length=length,
                draws=acceptance["bootstrap_draws"], seed=acceptance["bootstrap_seed"],
            )
            comparisons.append(comparison)
            result = comparison["metrics"]["rank_ic"]
            if (
                result["lower_95"] is None
                or result["paired_usable_dates"] < acceptance["minimum_sessions"]
                or result["lower_95"] <= acceptance["minimum_paired_rank_ic_gain"]
            ):
                quality_reasons.append(f"paired_rank_ic_gain_not_established_{length}")
    else:
        quality_reasons.append("no_finite_paired_labels")
    quality_status = "pending" if not quality_closed or missing_labels else (
        "failed" if quality_reasons else "passed"
    )
    role = protocol["specification"]["evaluation_role"]
    if role == "operations_pilot":
        quality_status = "research_only"
        quality_reasons.append("operations_pilot_cannot_establish_confirmation")
        status = ("pending" if operational_status == "pending" else
                  "operations_pilot_complete" if operational_status == "passed" else "failed")
    else:
        status = "pending" if "pending" in (operational_status, quality_status) else (
            "passed_for_review" if operational_status == quality_status == "passed" else "failed"
        )
    return {
        "protocol_id": protocol["protocol_id"], "assessed_at": now.isoformat(),
        "status": status, "evaluation_role": role, "operational_acceptance": {
            "status": operational_status, "reasons": operational_reasons,
            "expected_sessions": len(sessions), "captured_sessions": len(captured),
            "success_fraction": success_fraction, "failed_attempts": len(incidents),
        },
        "candidate_quality": {
            "status": quality_status, "reasons": quality_reasons,
            "paired_mature_sessions": usable_days, "label_coverage": coverage,
            "metrics": metrics, "paired_comparisons": comparisons,
        },
        "missing_sessions": missing, "future_sessions": future,
        "unmatured_sessions": unmatured, "missing_label_sessions": missing_labels,
        "incident_owner": protocol["specification"]["incident_owner"],
        "production_ready": False, "orders_submitted": False,
        "remaining_deployment_gates": [
            "account_settlement_and_execution_reconciliation", "deployment_fault_rehearsals",
            "model_and_data_rights", "risk_limits_and_recorded_deployment_approval",
        ],
    }
