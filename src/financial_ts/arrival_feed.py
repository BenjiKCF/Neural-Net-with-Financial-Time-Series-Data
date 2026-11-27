"""Observe and retain real chenditc arrivals before a prospective cutoff.

These are local-clock observations, not an independent timestamp authority.
The collector never derives an arrival time from the trading date or an older
download. The source publication, completed acquisition, complete snapshot audit
and durable admission all precede the decision cutoff. Forecasting is separate.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from financial_ts import chenditc
from financial_ts.bundles import (
    atomic_json,
    canonical_json,
    environment_versions,
    file_hash,
    write_json,
)
from financial_ts.workflow import _provider_fingerprint


def _now():
    # Deliberately no public wall-clock override. Tests replace this private seam.
    return pd.Timestamp.now(tz="UTC")


def _aware(value, name):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is None:
        raise ValueError(f"{name} requires a finite timezone-aware timestamp.")
    return result.tz_convert("UTC")


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Arrival evidence must be a regular, non-symlinked file.")
    return chenditc._json_bytes(path.read_bytes())


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _collector_source():
    """Retain the acquisition/audit implementation without changing inference pins."""
    root = Path(__file__).parent
    sources = {}
    for name in ("arrival_feed.py", "chenditc.py", "workflow.py", "bundles.py"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Collector source must be retained from regular files.")
        raw = path.read_bytes()
        sources[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8"),
        }
    body = {
        "schema_version": 1,
        "kind": "arrival_collector_source",
        "sources": sources,
        "environment": environment_versions(),
    }
    return {**body, "collector_id": _digest(body)}


def _verify_collector_source(path):
    source = _read(path)
    body = {key: value for key, value in source.items() if key != "collector_id"}
    if (
        source.get("schema_version") != 1
        or source.get("kind") != "arrival_collector_source"
        or _digest(body) != source.get("collector_id")
        or set(source.get("sources", {}))
        != {"arrival_feed.py", "chenditc.py", "workflow.py", "bundles.py"}
    ):
        raise ValueError("Retained collector source identity changed.")
    for value in source["sources"].values():
        if (
            not isinstance(value.get("text"), str)
            or hashlib.sha256(value["text"].encode()).hexdigest() != value.get("sha256")
        ):
            raise ValueError("Retained collector source bytes changed.")
    return source


def _release_identity(metadata):
    assets = chenditc._assets(metadata)
    if metadata.get("draft") or metadata.get("prerelease"):
        raise ValueError("A stable, published source release is required.")
    if type(metadata.get("id")) is not int or metadata["id"] <= 0:
        raise ValueError("Source release needs an immutable numeric identity.")
    published = _aware(metadata.get("published_at"), "release published_at")
    records = {}
    times = [published]
    for name, asset in assets.items():
        if type(asset.get("id")) is not int or asset["id"] <= 0:
            raise ValueError("Source assets need immutable numeric identities.")
        created = _aware(asset.get("created_at"), "asset created_at")
        updated = _aware(asset.get("updated_at"), "asset updated_at")
        if updated < created:
            raise ValueError("Source asset update precedes its creation.")
        times.extend((created, updated))
        records[name] = {key: asset[key] for key in (
            "id", "name", "size", "digest", "browser_download_url", "created_at", "updated_at"
        )}
    return {
        "release_id": metadata["id"], "release_tag": metadata["tag_name"],
        "published_at": published.isoformat(), "assets": records,
        "data_available_at": max(times).isoformat(),
    }


def _safe_tree(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Snapshot must be a regular directory.")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Links and special files are not valid snapshot evidence.")


def _binary(path, calendar_length):
    if path.stat().st_size % 4:
        raise ValueError(f"Incomplete float32 binary: {path.name}.")
    array = np.fromfile(path, dtype="<f4")
    if (
        len(array) < 2 or not np.isfinite(array[0]) or array[0] != int(array[0])
        or array[0] < 0 or int(array[0]) + len(array) - 1 > calendar_length
        or np.isinf(array[1:]).any()
    ):
        raise ValueError(f"Invalid offset, extent or values in binary: {path.name}.")
    return int(array[0]), array[1:]


def _membership(path, through):
    result = []
    for line in path.read_text().splitlines():
        symbol, start, end = line.split("\t")
        if start <= through:
            result.append((symbol, start, min(end, through)))
    return sorted(result)


def audit_provider_snapshots(provider, *, previous_provider=None):
    """Stream every consumed binary, with exact historical-vintage comparisons.

    New observations may extend existing series. Historical cells, instrument
    membership and observed-calendar history may not change silently. This
    strict collector has no blanket revision override: explained adjustments
    require a separately retained and reviewed admission workflow. Examples are
    bounded in the summary; counts still include every changed historical cell.
    """
    provider = Path(provider)
    previous = Path(previous_provider) if previous_provider is not None else None
    _safe_tree(provider)
    coverage = chenditc.inspect_chenditc_provider(provider)
    calendar = (provider / "calendars/day.txt").read_text().splitlines()
    paths = {p.relative_to(provider).as_posix(): p for p in provider.glob("features/*/*.day.bin")}
    symbols = {line.split("\t")[0].lower() for line in (provider / "instruments/all.txt").read_text().splitlines()}
    for symbol in symbols:
        for field in ("open", "high", "low", "close", "volume", "factor"):
            if f"features/{symbol}/{field}.day.bin" not in paths:
                raise ValueError(f"Snapshot is missing required OHLCV/adjustment data: {symbol}/{field}.")
    errors, examples, changed_cells = [], [], 0
    old_calendar, old_paths = [], {}
    if previous is not None:
        _safe_tree(previous)
        old_calendar = (previous / "calendars/day.txt").read_text().splitlines()
        old_paths = {p.relative_to(previous).as_posix(): p for p in previous.glob("features/*/*.day.bin")}
        if calendar[:len(old_calendar)] != old_calendar:
            errors.append("observed_calendar_history_changed")
        if set(old_paths) - set(paths):
            errors.append("historical_feature_files_removed")
        for name in chenditc.UNIVERSES:
            if _membership(provider / f"instruments/{name}.txt", old_calendar[-1]) != _membership(
                previous / f"instruments/{name}.txt", old_calendar[-1]
            ):
                errors.append(f"historical_membership_changed:{name}")
    for name, path in sorted(paths.items()):
        start, values = _binary(path, len(calendar))
        if name not in old_paths:
            continue
        old_start, old_values = _binary(old_paths[name], len(old_calendar))
        if start > old_start or start + len(values) < old_start + len(old_values):
            errors.append(f"historical_rows_removed:{name}")
            continue
        if calendar[:len(old_calendar)] != old_calendar:
            continue  # Index alignment is invalid; quarantine already applies.
        common = values[old_start - start:old_start - start + len(old_values)]
        changed = np.flatnonzero(~((common == old_values) | (np.isnan(common) & np.isnan(old_values))))
        changed_cells += len(changed)
        for offset in changed[:max(0, 100 - len(examples))]:
            def scalar(value):
                return float(value) if np.isfinite(value) else None
            examples.append({
                "path": name, "date": old_calendar[old_start + int(offset)],
                "previous": scalar(old_values[offset]), "current": scalar(common[offset]),
            })
    if changed_cells:
        errors.append("unexplained_historical_revisions")
    return {
        "schema_version": 1, "status": "quarantined" if errors else "admitted",
        "errors": errors, "coverage": coverage, "feature_files_checked": len(paths),
        "previous_feature_files": len(old_paths), "new_feature_files": len(set(paths) - set(old_paths)),
        "historical_cells_changed": int(changed_cells), "revision_examples": examples,
        "revision_examples_limit": 100, "complete_vintages_retained": True,
        "comparison": "complete prior observed history" if previous else "initial snapshot; no prior-vintage comparison",
        "policy": "No unexplained historical value, membership, row or calendar revisions",
        "prospective_forecasts_captured": False,
    }


def _snapshot_files(directory):
    directory = Path(directory)
    _safe_tree(directory)
    return {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}


def verify_arrival_metadata(receipt_path):
    """Audit retained receipt/source metadata without claiming materialized data bytes.

    This remains usable after an independently verified archive offload removes
    an older provider directory. It is insufficient for inference, new label
    computation or the next snapshot's complete historical-vintage comparison.
    """
    receipt_path = Path(receipt_path)
    receipt = _read(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    if _digest(body) != receipt.get("receipt_id"):
        raise ValueError("Arrival receipt identity changed.")
    if receipt.get("status") != "admitted" or receipt.get("snapshot_audit_status") != "admitted":
        raise ValueError("Only an admitted snapshot has reusable source metadata.")
    if receipt.get("publication_status") != "committed":
        raise ValueError("Arrival has no completed durable publication receipt.")
    feed = receipt_path.parent.parent.parent
    snapshot = feed / receipt["snapshot_relative_path"]
    if snapshot.resolve().parent != (feed / "snapshots").resolve():
        raise ValueError("Snapshot reference escapes the retained feed store.")
    if _snapshot_files(snapshot) != receipt["snapshot_files"]:
        raise ValueError("Retained snapshot metadata or audit changed.")
    provider = snapshot / "provider"
    acquisition = _read(snapshot / "acquisition.json")
    collector = _verify_collector_source(snapshot / "collector_source.json")
    if (
        collector["collector_id"] != receipt.get("collector_id")
        or collector["collector_id"] != acquisition.get("collector_id")
    ):
        raise ValueError("Collector identity differs from the retained acquisition/arrival.")
    retained_metadata_path = snapshot / "provider_metadata.json"
    if file_hash(retained_metadata_path) != acquisition["provider_metadata_sha256"]:
        raise ValueError("Retained provider provenance changed after admission.")
    metadata = _read(retained_metadata_path)
    source = _release_identity(_read(snapshot / "release.json"))
    installed_source = _release_identity({
        "id": metadata["release_id"], "tag_name": metadata["release_tag"],
        "published_at": metadata["release_published_at"],
        "assets": list(metadata["assets"].values()),
    })
    manifest = _read(snapshot / "source_manifest.json")
    audit = _read(snapshot / "snapshot_audit.json")
    if (
        source != receipt["source_release"]
        or source != acquisition["source_identity"]
        or source != installed_source
        or source["release_tag"] != receipt["expected_session"]
        or source["data_available_at"] != receipt["data_available_at"]
        or metadata.get("source") != chenditc.REPOSITORY
        or metadata.get("synthetic") is not False
        or metadata.get("region") != "cn"
        or metadata.get("frequency") != "day"
        or metadata.get("target_trade_date") != receipt["expected_session"]
        or manifest != metadata["validation"]["manifest"]
        or "sha256:" + file_hash(snapshot / "source_manifest.json")
        != metadata["validation"]["manifest_sha256"]
        or audit.get("status") != "admitted"
        or any(metadata.get(key) != value for key, value in audit["coverage"].items())
    ):
        raise ValueError("Retained source, provenance and admission audit disagree.")
    for key in (
        "expected_session", "decision_cutoff", "started_at", "discovered_at",
        "acquisition_complete_at", "observed_at", "provider_fingerprint", "previous_receipt_id",
    ):
        if acquisition.get(key) != receipt.get(key):
            raise ValueError("Retained acquisition and arrival metadata disagree.")
    published, started, discovered, acquired, observed, committed, cutoff = (
        _aware(receipt[key], key) for key in (
            "data_available_at", "started_at", "discovered_at", "acquisition_complete_at",
            "observed_at", "committed_at", "decision_cutoff",
        )
    )
    if (
        not published <= discovered
        or not started <= discovered <= acquired <= observed <= committed <= cutoff
        or committed > _now()
        or cutoff.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != receipt["expected_session"]
        or started.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != receipt["expected_session"]
    ):
        raise ValueError("Arrival clock chronology is invalid.")
    return {
        **receipt, "provider_dir": str(provider.absolute()),
        "arrival_receipt": str(receipt_path.absolute()),
        "provider_bytes_verified": False,
        "verification_scope": "receipt_and_retained_metadata_only",
    }


def verify_arrival(receipt_path):
    """Verify retained metadata and every materialized provider byte before use."""
    receipt = verify_arrival_metadata(receipt_path)
    provider = Path(receipt["provider_dir"])
    if not provider.is_dir() or provider.is_symlink():
        raise ValueError("Provider bytes are not materialized; restore the verified source archive before use.")
    snapshot = provider.parent
    if file_hash(provider / "metadata.json") != file_hash(snapshot / "provider_metadata.json"):
        raise ValueError("Retained provider provenance changed after admission.")
    if _provider_fingerprint(provider) != receipt["provider_fingerprint"]:
        raise ValueError("Retained provider bytes changed after admission.")
    return {
        **receipt, "provider_bytes_verified": True,
        "verification_scope": "retained_metadata_and_full_provider_bytes",
    }


def _previous(feed, session):
    paths = sorted((feed / "sessions").glob("*/arrival.json"))
    earlier = [path for path in paths if path.parent.name < session]
    if any(path.parent.name > session for path in paths):
        raise ValueError("Cannot insert a historical arrival before an existing later admission.")
    return verify_arrival(earlier[-1]) if earlier else None


def collect_arrival(feed_root, *, expected_session, decision_cutoff, cache_dir=None):
    """Acquire one exact-date source release using actual observation timestamps.

    A successful repeated call verifies the original receipt, including when
    retried after cutoff. Failed attempts remain separate; a retry cannot reset
    their clocks or admit a snapshot whose acquisition/audit ran late.
    """
    session = chenditc._date(expected_session)
    cutoff = _aware(decision_cutoff, "decision_cutoff")
    if cutoff.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != session:
        raise ValueError("Decision cutoff and expected exchange session differ.")
    feed = Path(feed_root).expanduser().absolute()
    if feed.is_symlink():
        raise ValueError("Feed root cannot be a symlink.")
    feed.mkdir(parents=True, exist_ok=True)
    for name in ("attempts", "snapshots", "sessions"):
        directory = feed / name
        if directory.is_symlink():
            raise ValueError("Feed evidence directories cannot be symlinks.")
        directory.mkdir(exist_ok=True)
    with FileLock(str(feed / ".collection.lock"), timeout=0):
        receipt_path = feed / "sessions" / session / "arrival.json"
        if receipt_path.exists():
            existing = verify_arrival(receipt_path)
            if _aware(existing["decision_cutoff"], "cutoff") != cutoff:
                raise ValueError("Existing arrival has a different immutable cutoff.")
            return {**existing, "reused": True}
        started = _now()
        attempt = feed / "attempts" / (started.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex)
        attempt.mkdir()
        request = {
            "schema_version": 1, "expected_session": session,
            "decision_cutoff": cutoff.isoformat(), "started_at": started.isoformat(),
            "clock_qualification": "trusted local wall clock; not an independent timestamp authority",
        }
        write_json(attempt / "request.json", request)
        temporary = None
        try:
            if started > cutoff or started.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") != session:
                raise ValueError("First arrival collection must occur on its exchange session before cutoff.")
            collector = _collector_source()
            write_json(attempt / "collector_source.json", collector)
            previous = _previous(feed, session)
            metadata = chenditc.resolve_chenditc_release(session)
            discovered = _now()
            write_json(attempt / "release.json", metadata)
            source_identity = _release_identity(metadata)
            if source_identity["release_tag"] != session:
                raise ValueError("Source release does not match the expected session.")
            available = _aware(source_identity["data_available_at"], "source availability")
            if available > discovered or discovered > cutoff:
                raise ValueError("Source release/assets are future-dated or arrived after cutoff.")
            # Leave room for the compressed asset, expanded immutable snapshot
            # and failed-attempt custody; do not silently delete older evidence.
            required_space = max(3 * 1024**3, 4 * source_identity["assets"]["qlib_bin.tar.gz"]["size"])
            free_space = shutil.disk_usage(feed).free
            write_json(attempt / "space_check.json", {
                "checked_at": _now().isoformat(), "free_bytes": free_space,
                "required_bytes": required_space, "status": "passed" if free_space >= required_space else "failed",
            })
            if free_space < required_space:
                raise ValueError("Insufficient free space to retain a complete immutable arrival snapshot.")
            temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=feed / "snapshots"))
            provider = temporary / "provider"
            download_cache = Path(cache_dir).expanduser() if cache_dir else feed / "downloads" / session
            downloaded = chenditc.download_chenditc(provider, release=session, cache_dir=download_cache)
            acquired = _now()
            write_json(temporary / "release.json", metadata)
            shutil.copyfile(download_cache / "qlib_bin.manifest.json", temporary / "source_manifest.json")
            installed_identity = _release_identity({
                "id": downloaded["release_id"], "tag_name": downloaded["release_tag"],
                "published_at": downloaded["release_published_at"],
                "assets": list(downloaded["assets"].values()),
            })
            if source_identity != installed_identity:
                raise ValueError("Release assets changed between discovery and acquisition.")
            if downloaded["target_trade_date"] != session:
                raise ValueError("Released provider is stale for the expected session.")
            if acquired > cutoff:
                raise ValueError("Provider acquisition completed after the decision cutoff.")
            audit = audit_provider_snapshots(
                provider, previous_provider=previous["provider_dir"] if previous else None
            )
            write_json(temporary / "snapshot_audit.json", audit)
            if audit["status"] != "admitted":
                raise ValueError("Provider snapshot quarantined: " + ", ".join(audit["errors"]))
            fingerprint = _provider_fingerprint(provider)
            if _collector_source()["collector_id"] != collector["collector_id"]:
                raise ValueError("Collector source or environment changed during acquisition.")
            observed = _now()
            if observed > cutoff:
                raise ValueError("Snapshot verification completed after the decision cutoff.")
            write_json(temporary / "acquisition.json", {
                **request, "discovered_at": discovered.isoformat(),
                "acquisition_complete_at": acquired.isoformat(), "observed_at": observed.isoformat(),
                "source_identity": source_identity, "provider_fingerprint": fingerprint,
                "previous_receipt_id": previous["receipt_id"] if previous else None,
                "provider_metadata_sha256": file_hash(provider / "metadata.json"),
                "collector_id": collector["collector_id"],
            })
            write_json(temporary / "collector_source.json", collector)
            shutil.copyfile(provider / "metadata.json", temporary / "provider_metadata.json")
            identity = _digest({"provider": fingerprint, "source": source_identity, "observed": observed.isoformat()})
            destination = feed / "snapshots" / f"{session}-{identity}"
            os.rename(temporary, destination)
            temporary = None
            body = {
                **request, "status": "admitted", "snapshot_audit_status": "admitted",
                "publication_status": "prepared",
                "data_available_at": available.isoformat(), "discovered_at": discovered.isoformat(),
                "acquisition_complete_at": acquired.isoformat(), "observed_at": observed.isoformat(),
                "committed_at": _now().isoformat(), "provider_fingerprint": fingerprint,
                "snapshot_relative_path": str(destination.relative_to(feed)),
                "snapshot_files": _snapshot_files(destination),
                "source": str((destination / "snapshot_audit.json").absolute()),
                "source_release": source_identity, "previous_receipt_id": previous["receipt_id"] if previous else None,
                "collector_id": collector["collector_id"],
                "orders_submitted": False, "prospective_forecasts_captured": False,
            }
            if _aware(body["committed_at"], "committed_at") > cutoff:
                raise ValueError("Durable snapshot admission completed after the decision cutoff.")
            receipt_path.parent.mkdir(exist_ok=False)
            atomic_json(receipt_path, {**body, "receipt_id": _digest(body)})
            committed = _now()
            if committed > cutoff:
                rejected = attempt / "late_arrival.json"
                os.rename(receipt_path, rejected)
                receipt_path.parent.rmdir()
                raise ValueError("Durable arrival publication completed after the decision cutoff.")
            # Include the final receipt write in the actual observation chronology.
            body["committed_at"] = committed.isoformat()
            body["publication_status"] = "committed"
            atomic_json(receipt_path, {**body, "receipt_id": _digest(body)})
            if _now() > cutoff:
                os.rename(receipt_path, attempt / "late_arrival.json")
                receipt_path.parent.rmdir()
                raise ValueError("Final arrival receipt write exceeded the decision cutoff.")
            result = verify_arrival(receipt_path)
            write_json(attempt / "outcome.json", {
                "status": "admitted", "receipt_id": result["receipt_id"], "finished_at": _now().isoformat(),
            })
            return {**result, "reused": False}
        except Exception as error:
            if temporary is not None and temporary.exists():
                # Retain quarantined/late bytes rather than deleting failed evidence.
                os.rename(temporary, attempt / "rejected_snapshot")
                temporary = None
            write_json(attempt / "outcome.json", {
                "status": "failed", "finished_at": _now().isoformat(),
                "error_type": type(error).__name__, "error": str(error),
            })
            raise
        finally:
            if temporary is not None:
                shutil.rmtree(temporary, ignore_errors=True)
