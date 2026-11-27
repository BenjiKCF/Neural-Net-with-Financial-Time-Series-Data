#!/usr/bin/env python3
"""Package exact arrival bytes and prospective state, or verify/restore them.

This command never uploads files or deletes an existing provider/download.
Checkpoint custody is distinct from a successful durable save and from the
registered experiment's operational or statistical acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath

from filelock import FileLock

MAX_PART_BYTES = 280 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular evidence file: {path}")
    return path


def _identity(path):
    path = _regular(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _read(path):
    from financial_ts.chenditc import _json_bytes

    return _json_bytes(_regular(path).read_bytes())


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Unsafe checkpoint member path")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(part in {".", ".."} for part in path.parts):
        raise ValueError("Unsafe checkpoint member path")
    return path


def _files(root, *, prune_provider=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Checkpoint source must be a real directory")
    result = {}
    for directory, names, files in os.walk(root):
        base = Path(directory)
        for name in names + files:
            path = base / name
            if path.is_symlink():
                raise ValueError("Checkpoint sources cannot contain symlinks")
        if prune_provider:
            names[:] = [name for name in names if name != "provider"]
        for name in sorted(files):
            path = _regular(base / name)
            # Lock files are process coordination, not experiment evidence.
            if name.endswith(".lock"):
                continue
            if prune_provider and (name == "qlib_bin.tar.gz" or name.startswith("qlib_bin.tar.gz.part")):
                continue
            result[path.relative_to(root).as_posix()] = path
    return result


def _inventory(store, feed, runtime):
    entries = {f"store/{name}": path for name, path in _files(store).items()}
    for kind in ("sessions", "snapshots", "attempts"):
        directory = feed / kind
        if directory.exists():
            entries.update({f"feed/{kind}/{name}": path for name, path in _files(directory, prune_provider=True).items()})
    if runtime is not None:
        entries.update({f"runtime/{name}": path for name, path in _files(runtime).items()})
    return dict(sorted(entries.items()))


def _verify_store(store):
    from financial_ts.bundles import verify_bundle

    protocol = _read(store / "protocol.json")
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if protocol.get("schema_version") != 1 or protocol.get("protocol_id") != _digest(body):
        raise ValueError("Prospective protocol identity changed")
    if set(protocol.get("bundle_ids", {})) != {"candidate", "reference"}:
        raise ValueError("Prospective frozen bundle pair is missing")
    for identifier in protocol["bundle_ids"].values():
        if not isinstance(identifier, str) or len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid prospective bundle identity")
        if verify_bundle(store / "registry/bundles" / identifier)["bundle_id"] != identifier:
            raise ValueError("Prospective frozen bundle identity changed")
    return protocol


def _verify_runtime(runtime, protocol):
    if runtime is None:
        return
    candidates = (runtime / "financial_ts", runtime / "src/financial_ts", runtime)
    package = next((path for path in candidates if (path / "prospective.py").is_file()), None)
    if package is None:
        raise ValueError("Optional frozen runtime has no financial_ts source package")
    for name, expected in protocol["source_files"].items():
        relative = _relative(name)
        if _identity(package.joinpath(*relative.parts))["sha256"] != expected:
            raise ValueError("Optional runtime differs from the registered source identity")


def _validate_archive_identity(record, source_manifest, release):
    from financial_ts.chenditc import _assets

    asset = _assets(release)["qlib_bin.tar.gz"]
    if (record["bytes"] != source_manifest["archive_size_bytes"]
            or "sha256:" + record["sha256"] != source_manifest["archive_sha256"]
            or record["bytes"] != asset["size"] or "sha256:" + record["sha256"] != asset["digest"]):
        raise ValueError("Checkpoint source archive differs from the original release/manifest")


def package_checkpoint(store, feed, session, output, *, runtime_root=None, part_bytes=MAX_PART_BYTES):
    """Retain one exact source archive and a stable snapshot of all small state."""
    from financial_ts.arrival_feed import verify_arrival_metadata
    from financial_ts.chenditc import validate_chenditc_archive

    if type(part_bytes) is not int or not 1 <= part_bytes <= MAX_PART_BYTES:
        raise ValueError("Part size must be between 1 byte and 280 MiB")
    if date.fromisoformat(session).isoformat() != session:
        raise ValueError("A canonical session date is required")
    store, feed, output = [Path(path).expanduser().absolute() for path in (store, feed, output)]
    runtime = Path(runtime_root).expanduser().absolute() if runtime_root is not None else None
    roots = [store, feed] + ([runtime] if runtime else [])
    if any(root.is_symlink() or not root.is_dir() for root in roots):
        raise ValueError("Checkpoint roots must be real existing directories")
    if any(a.resolve() == b.resolve() or a.resolve() in b.resolve().parents or b.resolve() in a.resolve().parents
           for index, a in enumerate(roots) for b in roots[index + 1:]):
        raise ValueError("Store, feed and optional runtime roots must be separate")
    if output.is_symlink() or output.exists():
        raise FileExistsError("Use a new checkpoint output directory; verify an existing checkpoint separately")
    if any(output.resolve() == root.resolve() or root.resolve() in output.resolve().parents
           or output.resolve() in root.resolve().parents for root in roots):
        raise ValueError("Checkpoint output must be separate from every input root")
    with contextlib.ExitStack() as locks:
        # Same outer order as the service; direct captures also use the latter locks.
        for path in (store / ".service.lock", feed / ".collection.lock",
                     store / ".observations.lock", store / ".labels.lock"):
            locks.enter_context(FileLock(str(path), timeout=0))
        protocol = _verify_store(store)
        _verify_runtime(runtime, protocol)
        arrival_path = feed / "sessions" / session / "arrival.json"
        arrival = verify_arrival_metadata(arrival_path)
        snapshot = feed / arrival["snapshot_relative_path"]
        source = feed / "downloads" / session / "qlib_bin.tar.gz"
        source_manifest, release = _read(snapshot / "source_manifest.json"), _read(snapshot / "release.json")
        source_identity = _identity(source)
        _validate_archive_identity(source_identity, source_manifest, release)
        validation = validate_chenditc_archive(source, snapshot / "source_manifest.json",
                                               expected_tag=session, release_metadata=release)
        if validation != _read(snapshot / "provider_metadata.json")["validation"]:
            raise ValueError("Original archive validation differs from admitted provider provenance")
        entries = _inventory(store, feed, runtime)
        identities = {name: _identity(path) for name, path in entries.items()}
        output.mkdir(parents=True)
        parts = []
        with source.open("rb") as stream:
            index = 0
            while True:
                first = stream.read(min(BLOCK_BYTES, part_bytes))
                if not first:
                    break
                name = f"qlib_bin.tar.gz.part-{index:04d}"
                digest, written = hashlib.sha256(), 0
                with (output / name).open("xb") as part:
                    chunk = first
                    while chunk:
                        part.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        chunk = stream.read(min(BLOCK_BYTES, part_bytes - written)) if written < part_bytes else b""
                    part.flush()
                    os.fsync(part.fileno())
                parts.append({"name": name, "bytes": written, "sha256": digest.hexdigest()})
                index += 1
        state = output / "state.tar.gz"
        with tarfile.open(state, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            for name, path in entries.items():
                info = archive.gettarinfo(str(path), arcname=name)
                with path.open("rb") as incoming:
                    archive.addfile(info, incoming)
        # Detect changing files or membership instead of certifying a mixed-time snapshot.
        after = _inventory(store, feed, runtime)
        if set(after) != set(entries) or any(_identity(path) != identities[name] for name, path in after.items()):
            raise ValueError("Prospective evidence changed while checkpointing; preserve this attempt and retry")
        if _identity(source) != source_identity:
            raise ValueError("Source archive changed while checkpointing")
        body = {
            "schema_version": 1, "kind": "prospective_source_and_state_checkpoint", "created_at": _now(),
            "protocol_id": protocol["protocol_id"], "session": session,
            "arrival_receipt_id": arrival["receipt_id"], "arrival_receipt_sha256": _identity(arrival_path)["sha256"],
            "snapshot_relative_path": arrival["snapshot_relative_path"], "provider_fingerprint": arrival["provider_fingerprint"],
            "original_roots": {"store": str(store), "feed": str(feed), "runtime": str(runtime) if runtime else None},
            "source_archive": {"name": "qlib_bin.tar.gz", **source_identity}, "source_parts": parts,
            "maximum_part_bytes": part_bytes,
            "state_archive": {"name": state.name, **_identity(state)}, "state_files": identities,
            "source_manifest_sha256": _identity(snapshot / "source_manifest.json")["sha256"],
            "release_metadata_sha256": _identity(snapshot / "release.json")["sha256"],
            "provider_metadata_sha256": _identity(snapshot / "provider_metadata.json")["sha256"],
            "exclusions": ["materialized provider directories", "compressed download caches (selected source retained only as parts)",
                           "process lock files"],
            "receipt_bytes_rewritten": False, "durable_save_status": "not_recorded", "eviction_performed": False,
            "registered_source_files_retained": runtime is not None,
            "qualification": "Local byte custody only. Original publication, arrival and forecast times are unchanged. "
                             "Upload success and operational/statistical acceptance require their own evidence.",
        }
        manifest = {**body, "checkpoint_id": _digest(body)}
        (output / "manifest.json").write_bytes(_canonical(manifest))
    result = verify_checkpoint(output)
    return {**result, "manifest": str(output / "manifest.json"), "durable_save_status": "not_recorded"}


def _checkpoint(output):
    output = Path(output).expanduser().absolute()
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Checkpoint must be a real directory")
    manifest = _read(output / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "checkpoint_id"}
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "prospective_source_and_state_checkpoint"
            or manifest.get("checkpoint_id") != _digest(body)):
        raise ValueError("Checkpoint manifest identity changed")
    if type(manifest["maximum_part_bytes"]) is not int or not 1 <= manifest["maximum_part_bytes"] <= MAX_PART_BYTES:
        raise ValueError("Checkpoint part bound is invalid")
    expected = {"manifest.json", manifest["state_archive"]["name"]}
    for index, part in enumerate(manifest["source_parts"]):
        if part["name"] != f"qlib_bin.tar.gz.part-{index:04d}":
            raise ValueError("Source parts must have contiguous, ordered identities")
        if type(part["bytes"]) is not int or not 1 <= part["bytes"] <= manifest["maximum_part_bytes"]:
            raise ValueError("Source part exceeds the declared bound")
        expected.add(part["name"])
    if not manifest["source_parts"] or manifest["state_archive"]["name"] != "state.tar.gz":
        raise ValueError("Missing source parts or unexpected state archive")
    if {path.name for path in output.iterdir()} != expected:
        raise ValueError("Checkpoint file inventory differs")
    for name in expected:
        _relative(name)
        _regular(output / name)
    return output, manifest


def _state_members(output, manifest, *, extract_to=None):
    state = output / manifest["state_archive"]["name"]
    if _identity(state) != {key: manifest["state_archive"][key] for key in ("bytes", "sha256")}:
        raise ValueError("State archive bytes changed")
    seen, selected = set(), {}
    snapshot = "feed/" + manifest["snapshot_relative_path"]
    wanted = {"store/protocol.json", f"feed/sessions/{manifest['session']}/arrival.json",
              snapshot + "/source_manifest.json", snapshot + "/release.json", snapshot + "/provider_metadata.json"}
    with tarfile.open(state, "r|gz") as archive:
        for member in archive:
            name = _relative(member.name).as_posix()
            if (not member.isfile() or member.issym() or member.islnk() or getattr(member, "sparse", None)
                    or name in seen or name not in manifest["state_files"]
                    or name.split("/", 1)[0] not in {"store", "feed", "runtime"}):
                raise ValueError("Unsafe, duplicate or unregistered state archive member")
            parts = PurePosixPath(name).parts
            if parts[0] == "feed" and (len(parts) < 3 or parts[1] not in {"sessions", "snapshots", "attempts"}
                                       or "provider" in parts or parts[-1].startswith("qlib_bin.tar.gz")):
                raise ValueError("State archive must not duplicate provider/download bytes")
            expected = manifest["state_files"][name]
            if member.size != expected["bytes"]:
                raise ValueError("State archive member size changed")
            incoming = archive.extractfile(member)
            target = None
            if extract_to is not None:
                destination = extract_to.joinpath(*PurePosixPath(name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                target = destination.open("xb")
            digest, captured = hashlib.sha256(), bytearray()
            try:
                for chunk in iter(lambda: incoming.read(BLOCK_BYTES), b""):
                    digest.update(chunk)
                    if target is not None:
                        target.write(chunk)
                    if name in wanted:
                        if len(captured) + len(chunk) > 16 * 1024 * 1024:
                            raise ValueError("Oversized checkpoint identity document")
                        captured.extend(chunk)
            finally:
                if target is not None:
                    target.close()
                incoming.close()
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("State archive member digest changed")
            if name in wanted:
                selected[name] = bytes(captured)
            seen.add(name)
    if seen != set(manifest["state_files"]) or set(selected) != wanted:
        raise ValueError("State archive inventory is incomplete")
    protocol = json.loads(selected["store/protocol.json"])
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    arrival_name = f"feed/sessions/{manifest['session']}/arrival.json"
    arrival = json.loads(selected[arrival_name])
    arrival_body = {key: value for key, value in arrival.items() if key != "receipt_id"}
    if (protocol["protocol_id"] != manifest["protocol_id"] or _digest(protocol_body) != protocol["protocol_id"]
            or _digest(arrival_body) != arrival["receipt_id"] or arrival["receipt_id"] != manifest["arrival_receipt_id"]
            or hashlib.sha256(selected[arrival_name]).hexdigest() != manifest["arrival_receipt_sha256"]
            or arrival["snapshot_relative_path"] != manifest["snapshot_relative_path"]
            or arrival["provider_fingerprint"] != manifest["provider_fingerprint"]):
        raise ValueError("Checkpoint protocol/arrival identity changed")
    for filename, field in (("source_manifest.json", "source_manifest_sha256"),
                            ("release.json", "release_metadata_sha256"), ("provider_metadata.json", "provider_metadata_sha256")):
        if hashlib.sha256(selected[snapshot + "/" + filename]).hexdigest() != manifest[field]:
            raise ValueError("Checkpoint source provenance identity changed")
    _validate_archive_identity(manifest["source_archive"], json.loads(selected[snapshot + "/source_manifest.json"]),
                               json.loads(selected[snapshot + "/release.json"]))
    return len(seen)


def verify_checkpoint(output):
    """Verify every part, combined source identity and every archived state file."""
    output, manifest = _checkpoint(output)
    digest, size = hashlib.sha256(), 0
    for part in manifest["source_parts"]:
        path = output / part["name"]
        if _identity(path) != {key: part[key] for key in ("bytes", "sha256")}:
            raise ValueError("Source archive part bytes changed")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(BLOCK_BYTES), b""):
                digest.update(chunk)
                size += len(chunk)
    if {"name": "qlib_bin.tar.gz", "bytes": size, "sha256": digest.hexdigest()} != manifest["source_archive"]:
        raise ValueError("Reassembled source archive identity changed")
    count = _state_members(output, manifest)
    return {"status": "verified", "checkpoint_id": manifest["checkpoint_id"], "session": manifest["session"],
            "source_archive_sha256": digest.hexdigest(), "source_archive_bytes": size,
            "source_parts": len(manifest["source_parts"]), "state_files_verified": count,
            "provider_materialized": False, "receipt_bytes_rewritten": False, "eviction_performed": False}


def restore_state(output, destination):
    """Recover portable state to a fresh review root without rewriting receipts."""
    verify_checkpoint(output)
    output, manifest = _checkpoint(output)
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("State restore requires a new destination")
    destination.mkdir(parents=True)
    count = _state_members(output, manifest, extract_to=destination)
    # Empty evidence directories have no byte inventory entry but are required
    # by the frozen evaluator, including a checkpoint before its first forecast.
    for relative in (
        "store/observations", "store/labels", "store/incidents",
        "feed/sessions", "feed/snapshots", "feed/attempts",
    ):
        (destination / relative).mkdir(parents=True, exist_ok=True)
    _verify_store(destination / "store")
    if manifest["registered_source_files_retained"]:
        _verify_runtime(destination / "runtime", _read(destination / "store/protocol.json"))
    from financial_ts.arrival_feed import verify_arrival_metadata
    for path in sorted((destination / "feed/sessions").glob("*/arrival.json")):
        verify_arrival_metadata(path)
    return {"status": "restored", "checkpoint_id": manifest["checkpoint_id"], "state_files_verified": count,
            "destination": str(destination), "original_roots": manifest["original_roots"],
            "receipt_bytes_rewritten": False, "provider_materialized": False,
            "operational_resume_ready": False,
            "qualification": "Portable custody restoration preserves absolute original provenance references. "
                             "Restore original operational paths and registered runtime before resuming."}


class _PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths, self.current = iter(paths), None
        self.digest, self.bytes_read = hashlib.sha256(), 0

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Part restoration requires bounded streaming reads")
        pieces, remaining = [], size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = _regular(path).open("rb")
            chunk = self.current.read(remaining)
            if not chunk:
                self.current.close()
                self.current = None
                continue
            pieces.append(chunk)
            remaining -= len(chunk)
            self.digest.update(chunk)
            self.bytes_read += len(chunk)
        return b"".join(pieces)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def restore_provider(output, feed):
    """Restore the selected offloaded provider at its original relative location."""
    from financial_ts.arrival_feed import verify_arrival, verify_arrival_metadata
    from financial_ts.chenditc import _member_path, inspect_chenditc_provider
    from financial_ts.workflow import _provider_fingerprint

    verify_checkpoint(output)
    output, manifest = _checkpoint(output)
    feed = Path(feed).expanduser().absolute()
    if feed.is_symlink() or not feed.is_dir():
        raise ValueError("Restore requires the existing retained feed metadata")
    with FileLock(str(feed / ".collection.lock"), timeout=0):
        receipt_path = feed / "sessions" / manifest["session"] / "arrival.json"
        arrival = verify_arrival_metadata(receipt_path)
        if arrival["receipt_id"] != manifest["arrival_receipt_id"] or _identity(receipt_path)["sha256"] != manifest["arrival_receipt_sha256"]:
            raise ValueError("Selected checkpoint belongs to a different arrival")
        provider = Path(arrival["provider_dir"])
        if provider.exists() or provider.is_symlink():
            existing = verify_arrival(receipt_path)
            return {"status": "verified_existing", "provider_dir": existing["provider_dir"],
                    "provider_bytes_verified": True, "checkpoint_id": manifest["checkpoint_id"],
                    "receipt_bytes_rewritten": False, "eviction_performed": False}
        snapshot = provider.parent
        for filename, key in (("provider_metadata.json", "provider_metadata_sha256"),
                              ("release.json", "release_metadata_sha256"), ("source_manifest.json", "source_manifest_sha256")):
            if _identity(snapshot / filename)["sha256"] != manifest[key]:
                raise ValueError("Retained restoration provenance differs from the checkpoint")
        metadata = _read(snapshot / "provider_metadata.json")
        stage = Path(tempfile.mkdtemp(prefix=".provider-restore-", dir=feed.parent))
        seen = set()
        try:
            with _PartsReader(output / part["name"] for part in manifest["source_parts"]) as incoming:
                with tarfile.open(fileobj=incoming, mode="r|gz") as archive:
                    for member in archive:
                        name = _member_path(member)
                        if name in seen:
                            raise ValueError("Duplicate original source archive member")
                        seen.add(name)
                        if not name:
                            continue
                        destination = stage.joinpath(*PurePosixPath(name).parts)
                        if member.isdir():
                            destination.mkdir(parents=True, exist_ok=True)
                        else:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with archive.extractfile(member) as source, destination.open("xb") as target:
                                shutil.copyfileobj(source, target, BLOCK_BYTES)
                while incoming.read(BLOCK_BYTES):
                    pass
                if (incoming.bytes_read != manifest["source_archive"]["bytes"]
                        or incoming.digest.hexdigest() != manifest["source_archive"]["sha256"]):
                    raise ValueError("Source parts changed during provider restoration")
            if len(seen) != metadata["validation"]["archive_member_count"]:
                raise ValueError("Restored original archive member count differs")
            coverage = inspect_chenditc_provider(stage, target_trade_date=manifest["session"])
            if any(metadata.get(key) != value for key, value in coverage.items()):
                raise ValueError("Restored coverage differs from original provenance")
            (stage / "metadata.json").write_bytes((snapshot / "provider_metadata.json").read_bytes())
            if _provider_fingerprint(stage) != arrival["provider_fingerprint"]:
                raise ValueError("Restored provider bytes differ from original admission")
            stage.rename(provider)
            verified = verify_arrival(receipt_path)
        except Exception as error:
            raise ValueError(f"Provider restore failed; original receipts unchanged and staging retained at {stage}: {error}") from error
    return {"status": "restored", "restored_at": _now(), "checkpoint_id": manifest["checkpoint_id"],
            "arrival_receipt_id": verified["receipt_id"], "provider_dir": verified["provider_dir"],
            "provider_fingerprint": verified["provider_fingerprint"], "provider_bytes_verified": True,
            "receipt_bytes_rewritten": False, "eviction_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    for name in ("store", "feed", "session", "output"):
        package.add_argument("--" + name, required=True)
    package.add_argument("--runtime-root")
    package.add_argument("--part-bytes", type=int, default=MAX_PART_BYTES)
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    state = commands.add_parser("restore-state")
    state.add_argument("--checkpoint", required=True)
    state.add_argument("--destination", required=True)
    provider = commands.add_parser("restore-provider")
    provider.add_argument("--checkpoint", required=True)
    provider.add_argument("--feed", required=True)
    args = parser.parse_args()
    if args.command == "package":
        result = package_checkpoint(args.store, args.feed, args.session, args.output,
                                    runtime_root=args.runtime_root, part_bytes=args.part_bytes)
    elif args.command == "verify":
        result = verify_checkpoint(args.checkpoint)
    elif args.command == "restore-state":
        result = restore_state(args.checkpoint, args.destination)
    else:
        result = restore_provider(args.checkpoint, args.feed)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
