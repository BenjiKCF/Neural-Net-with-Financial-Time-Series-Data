#!/usr/bin/env python3
"""Export and independently audit a portable rolling-campaign evidence archive.

Source files, protocol identities and attempt receipts are never rewritten.
Large Qlib handler caches and unused MLflow internals are excluded, but every
receipt-listed artifact is retained, including native checkpoint bytes. Native
pickle files are hashed as opaque evidence and are never deserialized here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import math
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = 1
ARCHIVE_NAME = "rolling_evidence.tar.xz"
ARCHIVE_ROOT = "campaign"
EXCLUDED_DIRECTORIES = {"cache", "mlruns", "__pycache__"}
AUDITED_PREFIX = "audited_summary"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Archive paths must be nonempty POSIX relative paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"..", "."} for part in path.parts):
        raise ValueError(f"Unsafe relative evidence path: {value}")
    if str(path) != value or not path.parts:
        raise ValueError(f"Noncanonical relative evidence path: {value}")
    return path


def regular_file(root, relative):
    relative = safe_relative(relative)
    current = Path(root)
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlinks are not permitted in retained evidence: {relative}")
    if not current.is_file():
        raise ValueError(f"Required evidence file is missing: {relative}")
    return current


def validate_protocol(protocol):
    body = {
        key: value
        for key, value in protocol.items()
        if key not in {"protocol_sha256", "registered_at_utc"}
    }
    if protocol.get("protocol_sha256") != digest(body):
        raise ValueError("Original campaign protocol identity is invalid.")
    trials = protocol.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("Protocol must register a nonempty trial list.")
    ids = [trial.get("id") for trial in trials]
    if len(set(ids)) != len(ids):
        raise ValueError("Registered trial identifiers are duplicated.")
    for trial_id in ids:
        safe_relative(trial_id)
    return {trial["id"]: trial for trial in trials}


def validate_receipts(protocol, receipts, *, artifact_identities=None):
    """Validate every attempt and, when supplied, every artifact's measured hash."""
    trials = validate_protocol(protocol)
    attempts = {trial_id: [] for trial_id in trials}
    expected_artifacts = {}
    for path, receipt in receipts.items():
        parts = safe_relative(path).parts
        if len(parts) < 4 or parts[0] != "trials" or parts[-1] != "receipt.json":
            raise ValueError(f"Unexpected receipt location: {path}")
        trial_id = "/".join(parts[1:-2])
        attempt_name = parts[-2]
        if trial_id not in trials or not attempt_name.startswith("attempt-"):
            raise ValueError(f"Receipt has no registered trial: {path}")
        try:
            attempt_number = int(attempt_name.removeprefix("attempt-"))
        except ValueError as error:
            raise ValueError(f"Malformed attempt directory: {path}") from error
        if attempt_name != f"attempt-{attempt_number:04d}" or attempt_number < 1:
            raise ValueError(f"Malformed attempt sequence: {path}")
        if (
            receipt.get("trial_id") != trial_id
            or receipt.get("trial_sha256") != digest(trials[trial_id])
            or receipt.get("protocol_sha256") != protocol["protocol_sha256"]
            or receipt.get("attempt") != attempt_number
        ):
            raise ValueError(f"Attempt receipt does not match original registration: {path}")
        status = receipt.get("status")
        if status not in {"completed", "failed"}:
            raise ValueError(f"Campaign still contains a nonterminal attempt: {path}")
        if status == "completed" and not receipt.get("artifacts"):
            raise ValueError(f"Completed attempt lacks an artifact manifest: {path}")
        expected_previous = (
            "/".join(parts[:-2]) + f"/attempt-{attempt_number - 1:04d}"
            if attempt_number > 1
            else None
        )
        if receipt.get("previous_attempt") != expected_previous:
            raise ValueError(f"Attempt predecessor chain is inconsistent: {path}")
        attempts[trial_id].append((attempt_number, path, receipt))
        for relative, identity in receipt.get("artifacts", {}).items():
            safe_relative(relative)
            full = str(PurePosixPath(path).parent / "workflow" / relative)
            if (
                not isinstance(identity, dict)
                or type(identity.get("bytes")) is not int
                or identity["bytes"] < 0
                or not isinstance(identity.get("sha256"), str)
                or len(identity["sha256"]) != 64
            ):
                raise ValueError(f"Malformed artifact identity: {full}")
            if full in expected_artifacts and expected_artifacts[full] != identity:
                raise ValueError(f"Contradictory artifact identities: {full}")
            expected_artifacts[full] = identity
    latest, historical_failures = [], 0
    for trial_id, rows in attempts.items():
        if not rows:
            raise ValueError(f"Registered trial has no terminal receipt: {trial_id}")
        rows.sort()
        if [row[0] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Attempt numbers are duplicated or incomplete: {trial_id}")
        for _, _, receipt in rows:
            historical_failures += receipt["status"] == "failed"
        number, path, receipt = rows[-1]
        latest.append(
            {
                "trial_id": trial_id,
                "attempt": number,
                "receipt_path": path,
                "status": receipt["status"],
                "error": receipt.get("error"),
            }
        )
    if artifact_identities is not None:
        for path, expected in expected_artifacts.items():
            actual = artifact_identities.get(path)
            if actual is None:
                raise ValueError(f"Receipt-listed artifact missing from archive: {path}")
            if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
                raise ValueError(f"Receipt-listed artifact failed independent hash audit: {path}")
    return {
        "registered_trials": len(trials),
        "attempts": len(receipts),
        "completed_trials": sum(row["status"] == "completed" for row in latest),
        "failed_trials": sum(row["status"] == "failed" for row in latest),
        "failed_attempts": historical_failures,
        "receipt_listed_artifacts": len(expected_artifacts),
        "latest_trials": latest,
    }, expected_artifacts


def collect_source(source):
    source = Path(source).resolve()
    protocol_path = regular_file(source, "protocol.json")
    protocol = json.loads(protocol_path.read_text())
    trials = validate_protocol(protocol)
    receipts = {}
    for trial_id in trials:
        parent = source / "trials" / trial_id
        if parent.is_symlink():
            raise ValueError("Symlinked trial directories are not permitted.")
        for attempt in sorted(parent.glob("attempt-*")):
            if not attempt.is_dir() or attempt.is_symlink():
                raise ValueError("Attempts must be real directories.")
            relative = str((attempt / "receipt.json").relative_to(source))
            receipts[relative] = json.loads(regular_file(source, relative).read_text())
    summary, required = validate_receipts(protocol, receipts)
    selected = set()
    for directory, dirnames, filenames in os.walk(source, followlinks=False):
        root = Path(directory)
        kept = []
        for name in dirnames:
            path = root / name
            if path.is_symlink():
                raise ValueError(f"Symlinked evidence directory is not allowed: {path}")
            if name not in EXCLUDED_DIRECTORIES:
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            path = root / name
            if name.endswith(".lock") or name.endswith(".pyc"):
                continue
            relative = str(path.relative_to(source))
            regular_file(source, relative)
            selected.add(relative)
    # An exact receipt-listed path overrides the broad MLflow/cache exclusion.
    # In this campaign this preserves only compact native params.pkl snapshots.
    exceptions = sorted(set(required) - selected)
    selected.update(required)
    identities = {}
    for relative in sorted(selected):
        path = regular_file(source, relative)
        identities[relative] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    validate_receipts(protocol, receipts, artifact_identities=identities)
    comparison = json.loads(regular_file(source, "comparison.json").read_text())
    expected_status = "failed" if summary["failed_trials"] else "completed"
    if (
        comparison.get("protocol_sha256") != protocol["protocol_sha256"]
        or comparison.get("registered_trials") != summary["registered_trials"]
        or comparison.get("completed_trials") != summary["completed_trials"]
        or comparison.get("status") != expected_status
    ):
        raise ValueError("Campaign comparison is stale or not finalized for all terminal trials.")
    unknown_receipts = {
        path
        for path in selected
        if path.startswith("trials/")
        and path.endswith("/receipt.json")
        and "/workflow/" not in path
    } - set(receipts)
    if unknown_receipts:
        raise ValueError(f"Unregistered attempt receipts found: {sorted(unknown_receipts)}")
    return protocol, receipts, summary, identities, exceptions, comparison


def add_audited_summary(source, audited, protocol, summary, identities):
    audited = Path(audited).expanduser().resolve()
    if audited == source or source in audited.parents or audited in source.parents:
        raise ValueError("Audited summary must be separate from the original campaign.")
    if any(PurePosixPath(path).parts[0] == AUDITED_PREFIX for path in identities):
        raise ValueError("Reserved audited-summary archive prefix already exists in source.")
    receipt = json.loads(regular_file(audited, "analysis_receipt.json").read_text())
    comparison = json.loads(regular_file(audited, "comparison.json").read_text())
    if (
        receipt.get("protocol_sha256") != protocol["protocol_sha256"]
        or receipt.get("registered_source_code_sha256") != protocol["source_code_sha256"]
        or receipt.get("original_comparison", {}).get("sha256")
        != identities["comparison.json"]["sha256"]
    ):
        raise ValueError("Audited analysis receipt does not bind the original campaign summary.")
    if (
        comparison.get("protocol_sha256") != protocol["protocol_sha256"]
        or comparison.get("registered_trials") != summary["registered_trials"]
        or comparison.get("completed_trials") != summary["completed_trials"]
        or comparison.get("status") != ("failed" if summary["failed_trials"] else "completed")
    ):
        raise ValueError("Audited comparison is not finalized for the original trials.")
    for directory, dirnames, filenames in os.walk(audited, followlinks=False):
        root = Path(directory)
        for name in dirnames:
            if (root / name).is_symlink():
                raise ValueError("Symlinked audited-summary directories are not permitted.")
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRECTORIES]
        for name in filenames:
            if name.endswith((".lock", ".pyc")):
                continue
            relative = str((root / name).relative_to(audited))
            path = regular_file(audited, relative)
            identities[f"{AUDITED_PREFIX}/{relative}"] = {
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            }
    return audited, receipt, comparison


def source_file(source, relative, audited_source=None):
    parts = safe_relative(relative).parts
    if audited_source is not None and parts[0] == AUDITED_PREFIX:
        return regular_file(audited_source, str(PurePosixPath(*parts[1:])))
    return regular_file(source, relative)


class HashingReader:
    """Check exactly the source bytes delivered to the compressor."""

    def __init__(self, stream):
        self.stream, self.sha256, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.sha256.update(value)
        self.count += len(value)
        return value


def write_archive(source, destination, identities, *, dictionary_mib=64, audited_source=None):
    if dictionary_mib not in {16, 32, 64, 128}:
        raise ValueError("XZ dictionary must be one of 16, 32, 64 or 128 MiB.")
    # Group identical artifact names together; repeated NPZ fixture members then
    # remain within the XZ dictionary even when workflow directories are large.
    order = sorted(identities, key=lambda value: (PurePosixPath(value).name, value))
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": dictionary_mib * 1024 * 1024}]
    with lzma.open(
        destination, "wb", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, filters=filters
    ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for relative in order:
                path = source_file(source, relative, audited_source)
                expected = identities[relative]
                entry = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{relative}")
                entry.size = expected["bytes"]
                entry.mode = 0o644
                entry.uid = entry.gid = 0
                entry.uname = entry.gname = ""
                entry.mtime = 0
                with path.open("rb") as stream:
                    checked = HashingReader(stream)
                    archive.addfile(entry, checked)
                if (
                    checked.count != expected["bytes"]
                    or checked.sha256.hexdigest() != expected["sha256"]
                ):
                    raise ValueError(f"Source changed while writing the archive: {relative}")
    return order


def audit_archive(archive_path, manifest):
    """Stream every member, reject unsafe paths/types, verify every artifact.

    This operation neither extracts files nor loads a model or pickle. It can
    run on another machine without access to the original absolute source paths.
    """
    archive_path = Path(archive_path)
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported export manifest schema.")
    identity = manifest["archive"]
    if (
        archive_path.stat().st_size != identity["bytes"]
        or file_hash(archive_path) != identity["sha256"]
    ):
        raise ValueError("Archive file hash or size does not match the export manifest.")
    expected = manifest["files"]
    seen, identities, receipts = set(), {}, {}
    protocol = comparison = audited_comparison = analysis_receipt = None
    with tarfile.open(archive_path, "r|xz") as archive:
        for member in archive:
            name = safe_relative(member.name)
            if name.parts[0] != ARCHIVE_ROOT or len(name.parts) < 2:
                raise ValueError("Every archive member must live under the campaign prefix.")
            relative = str(PurePosixPath(*name.parts[1:]))
            if relative in seen:
                raise ValueError(f"Duplicate archive member: {relative}")
            seen.add(relative)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"Only regular file archive members are permitted: {relative}")
            if relative not in expected:
                raise ValueError(f"Unregistered archive member: {relative}")
            if member.size != expected[relative]["bytes"]:
                raise ValueError(f"Archived member size differs from inventory: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {relative}")
            parse = relative in {
                "protocol.json",
                "comparison.json",
                f"{AUDITED_PREFIX}/comparison.json",
                f"{AUDITED_PREFIX}/analysis_receipt.json",
            } or (
                relative.startswith("trials/")
                and relative.endswith("/receipt.json")
                and "/workflow/" not in relative
            )
            buffer = io.BytesIO() if parse else None
            sha, count = hashlib.sha256(), 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                sha.update(block)
                count += len(block)
                if buffer is not None:
                    buffer.write(block)
            observed = {"sha256": sha.hexdigest(), "bytes": count}
            if observed != expected[relative]:
                raise ValueError(f"Archived member failed SHA256 audit: {relative}")
            identities[relative] = observed
            if parse:
                value = json.loads(buffer.getvalue())
                if relative == "protocol.json":
                    protocol = value
                elif relative == "comparison.json":
                    comparison = value
                elif relative == f"{AUDITED_PREFIX}/comparison.json":
                    audited_comparison = value
                elif relative == f"{AUDITED_PREFIX}/analysis_receipt.json":
                    analysis_receipt = value
                else:
                    receipts[relative] = value
    if seen != set(expected):
        raise ValueError(f"Archive is missing inventory entries: {sorted(set(expected) - seen)}")
    if protocol is None or comparison is None:
        raise ValueError("Archive is missing its original protocol or finalized comparison.")
    summary, required = validate_receipts(protocol, receipts, artifact_identities=identities)
    if manifest["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Archived protocol differs from exported campaign identity.")
    if (
        comparison["protocol_sha256"] != protocol["protocol_sha256"]
        or comparison.get("registered_trials") != summary["registered_trials"]
        or comparison["completed_trials"] != summary["completed_trials"]
        or comparison.get("status") != ("failed" if summary["failed_trials"] else "completed")
    ):
        raise ValueError("Archived comparison does not match audited attempts.")
    if manifest.get("attached_audited_summary"):
        if analysis_receipt is None or audited_comparison is None:
            raise ValueError("Archive is missing the attached audited analysis.")
        if (
            analysis_receipt.get("protocol_sha256") != protocol["protocol_sha256"]
            or analysis_receipt.get("registered_source_code_sha256")
            != protocol["source_code_sha256"]
            or analysis_receipt.get("original_comparison", {}).get("sha256")
            != identities["comparison.json"]["sha256"]
            or audited_comparison.get("protocol_sha256") != protocol["protocol_sha256"]
            or audited_comparison.get("registered_trials") != summary["registered_trials"]
            or audited_comparison.get("completed_trials") != summary["completed_trials"]
            or audited_comparison.get("status")
            != ("failed" if summary["failed_trials"] else "completed")
        ):
            raise ValueError("Audited analysis does not bind the archived original summary/trials.")
    return {
        "schema_version": 1,
        "status": "passed",
        "audited_summary_verified": analysis_receipt is not None,
        "checked_at_utc": now(),
        "archive_sha256": identity["sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "files_verified": len(identities),
        "uncompressed_bytes_verified": sum(row["bytes"] for row in identities.values()),
        "receipt_artifacts_verified": len(required),
        **summary,
        "pickle_deserialized": False,
        "files_extracted": False,
        "original_absolute_paths_changed": False,
    }


def reproduction_text(manifest):
    return f"""# Portable rolling campaign evidence

This directory contains a complete, independently hash-verified evidence archive
for protocol `{manifest["protocol_sha256"]}`. The original campaign files and
all attempt receipts were preserved byte-for-byte. Their absolute source and
output paths are historical provenance and have deliberately not been rewritten.

## Inspect and audit

Run from an installed checkout of this repository:

```bash
python scripts/export_rolling_evidence.py --audit PATH/TO/archive_manifest.json
```

The audit streams every archived file, rejects duplicate or unsafe paths and
non-regular entries, checks the complete inventory, verifies the original
protocol/trial identities, follows every attempt chain, and validates every
receipt-listed artifact hash. It does not load native model pickles or extract
files. Both completed and failed historical attempts remain visible.

After a successful audit, extract to a fresh inspection directory if needed:

```bash
mkdir -p runs/rolling-evidence-inspection
tar -xJf PATH/TO/{ARCHIVE_NAME} -C runs/rolling-evidence-inspection
```

The extracted tree is under `campaign/`. Numerical inference bundles use the
verified JSON/NPZ loader; native `params.pkl` files are retained solely as opaque
historical checkpoint evidence. Do not load a pickle from an untrusted source.

## Repeat the experiment at fresh paths

Do not edit the retained `protocol.json` to make its original paths fit a new
machine. Use its registered specification to create a new campaign identity:

```python
import json
from pathlib import Path
from financial_ts.rolling import prepare_rolling_campaign, run_rolling_campaign

old = json.loads(Path("PATH/TO/protocol.json").read_text())
new = prepare_rolling_campaign(
    provider=".data/chenditc/2026-11-26",
    output="runs/rolling-reproduction-new",
    spec=old["specification"],
)
result = run_rolling_campaign(new)
```

Restore the pinned provider release and confirm its recorded fingerprint. Use
matching source/package versions when reproducing numerical results. A corrected
analysis implementation or a different numerical environment needs its own
identity and an explicit comparison with the retained original/follow-up
receipts. Never treat inspected historical folds as a new confirmation holdout.

`comparison.json`, `comparison.csv`, `aggregation_summary.json`, and individual
receipt copies are readable without opening the archive. All other retained
campaign evidence, including analysis corrections or frozen-original reports,
is included inside the archive and listed in `archive_manifest.json`.

If an audited follow-up was supplied, its derived files are attached inside
`campaign/audited_summary/`. Readable `comparison.json` and `comparison.csv`
then show that corrected analysis; `original_comparison.json` and
`original_comparison.csv` preserve the registered implementation's reports.
`analysis_receipt.json` binds both source-code identities and the untouched
original comparison hash. No experimental predictions or trial receipts are
replaced by the follow-up analysis.

## Storage and exclusions

Compression is deterministic tar metadata plus XZ/LZMA2 preset 6 with a
{manifest["compression"]["dictionary_mib"]} MiB dictionary. Artifacts with the same
filename are adjacent in the stream so repeated large fixture arrays can be
reused by the compressor. Every inference bundle, prediction file, scenario
ledger and attempt is retained.

Generic handler caches, lock files, Python bytecode caches and unused MLflow
internals are excluded. Exact files explicitly listed in an attempt receipt
are always retained, even inside an excluded directory; the manifest lists
these exceptions. No original campaign evidence was deleted.
"""


def export_campaign(
    source, output, *, dictionary_mib=64, maximum_archive_mb=75, audited_summary=None
):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Export output must be separate from the original campaign.")
    if output.exists():
        raise ValueError("Export destination must be new; existing exports are immutable.")
    if (
        type(maximum_archive_mb) not in {int, float}
        or not math.isfinite(maximum_archive_mb)
        or maximum_archive_mb <= 0
    ):
        raise ValueError("Maximum archive size must be finite and positive.")
    protocol, receipts, summary, identities, exceptions, comparison = collect_source(source)
    audited_source, analysis_receipt = None, None
    comparison_relative = "comparison.json"
    if audited_summary is not None:
        audited_source, analysis_receipt, comparison = add_audited_summary(
            source, audited_summary, protocol, summary, identities
        )
        if (
            output == audited_source
            or output in audited_source.parents
            or audited_source in output.parents
        ):
            raise ValueError("Export output must be separate from the audited analysis.")
        comparison_relative = f"{AUDITED_PREFIX}/comparison.json"
    output.mkdir(parents=True)
    temporary = output / ".archive-incomplete.tar.xz"
    started = now()
    try:
        order = write_archive(
            source,
            temporary,
            identities,
            dictionary_mib=dictionary_mib,
            audited_source=audited_source,
        )
        # Recheck the live source inventory after streaming: a concurrent report
        # edit cannot silently create a mixed export of original and newer files.
        for relative, identity in identities.items():
            path = source_file(source, relative, audited_source)
            if path.stat().st_size != identity["bytes"] or file_hash(path) != identity["sha256"]:
                raise ValueError(f"Source changed before export finalization: {relative}")
        archive = output / ARCHIVE_NAME
        os.replace(temporary, archive)
        size = archive.stat().st_size
        maximum = int(maximum_archive_mb * 1_000_000)
        manifest = {
            "schema_version": SCHEMA,
            "source_campaign": str(source),
            "original_protocol_output_dir": protocol["output_dir"],
            "attached_audited_summary": {
                "source_directory": str(audited_source),
                "archive_prefix": f"{ARCHIVE_ROOT}/{AUDITED_PREFIX}",
                "analysis_receipt": analysis_receipt,
                "readable_comparison": "audited follow-up; original comparison also retained",
            }
            if audited_source is not None
            else None,
            "protocol_sha256": protocol["protocol_sha256"],
            "started_at_utc": started,
            "exported_at_utc": now(),
            "archive": {"name": ARCHIVE_NAME, "sha256": file_hash(archive), "bytes": size},
            "compression": {
                "format": "tar.xz",
                "codec": "LZMA2",
                "preset": 6,
                "dictionary_mib": dictionary_mib,
                "xz_check": "CRC64",
                "ordering": "artifact basename then full relative path",
                "tar_metadata": "mtime=0 uid=gid=0 regular mode0644",
            },
            "files": identities,
            "archive_member_order_sha256": digest(order),
            "uncompressed_bytes": sum(row["bytes"] for row in identities.values()),
            "file_count": len(identities),
            "receipt_artifacts_retained_from_excluded_directories": exceptions,
            "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
            "excluded_filename_suffixes": [".lock", ".pyc"],
            "original_files_mutated": False,
            "pickle_loading": "never; native checkpoints are opaque bytes for hashing only",
            "absolute_path_policy": "original protocol/config paths retained unchanged as provenance; register fresh paths for reproduction",
            "terminal_trials": summary,
            "size_limit": {
                "maximum_bytes": maximum,
                "status": "passed" if size <= maximum else "exceeded",
                "safe_alternative_if_exceeded": "Keep this complete verified archive. Store it as a release/download artifact, or split the identical byte stream into independently hashed <=50 MB chunks and reassemble before verifying the original SHA256. Do not delete or omit source evidence to meet a size target.",
            },
        }
        audit = audit_archive(archive, manifest)
        save_json(output / "archive_manifest.json", manifest)
        save_json(output / "audit.json", audit)
        for name in ("protocol.json", "comparison.json", "comparison.csv", "trial_history.json"):
            path = source / name
            if path.is_file() and name in identities:
                shutil.copyfile(path, output / name)
                if file_hash(output / name) != identities[name]["sha256"]:
                    raise ValueError(f"Readable sidecar changed while copying: {name}")
        if audited_source is not None:
            for name in ("comparison.json", "comparison.csv", "trial_history.json"):
                key = f"{AUDITED_PREFIX}/{name}"
                if key not in identities:
                    continue
                if (output / name).exists():
                    os.replace(output / name, output / ("original_" + name))
                shutil.copyfile(audited_source / name, output / name)
                if file_hash(output / name) != identities[key]["sha256"]:
                    raise ValueError(f"Readable audited sidecar changed: {name}")
            shutil.copyfile(
                audited_source / "analysis_receipt.json", output / "analysis_receipt.json"
            )
            if (
                file_hash(output / "analysis_receipt.json")
                != identities[f"{AUDITED_PREFIX}/analysis_receipt.json"]["sha256"]
            ):
                raise ValueError("Readable audited analysis receipt changed while copying.")
        for relative in receipts:
            parts = PurePosixPath(relative).parts
            destination = output / "receipts" / Path(*parts[1:-2]) / (parts[-2] + ".json")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            if file_hash(destination) != identities[relative]["sha256"]:
                raise ValueError(f"Readable receipt differs from original: {relative}")
        aggregation = {
            "protocol_sha256": protocol["protocol_sha256"],
            "source_comparison_sha256": identities[comparison_relative]["sha256"],
            "comparison_variant": "audited_followup"
            if audited_source is not None
            else "registered_original",
            "original_comparison_sha256": identities["comparison.json"]["sha256"],
            "status": comparison["status"],
            "trial_audit": summary,
            "selections": comparison.get("selections", []),
            "fold_seed_aggregates": comparison.get("fold_seed_aggregates", []),
            "interpretation": comparison.get("interpretation", {}),
            "promotion": comparison.get("promotion", {}),
            "analysis_related_files": sorted(
                path
                for path in identities
                if "/" not in path
                and any(
                    word in path.lower()
                    for word in ("comparison", "analysis", "bootstrap", "audit")
                )
            ),
        }
        save_json(output / "aggregation_summary.json", aggregation)
        (output / "README.md").write_text(reproduction_text(manifest), encoding="utf-8")
        return {
            "status": "verified" if size <= maximum else "verified_but_size_limit_exceeded",
            "output": str(output),
            "archive_bytes": size,
            "archive_sha256": manifest["archive"]["sha256"],
            "files_verified": audit["files_verified"],
            "receipt_artifacts_verified": audit["receipt_artifacts_verified"],
            "registered_trials": summary["registered_trials"],
            "attempts": summary["attempts"],
            "failed_attempts": summary["failed_attempts"],
            "size_limit": manifest["size_limit"],
        }
    except Exception as error:
        save_json(
            output / "export_failure.json",
            {
                "status": "failed",
                "at": now(),
                "error": f"{type(error).__name__}: {error}",
                "source_files_mutated": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign")
    parser.add_argument("--output")
    parser.add_argument("--dictionary-mib", type=int, default=64)
    parser.add_argument("--maximum-archive-mb", type=float, default=75)
    parser.add_argument(
        "--audited-summary",
        help="Separate corrected-analysis output to attach without replacing original files",
    )
    parser.add_argument(
        "--audit", help="Audit an existing archive_manifest.json without source access"
    )
    args = parser.parse_args()
    if args.audit:
        manifest_path = Path(args.audit).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text())
        name = manifest["archive"]["name"]
        if safe_relative(name).name != name:
            raise ValueError("Archive manifest must name a sibling file.")
        result = audit_archive(manifest_path.parent / name, manifest)
    else:
        if not args.campaign or not args.output:
            parser.error("--campaign and --output are required for export")
        result = export_campaign(
            args.campaign,
            args.output,
            dictionary_mib=args.dictionary_mib,
            maximum_archive_mb=args.maximum_archive_mb,
            audited_summary=args.audited_summary,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
