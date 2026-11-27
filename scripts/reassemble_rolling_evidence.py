#!/usr/bin/env python3
"""Reassemble the exact rolling evidence archive from verified sibling parts.

Uses only the Python standard library. This script does not decompress an
archive, extract files, or deserialize native model checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path

ARCHIVE_NAME = "rolling_evidence.tar.xz"
CHUNK_BYTES = 1024 * 1024


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _regular_file(path: Path):
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"Required evidence file is missing: {path.name}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"Evidence files must be regular files, not symlinks: {path.name}")


def _identity(value, expected_name):
    if not isinstance(value, dict) or set(value) != {"name", "bytes", "sha256"}:
        raise ValueError("Each archive/part identity requires exactly name, bytes and sha256.")
    if value["name"] != expected_name:
        raise ValueError(f"Unsafe or unordered sibling name; expected {expected_name}.")
    if type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise ValueError("Evidence byte counts must be positive integers.")
    if not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
        raise ValueError("Evidence SHA256 values must be 64 lowercase hexadecimal characters.")


def load_manifest(path):
    path = Path(path).expanduser().absolute()
    _regular_file(path)
    manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "archive", "parts"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported archive-parts manifest; expected schema version 1.")
    _identity(manifest["archive"], ARCHIVE_NAME)
    parts = manifest["parts"]
    if not isinstance(parts, list) or not parts:
        raise ValueError("The manifest must contain a nonempty ordered parts list.")
    for index, part in enumerate(parts):
        # Exact contiguous names reject traversal, aliases, duplicates and
        # reordering before any part or destination is opened.
        _identity(part, f"{ARCHIVE_NAME}.part-{index:03d}")
    if sum(part["bytes"] for part in parts) != manifest["archive"]["bytes"]:
        raise ValueError("Part byte counts do not sum to the original archive size.")
    return path, manifest


def _verify_existing(path: Path, identity):
    _regular_file(path)
    actual = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            actual.update(chunk)
            size += len(chunk)
    if size != identity["bytes"] or actual.hexdigest() != identity["sha256"]:
        raise ValueError("Existing archive differs from the manifest; it will not be overwritten.")


def reassemble(manifest_path):
    """Verify every part and atomically publish an unchanged archive without replacement."""
    manifest_path, manifest = load_manifest(manifest_path)
    root = manifest_path.parent
    output = root / ARCHIVE_NAME
    expected = manifest["archive"]
    if output.exists() or output.is_symlink():
        _verify_existing(output, expected)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".rolling-reassembly-", dir=root)
    temporary = Path(temporary_name)
    combined = hashlib.sha256()
    combined_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            for part in manifest["parts"]:
                path = root / part["name"]
                _regular_file(path)
                if path.stat().st_size != part["bytes"]:
                    raise ValueError(f"Part size mismatch: {part['name']}")
                digest = hashlib.sha256()
                part_bytes = 0
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                        target.write(chunk)
                        digest.update(chunk)
                        combined.update(chunk)
                        part_bytes += len(chunk)
                        combined_bytes += len(chunk)
                if part_bytes != part["bytes"] or digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Part integrity check failed: {part['name']}")
            if combined_bytes != expected["bytes"] or combined.hexdigest() != expected["sha256"]:
                raise ValueError("Combined archive integrity check failed.")
            target.flush()
            os.fsync(target.fileno())
        # A hard link atomically creates the destination only if absent. Unlike
        # rename/replace, it cannot overwrite a different concurrently-created
        # archive. Both files live on the same filesystem.
        try:
            os.link(temporary, output)
            reused = False
        except FileExistsError:
            _verify_existing(output, expected)
            reused = True
        return {
            "status": "verified",
            "archive": str(output),
            "bytes": combined_bytes,
            "sha256": combined.hexdigest(),
            "parts_verified": len(manifest["parts"]),
            "reused": reused,
            "files_extracted": False,
            "pickle_deserialized": False,
        }
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Path to archive_parts.json")
    args = parser.parse_args()
    try:
        result = reassemble(args.manifest)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
