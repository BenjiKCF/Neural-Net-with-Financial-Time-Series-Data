"""Exact archive-byte recovery, no-clobber publication and part integrity."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reassemble_rolling_evidence.py"
SPEC = importlib.util.spec_from_file_location("rolling_reassembly", MODULE_PATH)
reassembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reassembly)


def _identity(name, data):
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _parts(tmp_path):
    # Deliberately opaque bytes: the tool must not interpret a tar or checkpoint.
    data = b"opaque evidence\x00\xff\n" * 17
    chunks = [data[:100], data[100:200], data[200:]]
    parts = []
    for index, chunk in enumerate(chunks):
        name = f"{reassembly.ARCHIVE_NAME}.part-{index:03d}"
        (tmp_path / name).write_bytes(chunk)
        parts.append(_identity(name, chunk))
    manifest = {
        "schema_version": 1,
        "archive": _identity(reassembly.ARCHIVE_NAME, data),
        "parts": parts,
    }
    path = tmp_path / "archive_parts.json"
    _write_manifest(path, manifest)
    return path, manifest, data


def _assert_clean_failure(tmp_path):
    assert not (tmp_path / reassembly.ARCHIVE_NAME).exists()
    assert not list(tmp_path.glob(".rolling-reassembly-*"))


def test_exact_reassembly_and_verified_idempotency(tmp_path):
    path, manifest, data = _parts(tmp_path)
    before = {part["name"]: (tmp_path / part["name"]).read_bytes() for part in manifest["parts"]}
    first = reassembly.reassemble(path)
    assert first["status"] == "verified" and not first["reused"]
    assert first["parts_verified"] == 3 and first["bytes"] == len(data)
    assert first["sha256"] == hashlib.sha256(data).hexdigest()
    assert not first["files_extracted"] and not first["pickle_deserialized"]
    output = tmp_path / reassembly.ARCHIVE_NAME
    assert output.read_bytes() == data
    original_stat = output.stat()
    assert reassembly.reassemble(path)["reused"]
    assert output.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert {name: (tmp_path / name).read_bytes() for name in before} == before
    assert not list(tmp_path.glob(".rolling-reassembly-*"))


def test_portable_cli_uses_only_the_parts_manifest(tmp_path):
    path, _, data = _parts(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-I", str(MODULE_PATH), "--manifest", str(path)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    assert json.loads(completed.stdout)["status"] == "verified"
    assert (tmp_path / reassembly.ARCHIVE_NAME).read_bytes() == data


def test_registered_prospective_store_uses_the_same_integrity_and_no_clobber_checks(tmp_path):
    path, manifest, data = _parts(tmp_path)
    manifest["archive"]["name"] = "registered_store.tar.xz"
    for index, part in enumerate(manifest["parts"]):
        new_name = f"registered_store.tar.xz.part-{index:03d}"
        (tmp_path / part["name"]).rename(tmp_path / new_name)
        part["name"] = new_name
    _write_manifest(path, manifest)
    result = reassembly.reassemble(path)
    output = tmp_path / "registered_store.tar.xz"
    assert result["status"] == "verified" and output.read_bytes() == data
    output.write_bytes(b"unrelated existing store")
    with pytest.raises(ValueError, match="will not be overwritten"):
        reassembly.reassemble(path)


@pytest.mark.parametrize("fault", ["tampered", "truncated", "missing"])
def test_bad_parts_never_publish_partial_archive(tmp_path, fault):
    path, manifest, _ = _parts(tmp_path)
    part = tmp_path / manifest["parts"][1]["name"]
    if fault == "missing":
        part.unlink()
    elif fault == "truncated":
        part.write_bytes(part.read_bytes()[:-1])
    else:
        original = part.read_bytes()
        part.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="missing|size mismatch|integrity"):
        reassembly.reassemble(path)
    _assert_clean_failure(tmp_path)


@pytest.mark.parametrize("fault", ["reordered", "duplicate", "unsafe_part", "unsafe_archive"])
def test_manifest_rejects_unordered_duplicate_and_unsafe_names(tmp_path, fault):
    path, manifest, _ = _parts(tmp_path)
    if fault == "reordered":
        manifest["parts"].reverse()
    elif fault == "duplicate":
        manifest["parts"][1] = manifest["parts"][0].copy()
    elif fault == "unsafe_part":
        manifest["parts"][0]["name"] = "../outside.part"
    else:
        manifest["archive"]["name"] = "../outside.tar.xz"
    _write_manifest(path, manifest)
    with pytest.raises(ValueError, match="sibling name"):
        reassembly.reassemble(path)
    _assert_clean_failure(tmp_path)


@pytest.mark.parametrize("target", ["manifest", "part", "output"])
def test_symlinked_files_are_rejected(tmp_path, target):
    path, manifest, data = _parts(tmp_path)
    candidate = (
        path if target == "manifest" else tmp_path / manifest["parts"][0]["name"]
        if target == "part" else tmp_path / reassembly.ARCHIVE_NAME
    )
    real = tmp_path / "real-evidence"
    real.write_bytes(data if target == "output" else candidate.read_bytes())
    candidate.unlink(missing_ok=True)
    candidate.symlink_to(real)
    with pytest.raises(ValueError, match="regular files, not symlinks"):
        reassembly.reassemble(path)
    assert candidate.is_symlink()
    assert not list(tmp_path.glob(".rolling-reassembly-*"))


def test_existing_different_output_is_never_overwritten(tmp_path):
    path, _, _ = _parts(tmp_path)
    output = tmp_path / reassembly.ARCHIVE_NAME
    output.write_bytes(b"different pre-existing evidence")
    with pytest.raises(ValueError, match="will not be overwritten"):
        reassembly.reassemble(path)
    assert output.read_bytes() == b"different pre-existing evidence"
    assert not list(tmp_path.glob(".rolling-reassembly-*"))


def test_aggregate_hash_mismatch_removes_temporary_output(tmp_path):
    path, manifest, _ = _parts(tmp_path)
    manifest["archive"]["sha256"] = "0" * 64
    _write_manifest(path, manifest)
    with pytest.raises(ValueError, match="Combined archive integrity"):
        reassembly.reassemble(path)
    _assert_clean_failure(tmp_path)


def test_matching_existing_archive_does_not_hide_tampered_parts(tmp_path):
    path, manifest, data = _parts(tmp_path)
    output = tmp_path / reassembly.ARCHIVE_NAME
    output.write_bytes(data)
    part = tmp_path / manifest["parts"][0]["name"]
    part.write_bytes(b"X" * part.stat().st_size)
    with pytest.raises(ValueError, match="Part integrity"):
        reassembly.reassemble(path)
    assert output.read_bytes() == data
    assert not list(tmp_path.glob(".rolling-reassembly-*"))


def test_concurrent_different_output_is_preserved(tmp_path, monkeypatch):
    path, _, _ = _parts(tmp_path)
    link = reassembly.os.link

    def competing_writer(source, destination):
        destination.write_bytes(b"concurrent evidence")
        return link(source, destination)

    monkeypatch.setattr(reassembly.os, "link", competing_writer)
    with pytest.raises(ValueError, match="will not be overwritten"):
        reassembly.reassemble(path)
    assert (tmp_path / reassembly.ARCHIVE_NAME).read_bytes() == b"concurrent evidence"
    assert not list(tmp_path.glob(".rolling-reassembly-*"))
