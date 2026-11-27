"""Portable archival custody and tar safety; native checkpoints stay opaque."""

import copy
import importlib.util
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_rolling_evidence.py"
SPEC = importlib.util.spec_from_file_location("rolling_export", MODULE_PATH)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _campaign(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    trials = [
        {"id": "fold-000/zero/seed-7", "model": "zero", "seed": 7},
        {"id": "fold-000/ridge/seed-7", "model": "ridge", "seed": 7},
    ]
    identity = {
        "schema_version": 1,
        "output_dir": str(source),
        "provider_uri": "/original/machine/provider",
        "source_code_sha256": "registered-original-source",
        "trials": trials,
        "specification": {"models": ["zero", "ridge"], "seeds": [7]},
    }
    protocol = {
        **identity,
        "registered_at_utc": "2026-11-27T00:00:00+00:00",
        "protocol_sha256": exporter.digest(identity),
    }
    exporter.save_json(source / "protocol.json", protocol)
    for index, trial in enumerate(trials):
        attempts = ["failed", "completed"] if index == 0 else ["completed"]
        for number, status in enumerate(attempts, start=1):
            parent = source / "trials" / trial["id"] / f"attempt-{number:04d}"
            parent.mkdir(parents=True)
            receipt = {
                "trial_id": trial["id"],
                "trial_sha256": exporter.digest(trial),
                "protocol_sha256": protocol["protocol_sha256"],
                "attempt": number,
                "status": status,
                "previous_attempt": f"trials/{trial['id']}/attempt-{number - 1:04d}"
                if number > 1
                else None,
            }
            if status == "failed":
                receipt["error"] = "Intentional retained failed trial"
                _write(parent / "workflow" / "partial.txt", b"partial attempt evidence")
            else:
                files = {
                    "predictions.csv": b"datetime,instrument,score,label\n2024-01-02,A,0.01,0.02\n",
                    "inference_bundle/weights.npz": b"numerical weights opaque archive fixture",
                    "inference_bundle/reload_fixture.npz": b"repeated numerical fixture" * 2048,
                    "scenarios/base/positions.json.gz": b"scenario position evidence",
                    "mlruns/1/run/artifacts/params.pkl": b"NOT A PICKLE; audit must never deserialize",
                }
                artifacts = {}
                for name, data in files.items():
                    path = parent / "workflow" / name
                    _write(path, data)
                    artifacts[name] = {"sha256": exporter.file_hash(path), "bytes": len(data)}
                _write(
                    parent / "workflow" / "mlruns" / "irrelevant.bin", b"excluded MLflow internals"
                )
                receipt["artifacts"] = artifacts
            exporter.save_json(parent / "receipt.json", receipt)
    _write(source / "cache" / "handler.pkl", b"large handler not in evidence" * 4096)
    _write(source / ".registration.lock", b"")
    comparison = {
        "status": "completed",
        "protocol_sha256": protocol["protocol_sha256"],
        "registered_trials": 2,
        "completed_trials": 2,
        "selections": [{"version": "original"}],
        "fold_seed_aggregates": [],
        "interpretation": {"holdout": "exploratory"},
    }
    exporter.save_json(source / "comparison.json", comparison)
    _write(source / "comparison.csv", b"model,score\nzero,0\n")
    exporter.save_json(source / "trial_history.json", {"kept": "original"})
    return source, protocol


def _audited(tmp_path, source, protocol):
    root = tmp_path / "corrected-analysis"
    root.mkdir()
    comparison = json.loads((source / "comparison.json").read_text())
    comparison["selections"] = [{"version": "audited correction"}]
    exporter.save_json(root / "comparison.json", comparison)
    _write(root / "comparison.csv", b"model,score\nzero,1\n")
    exporter.save_json(
        root / "analysis_receipt.json",
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "registered_source_code_sha256": protocol["source_code_sha256"],
            "analysis_source_code_sha256": "new-analysis-source",
            "original_comparison": {"sha256": exporter.file_hash(source / "comparison.json")},
        },
    )
    return root


def _inventory(root):
    return {
        str(path.relative_to(root)): exporter.file_hash(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_export_is_portable_complete_and_preserves_original_and_corrected_reports(tmp_path):
    source, protocol = _campaign(tmp_path)
    audited = _audited(tmp_path, source, protocol)
    before = _inventory(source)
    result = exporter.export_campaign(
        source, tmp_path / "export", dictionary_mib=16, audited_summary=audited
    )
    assert result["status"] == "verified"
    assert result["registered_trials"] == 2 and result["attempts"] == 3
    assert result["failed_attempts"] == 1 and result["receipt_artifacts_verified"] == 10
    assert _inventory(source) == before
    root = tmp_path / "export"
    manifest = json.loads((root / "archive_manifest.json").read_text())
    assert len(manifest["receipt_artifacts_retained_from_excluded_directories"]) == 2
    assert not any(
        "irrelevant.bin" in path or path.startswith("cache/") for path in manifest["files"]
    )
    assert (root / "original_comparison.json").read_bytes() == (
        source / "comparison.json"
    ).read_bytes()
    assert (root / "comparison.json").read_bytes() == (audited / "comparison.json").read_bytes()
    assert (root / "protocol.json").read_bytes() == (source / "protocol.json").read_bytes()
    assert len(list((root / "receipts").rglob("attempt-*.json"))) == 3
    # A portable audit has no dependency on original absolute paths.
    source.rename(tmp_path / "source-hidden-from-auditor")
    audited.rename(tmp_path / "analysis-hidden-from-auditor")
    copied = tmp_path / "new-machine"
    copied.mkdir()
    shutil.copy(root / exporter.ARCHIVE_NAME, copied)
    shutil.copy(root / "archive_manifest.json", copied)
    check = exporter.audit_archive(copied / exporter.ARCHIVE_NAME, copied / "archive_manifest.json")
    assert check["status"] == "passed" and check["audited_summary_verified"]
    assert not check["pickle_deserialized"] and not check["files_extracted"]


def test_deterministic_archive_and_explicit_size_overflow_keep_all_evidence(tmp_path):
    source, _ = _campaign(tmp_path)
    one = exporter.export_campaign(source, tmp_path / "one", dictionary_mib=16)
    two = exporter.export_campaign(
        source, tmp_path / "two", dictionary_mib=16, maximum_archive_mb=0.000001
    )
    assert one["archive_sha256"] == two["archive_sha256"]
    assert two["status"] == "verified_but_size_limit_exceeded"
    assert two["files_verified"] == one["files_verified"]
    assert (tmp_path / "two" / exporter.ARCHIVE_NAME).exists()


def test_nonterminal_receipt_and_source_artifact_corruption_fail_before_export(tmp_path):
    source, _ = _campaign(tmp_path)
    path = next(source.glob("trials/**/attempt-0002/receipt.json"))
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed["status"] = "running"
    exporter.save_json(path, changed)
    with pytest.raises(ValueError, match="nonterminal"):
        exporter.export_campaign(source, tmp_path / "not-complete", dictionary_mib=16)
    assert not (tmp_path / "not-complete").exists()
    exporter.save_json(path, original)
    artifact = path.parent / "workflow" / "predictions.csv"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash audit"):
        exporter.export_campaign(source, tmp_path / "corrupt", dictionary_mib=16)


def _rewrite_archive(original, target, *, omit=None, extra=None):
    with (
        tarfile.open(original, "r:xz") as incoming,
        tarfile.open(target, "w:xz", preset=0) as outgoing,
    ):
        for member in incoming:
            if member.name == omit:
                continue
            outgoing.addfile(member, incoming.extractfile(member))
        if extra is not None:
            member, data = extra
            outgoing.addfile(member, io.BytesIO(data) if data is not None else None)


def test_missing_receipt_artifact_cannot_be_hidden_by_editing_export_inventory(tmp_path):
    source, _ = _campaign(tmp_path)
    exporter.export_campaign(source, tmp_path / "export", dictionary_mib=16)
    root = tmp_path / "export"
    manifest = json.loads((root / "archive_manifest.json").read_text())
    omitted = next(path for path in manifest["files"] if path.endswith("params.pkl"))
    target = tmp_path / "omitted.tar.xz"
    _rewrite_archive(root / exporter.ARCHIVE_NAME, target, omit="campaign/" + omitted)
    del manifest["files"][omitted]
    manifest["archive"].update(bytes=target.stat().st_size, sha256=exporter.file_hash(target))
    with pytest.raises(ValueError, match="Receipt-listed artifact missing"):
        exporter.audit_archive(target, manifest)


@pytest.mark.parametrize("attack", ["parent_path", "duplicate", "symlink"])
def test_tar_path_duplicate_and_link_attacks_rejected_without_extraction(tmp_path, attack):
    source, _ = _campaign(tmp_path)
    exporter.export_campaign(source, tmp_path / "export", dictionary_mib=16)
    root = tmp_path / "export"
    manifest = json.loads((root / "archive_manifest.json").read_text())
    if attack == "parent_path":
        member, data = tarfile.TarInfo("campaign/../escape.txt"), b"unsafe"
        member.size = len(data)
    elif attack == "duplicate":
        member, data = (
            tarfile.TarInfo("campaign/protocol.json"),
            (source / "protocol.json").read_bytes(),
        )
        member.size = len(data)
    else:
        member, data = tarfile.TarInfo("campaign/link"), None
        member.type, member.linkname = tarfile.SYMTYPE, "../../outside"
    target = tmp_path / "unsafe.tar.xz"
    _rewrite_archive(root / exporter.ARCHIVE_NAME, target, extra=(member, data))
    manifest["archive"].update(bytes=target.stat().st_size, sha256=exporter.file_hash(target))
    with pytest.raises(ValueError, match="Unsafe|Duplicate|regular file"):
        exporter.audit_archive(target, manifest)
    assert not (tmp_path / "escape.txt").exists()


def test_audited_summary_must_bind_the_exact_original_summary(tmp_path):
    source, protocol = _campaign(tmp_path)
    audited = _audited(tmp_path, source, protocol)
    path = audited / "analysis_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["original_comparison"]["sha256"] = "0" * 64
    exporter.save_json(path, receipt)
    with pytest.raises(ValueError, match="does not bind"):
        exporter.export_campaign(
            source, tmp_path / "export", dictionary_mib=16, audited_summary=audited
        )
