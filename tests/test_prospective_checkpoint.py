"""Portable source/state custody, bounded parts and exact provider restoration."""

import copy
import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock, Timeout
from test_arrival_feed import _source
from test_prospective import setup as prospective_setup

from financial_ts import arrival_feed, chenditc, prospective

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/package_prospective_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("prospective_checkpoint", MODULE_PATH)
checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint)


@pytest.fixture
def retained(tmp_path, monkeypatch):
    cache, metadata = _source(tmp_path)
    monkeypatch.setattr(chenditc, "resolve_chenditc_release", lambda release: copy.deepcopy(metadata))
    monkeypatch.setattr(arrival_feed, "_now", lambda: pd.Timestamp("2026-11-26T10:20:00Z"))
    monkeypatch.setattr(prospective, "_now", lambda: pd.Timestamp("2026-11-27T09:00:00Z"))
    feed = tmp_path / "feed"
    arrival = arrival_feed.collect_arrival(feed, expected_session="2026-11-26",
                                           decision_cutoff="2026-11-26T19:00:00+08:00", cache_dir=cache)
    downloads = feed / "downloads/2026-11-26"
    shutil.copytree(cache, downloads)
    store, protocol = prospective_setup(tmp_path, [pd.Timestamp("2026-11-27T09:00:00Z")])
    (store / "service_ticks").mkdir()
    (store / "service_ticks/observed.json").write_text('{"status":"idle","prospective":false}\n')
    runtime = tmp_path / "runtime"
    for name in protocol["source_files"]:
        source = Path(prospective.__file__).parent / name
        destination = runtime / "financial_ts" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return {"store": store, "feed": feed, "runtime": runtime, "arrival": arrival,
            "protocol": protocol, "output": tmp_path / "checkpoint", "cache": downloads}


def package(retained, **kwargs):
    return checkpoint.package_checkpoint(retained["store"], retained["feed"], "2026-11-26", retained["output"],
                                         runtime_root=retained["runtime"], part_bytes=256, **kwargs)


def test_bounded_parts_and_complete_state_preserve_exact_provenance(retained):
    arrival_path = Path(retained["arrival"]["arrival_receipt"])
    original = arrival_path.read_bytes()
    result = package(retained)
    manifest = json.loads((retained["output"] / "manifest.json").read_text())
    assert result["status"] == "verified" and result["source_parts"] > 1
    assert result["durable_save_status"] == "not_recorded"
    assert manifest["registered_source_files_retained"]
    assert all(part["bytes"] <= 256 for part in manifest["source_parts"])
    reconstructed = b"".join((retained["output"] / part["name"]).read_bytes() for part in manifest["source_parts"])
    assert reconstructed == (retained["cache"] / "qlib_bin.tar.gz").read_bytes()
    assert manifest["source_archive"]["sha256"] == hashlib.sha256(reconstructed).hexdigest()
    assert "store/service_ticks/observed.json" in manifest["state_files"]
    assert any(name.startswith("store/registry/bundles/") for name in manifest["state_files"])
    assert any(name.startswith("feed/attempts/") for name in manifest["state_files"])
    assert not any("/provider/" in name or name.startswith("feed/downloads/") for name in manifest["state_files"])
    assert arrival_path.read_bytes() == original
    assert Path(retained["arrival"]["provider_dir"]).is_dir()  # Packaging never evicts it.
    with pytest.raises(FileExistsError, match="new checkpoint"):
        package(retained)


def test_portable_state_and_exact_provider_restore_keep_original_receipts(retained, tmp_path):
    package(retained)
    restored_root = tmp_path / "portable-review"
    result = checkpoint.restore_state(retained["output"], restored_root)
    assert result["status"] == "restored" and not result["operational_resume_ready"]
    for directory in ("observations", "labels", "incidents"):
        assert (restored_root / "store" / directory).is_dir()
    status = prospective.evaluate_prospective(restored_root / "store")
    assert status["operational_acceptance"]["captured_sessions"] == 0
    original_arrival = Path(retained["arrival"]["arrival_receipt"]).read_bytes()
    recovered_arrival = restored_root / "feed/sessions/2026-11-26/arrival.json"
    assert recovered_arrival.read_bytes() == original_arrival
    assert not Path(arrival_feed.verify_arrival_metadata(recovered_arrival)["provider_dir"]).exists()
    result = checkpoint.restore_provider(retained["output"], restored_root / "feed")
    assert result["status"] == "restored" and result["provider_bytes_verified"]
    assert arrival_feed.verify_arrival(recovered_arrival)["provider_fingerprint"] == retained["arrival"]["provider_fingerprint"]
    assert recovered_arrival.read_bytes() == original_arrival
    snapshot = Path(result["provider_dir"]).parent
    assert (snapshot / "provider/metadata.json").read_bytes() == (snapshot / "provider_metadata.json").read_bytes()
    assert checkpoint.restore_provider(retained["output"], restored_root / "feed")["status"] == "verified_existing"
    with pytest.raises(FileExistsError, match="new destination"):
        checkpoint.restore_state(retained["output"], restored_root)


def test_selected_offloaded_original_provider_can_be_restored_without_retimestamping(retained):
    package(retained)
    before = Path(retained["arrival"]["arrival_receipt"]).read_bytes()
    provider = Path(retained["arrival"]["provider_dir"])
    # Explicit fixture offload simulates a separately authorized, durably saved eviction.
    shutil.rmtree(provider)
    shutil.rmtree(retained["cache"])
    result = checkpoint.restore_provider(retained["output"], retained["feed"])
    assert result["provider_dir"] == str(provider)
    assert result["provider_bytes_verified"]
    assert Path(retained["arrival"]["arrival_receipt"]).read_bytes() == before
    assert not retained["cache"].exists()  # No second compressed cache is reconstructed.


@pytest.mark.parametrize("kind", ["source_part", "state_archive", "manifest"])
def test_corrupt_or_replaced_checkpoint_is_rejected_before_restoration(retained, tmp_path, kind):
    package(retained)
    if kind == "source_part":
        target = retained["output"] / "qlib_bin.tar.gz.part-0000"
    elif kind == "state_archive":
        target = retained["output"] / "state.tar.gz"
    else:
        target = retained["output"] / "manifest.json"
    raw = target.read_bytes()
    target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        checkpoint.restore_state(retained["output"], tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("size", [0, -1, True, checkpoint.MAX_PART_BYTES + 1])
def test_part_size_never_exceeds_upload_bound(retained, size):
    with pytest.raises(ValueError, match="280 MiB"):
        checkpoint.package_checkpoint(retained["store"], retained["feed"], "2026-11-26", retained["output"], part_bytes=size)


def test_wrong_source_archive_and_changed_provenance_do_not_get_certified(retained):
    archive = retained["cache"] / "qlib_bin.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="original release/manifest"):
        package(retained)
    assert not retained["output"].exists()


def test_checkpoint_refuses_symlink_state_and_active_service(retained, tmp_path):
    linked = retained["store"] / "link"
    linked.symlink_to(tmp_path / "external")
    with pytest.raises(ValueError, match="symlink"):
        package(retained)
    linked.unlink()
    with FileLock(str(retained["store"] / ".service.lock"), timeout=0):
        with pytest.raises(Timeout):
            package(retained)


def test_runtime_must_match_frozen_protocol_when_supplied(retained):
    path = retained["runtime"] / "financial_ts/prospective.py"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="registered source identity"):
        package(retained)


def test_restore_does_not_overwrite_changed_existing_provider(retained):
    package(retained)
    provider = Path(retained["arrival"]["provider_dir"])
    path = provider / "calendars/day.txt"
    path.write_text(path.read_text() + "2026-11-27\n")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="provider bytes changed"):
        checkpoint.restore_provider(retained["output"], retained["feed"])
    assert path.read_bytes() == original


def test_state_archive_does_not_duplicate_failed_attempt_providers(retained):
    rejected = retained["feed"] / "attempts/rejected/rejected_snapshot"
    (rejected / "provider/features").mkdir(parents=True)
    (rejected / "provider/features/must-not-copy.bin").write_bytes(b"large fixture data")
    (rejected / "snapshot_audit.json").write_text('{"status":"quarantined"}')
    package(retained)
    with tarfile.open(retained["output"] / "state.tar.gz") as archive:
        names = archive.getnames()
    assert "feed/attempts/rejected/rejected_snapshot/snapshot_audit.json" in names
    assert not any("must-not-copy.bin" in name for name in names)
