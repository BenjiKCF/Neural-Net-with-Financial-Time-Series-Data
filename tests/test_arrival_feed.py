"""Exercise real acquisition custody, late arrivals and historical revisions."""

import copy
import hashlib
import io
import json
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from financial_ts import arrival_feed, chenditc


def _sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _source(tmp_path):
    cache = tmp_path / "source"
    cache.mkdir()
    days = "2026-11-24\n2026-11-25\n2026-11-26\n"
    members = {
        "qlib_bin/calendars/day.txt": days.encode(),
        "qlib_bin/calendars/day_future.txt": (days + "2026-11-27\n2026-11-30\n").encode(),
    }
    for name in chenditc.UNIVERSES:
        members[f"qlib_bin/instruments/{name}.txt"] = b"SH600000\t2026-11-24\t2026-11-26\n"
    for field in ("open", "high", "low", "close", "volume", "factor"):
        members[f"qlib_bin/features/sh600000/{field}.day.bin"] = np.array(
            [0, 1, np.nan, 1.2], dtype="<f4"
        ).tobytes()
    archive = cache / "qlib_bin.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, raw in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            output.addfile(member, io.BytesIO(raw))
    manifest = {
        "release_tag": "2026-11-26", "target_trade_date": "2026-11-26",
        "future_start_date": "2026-11-27", "future_end_date": "2026-11-30",
        "dolt_commit": "a" * 32, "investment_data_commit": "b" * 40,
        "qlib_commit": "c" * 40, "image_digest": "sha256:" + "d" * 64,
        "archive_size_bytes": archive.stat().st_size, "archive_sha256": _sha(archive.read_bytes()),
    }
    (cache / "qlib_bin.manifest.json").write_text(json.dumps(manifest))
    metadata = {
        "id": 123, "tag_name": "2026-11-26",
        "html_url": "https://github.com/chenditc/investment_data/releases/tag/2026-11-26",
        "published_at": "2026-11-26T10:08:41Z", "assets": [
            {
                "id": index + 1, "name": name, "size": (cache / name).stat().st_size,
                "digest": _sha((cache / name).read_bytes()),
                "created_at": "2026-11-26T10:08:45Z",
                "updated_at": "2026-11-26T10:09:25Z",
                "browser_download_url": f"https://github.com/chenditc/investment_data/releases/download/2026-11-26/{name}",
            }
            for index, name in enumerate(chenditc.ASSET_NAMES)
        ],
    }
    return cache, metadata


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cache, metadata = _source(tmp_path)
    monkeypatch.setattr(chenditc, "resolve_chenditc_release", lambda release: copy.deepcopy(metadata))
    clock = {"now": pd.Timestamp("2026-11-26T10:20:00Z")}
    monkeypatch.setattr(arrival_feed, "_now", lambda: clock["now"])
    return tmp_path / "feed", cache, metadata, clock


def _collect(setup):
    feed, cache, _, _ = setup
    return arrival_feed.collect_arrival(
        feed, expected_session="2026-11-26", decision_cutoff="2026-11-26T19:00:00+08:00",
        cache_dir=cache,
    )


def _outcome(feed):
    return json.loads(next((feed / "attempts").glob("*/outcome.json")).read_text())


def test_verified_acquisition_uses_actual_clock_and_retains_source(setup):
    feed, _, _, clock = setup
    receipt = _collect(setup)
    assert receipt["data_available_at"] == "2026-11-26T10:09:25+00:00"
    assert receipt["observed_at"] == clock["now"].isoformat()
    assert receipt["observed_at"] != receipt["data_available_at"]
    assert receipt["status"] == receipt["snapshot_audit_status"] == "admitted"
    assert receipt["publication_status"] == "committed"
    assert receipt["prospective_forecasts_captured"] is False
    assert receipt["orders_submitted"] is False
    snapshot = feed / receipt["snapshot_relative_path"]
    audit = json.loads((snapshot / "snapshot_audit.json").read_text())
    assert audit["feature_files_checked"] == 6
    assert audit["comparison"] == "initial snapshot; no prior-vintage comparison"
    clock["now"] = pd.Timestamp("2026-11-27T12:00:00Z")
    repeated = _collect(setup)
    assert repeated["reused"] is True
    assert repeated["receipt_id"] == receipt["receipt_id"]
    assert repeated["observed_at"] == receipt["observed_at"]
    assert len(list((feed / "attempts").iterdir())) == 1


def test_collector_source_bytes_environment_and_identity_are_retained(setup):
    feed, _, _, _ = setup
    receipt = _collect(setup)
    snapshot = feed / receipt["snapshot_relative_path"]
    source = json.loads((snapshot / "collector_source.json").read_text())
    acquisition = json.loads((snapshot / "acquisition.json").read_text())
    assert source["collector_id"] == receipt["collector_id"] == acquisition["collector_id"]
    assert source["environment"]["python"]
    assert set(source["sources"]) == {
        "arrival_feed.py", "chenditc.py", "workflow.py", "bundles.py",
    }
    for value in source["sources"].values():
        assert hashlib.sha256(value["text"].encode()).hexdigest() == value["sha256"]
    assert next((feed / "attempts").glob("*/collector_source.json")).read_bytes() == (
        snapshot / "collector_source.json"
    ).read_bytes()
    (snapshot / "collector_source.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        arrival_feed.verify_arrival(receipt["arrival_receipt"])


def test_metadata_audit_survives_provider_offload_but_full_verification_requires_bytes(setup):
    feed, _, _, _ = setup
    receipt = _collect(setup)
    assert receipt["provider_bytes_verified"] is True
    snapshot = feed / receipt["snapshot_relative_path"]
    assert (snapshot / "provider_metadata.json").read_bytes() == (
        snapshot / "provider/metadata.json"
    ).read_bytes()
    shutil.rmtree(receipt["provider_dir"])
    metadata = arrival_feed.verify_arrival_metadata(receipt["arrival_receipt"])
    assert metadata["receipt_id"] == receipt["receipt_id"]
    assert metadata["provider_dir"] == receipt["provider_dir"]
    assert metadata["provider_bytes_verified"] is False
    assert metadata["verification_scope"] == "receipt_and_retained_metadata_only"
    with pytest.raises(ValueError, match="not materialized"):
        arrival_feed.verify_arrival(receipt["arrival_receipt"])


def test_metadata_scope_does_not_claim_to_verify_changed_market_bytes(setup):
    receipt = _collect(setup)
    from pathlib import Path
    path = Path(receipt["provider_dir"]) / "features/sh600000/close.day.bin"
    path.write_bytes(np.array([0, 1, np.nan, 99], dtype="<f4").tobytes())
    assert arrival_feed.verify_arrival_metadata(receipt["arrival_receipt"])[
        "provider_bytes_verified"
    ] is False
    with pytest.raises(ValueError, match="provider bytes changed"):
        arrival_feed.verify_arrival(receipt["arrival_receipt"])


@pytest.mark.parametrize("name", ["provider_metadata.json", "snapshot_audit.json", "source_manifest.json"])
@pytest.mark.parametrize("change", ["delete", "modify"])
def test_offloaded_metadata_custody_still_rejects_missing_or_changed_evidence(setup, name, change):
    feed, _, _, _ = setup
    receipt = _collect(setup)
    snapshot = feed / receipt["snapshot_relative_path"]
    shutil.rmtree(receipt["provider_dir"])
    path = snapshot / name
    if change == "delete":
        path.unlink()
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="metadata or audit changed"):
        arrival_feed.verify_arrival_metadata(receipt["arrival_receipt"])


def test_source_or_environment_change_during_acquisition_cannot_be_admitted(setup, monkeypatch):
    feed, _, _, _ = setup
    original = arrival_feed._collector_source
    calls = []

    def changed():
        source = original()
        calls.append(None)
        if len(calls) > 1:
            source["collector_id"] = "different source or environment"
        return source

    monkeypatch.setattr(arrival_feed, "_collector_source", changed)
    with pytest.raises(ValueError, match="changed during acquisition"):
        _collect(setup)
    assert _outcome(feed)["status"] == "failed"
    assert next((feed / "attempts").glob("*/collector_source.json")).is_file()
    assert next((feed / "attempts").glob("*/rejected_snapshot/provider/metadata.json")).is_file()
    assert not list((feed / "sessions").glob("*/arrival.json"))


@pytest.mark.parametrize("timestamp", ["2026-11-26T10:20:01Z", "2026-11-26T11:01:00Z"])
def test_future_or_late_asset_publication_cannot_be_admitted(setup, timestamp):
    feed, _, metadata, _ = setup
    metadata["assets"][0]["updated_at"] = timestamp
    with pytest.raises(ValueError, match="future-dated|after cutoff"):
        _collect(setup)
    assert _outcome(feed)["status"] == "failed"
    assert not list((feed / "sessions").glob("*/arrival.json"))


def test_download_finishing_after_cutoff_retains_rejected_bytes(setup, monkeypatch):
    feed, _, _, clock = setup
    original = chenditc.download_chenditc

    def late(*args, **kwargs):
        result = original(*args, **kwargs)
        clock["now"] = pd.Timestamp("2026-11-26T11:00:01Z")
        return result

    monkeypatch.setattr(chenditc, "download_chenditc", late)
    with pytest.raises(ValueError, match="acquisition completed after"):
        _collect(setup)
    assert next((feed / "attempts").glob("*/rejected_snapshot/provider/metadata.json")).is_file()
    assert not list((feed / "sessions").glob("*/arrival.json"))


def test_source_replacement_between_discovery_and_download_is_rejected(setup, monkeypatch):
    feed, _, metadata, _ = setup
    count = 0

    def replacing(release):
        nonlocal count
        count += 1
        result = copy.deepcopy(metadata)
        if count > 1:
            result["assets"][0]["id"] += 100
        return result

    monkeypatch.setattr(chenditc, "resolve_chenditc_release", replacing)
    with pytest.raises(ValueError, match="changed between"):
        _collect(setup)
    assert _outcome(feed)["status"] == "failed"


def test_low_disk_space_fails_before_download_without_discarding_history(setup, monkeypatch):
    feed, _, _, _ = setup
    monkeypatch.setattr(arrival_feed.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024))
    with pytest.raises(ValueError, match="Insufficient free space"):
        _collect(setup)
    assert _outcome(feed)["status"] == "failed"
    check = json.loads(next((feed / "attempts").glob("*/space_check.json")).read_text())
    assert check["status"] == "failed"
    assert not list((feed / "snapshots").iterdir())


@pytest.mark.parametrize("which", ["provider", "metadata", "audit", "receipt"])
def test_any_retained_admission_input_tampering_is_rejected(setup, which):
    feed, _, _, _ = setup
    receipt = _collect(setup)
    snapshot = feed / receipt["snapshot_relative_path"]
    if which == "provider":
        path = snapshot / "provider/features/sh600000/close.day.bin"
        path.write_bytes(np.array([0, 1, np.nan, 99], dtype="<f4").tobytes())
    elif which == "metadata":
        path = snapshot / "provider/metadata.json"
        path.write_text(path.read_text() + "\n")
    elif which == "audit":
        (snapshot / "snapshot_audit.json").write_text("{}")
    else:
        path = feed / "sessions/2026-11-26/arrival.json"
        value = json.loads(path.read_text())
        value["observed_at"] = "2026-11-26T09:00:00Z"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        arrival_feed.verify_arrival(receipt["arrival_receipt"])


def test_final_durable_write_crossing_cutoff_cannot_create_success(setup, monkeypatch):
    feed, _, _, clock = setup
    original = arrival_feed.atomic_json

    def slow(path, value):
        original(path, value)
        if value.get("publication_status") == "committed":
            clock["now"] = pd.Timestamp("2026-11-26T11:00:01Z")

    monkeypatch.setattr(arrival_feed, "atomic_json", slow)
    with pytest.raises(ValueError, match="Final arrival receipt write"):
        _collect(setup)
    assert not list((feed / "sessions").glob("*/arrival.json"))
    assert next((feed / "attempts").glob("*/late_arrival.json")).is_file()


def test_new_cutoff_cannot_replace_a_captured_arrival(setup):
    feed, cache, _, _ = setup
    _collect(setup)
    with pytest.raises(ValueError, match="different immutable cutoff"):
        arrival_feed.collect_arrival(
            feed, expected_session="2026-11-26", decision_cutoff="2026-11-26T19:30:00+08:00", cache_dir=cache,
        )


@pytest.mark.parametrize("now", ["2026-11-25T10:00:00Z", "2026-11-26T11:00:01Z", "2026-11-27T10:00:00Z"])
def test_historical_and_future_session_capture_cannot_be_faked(setup, now):
    feed, _, _, clock = setup
    clock["now"] = pd.Timestamp(now)
    with pytest.raises(ValueError, match="exchange session before cutoff"):
        _collect(setup)
    assert _outcome(feed)["status"] == "failed"


def _provider_pair(setup, tmp_path):
    receipt = _collect(setup)
    old = receipt["provider_dir"]
    new = tmp_path / "next-provider"
    shutil.copytree(old, new)
    day = (new / "calendars/day.txt").read_text()
    (new / "calendars/day.txt").write_text(day + "2026-11-27\n")
    for name in chenditc.UNIVERSES:
        path = new / f"instruments/{name}.txt"
        path.write_text(path.read_text().replace("2026-11-26", "2026-11-27"))
    for path in new.glob("features/*/*.day.bin"):
        path.write_bytes(path.read_bytes() + np.array([1.3], dtype="<f4").tobytes())
    return old, new


def test_extension_preserves_missing_values_and_complete_prior_history(setup, tmp_path):
    old, new = _provider_pair(setup, tmp_path)
    audit = arrival_feed.audit_provider_snapshots(new, previous_provider=old)
    assert audit["status"] == "admitted"
    assert audit["historical_cells_changed"] == 0
    assert audit["coverage"]["end_date"] == "2026-11-27"


def test_historical_price_or_adjustment_revision_is_quarantined(setup, tmp_path):
    old, new = _provider_pair(setup, tmp_path)
    for field in ("close", "factor"):
        path = new / f"features/sh600000/{field}.day.bin"
        values = np.fromfile(path, dtype="<f4")
        values[1] = 1.1
        values.tofile(path)
    audit = arrival_feed.audit_provider_snapshots(new, previous_provider=old)
    assert audit["status"] == "quarantined"
    assert audit["historical_cells_changed"] == 2
    assert "unexplained_historical_revisions" in audit["errors"]
    assert audit["revision_examples"][0]["date"] == "2026-11-24"


def test_historical_membership_revision_is_quarantined(setup, tmp_path):
    old, new = _provider_pair(setup, tmp_path)
    (new / "instruments/csi300.txt").write_text("SH600000\t2026-11-25\t2026-11-27\n")
    audit = arrival_feed.audit_provider_snapshots(new, previous_provider=old)
    assert "historical_membership_changed:csi300" in audit["errors"]


def test_missing_required_feature_cannot_be_admitted(setup, tmp_path):
    old, new = _provider_pair(setup, tmp_path)
    (new / "features/sh600000/open.day.bin").unlink()
    with pytest.raises(ValueError, match="missing required OHLCV"):
        arrival_feed.audit_provider_snapshots(new, previous_provider=old)
