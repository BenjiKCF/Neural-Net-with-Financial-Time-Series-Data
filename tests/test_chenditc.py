"""Protect release identity, safe extraction, and observed-date coverage."""

import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from financial_ts import chenditc


def _sha(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _fixture(tmp_path, *, mutate_members=None, mutate_manifest=None):
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    day = "2026-11-24\n2026-11-25\n2026-11-26\n"
    members = {
        "qlib_bin/calendars/day.txt": day.encode(),
        "qlib_bin/calendars/day_future.txt": (day + "2026-11-27\n2026-11-30\n").encode(),
    }
    rows = "SH600000\t2026-11-24\t2026-11-26\nSZ000001\t2026-11-25\t2026-11-26\n"
    for name in chenditc.UNIVERSES:
        members[f"qlib_bin/instruments/{name}.txt"] = rows.encode()
    members["qlib_bin/features/sh600000/close.day.bin"] = np.array(
        [0, 1, np.nan, 1.2], dtype="<f4"
    ).tobytes()
    members["qlib_bin/features/sz000001/close.day.bin"] = np.array(
        [1, 2, 2.1], dtype="<f4"
    ).tobytes()
    if mutate_members:
        mutate_members(members)
    archive = cache / "qlib_bin.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, raw in members.items():
            info = tarfile.TarInfo(name)
            if isinstance(raw, tarfile.TarInfo):
                output.addfile(raw)
            else:
                info.size = len(raw)
                output.addfile(info, io.BytesIO(raw))
    manifest = {
        "release_tag": "2026-11-26",
        "target_trade_date": "2026-11-26",
        "future_start_date": "2026-11-27",
        "future_end_date": "2026-11-30",
        "dolt_commit": "a" * 32,
        "investment_data_commit": "b" * 40,
        "qlib_commit": "c" * 40,
        "image_digest": "sha256:" + "d" * 64,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha(archive.read_bytes()),
    }
    if mutate_manifest:
        mutate_manifest(manifest)
    manifest_path = cache / "qlib_bin.manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    metadata = {
        "id": 123,
        "tag_name": "2026-11-26",
        "html_url": "https://github.com/chenditc/investment_data/releases/tag/2026-11-26",
        "published_at": "2026-11-26T10:08:41Z",
        "assets": [
            {
                "name": name,
                "id": index,
                "size": (cache / name).stat().st_size,
                "digest": _sha((cache / name).read_bytes()),
                "browser_download_url": (
                    "https://github.com/chenditc/investment_data/releases/download/"
                    f"2026-11-26/{name}"
                ),
            }
            for index, name in enumerate(chenditc.ASSET_NAMES)
        ],
    }
    return cache, metadata


def _validate(cache, metadata):
    return chenditc.validate_chenditc_archive(
        cache / "qlib_bin.tar.gz",
        cache / "qlib_bin.manifest.json",
        expected_tag="2026-11-26",
        release_metadata=metadata,
    )


def test_install_verified_snapshot_and_exclude_future_dates(tmp_path, monkeypatch):
    cache, metadata = _fixture(tmp_path)
    monkeypatch.setattr(chenditc, "resolve_chenditc_release", lambda release: metadata)
    target = tmp_path / "provider"
    result = chenditc.download_chenditc(target, cache_dir=cache)
    assert result["target_trade_date"] == result["end_date"] == "2026-11-26"
    assert result["start_date"] == "2026-11-24"
    assert result["sessions"] == 3
    assert result["future_calendar_sessions"] == 2
    assert result["observations"] == 4  # Missing close remains missing.
    assert result["finite_closes_on_target_date"] == 2
    assert result["future_dates_excluded_from_observations"] is True
    assert result["universes"]["csi300"]["unique_instruments"] == 2
    assert json.loads((target / "metadata.json").read_text()) == result
    assert (target / "features/sh600000/close.day.bin").is_file()
    assert not (target / "qlib_bin").exists()


def test_archive_tamper_rejected_before_extraction(tmp_path):
    cache, metadata = _fixture(tmp_path)
    path = cache / "qlib_bin.tar.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Archive bytes"):
        _validate(cache, metadata)


def test_github_digest_independently_protects_manifest(tmp_path):
    cache, metadata = _fixture(tmp_path)
    path = cache / "qlib_bin.manifest.json"
    content = json.loads(path.read_text())
    content["dolt_commit"] = "e" * 32
    path.write_text(json.dumps(content) + "\n")
    with pytest.raises(ValueError, match="asset SHA256"):
        _validate(cache, metadata)


@pytest.mark.parametrize("name", ["../escaped", "/tmp/escaped", "qlib_bin/../escaped"])
def test_traversal_member_rejected(tmp_path, name):
    cache, metadata = _fixture(tmp_path, mutate_members=lambda rows: rows.update({name: b"x"}))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        _validate(cache, metadata)


def test_archive_symlink_rejected(tmp_path):
    info = tarfile.TarInfo("qlib_bin/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "/tmp"
    cache, metadata = _fixture(tmp_path, mutate_members=lambda rows: rows.update({info.name: info}))
    with pytest.raises(ValueError, match="Unsafe archive member"):
        _validate(cache, metadata)


def test_calendar_must_reach_manifest_target(tmp_path):
    cache, metadata = _fixture(
        tmp_path,
        mutate_members=lambda rows: rows.update(
            {"qlib_bin/calendars/day.txt": b"2026-11-24\n2026-11-25\n"}
        ),
    )
    with pytest.raises(ValueError, match="calendars disagree"):
        _validate(cache, metadata)


def test_future_calendar_must_extend_observed_prefix(tmp_path):
    cache, metadata = _fixture(
        tmp_path,
        mutate_members=lambda rows: rows.update(
            {"qlib_bin/calendars/day_future.txt": b"2026-11-27\n2026-11-30\n"}
        ),
    )
    with pytest.raises(ValueError, match="calendars disagree"):
        _validate(cache, metadata)


def test_manifest_requires_publishable_provenance(tmp_path):
    cache, metadata = _fixture(
        tmp_path, mutate_manifest=lambda manifest: manifest.update(image_digest=None)
    )
    with pytest.raises(ValueError, match="publishable provenance"):
        _validate(cache, metadata)


def test_duplicate_json_keys_rejected(tmp_path):
    cache, metadata = _fixture(tmp_path)
    path = cache / "qlib_bin.manifest.json"
    content = path.read_text().replace("{", '{"release_tag":"2026-11-26",', 1)
    path.write_text(content)
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        _validate(cache, metadata)


def test_historical_membership_intervals_preserve_exit_and_reentry(tmp_path):
    rows = (
        "SH600000\t2026-11-24\t2026-11-24\n"
        "SH600000\t2026-11-26\t2026-11-26\n"
        "SZ000001\t2026-11-25\t2026-11-26\n"
    )
    cache, metadata = _fixture(
        tmp_path,
        mutate_members=lambda members: members.update(
            {"qlib_bin/instruments/csi300.txt": rows.encode()}
        ),
    )
    result = _validate(cache, metadata)
    assert result["universes"]["csi300"]["membership_intervals"] == 3
    assert result["universes"]["csi300"]["unique_instruments"] == 2


def test_binary_cannot_extend_into_future_calendar(tmp_path, monkeypatch):
    cache, metadata = _fixture(
        tmp_path,
        mutate_members=lambda rows: rows.update(
            {
                "qlib_bin/features/sh600000/close.day.bin": np.array(
                    [0, 1, 1.1, 1.2, 1.3], dtype="<f4"
                ).tobytes()
            }
        ),
    )
    monkeypatch.setattr(chenditc, "resolve_chenditc_release", lambda release: metadata)
    target = tmp_path / "provider"
    with pytest.raises(ValueError, match="observations beyond day.txt"):
        chenditc.download_chenditc(target, cache_dir=cache)
    assert not target.exists()
    assert not list(tmp_path.glob(".provider-*"))


def test_existing_provider_is_preserved_without_network(tmp_path, monkeypatch):
    target = tmp_path / "provider"
    target.mkdir()
    (target / "keep.txt").write_text("existing data")
    monkeypatch.setattr(
        chenditc, "resolve_chenditc_release", lambda release: pytest.fail("No lookup allowed")
    )
    with pytest.raises(FileExistsError):
        chenditc.download_chenditc(target)
    assert (target / "keep.txt").read_text() == "existing data"


def test_asset_url_must_remain_in_pinned_release(tmp_path):
    _, metadata = _fixture(tmp_path)
    metadata["assets"][0]["browser_download_url"] = "https://example.com/unrelated"
    with pytest.raises(ValueError, match="Unexpected download URL"):
        chenditc._assets(metadata)
