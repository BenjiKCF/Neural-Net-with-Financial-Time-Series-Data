"""Pinned, verified installation of chenditc's daily China Qlib provider.

The moving GitHub ``latest`` endpoint is resolved once. Both assets are pinned
to that release and their GitHub digests; the archive is additionally checked
against its provenance manifest. ``day_future.txt`` is retained for Qlib's
calendar mechanics, but only ``day.txt`` contains observed market sessions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

import numpy as np

REPOSITORY = "chenditc/investment_data"
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}"
ASSET_NAMES = ("qlib_bin.manifest.json", "qlib_bin.tar.gz")
UNIVERSES = ("all", "csi300", "csi500", "csi800", "csi1000", "csiall")
REQUIRED_MEMBERS = ("calendars/day.txt", "calendars/day_future.txt") + tuple(
    f"instruments/{name}.txt" for name in UNIVERSES
)
MANIFEST_KEYS = {
    "release_tag",
    "target_trade_date",
    "future_start_date",
    "future_end_date",
    "dolt_commit",
    "investment_data_commit",
    "qlib_commit",
    "image_digest",
    "archive_size_bytes",
    "archive_sha256",
}


def _date(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"Expected a canonical YYYY-MM-DD date, got {value!r}.")
    dt.date.fromisoformat(value)
    return value


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return "sha256:" + result.hexdigest()


def _json_bytes(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}.")
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def resolve_chenditc_release(release: str = "latest") -> dict:
    """Resolve a release and retain its exact API response for provenance."""
    endpoint = "latest" if release == "latest" else f"tags/{_date(release)}"
    request = Request(
        f"{API_ROOT}/releases/{endpoint}",
        headers={"User-Agent": "financial-ts-qlib", "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=60) as response:
        metadata = _json_bytes(response.read())
    tag = _date(metadata["tag_name"])
    if release != "latest" and tag != release:
        raise ValueError("The returned release tag differs from the requested tag.")
    if metadata.get("draft") or metadata.get("prerelease"):
        raise ValueError("A stable, published data release is required.")
    _assets(metadata)
    return metadata


def _assets(metadata: dict) -> dict:
    tag = _date(metadata["tag_name"])
    assets = {}
    for name in ASSET_NAMES:
        matching = [asset for asset in metadata.get("assets", []) if asset.get("name") == name]
        if len(matching) != 1:
            raise ValueError(f"Release must contain exactly one {name} asset.")
        asset = matching[0]
        expected_url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{name}"
        if asset.get("browser_download_url") != expected_url:
            raise ValueError(f"Unexpected download URL for {name}.")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", asset.get("digest") or ""):
            raise ValueError(f"Release has no usable SHA256 digest for {name}.")
        size = asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid asset size for {name}.")
        assets[name] = asset
    return assets


def _check_asset(path: Path, asset: dict) -> None:
    if path.is_symlink() or path.stat().st_size != asset["size"]:
        raise ValueError(f"Release asset size/type mismatch: {path.name}.")
    if _digest(path) != asset["digest"]:
        raise ValueError(f"Release asset SHA256 mismatch: {path.name}.")


def _download(asset: dict, destination: Path) -> None:
    if destination.exists():
        _check_asset(destination, asset)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    # Reopen through a normal file object so every failure closes the descriptor.
    import os

    os.close(handle)
    temporary = Path(name)
    try:
        request = Request(
            asset["browser_download_url"], headers={"User-Agent": "financial-ts-qlib"}
        )
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                downloaded += len(chunk)
                if downloaded > asset["size"]:
                    raise ValueError("Download exceeds its pinned release asset size.")
                output.write(chunk)
        _check_asset(temporary, asset)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _member_path(member: tarfile.TarInfo) -> str:
    name = member.name
    trimmed = name.rstrip("/")
    parts = PurePosixPath(trimmed).parts
    if (
        not name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or not parts
        or parts[0] != "qlib_bin"
        or str(PurePosixPath(trimmed)) != trimmed
        or any(part in (".", "..") for part in parts)
        or not (member.isdir() or member.isfile())
        or getattr(member, "sparse", None)
    ):
        raise ValueError(f"Unsafe archive member: {name!r}.")
    return "/".join(parts[1:])


def _lines(raw: bytes) -> list[str]:
    text = raw.decode("utf-8")
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("Required data files must be newline-terminated UTF-8.")
    lines = text[:-1].split("\n")
    if not lines or any(not value for value in lines):
        raise ValueError("Required data files may not be empty or contain blank rows.")
    return lines


def _calendars(members: dict[str, bytes], target: str) -> tuple[list[str], list[str]]:
    day = [_date(value) for value in _lines(members["calendars/day.txt"])]
    future = [_date(value) for value in _lines(members["calendars/day_future.txt"])]
    if day != sorted(set(day)) or future != sorted(set(future)):
        raise ValueError("Calendars must be unique and chronologically sorted.")
    if day[-1] != target or len(future) <= len(day) or future[: len(day)] != day:
        raise ValueError("Observed/future calendars disagree with the target trading date.")
    return day, future


def _instrument_summary(raw: bytes, target: str) -> dict:
    rows = []
    for line in _lines(raw):
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError("Instrument rows must contain symbol, start, and end.")
        symbol, start, end = fields
        if not re.fullmatch(r"[A-Za-z0-9._-]+", symbol):
            raise ValueError("Unsafe or empty instrument symbol.")
        if _date(start) > _date(end) or end > target:
            raise ValueError("Instrument membership interval exceeds the observed calendar.")
        rows.append((symbol, start, end))
    if rows != sorted(set(rows)) or max(row[2] for row in rows) != target:
        raise ValueError("Instrument intervals must be sorted, unique, and reach the target date.")
    last_end = {}
    for symbol, start, end in rows:
        if start <= last_end.get(symbol, ""):
            raise ValueError(f"Overlapping membership intervals for {symbol}.")
        last_end[symbol] = end
    return {
        "unique_instruments": len({row[0] for row in rows}),
        "membership_intervals": len(rows),
        "active_on_target_date": len({row[0] for row in rows if row[1] <= target <= row[2]}),
        "start_date": min(row[1] for row in rows),
        "end_date": max(row[2] for row in rows),
    }


def validate_chenditc_archive(
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_tag: str,
    release_metadata: dict | None = None,
) -> dict:
    """Validate archive identity, every member path, and calendar/date semantics."""
    archive, manifest_path = Path(archive_path), Path(manifest_path)
    _date(expected_tag)
    manifest = _json_bytes(manifest_path.read_bytes())
    if set(manifest) != MANIFEST_KEYS or manifest["release_tag"] != expected_tag:
        raise ValueError("Manifest fields or release tag are inconsistent.")
    for name in ("release_tag", "target_trade_date", "future_start_date", "future_end_date"):
        _date(manifest[name])
    if manifest["target_trade_date"] > expected_tag:
        raise ValueError("Target trading date is after the release date.")
    for name in ("investment_data_commit", "qlib_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest[name]):
            raise ValueError(f"Invalid immutable source identity: {name}.")
    if not re.fullmatch(r"[0-9a-v]{32}", manifest["dolt_commit"]):
        raise ValueError("Invalid immutable Dolt identity.")
    for name in ("image_digest", "archive_sha256"):
        if not isinstance(manifest[name], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", manifest[name]
        ):
            raise ValueError(f"Missing publishable provenance: {name}.")
    size = manifest["archive_size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Invalid archive size in manifest.")
    if archive.stat().st_size != size or _digest(archive) != manifest["archive_sha256"]:
        raise ValueError("Archive bytes do not match the provenance manifest.")
    if release_metadata is not None:
        if release_metadata["tag_name"] != expected_tag:
            raise ValueError("GitHub release identity does not match expected tag.")
        assets = _assets(release_metadata)
        _check_asset(manifest_path, assets["qlib_bin.manifest.json"])
        _check_asset(archive, assets["qlib_bin.tar.gz"])
    required, seen = {}, set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            name = _member_path(member)
            if name in seen:
                raise ValueError(f"Duplicate archive member: {name}.")
            seen.add(name)
            if name in REQUIRED_MEMBERS:
                if not member.isfile() or member.size > 16 * 1024 * 1024:
                    raise ValueError(f"Invalid required archive member: {name}.")
                required[name] = source.extractfile(member).read()
    if set(required) != set(REQUIRED_MEMBERS):
        raise ValueError("Archive is missing required calendar or instrument files.")
    day, future = _calendars(required, manifest["target_trade_date"])
    if (
        future[len(day)] != manifest["future_start_date"]
        or future[-1] != manifest["future_end_date"]
    ):
        raise ValueError("Future calendar range does not match the manifest.")
    universes = {
        name: _instrument_summary(required[f"instruments/{name}.txt"], day[-1])
        for name in UNIVERSES
    }
    return {
        "manifest": manifest,
        "manifest_sha256": _digest(manifest_path),
        "archive_member_count": len(seen),
        "start_date": day[0],
        "end_date": day[-1],
        "sessions": len(day),
        "future_calendar_sessions": len(future) - len(day),
        "universes": universes,
    }


def inspect_chenditc_provider(
    provider_dir: str | Path, *, target_trade_date: str | None = None
) -> dict:
    """Inspect observed calendar, historical memberships, and close-file coverage."""
    provider = Path(provider_dir)
    members = {name: (provider / name).read_bytes() for name in REQUIRED_MEMBERS}
    target = target_trade_date or _lines(members["calendars/day.txt"])[-1]
    day, future = _calendars(members, _date(target))
    universes = {
        name: _instrument_summary(members[f"instruments/{name}.txt"], target) for name in UNIVERSES
    }
    observations = 0
    observed_on_target = 0
    feature_symbols = []
    all_symbols = {line.split("\t")[0].lower() for line in _lines(members["instruments/all.txt"])}
    for path in sorted((provider / "features").glob("*/close.day.bin")):
        if path.stat().st_size % 4:
            raise ValueError(f"Incomplete float32 close binary: {path.parent.name}.")
        values = np.fromfile(path, dtype="<f4")
        if len(values) < 2 or not np.isfinite(values[0]) or values[0] != int(values[0]):
            raise ValueError(f"Invalid close binary offset: {path.parent.name}.")
        start = int(values[0])
        count = len(values) - 1
        if start < 0 or start + count > len(day):
            raise ValueError("A close binary contains observations beyond day.txt.")
        finite = np.isfinite(values[1:])
        if np.isinf(values[1:]).any() or (values[1:][finite] <= 0).any():
            raise ValueError(f"Invalid close values: {path.parent.name}.")
        observations += int(finite.sum())
        if start + count == len(day) and finite[-1]:
            observed_on_target += 1
        feature_symbols.append(path.parent.name)
    if not feature_symbols or set(feature_symbols) != all_symbols:
        raise ValueError("Close binaries and the all-instruments universe disagree.")
    return {
        "start_date": day[0],
        "end_date": day[-1],
        "sessions": len(day),
        "observations": observations,
        "feature_instruments": len(feature_symbols),
        "finite_closes_on_target_date": observed_on_target,
        "future_calendar_sessions": len(future) - len(day),
        "future_dates_excluded_from_observations": True,
        "universes": universes,
    }


def download_chenditc(
    target_dir: str | Path, *, release: str = "latest", cache_dir: str | Path | None = None
) -> dict:
    """Download, verify, and atomically install a new provider with metadata.json.

    ``target_dir`` is the provider root itself, not a parent directory. Existing
    nonempty destinations are rejected; pick a dated location for each snapshot.
    """
    target = Path(target_dir).expanduser().absolute()
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise FileExistsError("Choose a new or empty provider destination.")
    metadata = resolve_chenditc_release(release)
    tag = metadata["tag_name"]
    assets = _assets(metadata)
    cache = Path(cache_dir).expanduser() if cache_dir else target.parent / "downloads" / tag
    cache.mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        _download(assets[name], cache / name)
    (cache / "release.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validation = validate_chenditc_archive(
        cache / "qlib_bin.tar.gz",
        cache / "qlib_bin.manifest.json",
        expected_tag=tag,
        release_metadata=metadata,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        with tarfile.open(cache / "qlib_bin.tar.gz", "r|gz") as source:
            for member in source:
                name = _member_path(member)
                if not name:
                    continue
                destination = stage.joinpath(*PurePosixPath(name).parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as incoming, destination.open("xb") as output:
                        shutil.copyfileobj(incoming, output)
        coverage = inspect_chenditc_provider(
            stage, target_trade_date=validation["manifest"]["target_trade_date"]
        )
        provenance = {
            "schema_version": 1,
            "source": REPOSITORY,
            "region": "cn",
            "frequency": "day",
            "synthetic": False,
            "calendar": "SSE observed trading sessions; future calendar is not market data",
            "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "release_tag": tag,
            "release_url": metadata["html_url"],
            "release_id": metadata["id"],
            "release_published_at": metadata.get("published_at"),
            "target_trade_date": validation["manifest"]["target_trade_date"],
            "assets": assets,
            "validation": validation,
            **coverage,
            "missing_bars": "NaN; no forward fill",
            "membership": "Historical intervals from provider index snapshots; not a PIT guarantee",
        }
        (stage / "metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if target.exists():
            target.rmdir()
        stage.rename(target)
        return provenance
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="New or empty Qlib provider directory")
    parser.add_argument("--release", default="latest", help="latest or YYYY-MM-DD")
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    print(
        json.dumps(download_chenditc(args.target, release=args.release, cache_dir=args.cache_dir))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
