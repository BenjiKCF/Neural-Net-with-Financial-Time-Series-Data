"""Prepare cached data and restore a manifest only after registered-byte verification.

The new host's observed retrieval metadata is retained separately; the immutable
campaign identity is never represented as the time of a new retrieval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from filelock import FileLock

from financial_ts import rolling
from financial_ts.chenditc import REPOSITORY, download_chenditc
from financial_ts.workflow import _provider_fingerprint


def ensure_provider_available(provider, release, *, downloader=None):
    """Download only into an empty location; keep existing cached bytes intact.

    Availability is deliberately separate from ``restore_provider``'s complete
    registered fingerprint check. A cache hit is not scientific acceptance.
    """
    provider = Path(provider).expanduser().absolute()
    if provider.is_symlink() or (provider.exists() and not provider.is_dir()):
        raise ValueError("Provider cache must be a real directory.")
    populated = provider.exists() and any(provider.iterdir())
    if not populated:
        (downloader or download_chenditc)(provider, release=release)
    metadata_path = provider / "metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError(
            "Existing provider cache has no regular metadata.json; preserve it for inspection."
        )
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("source") != REPOSITORY
        or metadata.get("release_tag") != release
        or metadata.get("region") != "cn"
        or metadata.get("frequency") != "day"
        or metadata.get("synthetic") is not False
    ):
        raise ValueError("Provider cache source/release differs from the pinned request.")
    return {
        "provider_available": True,
        "downloaded": not populated,
        "release": release,
        "qualification": "Availability only; registered provider bytes must still be verified.",
    }


def semantic_metadata(value):
    result = copy.deepcopy(value)
    result.pop("retrieved_at_utc", None)
    for asset in result.get("assets", {}).values():
        # Download counters and uploader profile fields are current API metadata;
        # asset IDs, digests, sizes, dates, URLs and validated archive remain pinned.
        asset.pop("download_count", None)
        asset.pop("uploader", None)
    return result


def restore_provider(output, provider, registered_metadata, receipts_dir):
    protocol = rolling._load_protocol(output)
    provider = Path(provider).expanduser().resolve()
    receipts_dir = Path(receipts_dir).expanduser().resolve()
    if provider != Path(protocol["provider_uri"]):
        raise ValueError("Provider path differs from the registered absolute path.")
    if receipts_dir == provider or receipts_dir.is_relative_to(provider):
        raise ValueError("Retrieval receipts must be outside the provider.")
    registered = Path(registered_metadata).read_bytes()
    registered_hash = hashlib.sha256(registered).hexdigest()
    if registered_hash != protocol["data_manifest_sha256"]:
        raise ValueError("Retained provider metadata differs from the protocol hash.")
    original = json.loads(registered)
    if original != protocol["data_manifest"]:
        raise ValueError("Retained provider metadata differs from the registered contents.")
    receipts_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(receipts_dir / ".restore.lock"), timeout=0):
        observed_bytes = (provider / "metadata.json").read_bytes()
        observed = json.loads(observed_bytes)
        if semantic_metadata(observed) != semantic_metadata(original):
            raise ValueError(
                "Downloaded source, archive or semantic metadata differs from registration."
            )
        fingerprint = _provider_fingerprint(provider)
        if fingerprint != protocol["provider_fingerprint"]:
            raise ValueError("Downloaded provider bytes differ from the registered fingerprint.")
        observed_hash = hashlib.sha256(observed_bytes).hexdigest()
        receipt = {
            "schema_version": 1,
            "observed_at_utc": rolling._now(),
            "protocol_sha256": protocol["protocol_sha256"],
            "provider_fingerprint": fingerprint,
            "registered_metadata_sha256": registered_hash,
            "observed_metadata_sha256": observed_hash,
            "observed_metadata_text": observed_bytes.decode("utf-8"),
            "registered_retrieved_at_utc": original.get("retrieved_at_utc"),
            "observed_retrieved_at_utc": observed.get("retrieved_at_utc"),
            "manifest_restored": observed_hash != registered_hash,
            "qualification": "Observed host metadata is retained here. Restored metadata is the immutable campaign identity, not a claim that this host retrieved the data at the original time.",
            "nonsemantic_fields": [
                "retrieved_at_utc",
                "assets.*.download_count",
                "assets.*.uploader",
            ],
        }
        receipt_path = receipts_dir / ("retrieval-" + str(time.time_ns()) + ".json")
        # Retain retrieval provenance before replacing any manifest bytes.
        rolling._save(receipt_path, receipt)
        if observed_hash != registered_hash:
            temporary = provider / "metadata.restoring.json"
            temporary.write_bytes(registered)
            temporary.replace(provider / "metadata.json")
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "receipt": str(receipt_path),
        "manifest_restored": receipt["manifest_restored"],
        "provider_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--registered-metadata", type=Path)
    parser.add_argument("--receipts-dir", type=Path)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--release")
    args = parser.parse_args()
    if args.download_if_missing and not args.release:
        parser.error("--download-if-missing requires an explicit --release")
    restore_arguments = (args.output, args.registered_metadata, args.receipts_dir)
    if any(restore_arguments) and not all(restore_arguments):
        parser.error("--output, --registered-metadata and --receipts-dir must be supplied together")
    if not args.download_if_missing and not all(restore_arguments):
        parser.error("Choose --download-if-missing or supply all restoration arguments")
    result = {}
    if args.download_if_missing:
        result.update(ensure_provider_available(args.provider, args.release))
    if all(restore_arguments):
        result.update(
            restore_provider(
                args.output, args.provider, args.registered_metadata, args.receipts_dir
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
