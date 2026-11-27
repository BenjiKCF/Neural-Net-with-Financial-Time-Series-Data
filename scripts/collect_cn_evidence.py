"""Combine verified forecast evidence with its deterministic portfolio replay.

The original numerical export is retained intact. No model weights, adapters,
pickles, provider binaries or inference caches are copied into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SOURCE_FILES = {
    "metrics.json",
    "resolved_config.json",
    "run.json",
    "source.json",
    "data_source.json",
    "protocol.json",
    "qlib_task.yaml",
    "learning_curves.json",
    "training_curves.json",
    "fit_manifest.json",
    "fit_samples.json",
    "daily_ic.csv",
    "validation_daily_ic.csv",
    "predictions.csv.gz",
    "validation_predictions.csv.gz",
    "portfolio.csv",
    "positions.json.gz",
    "audit.json",
    "checksums.json",
    "checkpoint_identity.json",
    "assembly.json",
}
REPLAY_FILES = {
    "portfolio.csv",
    "positions.json.gz",
    "metrics.json",
    "native_risk.csv",
    "source_gap_holdings.json",
    "replay_manifest.json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(source: Path, replay: Path, output: Path):
    if output.exists():
        raise ValueError("Collection requires a new destination; original evidence is preserved")
    original = json.loads((source / "run.json").read_text())
    forecasts = json.loads((source / "metrics.json").read_text())
    corrected = json.loads((replay / "metrics.json").read_text())
    receipt = json.loads((replay / "replay_manifest.json").read_text())
    if original["status"] not in {"completed", "forecasts_completed"}:
        raise ValueError("Source forecasts are incomplete")
    if receipt["original_run"] != original or receipt["provider"] != original["provider"]:
        raise ValueError("Replay is not bound to this exact forecast run and provider")
    for name, info in receipt["input_files"].items():
        if Path(name).name != name or digest(source / name) != info["sha256"]:
            raise ValueError("Replay forecast inputs have changed")
    for name, info in receipt["outputs"].items():
        if Path(name).name != name or digest(replay / name) != info["sha256"]:
            raise ValueError("Replay outputs have changed")
    for key, value in forecasts.items():
        if key not in {"portfolio", "qlib_metrics"} and corrected.get(key) != value:
            raise ValueError(f"Replay changed forecast metric {key}")
    if (source / "checksums.json").exists():
        exported = json.loads((source / "checksums.json").read_text())
        if exported["status"] != "completed":
            raise ValueError("Worker export is incomplete")
        for item in exported["files"]:
            name = item["path"]
            if name not in SOURCE_FILES or digest(source / name) != item["sha256"]:
                raise ValueError("Original worker export failed verification")
    elif not (source / "assembly.json").exists():
        raise ValueError("Expected a verified worker export or complete shard assembly")
    available = {path.name for path in source.iterdir() if path.is_file()}
    if available - SOURCE_FILES:
        raise ValueError(f"Unexpected source export files: {sorted(available - SOURCE_FILES)}")
    for name in REPLAY_FILES:
        if not (replay / name).is_file():
            raise ValueError(f"Replay lacks required numerical artifact {name}")
    (output / "original").mkdir(parents=True)
    for name in sorted(available):
        shutil.copyfile(source / name, output / "original" / name)
        if name not in {
            "metrics.json",
            "run.json",
            "portfolio.csv",
            "positions.json.gz",
            "checksums.json",
            "audit.json",
            "protocol.json",
        }:
            shutil.copyfile(source / name, output / name)
    for name in sorted(REPLAY_FILES):
        shutil.copyfile(replay / name, output / name)
    metadata = {
        **original,
        "status": "completed",
        "portfolio_status": "completed deterministic native Qlib replay",
        "portfolio_replay_manifest_sha256": digest(output / "replay_manifest.json"),
        "original_forecast_run_sha256": digest(output / "original/run.json"),
        "forecast_metrics_unchanged": True,
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    result = {
        "status": "completed",
        "source": str(source),
        "replay": str(replay),
        "original_numerical_evidence_retained": True,
        "forecast_metrics_unchanged": True,
        "export_policy": "Allowlisted numerical and metadata files only; no weights, adapters, pickles or raw provider",
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ],
    }
    (output / "collection.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = collect(args.source, args.replay, args.output)
    print(json.dumps({"status": result["status"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
