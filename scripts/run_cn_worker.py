"""Run one pinned CSI300 benchmark model and export numerical evidence only.

GitHub Actions invokes this worker once per matrix entry. Raw data, official
weights, fitted adapters, handler/model pickles, and inference caches stay in
its disposable local runtime. Only the explicit allowlist below is exported.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import traceback
from pathlib import Path

RELEASE = "2026-11-26"
ARCHIVE_SHA256 = "5c7d2391e569896f22fdccd146f9a0466fc363b7bbabb528c68bd2cc56f11559"
CHECKPOINT_REPOSITORY = "google/timesfm-3.0-pytorch"
CHECKPOINT_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
CHECKPOINT_FILES = {
    "config.json": {
        "bytes": 1273,
        "sha256": "ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6",
    },
    "model.safetensors": {
        "bytes": 1322898824,
        "sha256": "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da",
    },
    "LICENSE": {
        "bytes": 7270,
        "sha256": "3e36db7240d23adb6ac6d7d931892dcc5706a14a7d31581f08c35d2f25736dfd",
    },
}
SCRATCH_MODELS = (
    "lightgbm",
    "transformer",
    "deepseek_v41_mini",
    "glm54_mini",
    "qwen4_mini",
    "timesfm3_mini",
    "multiscale_residual_mixer",
    "regime_linear_moe",
)
OFFICIAL_MODELS = (
    "timesfm3_frozen_128",
    "timesfm3_frozen_256",
    "timesfm3_loop_128",
    "timesfm3_adapter_128",
)
COPY_FILES = (
    "metrics.json",
    "resolved_config.json",
    "run.json",
    "qlib_task.yaml",
    "learning_curves.json",
    "training_curves.json",
    "fit_manifest.json",
    "fit_samples.json",
    "daily_ic.csv",
    "validation_daily_ic.csv",
    "portfolio.csv",
)
COMPRESS_FILES = ("predictions.csv", "validation_predictions.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source(root: Path) -> dict:
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_ref": git("symbolic-ref", "--short", "-q", "HEAD")
        if (root / ".git").exists() and not os.environ.get("GITHUB_SHA")
        else os.environ.get("GITHUB_REF", "detached"),
        "tracked_source_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "workflow": {
            key: os.environ[key]
            for key in [
                "GITHUB_SHA",
                "GITHUB_REF",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_REPOSITORY",
                "GITHUB_SERVER_URL",
                "GITHUB_WORKFLOW_REF",
            ]
            if key in os.environ
        },
    }


def _verify_provider(provider: Path):
    metadata = json.loads((provider / "metadata.json").read_text())
    manifest = metadata.get("validation", {}).get("manifest", {})
    if (
        metadata.get("source") != "chenditc/investment_data"
        or metadata.get("release_tag") != RELEASE
        or metadata.get("target_trade_date") != RELEASE
        or manifest.get("archive_sha256") != f"sha256:{ARCHIVE_SHA256}"
        or metadata.get("synthetic")
    ):
        raise ValueError("Worker requires the hash-verified 2026-11-26 chenditc release")
    return metadata


def _checkpoint(path: Path, root: Path, *, download: bool):
    # Independently require agreement with the already published pilot evidence.
    pilot = json.loads((root / "reports/timesfm_loop_pilot/manifest.json").read_text())
    recorded = {Path(row["path"]).name: row for row in pilot["checkpoint_files"]}
    for name in ["config.json", "model.safetensors"]:
        expected = CHECKPOINT_FILES[name]
        if any(recorded[name][key] != expected[key] for key in ["sha256", "bytes"]):
            raise ValueError(
                "Worker checkpoint identity differs from the repository's pilot evidence"
            )
    if download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=CHECKPOINT_REPOSITORY,
            revision=CHECKPOINT_REVISION,
            allow_patterns=list(CHECKPOINT_FILES),
            local_dir=path,
        )
    for name, expected in CHECKPOINT_FILES.items():
        artifact = path / name
        if (
            not artifact.is_file()
            or artifact.stat().st_size != expected["bytes"]
            or sha256(artifact) != expected["sha256"]
        ):
            raise ValueError(f"Official checkpoint file failed pinned validation: {name}")
    return {
        "repository": CHECKPOINT_REPOSITORY,
        "revision": CHECKPOINT_REVISION,
        "files": CHECKPOINT_FILES,
        "distribution": "No checkpoint, adapter tensors, or model derivative is included in exported evidence.",
    }


def _auditor():
    path = Path(__file__).with_name("audit_cn_benchmark.py")
    spec = importlib.util.spec_from_file_location("cn_worker_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gzip_copy(source: Path, destination: Path):
    with source.open("rb") as raw, destination.open("wb") as compressed:
        # Fixed mtime and empty filename make the compression output reproducible.
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
            shutil.copyfileobj(raw, stream)


def _export_positions(run: Path, destination: Path, metadata: dict):
    path = (
        run
        / "mlruns"
        / metadata["experiment_id"]
        / metadata["recorder_id"]
        / "artifacts/portfolio_analysis/positions_normal_1day.pkl"
    )
    # This file was generated by the worker's own local Qlib backtest.
    with path.open("rb") as stream:
        positions = pickle.load(stream)
    rows = []
    for date, obj in sorted(positions.items()):
        position = obj.position
        stocks = [
            {
                "instrument": instrument,
                **{
                    field: float(value[field])
                    for field in ["amount", "price", "weight", "count_day"]
                    if field in value
                },
            }
            for instrument, value in sorted(position.items())
            if isinstance(value, dict)
        ]
        rows.append(
            {
                "datetime": str(date.date()),
                "cash": float(position["cash"]),
                "account": float(position["now_account_value"]),
                "holdings": stocks,
            }
        )
    payload = {
        "origin": "Native Qlib local position artifact converted to plain numerical JSON",
        "quantity_units": "Qlib adjusted share amounts; original-share units require the source factor",
        "source_artifact_sha256": sha256(path),
        "positions": rows,
    }
    with destination.open("wb") as compressed:
        with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
            stream.write(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode())
    return {"days": len(rows), "holding_records": sum(len(row["holdings"]) for row in rows)}


def export_worker(model, runs, provider, artifacts, source, checkpoint_identity=None):
    run = runs / model
    metadata = json.loads((run / "run.json").read_text())
    if metadata.get("status") != "completed":
        raise ValueError("Only a completed model can be exported as benchmark evidence")
    destination = artifacts / model
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Evidence output is not empty; choose another --artifacts directory")
    destination.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        path = run / name
        if path.exists():
            if path.is_symlink():
                raise ValueError("Numerical evidence must be a regular local file")
            shutil.copy2(path, destination / name)
    for name in COMPRESS_FILES:
        _gzip_copy(run / name, destination / (name + ".gz"))
    shutil.copy2(runs / "protocol.json", destination / "protocol.json")
    shutil.copy2(provider / "metadata.json", destination / "data_source.json")
    positions = _export_positions(run, destination / "positions.json.gz", metadata)
    audit = _auditor().audit(runs, provider, trusted_local_pickles=True)
    if model not in audit["audited_models"] or audit["status"] != "passed":
        _write_json(destination / "audit.json", audit)
        raise ValueError("The worker's independent numerical/source audit did not pass")
    # Each job exports its own independent result; no absent model is implied completed.
    audit["worker_model"] = model
    _write_json(destination / "audit.json", audit)
    _write_json(destination / "source.json", source)
    if checkpoint_identity is not None:
        _write_json(destination / "checkpoint_identity.json", checkpoint_identity)
    if "GITHUB_STEP_SUMMARY" in os.environ:
        metrics = json.loads((run / "metrics.json").read_text())
        text = (
            f"### {model}\n\n"
            f"- Completed {metrics['prediction_count']:,} test forecasts.\n"
            f"- Test RMSE: {metrics['rmse']:.8f}.\n"
            f"- Qlib simulated net return: {metrics['portfolio']['net_cumulative_return']:.2%}.\n"
            f"- {audit['backtest_qualification']}.\n"
            "- Numerical evidence exported; model weights and data providers excluded.\n"
        )
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write(text)
    manifest = {
        "schema_version": 1,
        "model": model,
        "status": "completed",
        "source": source,
        "release": RELEASE,
        "archive_sha256": ARCHIVE_SHA256,
        "provider_fingerprint": metadata["provider"],
        "positions": positions,
        "export_policy": "Explicit numerical/metadata allowlist. No raw provider, model/handler pickle, checkpoint, trained adapter tensor, or inference cache is exported.",
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(destination.iterdir())
            if path.is_file()
        ],
    }
    _write_json(destination / "checksums.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=[*SCRATCH_MODELS, *OFFICIAL_MODELS])
    parser.add_argument("--provider", type=Path, default=Path(".data/chenditc/2026-11-26"))
    parser.add_argument("--output", type=Path, default=Path("runs/chenditc-csi300-20261126"))
    parser.add_argument("--artifacts", type=Path, default=Path("worker-artifacts"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(".data/timesfm3-checkpoint"))
    parser.add_argument("--download-checkpoint", action="store_true")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Audit/export an existing locally completed run without training",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = _source(root)
    try:
        _verify_provider(args.provider)
        official = args.model in OFFICIAL_MODELS
        if args.download_checkpoint and not official:
            raise ValueError("Only official TimesFM variants need a downloaded checkpoint")
        checkpoint_identity = (
            _checkpoint(args.checkpoint_dir, root, download=args.download_checkpoint)
            if official
            else None
        )
        if not args.export_only:
            from financial_ts.cn_benchmark import prepare_cn_benchmark, run_cn_models

            protocol = prepare_cn_benchmark(
                args.provider,
                args.output,
                epochs=6,
                seed=42,
                device="cpu",
                num_threads=2,
                checkpoint=args.checkpoint_dir if official else None,
            )
            run_cn_models(protocol, models=[args.model])
        manifest = export_worker(
            args.model, args.output, args.provider, args.artifacts, source, checkpoint_identity
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "model": args.model,
                    "exported_files": len(manifest["files"]),
                }
            )
        )
        return 0
    except Exception as exc:
        destination = args.artifacts / args.model
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(
            destination / "failure.json",
            {
                "status": "failed",
                "model": args.model,
                "source": source,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
