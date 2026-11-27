"""Fit the registered official variant and infer one batch-aligned panel shard."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

from run_cn_worker import OFFICIAL_MODELS, _checkpoint, _source, _verify_provider, sha256


def run(args):
    import qlib
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config

    from financial_ts.cn_benchmark import prepare_cn_benchmark
    from financial_ts.cn_inference_shards import BatchAlignedDatasetView, state_sha256
    from financial_ts.workflow import _read_calendar, _save_json, build_task

    root = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    metadata = _verify_provider(args.provider)
    checkpoint = _checkpoint(args.checkpoint_dir, root, download=args.download_checkpoint)
    source = _source(root)
    destination = args.artifacts / args.model / f"shard-{args.shard}"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Use an empty shard output directory.")
    destination.mkdir(parents=True, exist_ok=True)
    protocol = prepare_cn_benchmark(
        args.provider,
        args.output,
        epochs=6,
        seed=42,
        device="cpu",
        num_threads=2,
        checkpoint=args.checkpoint_dir,
    )
    config = copy.deepcopy(protocol["configs"][args.model])
    qlib.init(
        provider_uri=str(args.provider.resolve()),
        region="cn",
        kernels=2,
        expression_cache=None,
        dataset_cache=None,
    )
    task, _ = build_task(config, _read_calendar(args.provider))
    dataset = init_instance_by_config(task["dataset"])
    model = init_instance_by_config(task["model"])
    model.fit(dataset)
    state_hash = state_sha256(model.adapter_state_)
    samples = getattr(model, "samples_", {})
    sample_hash = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    _save_json(destination / "fit_manifest.json", model.manifest_)
    _save_json(destination / "fit_samples.json", samples)
    _save_json(destination / "training_curves.json", model.training_curves_)
    _save_json(destination / "resolved_config.json", config)
    _save_json(destination / "data_source.json", metadata)
    _save_json(destination / "protocol.json", protocol)
    _save_json(destination / "source.json", source)
    # Serialize this completed fit before inference, without redistributing tensors.
    print(
        json.dumps(
            {
                "fit_completed": args.model,
                "shard": args.shard,
                "best_epoch": model.best_epoch_,
                "trained_state_sha256": state_hash,
                "fit_seconds": model.fit_seconds_,
            }
        ),
        flush=True,
    )
    view = BatchAlignedDatasetView(dataset, args.shard, args.count, model.batch_size)
    for segment, filename in (
        ("valid", "validation_predictions.csv.gz"),
        ("test", "predictions.csv.gz"),
    ):
        prediction = model.predict(view, segment=segment)
        # Labels are read only after this segment's forecasts have been produced.
        label = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
        joined = prediction.rename("score").to_frame().join(label.astype("float64").rename("label"))
        joined.to_csv(destination / filename, compression={"method": "gzip", "mtime": 0})
    if state_sha256(model.adapter_state_) != state_hash:
        raise ValueError("Selected trained state changed during inference.")
    if model.mode != "frozen" and state_sha256(model._backend.adapter_state_dict()) != state_hash:
        raise ValueError("Loaded adapter tensors changed during inference.")
    settings = {
        key: value
        for key, value in config["model"]["params"].items()
        if key not in {"checkpoint_path", "prediction_cache_dir"}
    }
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "model": args.model,
        "shard": args.shard,
        "count": args.count,
        "identity": {
            "model": args.model,
            "batch_size": model.batch_size,
            "settings": settings,
            "segments": protocol["segments"],
            "provider_fingerprint": protocol["provider_fingerprint"],
            "checkpoint": checkpoint,
            "source_files": model.manifest_["source_files"],
            "git_commit": source["git_commit"],
            "trained_state_sha256": state_hash,
            "fit_samples_sha256": sample_hash,
            "best_epoch": model.best_epoch_,
        },
        "segments": view.segments,
        "model_details": {
            "name": "timesfm3_panel",
            "mode": model.mode,
            "parameter_count": model.parameter_count_,
            "trainable_parameter_count": model.trainable_parameter_count_,
            "best_epoch": model.best_epoch_,
            "epochs_run": model.manifest_.get("epochs_run", 0),
            "fit_seconds": model.fit_seconds_,
            "pretrained": True,
            "selection_rmse": model.manifest_.get("selected_sampled_validation_rmse"),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "export_policy": "Predictions, hashes, fit metadata only; no weights, adapters, pickles or provider.",
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(destination.iterdir())
            if path.is_file()
        ],
    }
    _save_json(destination / "shard.json", manifest)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
            stream.write(
                f"### {args.model}, shard {args.shard + 1}/{args.count}\n\n"
                f"Exact trained-state SHA256: `{state_hash}`.\n\n"
                "Complete assigned validation/test batches exported; no model tensors exported.\n"
            )
    print(
        json.dumps({"completed": args.model, "shard": args.shard, "segments": view.segments}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=OFFICIAL_MODELS)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--count", default=4, type=int)
    parser.add_argument("--provider", type=Path, default=Path(".data/chenditc/2026-11-26"))
    parser.add_argument("--output", type=Path, default=Path("runs/chenditc-csi300-20261126"))
    parser.add_argument("--artifacts", type=Path, default=Path("worker-artifacts"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(".data/timesfm3-checkpoint"))
    parser.add_argument("--download-checkpoint", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.count:
        parser.error("--shard must be in [0, --count).")
    try:
        run(args)
    except Exception as exc:
        destination = args.artifacts / args.model / f"shard-{args.shard}"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "failure.json").write_text(
            json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}) + "\n"
        )
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
