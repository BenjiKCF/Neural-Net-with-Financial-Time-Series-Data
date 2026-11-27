"""Recover the fixed TimesFM pilot and compare selected variants on common dates.

The gated adapter-only control matches the loop model's trainable parameter
count and initial tensors. It applies ``h + (1 + gate(h)) * adapter(h)`` without
replaying the shared block. Its gate has a different role and it uses less
compute; this is not an exact compute-matched causal ablation of looping.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from financial_ts.loop_experiment import (
    BACKEND_SETTINGS,
    _change,
    _evaluate,
    _file_record,
    _read_closes,
    _select_samples,
    _snapshot,
    run_loop_experiment,
)
from financial_ts.timesfm_loop import LoopTimesFM3
from financial_ts.workflow import (
    LABEL,
    LABEL_HORIZON,
    _provider_fingerprint,
    _read_calendar,
    _save_json,
    resolve_segments,
)


class _GatedAdapterStack(nn.Module):
    """Use the same initial refinement tensors, with no shared-block replay."""

    def __init__(self, loop_stack):
        super().__init__()
        self.base = loop_stack.base
        self.width = loop_stack.width
        self.loop_end = loop_stack.loop_end
        self.refinements = loop_stack.refinements
        self.refinement_enabled = True

    def forward(
        self,
        input_embeddings,
        patch_mask,
        segment_ids=None,
        segment_pos=None,
        decode_cache=None,
        var_segment_pos=None,
    ):
        if input_embeddings.ndim != 4 or input_embeddings.shape[-1] != self.width:
            raise ValueError("Adapter stack expects [batch,variate,patch,hidden] embeddings.")
        if patch_mask.shape != input_embeddings.shape[:-1] or patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be boolean [batch,variate,patch].")
        if decode_cache is not None and (
            not isinstance(decode_cache, (list, tuple))
            or len(decode_cache) != len(self.base.layers)
            or any(cache is not None for cache in decode_cache)
        ):
            raise ValueError("Adapter control supports full-sequence decoding, not KV caches.")
        if not self.refinement_enabled:
            return self.base(
                input_embeddings,
                patch_mask,
                segment_ids,
                segment_pos,
                decode_cache,
                var_segment_pos,
            )
        hidden, caches, masks = input_embeddings, [], []
        for layer_index, layer in enumerate(self.base.layers):
            hidden, cache, attention_mask = layer(
                hidden, patch_mask, segment_ids, segment_pos, None, var_segment_pos
            )
            caches.append(cache)
            masks.append(attention_mask)
            if layer_index + 1 == self.loop_end:
                for refinement in self.refinements:
                    normalized = refinement.norm(hidden)
                    gate = refinement.gate_scale * torch.tanh(refinement.gate(normalized))
                    adapter = refinement.adapter_up(F.gelu(refinement.adapter_down(normalized)))
                    hidden = hidden + (1 + gate) * adapter
        return hidden, caches, masks


class GatedAdapterTimesFM3(LoopTimesFM3):
    """Parameter-matched adapter-only control with the original layer depth."""

    def __init__(self, checkpoint_path, **kwargs):
        super().__init__(checkpoint_path, **kwargs)
        # No parameters are created or reinitialized during conversion. Calling
        # either constructor with the same seed produces identical initial state.
        self.model.transformer_stack = _GatedAdapterStack(self.loop_stack)
        self.eval()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_recovery(original, recovered, *, allow_numerical_replication=False):
    """Reject a changed selection or materially different recovered forecasts."""
    expected_metrics = _read_json(original / "metrics.json")
    actual_metrics = _read_json(recovered / "metrics.json")
    original_curves = _read_json(original / "learning_curves.json")
    recovered_curves = _read_json(recovered / "learning_curves.json")
    if len(original_curves) != len(recovered_curves):
        raise ValueError("Recovery changed the number of training epochs.")
    differences, strict_failures = [], []
    for expected, actual in zip(original_curves, recovered_curves):
        if any(actual[key] != expected[key] for key in ("epoch", "updates", "train_measurement")):
            raise ValueError("Recovery changed the recorded training protocol.")
        for key in ("train_rmse", "validation_rmse"):
            differences.append(abs(actual[key] - expected[key]))
            if not np.isclose(actual[key], expected[key], rtol=1e-5, atol=1e-8):
                strict_failures.append(f"epoch {actual['epoch']} {key} curve")
    for key in (
        "best_epoch",
        "epochs_run",
        "optimizer_updates",
        "training_examples_seen",
        "train_samples",
        "validation_samples",
        "parameter_count",
        "trainable_parameter_count",
    ):
        if actual_metrics[key] != expected_metrics[key]:
            raise ValueError(f"Recovery changed {key}.")
    if _read_json(original / "samples.json") != _read_json(recovered / "samples.json"):
        raise ValueError("Recovery changed sampled contexts, labels, or their hashes.")
    original_predictions = pd.read_csv(
        original / "validation_predictions.csv", float_precision="round_trip"
    )
    recovered_predictions = pd.read_csv(
        recovered / "validation_predictions.csv", float_precision="round_trip"
    )
    columns = ["datetime", "instrument", "context_sha256", "label"]
    if not original_predictions[columns].equals(recovered_predictions[columns]):
        raise ValueError("Recovery changed validation prediction rows or labels.")
    for frame, metrics in (
        (original_predictions, expected_metrics),
        (recovered_predictions, actual_metrics),
    ):
        for score_name, metric_name in (
            ("baseline_score", "baseline_validation_rmse"),
            ("selected_score", "selected_validation_rmse"),
        ):
            rmse = float(np.sqrt(np.mean((frame[score_name] - frame["label"]) ** 2)))
            if not np.isclose(rmse, metrics[metric_name], rtol=1e-12, atol=1e-15):
                raise ValueError("Recorded recovery metrics disagree with the saved predictions.")
    score_differences, score_mae, score_rmse = {}, {}, {}
    for key in ("baseline_score", "selected_score"):
        expected = original_predictions[key].to_numpy()
        actual = recovered_predictions[key].to_numpy()
        if not np.isfinite(actual).all():
            raise ValueError("Recovered predictions must be finite.")
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-8):
            strict_failures.append(f"all-row {key} predictions")
        score_differences[key] = float(np.max(np.abs(actual - expected)))
        score_mae[key] = float(np.mean(np.abs(actual - expected)))
        score_rmse[key] = float(np.sqrt(np.mean((actual - expected) ** 2)))
    old_manifest, new_manifest = (
        _read_json(original / "manifest.json"),
        _read_json(recovered / "manifest.json"),
    )
    for key in ("provider_fingerprint", "segments", "target_normalization"):
        if old_manifest[key] != new_manifest[key]:
            raise ValueError(f"Recovery changed {key}.")
    for old, new in zip(old_manifest["checkpoint_files"], new_manifest["checkpoint_files"]):
        if old["sha256"] != new["sha256"]:
            raise ValueError("Recovery used different official checkpoint bytes.")
    if strict_failures and not allow_numerical_replication:
        raise ValueError("Strict numerical recovery failed: " + "; ".join(strict_failures))
    return {
        "status": "numerical_replication_only"
        if strict_failures
        else "reproduced_within_declared_tolerance",
        "strict_failures": strict_failures,
        "numerical_replication_explicitly_enabled": allow_numerical_replication,
        "interpretation": "fresh same-protocol retrain; not restoration of the original adapter"
        if strict_failures
        else "same fixed protocol and forecasts within the declared tolerance",
        "rtol": 1e-5,
        "atol": 1e-8,
        "validation_rows_checked": len(original_predictions),
        "sample_records_checked": sum(
            len(rows) for rows in _read_json(original / "samples.json").values()
        ),
        "max_curve_absolute_error": max(differences),
        "max_score_absolute_errors": score_differences,
        "mean_score_absolute_errors": score_mae,
        "rmse_between_prediction_sets": score_rmse,
        "original_selected_validation_rmse": expected_metrics["selected_validation_rmse"],
        "retrained_selected_validation_rmse": actual_metrics["selected_validation_rmse"],
        "selected_epoch": actual_metrics["best_epoch"],
        "adapter_archive_sha256_equal": old_manifest["adapter_file"]["sha256"]
        == new_manifest["adapter_file"]["sha256"],
        "original_packages": old_manifest["packages"],
        "recovered_packages": new_manifest["packages"],
    }


def _train_control(provider, checkpoint, output, settings, expected_parameters):
    output.mkdir()
    (output / ".gitignore").write_text("adapter.pt\n", encoding="utf-8")
    started = time.perf_counter()
    calendar = _read_calendar(provider)
    segments = resolve_segments(calendar)
    closes, cutoff = _read_closes(provider, calendar, segments)
    train = _select_samples(
        closes, calendar, segments["train"], settings["context_length"], settings["train_samples"]
    )
    valid = _select_samples(
        closes, calendar, segments["valid"], settings["context_length"], settings["valid_samples"]
    )
    _save_json(output / "samples.json", {"train": train.records, "valid": valid.records})
    seed, device, batch_size = (settings[key] for key in ("seed", "device", "batch_size"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(BACKEND_SETTINGS["num_threads"])
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = GatedAdapterTimesFM3(checkpoint, device=device, **BACKEND_SETTINGS)
    base = list(model.base_parameters())
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    initial = _snapshot(model)
    if not base or any(p.requires_grad for p in base):
        raise ValueError("The control backbone must be frozen.")
    if not trainable or set(trainable) != set(initial):
        raise ValueError("Only the saved gate/adapter tensors may be trainable.")
    if {id(p) for p in base} & {id(p) for p in trainable.values()}:
        raise ValueError("Control optimizer and frozen backbone must be disjoint.")
    if (
        model.parameter_count != expected_parameters["total"]
        or model.trainable_parameter_count != expected_parameters["trainable"]
    ):
        raise ValueError("Adapter-only control must match the loop parameter count.")
    target_mean, target_scale = float(train.labels.mean()), float(train.labels.std(ddof=0))
    target_scale = target_scale if target_scale > 1e-12 else 1.0
    baseline_train, _ = _evaluate(model, train, batch_size, device, baseline=True)
    baseline_valid, baseline_scores = _evaluate(model, valid, batch_size, device, baseline=True)
    initial_valid, initial_scores = _evaluate(model, valid, batch_size, device)
    if not np.array_equal(initial_scores, baseline_scores):
        raise ValueError("Control initialization must exactly preserve all baseline scores.")
    curves = [
        {
            "epoch": 0,
            "train_rmse": baseline_train,
            "validation_rmse": initial_valid,
            "updates": 0,
            "train_measurement": "fixed baseline",
        }
    ]
    best_epoch, best_rmse, best_state = 0, initial_valid, initial
    best_scores = initial_scores.copy()
    optimizer = torch.optim.AdamW(trainable.values(), lr=settings["lr"], weight_decay=0.0)
    generator, updates = np.random.default_rng(seed), 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order, squared_error, observed = generator.permutation(len(train.labels)), 0.0, 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            contexts = torch.as_tensor(train.contexts[indices], device=device)
            labels = torch.as_tensor(train.labels[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model.return_scores(contexts)
            if predictions.shape != labels.shape:
                raise ValueError("Control predictions must match the training labels.")
            error = ((predictions - target_mean) - (labels - target_mean)) / target_scale
            loss = error.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Control loss is nonfinite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            squared_error += float((predictions.detach().double() - labels).square().sum())
            observed += len(indices)
            updates += 1
        if any(p.grad is not None for p in base):
            raise ValueError("Control backbone unexpectedly received gradients.")
        valid_rmse, scores = _evaluate(model, valid, batch_size, device)
        curves.append(
            {
                "epoch": epoch,
                "train_rmse": math.sqrt(squared_error / observed),
                "validation_rmse": valid_rmse,
                "updates": updates,
                "train_measurement": "online pre-update batches",
            }
        )
        if valid_rmse < best_rmse:
            best_epoch, best_rmse = epoch, valid_rmse
            best_state, best_scores = _snapshot(model), scores.copy()
        _save_json(output / "learning_curves.json", curves)
        print(json.dumps({"variant": "gated_adapter_only", **curves[-1]}), flush=True)
    adapter_path = output / "adapter.pt"
    torch.save(best_state, adapter_path)
    model.load_adapter_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=True))
    _, replayed = _evaluate(model, valid, batch_size, device)
    if not np.array_equal(replayed, best_scores):
        raise ValueError("The selected control adapter must replay all validation scores exactly.")
    pd.DataFrame(valid.records).assign(
        baseline_score=baseline_scores, selected_score=best_scores
    ).to_csv(output / "validation_predictions.csv", index=False)
    metrics = {
        "best_epoch": best_epoch,
        "baseline_validation_rmse": baseline_valid,
        "selected_validation_rmse": best_rmse,
        "epochs_run": settings["epochs"],
        "optimizer_updates": updates,
        "selected_checkpoint_updates": best_epoch * math.ceil(len(train.labels) / batch_size),
        "training_examples_seen": settings["epochs"] * len(train.labels),
        "parameter_count": model.parameter_count,
        "trainable_parameter_count": model.trainable_parameter_count,
        "selected_gate_change_l2": _change(initial, best_state, gates_only=True),
        "selected_adapter_change_l2": _change(initial, best_state),
        "initial_baseline_max_score_error": float(np.max(np.abs(initial_scores - baseline_scores))),
        "adapter_replay_max_score_error": float(np.max(np.abs(replayed - best_scores))),
        "adapter_replay_contexts": len(replayed),
        "test_used_for_training_or_selection": False,
    }
    manifest = {
        "status": "completed",
        "variant": "gated_adapter_only",
        "formula": "h + (1 + gate(h)) * adapter(h)",
        "settings": settings,
        "segments": segments,
        "last_training_price_read": cutoff.strftime("%Y-%m-%d"),
        "sample_counts": {"train": len(train.labels), "valid": len(valid.labels)},
        "target_normalization": {
            "fit_partition": "train",
            "mean": target_mean,
            "scale": target_scale,
        },
        "base_frozen": True,
        "optimizer_names": list(trainable),
        "adapter_file": _file_record(adapter_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _save_json(output / "learning_curves.json", curves)
    _save_json(output / "metrics.json", metrics)
    _save_json(output / "manifest.json", manifest)
    del optimizer, model, base, trainable
    gc.collect()
    return metrics, manifest


def _read_evaluation_closes(provider, calendar, segments):
    import qlib
    from qlib.data import D
    from qlib.data.cache import H

    H.clear()
    qlib.init(
        provider_uri=str(provider),
        region="us",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    last = int(calendar.get_loc(pd.Timestamp(segments["test"][1]))) + LABEL_HORIZON
    instruments = sorted(
        D.list_instruments(
            D.instruments("all"),
            start_time=segments["valid"][0],
            end_time=segments["test"][1],
            as_list=True,
        )
    )
    frame = D.features(
        instruments, ["$close"], start_time=calendar[0], end_time=calendar[last], freq="day"
    )
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Evaluation closes require unique nonempty instrument/date rows.")
    return frame["$close"]


def _scores_summary(labels, scores):
    return {
        "rmse": float(np.sqrt(np.mean((scores - labels) ** 2))),
        "mae": float(np.mean(np.abs(scores - labels))),
        "directional_accuracy": float(np.mean(np.sign(scores) == np.sign(labels))),
    }


def run_timesfm_comparison(
    provider,
    checkpoint,
    output,
    original_evidence,
    *,
    resume=False,
    allow_numerical_replication=False,
):
    """Recover once, train one fixed control, then score common full partitions.

    No test-period data is requested until both selected adapter states have
    been saved. Full validation scoring is diagnostic; it does not reselect
    checkpoints. Existing test history is explicitly marked as already seen.
    Numerical drift requires an explicit flag and is reported as a fresh
    same-protocol retrain, never an exact restored adapter. A failed strict
    check can be resumed before control training, reusing the single retrain.
    """
    provider, checkpoint, output, evidence = (
        Path(value).expanduser().resolve()
        for value in (provider, checkpoint, output, original_evidence)
    )
    previous_attempt = None
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        if not resume:
            raise ValueError("Comparison output must be new or empty unless explicitly resumed.")
        previous_attempt = _read_json(output / "comparison.json")
        recovered_manifest = _read_json(output / "recovered_loop/manifest.json")
        if (
            previous_attempt["status"] != "failed"
            or recovered_manifest["status"] != "completed"
            or (output / "gated_adapter_only").exists()
            or "selection_frozen_at_utc" in previous_attempt
            or (output / "strict_recovery_attempt.json").exists()
        ):
            raise ValueError(
                "Resume is limited to a completed retrain stopped before control/test."
            )
    elif resume:
        raise ValueError("Resume requires an existing failed comparison and completed retrain.")
    for source in (provider, checkpoint, evidence):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Comparison output must be separate from data and original evidence.")
    original = _read_json(evidence / "manifest.json")
    settings = {
        name: original["settings"][name]
        for name in (
            "epochs",
            "train_samples",
            "valid_samples",
            "context_length",
            "batch_size",
            "lr",
            "seed",
            "device",
        )
    }
    if original["settings"]["backend"] != BACKEND_SETTINGS:
        raise ValueError("Original evidence uses a different loop architecture.")
    if original["status"] != "completed" or original["test_evaluated"] is not False:
        raise ValueError("Original evidence must be a completed validation-only pilot.")
    for record in original["source_files"]:
        name = Path(record["path"]).name
        if name in {"loop_experiment.py", "timesfm_loop.py"}:
            path = Path(__file__).with_name(name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Original loop training sources changed; recovery needs review.")
    fingerprint = _provider_fingerprint(provider)
    if fingerprint != original["provider_fingerprint"]:
        raise ValueError("Recovery requires the original provider bytes.")
    expected_checkpoint = {
        Path(record["path"]).name: record for record in original["checkpoint_files"]
    }
    checkpoint_files = [
        _file_record(checkpoint / name) for name in ("config.json", "model.safetensors")
    ]
    for record in checkpoint_files:
        expected = expected_checkpoint[Path(record["path"]).name]
        if any(record[key] != expected[key] for key in ("sha256", "bytes")):
            raise ValueError("Recovery requires the original official checkpoint bytes.")
    if previous_attempt is not None:
        if (
            previous_attempt["settings"] != settings
            or previous_attempt["provider_fingerprint"] != fingerprint
        ):
            raise ValueError("Resume settings and provider must match the failed attempt.")
        (output / "strict_recovery_attempt.json").write_bytes(
            (output / "comparison.json").read_bytes()
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "status": "running",
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "settings": settings,
        "label": LABEL,
        "original_evidence": str(evidence),
        "original_evidence_files": [
            _file_record(path) for path in sorted(evidence.iterdir()) if path.is_file()
        ],
        "provider_fingerprint": fingerprint,
        "checkpoint_files": checkpoint_files,
        "source_files": [_file_record(Path(__file__))],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "torch", "pyqlib")
        },
        "execution_environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_parallel_info": torch.__config__.parallel_info(),
            "configured_model_threads": BACKEND_SETTINGS["num_threads"],
            "original_pilot_cpu_details": "not recorded; exact cross-host identity is not assumed",
        },
        "resumed_without_retraining": previous_attempt is not None,
        "control": {
            "name": "gated_adapter_only",
            "formula": "h + (1 + gate(h)) * adapter(h)",
            "initial_parameters": "identical tensors under the same constructor seed",
            "parameter_matched": True,
            "compute_matched": False,
            "gate_role": "modulates the adapter, whereas the loop gate modulates shared-block replay",
        },
        "selection": "original sampled validation only; epoch zero and all fixed-budget epochs eligible",
        "test_used_for_training_or_selection": False,
        "full_validation_used_for_selection": False,
        "test_status": "previously inspected historical interval; not a fresh final holdout",
        "pretraining_overlap": "unknown",
    }
    _save_json(output / "comparison.json", report)
    try:
        recovered = output / "recovered_loop"
        if previous_attempt is None:
            print("Recovering the original fixed pilot once", flush=True)
            recovery_run = run_loop_experiment(provider, checkpoint, recovered, **settings)
        else:
            print(
                "Reusing the single completed same-protocol retrain; no additional training",
                flush=True,
            )
            recovery_run = {
                "metrics": _read_json(recovered / "metrics.json"),
                "run": _read_json(recovered / "manifest.json"),
            }
        report["recovery"] = _verify_recovery(
            evidence, recovered, allow_numerical_replication=allow_numerical_replication
        )
        report["loop_training"] = recovery_run["metrics"]
        _save_json(output / "comparison.json", report)
        print("Training the parameter-matched gated adapter-only control", flush=True)
        control = output / "gated_adapter_only"
        report["control_training"], report["control_manifest"] = _train_control(
            provider, checkpoint, control, settings, recovery_run["run"]["parameters"]
        )
        report["selection_frozen_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        _save_json(output / "comparison.json", report)
        # The first test-price read is deliberately below the selection freeze.
        calendar = _read_calendar(provider)
        segments = resolve_segments(calendar)
        closes = _read_evaluation_closes(provider, calendar, segments)
        samples = {
            name: _select_samples(
                closes, calendar, segments[name], settings["context_length"], len(closes)
            )
            for name in ("valid", "test")
        }
        frames = {name: pd.DataFrame(values.records) for name, values in samples.items()}
        device, batch_size = settings["device"], settings["batch_size"]
        model = LoopTimesFM3(checkpoint, device=device, **BACKEND_SETTINGS)
        model.load_adapter_state_dict(
            torch.load(recovered / "adapter.pt", map_location="cpu", weights_only=True)
        )
        for partition, values in samples.items():
            print(
                f"Scoring frozen and loop models on {partition}: {len(values.labels)} rows",
                flush=True,
            )
            _, frames[partition]["timesfm3_frozen_128"] = _evaluate(
                model, values, batch_size, device, baseline=True
            )
            _, frames[partition]["timesfm3_loop_128"] = _evaluate(model, values, batch_size, device)
        del model
        gc.collect()
        model = GatedAdapterTimesFM3(checkpoint, device=device, **BACKEND_SETTINGS)
        model.load_adapter_state_dict(
            torch.load(control / "adapter.pt", map_location="cpu", weights_only=True)
        )
        for partition, values in samples.items():
            print(
                f"Scoring gated adapter-only on {partition}: {len(values.labels)} rows", flush=True
            )
            _, frames[partition]["timesfm3_adapter_only_128"] = _evaluate(
                model, values, batch_size, device
            )
        del model
        gc.collect()
        report["segments"], report["evaluations"] = segments, {}
        for name, frame in frames.items():
            labels = frame["label"].to_numpy()
            frame.insert(0, "segment", name)
            frame["zero_return"] = 0.0
            report["evaluations"][name] = {
                "count": len(frame),
                "target_mean": float(labels.mean()),
                "target_std": float(labels.std(ddof=0)),
                "models": {
                    model_name: _scores_summary(labels, frame[model_name].to_numpy())
                    for model_name in (
                        "zero_return",
                        "timesfm3_frozen_128",
                        "timesfm3_loop_128",
                        "timesfm3_adapter_only_128",
                    )
                },
            }
        predictions_path = output / "all_period_predictions.csv"
        pd.concat(frames.values(), ignore_index=True).to_csv(predictions_path, index=False)
        report.update(
            status="completed",
            finished_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            predictions_file=_file_record(predictions_path),
        )
        _save_json(output / "comparison.json", report)
        return report
    except Exception as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(output / "comparison.json", report)
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-evidence", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a completed retrain stopped before control/test",
    )
    parser.add_argument(
        "--allow-numerical-replication",
        action="store_true",
        help="Explicitly label a numerically different same-protocol retrain instead of claiming exact recovery",
    )
    arguments = parser.parse_args()
    result = run_timesfm_comparison(
        arguments.provider,
        arguments.checkpoint,
        arguments.output,
        arguments.original_evidence,
        resume=arguments.resume,
        allow_numerical_replication=arguments.allow_numerical_replication,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
