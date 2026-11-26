"""Bounded adapter training with validation-only selection for official TimesFM3.

This runner deliberately has no test-evaluation path. It reads adjusted provider
closes only through the final validation label and saves only local adapter
tensors, never a copy of the frozen official checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from financial_ts.workflow import (
    LABEL,
    LABEL_HORIZON,
    _json_ready,
    _provider_fingerprint,
    _read_calendar,
    _save_json,
    resolve_segments,
)

BACKEND_SETTINGS = {
    "num_threads": 2,
    "loop_start": 8,
    "loop_end": 12,
    "loop_steps": 2,
    "adapter_rank": 8,
    "gate_scale": 0.1,
}


@dataclass
class _Samples:
    contexts: np.ndarray
    labels: np.ndarray
    records: list[dict]


def _file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _make_model(checkpoint, device):
    # The optional official package is needed only when this experiment is run.
    from financial_ts.timesfm_loop import LoopTimesFM3

    return LoopTimesFM3(checkpoint, device=device, **BACKEND_SETTINGS)


def _read_closes(provider, calendar, segments):
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
    last = int(calendar.get_loc(pd.Timestamp(segments["valid"][1]))) + LABEL_HORIZON
    cutoff = calendar[last]
    if cutoff >= pd.Timestamp(segments["test"][0]):
        raise ValueError("Validation labels must end before the test segment starts.")
    instruments = sorted(
        D.list_instruments(
            D.instruments("all"),
            start_time=segments["train"][0],
            end_time=segments["valid"][1],
            as_list=True,
        )
    )
    if not instruments:
        raise ValueError("No instruments cover the training/validation partitions.")
    # No label expressions and no test-period prices are requested from Qlib.
    frame = D.features(instruments, ["$close"], start_time=calendar[0], end_time=cutoff, freq="day")
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Provider closes must be nonempty with unique instrument/date rows.")
    return frame["$close"], cutoff


def _select_samples(closes, calendar, bounds, context_length, budget):
    """Balance instruments, then space each instrument's eligible dates evenly."""
    first, last = (int(calendar.get_loc(pd.Timestamp(date))) for date in bounds)
    eligible = {}
    for instrument in sorted(closes.index.get_level_values("instrument").unique()):
        prices = closes.xs(instrument, level="instrument").reindex(calendar).to_numpy()
        candidates = []
        for position in range(max(first, context_length - 1), last + 1):
            window = prices[position - context_length + 1 : position + LABEL_HORIZON + 1]
            if (
                len(window) == context_length + LABEL_HORIZON
                and np.isfinite(window).all()
                and (window > 0).all()
            ):
                candidates.append(position)
        if candidates:
            eligible[instrument] = (prices, candidates)
    if not eligible:
        raise ValueError("No complete positive close contexts and two-close labels in a partition.")
    instruments = list(eligible)
    if len(instruments) > budget:
        instruments = [
            instruments[i] for i in np.linspace(0, len(instruments) - 1, budget, dtype=int)
        ]
    quotas = dict.fromkeys(instruments, 0)
    remaining = min(budget, sum(len(eligible[name][1]) for name in instruments))
    while remaining:
        for instrument in instruments:
            if quotas[instrument] < len(eligible[instrument][1]):
                quotas[instrument] += 1
                remaining -= 1
                if not remaining:
                    break
    samples = []
    for instrument, count in quotas.items():
        prices, positions = eligible[instrument]
        for index in np.linspace(0, len(positions) - 1, count, dtype=int):
            position = positions[index]
            context = np.ascontiguousarray(
                prices[position - context_length + 1 : position + 1], dtype=np.float32
            )
            label = float(prices[position + 2] / prices[position + 1] - 1)
            if not np.isfinite(context).all() or not math.isfinite(label):
                raise ValueError("Close values or delayed-return labels overflowed.")
            samples.append(
                (
                    context,
                    label,
                    {
                        "instrument": instrument,
                        "datetime": calendar[position].strftime("%Y-%m-%d"),
                        "context_start": calendar[position - context_length + 1].strftime(
                            "%Y-%m-%d"
                        ),
                        "context_end": calendar[position].strftime("%Y-%m-%d"),
                        "entry_date": calendar[position + 1].strftime("%Y-%m-%d"),
                        "exit_date": calendar[position + 2].strftime("%Y-%m-%d"),
                        "context_sha256": hashlib.sha256(
                            context.astype("<f4").tobytes()
                        ).hexdigest(),
                        "label": label,
                    },
                )
            )
    samples.sort(key=lambda sample: (sample[2]["datetime"], sample[2]["instrument"]))
    return _Samples(
        np.stack([sample[0] for sample in samples]),
        np.asarray([sample[1] for sample in samples], dtype=np.float64),
        [sample[2] for sample in samples],
    )


def _evaluate(model, samples, batch_size, device, baseline=False):
    import torch

    model.eval()
    predict = model.baseline_return_scores if baseline else model.return_scores
    predictions = []
    with torch.no_grad():
        for start in range(0, len(samples.labels), batch_size):
            batch = torch.as_tensor(samples.contexts[start : start + batch_size], device=device)
            scores = predict(batch)
            if scores.shape != (len(batch),) or not torch.isfinite(scores).all():
                raise ValueError("The loop model must return one finite return per context.")
            predictions.append(scores.detach().cpu().numpy().astype(np.float64))
    predictions = np.concatenate(predictions)
    return float(np.sqrt(np.mean((predictions - samples.labels) ** 2))), predictions


def _snapshot(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.adapter_state_dict().items()
    }


def _change(initial, state, gates_only=False):
    keys = [
        name for name in initial if not gates_only or ".gate." in name or name.startswith("gate.")
    ]
    return math.sqrt(
        sum(float((state[name].double() - initial[name].double()).square().sum()) for name in keys)
    )


def run_loop_experiment(
    provider,
    checkpoint,
    output,
    epochs=3,
    train_samples=256,
    valid_samples=128,
    context_length=128,
    batch_size=4,
    lr=0.001,
    seed=42,
    device="cpu",
):
    """Run one declared budget and select adapters using validation RMSE only.

    Positive complete contexts are sampled uniformly within each instrument,
    with approximately balanced instrument quotas. Epoch zero is eligible for
    selection. All requested epochs run; no test labels are loaded or scored.
    """
    for name, value in {
        "epochs": epochs,
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "context_length": context_length,
        "batch_size": batch_size,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not 2 <= context_length <= 15360:
        raise ValueError("context_length must be between 2 and 15360.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if not isinstance(lr, (int, float)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive.")
    if not isinstance(device, str) or not (
        device == "cpu" or device == "cuda" or device.startswith("cuda:")
    ):
        raise ValueError("device must be cpu, cuda, or cuda:N.")
    provider = Path(provider).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Experiment output must be new or empty: {output}")
    for source in (provider, checkpoint):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Experiment output must be separate from provider and checkpoint.")
    for filename in ("config.json", "model.safetensors"):
        if not (checkpoint / filename).is_file():
            raise ValueError("checkpoint must contain local config.json and model.safetensors.")
    calendar = _read_calendar(provider)
    segments = resolve_segments(calendar)
    output.mkdir(parents=True, exist_ok=True)
    # Protect the local derived weights even if output is outside the usual runs/ directory.
    (output / ".gitignore").write_text("adapter.pt\n", encoding="utf-8")
    started = time.perf_counter()
    settings = {
        "provider": str(provider),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "epochs": epochs,
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "context_length": context_length,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "device": device,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "backend": dict(BACKEND_SETTINGS),
    }
    manifest = {
        "status": "running",
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "settings": settings,
        "segments": segments,
        "label": LABEL,
        "selection": "minimum sampled validation RMSE across epoch zero and all declared epochs",
        "sampling": "balanced instrument quotas; evenly spaced eligible dates within each instrument",
        "test_evaluated": False,
        "pretraining_overlap": "unknown",
        "python": platform.python_version(),
        "packages": {},
        "synthetic": False,
    }
    _save_json(output / "manifest.json", manifest)
    try:
        import torch

        for package in ("financial-ts-qlib", "pyqlib", "numpy", "pandas", "torch", "timesfm"):
            try:
                manifest["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest["provider_fingerprint"] = _provider_fingerprint(provider)
        manifest["checkpoint_files"] = [
            _file_record(checkpoint / name) for name in ("config.json", "model.safetensors")
        ]
        manifest["source_files"] = [
            _file_record(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("timesfm_loop.py"),
                Path(__file__).with_name("workflow.py"),
            )
            if path.is_file()
        ]
        if (provider / "metadata.json").is_file():
            manifest["data_manifest"] = json.loads((provider / "metadata.json").read_text())
            manifest["synthetic"] = bool(manifest["data_manifest"].get("synthetic", False))
        closes, cutoff = _read_closes(provider, calendar, segments)
        train = _select_samples(closes, calendar, segments["train"], context_length, train_samples)
        valid = _select_samples(closes, calendar, segments["valid"], context_length, valid_samples)
        _save_json(output / "samples.json", {"train": train.records, "valid": valid.records})
        manifest["samples_file"] = _file_record(output / "samples.json")
        manifest["last_price_read"] = cutoff.strftime("%Y-%m-%d")
        manifest["sample_counts"] = {"train": len(train.labels), "valid": len(valid.labels)}
        target_mean = float(train.labels.mean())
        target_scale = float(train.labels.std(ddof=0))
        target_scale = target_scale if target_scale > 1e-12 else 1.0
        manifest["target_normalization"] = {
            "fit_partition": "train",
            "mean": target_mean,
            "scale": target_scale,
        }
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(BACKEND_SETTINGS["num_threads"])
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        model = _make_model(str(checkpoint), device)
        base = list(model.base_parameters())
        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        initial = _snapshot(model)
        if not base or any(parameter.requires_grad for parameter in base):
            raise ValueError("Every pretrained base parameter must be frozen.")
        if not trainable or {id(parameter) for parameter in base} & {
            id(parameter) for parameter in trainable.values()
        }:
            raise ValueError("Only adapter/gate parameters may enter the optimizer.")
        if set(trainable) != set(initial):
            raise ValueError(
                "Adapter checkpoint keys must exactly match trainable parameter names."
            )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(parameter.numel() for parameter in trainable.values())
        if (
            parameter_count != model.parameter_count
            or trainable_count != model.trainable_parameter_count
        ):
            raise ValueError("Backend parameter counts disagree with the actual parameters.")
        manifest["parameters"] = {
            "total": parameter_count,
            "trainable": trainable_count,
            "base": sum(parameter.numel() for parameter in base),
            "base_frozen": True,
            "optimizer_names": list(trainable),
        }
        model.eval()
        parity_contexts = torch.as_tensor(valid.contexts[:batch_size], device=device)
        with torch.no_grad():
            baseline_forecast = model.baseline_forecast(parity_contexts)
            loop_forecast = model.forecast(parity_contexts)
        if (
            loop_forecast.shape != (len(parity_contexts), 2)
            or baseline_forecast.shape != loop_forecast.shape
            or not torch.isfinite(loop_forecast).all()
            or not torch.allclose(loop_forecast, baseline_forecast, rtol=1e-6, atol=1e-6)
        ):
            raise ValueError("Epoch-zero loop forecasts do not match the frozen baseline.")
        manifest["baseline_parity"] = {
            "contexts": len(parity_contexts),
            "max_forecast_absolute_error": float((loop_forecast - baseline_forecast).abs().max()),
            "rtol": 1e-6,
            "atol": 1e-6,
        }
        train_rmse, _ = _evaluate(model, train, batch_size, device, baseline=True)
        valid_rmse, baseline_scores = _evaluate(model, valid, batch_size, device, baseline=True)
        curves = [
            {
                "epoch": 0,
                "train_rmse": train_rmse,
                "validation_rmse": valid_rmse,
                "updates": 0,
                "train_measurement": "fixed baseline",
            }
        ]
        best_epoch, best_rmse, best_state, best_scores = (
            0,
            valid_rmse,
            initial,
            baseline_scores.copy(),
        )
        optimizer = torch.optim.AdamW(trainable.values(), lr=lr, weight_decay=0.0)
        generator, updates = np.random.default_rng(seed), 0
        _save_json(output / "manifest.json", manifest)
        _save_json(output / "learning_curves.json", curves)
        for epoch in range(1, epochs + 1):
            model.train()
            squared_error, observed = 0.0, 0
            order = generator.permutation(len(train.labels))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                contexts = torch.as_tensor(train.contexts[indices], device=device)
                labels = torch.as_tensor(train.labels[indices], device=device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model.return_scores(contexts)
                if predictions.shape != labels.shape:
                    raise ValueError("Training scores must match the label shape.")
                normalized_error = (
                    (predictions - target_mean) - (labels - target_mean)
                ) / target_scale
                loss = normalized_error.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Adapter training produced nonfinite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(trainable.values()), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                squared_error += float((predictions.detach().double() - labels).square().sum())
                observed += len(indices)
                updates += 1
                manifest["optimizer_updates"] = updates
                manifest["training_examples_seen"] = (epoch - 1) * len(train.labels) + observed
            if any(parameter.grad is not None for parameter in base):
                raise ValueError("Frozen pretrained parameters unexpectedly received gradients.")
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
                best_epoch, best_rmse, best_state, best_scores = (
                    epoch,
                    valid_rmse,
                    _snapshot(model),
                    scores.copy(),
                )
            _save_json(output / "learning_curves.json", curves)
            manifest["last_completed_epoch"] = epoch
            _save_json(output / "manifest.json", manifest)
            print(json.dumps(curves[-1]), flush=True)
        final_state = _snapshot(model)
        adapter_path = output / "adapter.pt"
        torch.save(best_state, adapter_path)
        # Tensor-only restoration of this run's own adapter artifact; no arbitrary pickle loading.
        model.load_adapter_state_dict(
            torch.load(adapter_path, map_location="cpu", weights_only=True)
        )
        replay_samples = _Samples(
            valid.contexts[:batch_size], valid.labels[:batch_size], valid.records[:batch_size]
        )
        _, replay = _evaluate(model, replay_samples, batch_size, device)
        if not np.allclose(replay, best_scores[: len(replay)], rtol=1e-6, atol=1e-8):
            raise ValueError("Saved adapter replay did not reproduce selected validation scores.")
        predictions_frame = pd.DataFrame(valid.records)
        predictions_frame["baseline_score"], predictions_frame["selected_score"] = (
            baseline_scores,
            best_scores,
        )
        predictions_frame.to_csv(output / "validation_predictions.csv", index=False)
        metrics = {
            "baseline_validation_rmse": curves[0]["validation_rmse"],
            "selected_validation_rmse": best_rmse,
            "best_epoch": best_epoch,
            "epochs_run": epochs,
            "optimizer_updates": updates,
            "training_examples_seen": epochs * len(train.labels),
            "train_samples": len(train.labels),
            "validation_samples": len(valid.labels),
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "last_epoch_gate_change_l2": _change(initial, final_state, gates_only=True),
            "selected_gate_change_l2": _change(initial, best_state, gates_only=True),
            "last_epoch_adapter_change_l2": _change(initial, final_state),
            "selected_adapter_change_l2": _change(initial, best_state),
            "adapter_replay_max_score_error": float(
                np.max(np.abs(replay - best_scores[: len(replay)]))
            ),
            "test_evaluated": False,
            "pretraining_overlap": "unknown",
        }
        _save_json(output / "metrics.json", metrics)
        manifest.update(
            status="completed",
            finished_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            adapter_file=_file_record(adapter_path),
            optimizer_updates=updates,
        )
        _save_json(output / "manifest.json", manifest)
        return _json_ready(
            {
                "output_dir": str(output),
                "metrics": metrics,
                "run": manifest,
                "learning_curves": curves,
            }
        )
    except Exception as exc:
        manifest.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(output / "manifest.json", manifest)
        raise
