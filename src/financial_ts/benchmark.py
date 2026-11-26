"""A fixed, inspectable comparison protocol; no test-set-driven model search."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from financial_ts.catalog import MODEL_NAMES, NEURAL_MODELS, TRAINABLE_MODELS
from financial_ts.workflow import _json_ready, _provider_fingerprint, _save_json, run_workflow


def benchmark_config(name, provider, output, seed=42, epochs=12, device="cpu", checkpoint=None):
    """Matched input/split/optimizer settings; architecture depth semantics differ."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")
    if name == "lightgbm":
        params = {"num_boost_round": 300, "early_stopping_rounds": 30, "num_threads": 2}
    elif name in NEURAL_MODELS:
        params = {
            "d_model": 32,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "n_epochs": epochs,
            "early_stop": 4,
            "batch_size": 128,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "device": device,
            "num_threads": 2,
            "aux_loss_weight": 0.01,
        }
    else:
        if checkpoint is None:
            raise ValueError("timesfm3_pretrained needs an explicit local checkpoint directory.")
        params = {
            "checkpoint_path": str(Path(checkpoint).expanduser().resolve()),
            "context_length": 256,
            "batch_size": 8,
            "device": device,
            "num_threads": 2,
        }
    return {
        "provider_uri": str(Path(provider).expanduser().resolve()),
        "output_dir": str(Path(output).expanduser().resolve()),
        "experiment_name": "architecture_comparison",
        "region": "us",
        "instruments": "all",
        "seed": seed,
        "warmup_sessions": 60,
        "sequence_length": 32,
        "segments": None,
        "model": {"name": name, "params": params},
        "backtest": {"enabled": False},
    }


def run_benchmark(
    provider, output, models=None, seeds=(7, 42, 2026), epochs=12, device="cpu", checkpoint=None
):
    """Run the declared models/seeds and preserve both runs and aggregate metrics."""
    models = list(TRAINABLE_MODELS if models is None else models)
    seeds = list(seeds)
    if not models or len(set(models)) != len(models) or any(m not in MODEL_NAMES for m in models):
        raise ValueError("models must be a nonempty list of distinct supported model names.")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int for seed in seeds):
        raise ValueError("seeds must be a nonempty list of distinct integers.")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    root = Path(output).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"Benchmark output must be new or empty: {root}")
    configs = []
    for name in models:
        for seed in [seeds[0]] if name == "timesfm3_pretrained" else seeds:
            configs.append(
                benchmark_config(
                    name, provider, root / name / f"seed-{seed}", seed, epochs, device, checkpoint
                )
            )
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "running",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "models": models,
        "seeds": seeds,
        "provider_fingerprint": _provider_fingerprint(Path(provider).expanduser().resolve()),
        "selection": "early stopping on validation only; no automatic winner selected from test scores",
        "comparability": "all scratch models share factors, target and chronological splits; neural models share 32-session context and AdamW budget; LightGBM uses one factor row and boosting; parameter counts and effective depth differ",
        "holdout": "exploratory historical comparison; a previously inspected interval is not a fresh final holdout",
        "pretrained": "official TimesFM3 is a separate fixed-weight replay with 256-session price context and unknown pretraining overlap",
        "configs": copy.deepcopy(configs),
    }
    _save_json(root / "protocol.json", protocol)
    rows = []
    try:
        for config in configs:
            name, seed = config["model"]["name"], config["seed"]
            print(f"Benchmark: {name}, seed={seed}", flush=True)
            result = run_workflow(config)
            metric = result["metrics"]
            details = metric["model_details"]
            row = {
                "model": name,
                "seed": None if details["pretrained"] else seed,
                "pretrained": details["pretrained"],
                "parameter_count": details["parameter_count"],
                "best_epoch": details["best_epoch"],
                "epochs_run": details["epochs_run"],
                "validation_rmse": details["validation_rmse"],
                "test_rmse": metric["rmse"],
                "zero_return_rmse": metric["zero_return_rmse"],
                "test_mae": metric["mae"],
                "directional_accuracy": metric["directional_accuracy"],
                "prediction_count": metric["prediction_count"],
                "fit_seconds": details["fit_seconds"],
                "elapsed_seconds": result["run"]["elapsed_seconds"],
            }
            rows.append(row)
            _save_json(root / "comparison.json", {"status": "running", "runs": rows})
            pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
            print(json.dumps(row), flush=True)
        aggregates = []
        frame = pd.DataFrame(rows)
        for name, group in frame.groupby("model", sort=False):
            aggregates.append(
                {
                    "model": name,
                    "runs": len(group),
                    "pretrained": bool(group["pretrained"].iloc[0]),
                    "parameter_count": group["parameter_count"].iloc[0],
                    "test_rmse_mean": group["test_rmse"].mean(),
                    "test_rmse_std_across_seeds": group["test_rmse"].std(ddof=0),
                    "zero_return_rmse": group["zero_return_rmse"].mean(),
                    "validation_rmse_mean": group["validation_rmse"].mean(),
                }
            )
        result = {
            "status": "completed",
            "runs": rows,
            "aggregate": aggregates,
            "interpretation": "seed dispersion is not a confidence interval; no profitability claim",
        }
        _save_json(root / "comparison.json", result)
        pd.DataFrame(aggregates).to_csv(root / "aggregate.csv", index=False)
        protocol.update(status="completed", finished_at_utc=pd.Timestamp.now(tz="UTC").isoformat())
        _save_json(root / "protocol.json", protocol)
        return _json_ready(result)
    except Exception as exc:
        protocol.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _save_json(root / "protocol.json", protocol)
        raise
