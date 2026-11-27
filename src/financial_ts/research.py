"""A fixed small-model campaign with selection locked before test prediction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import pandas as pd

from financial_ts.benchmark import benchmark_config
from financial_ts.workflow import (
    LABEL,
    _provider_fingerprint,
    _read_calendar,
    _save_json,
    build_task,
)

RESEARCH_MODELS = ("timesfm3_mini", "multiscale_residual_mixer", "regime_linear_moe")
SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _finite_vector(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector.")
    return array


def rmse(labels, predictions):
    labels, predictions = _finite_vector(labels, "labels"), _finite_vector(predictions, "scores")
    if labels.shape != predictions.shape:
        raise ValueError("Labels and scores must have the same shape.")
    result = float(np.sqrt(np.mean((predictions - labels) ** 2)))
    if not np.isfinite(result):
        raise ValueError("Forecast error overflowed.")
    return result


def select_shrinkage(labels, predictions, training_mean):
    """Choose a convex shrinkage toward training drift using validation only."""
    if not np.isfinite(training_mean):
        raise ValueError("training_mean must be finite.")
    predictions = _finite_vector(predictions, "scores")
    candidates = [
        {
            "gamma": gamma,
            "validation_rmse": rmse(labels, training_mean + gamma * (predictions - training_mean)),
        }
        for gamma in SHRINKAGE_GRID
    ]
    # Ties prefer the smaller coefficient, a fixed lower-variance convention.
    best = min(candidates, key=lambda row: (row["validation_rmse"], row["gamma"]))
    return {**best, "anchor": float(training_mean), "candidates": candidates}


def paired_block_interval(labels, candidate, reference, *, block_length=20, draws=2000, seed=913):
    """Circular block-bootstrap interval for paired RMSE difference.

    Negative values favor the candidate. This conditional historical diagnostic
    does not correct for repeated research on the same test interval.
    """
    y = _finite_vector(labels, "labels")
    candidate = _finite_vector(candidate, "candidate")
    reference = _finite_vector(reference, "reference")
    if y.shape != candidate.shape or y.shape != reference.shape:
        raise ValueError("Paired series must have identical shapes.")
    if type(block_length) is not int or not 1 <= block_length <= len(y):
        raise ValueError("block_length must be between one and the observation count.")
    if type(draws) is not int or draws < 100:
        raise ValueError("Use at least 100 bootstrap draws.")
    rng = np.random.default_rng(seed)
    errors, base_errors = (candidate - y) ** 2, (reference - y) ** 2
    differences = np.empty(draws)
    blocks = (len(y) + block_length - 1) // block_length
    for draw in range(draws):
        starts = rng.integers(0, len(y), size=blocks)
        indices = ((starts[:, None] + np.arange(block_length)) % len(y)).ravel()[: len(y)]
        differences[draw] = np.sqrt(errors[indices].mean()) - np.sqrt(base_errors[indices].mean())
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "rmse_difference": rmse(y, candidate) - rmse(y, reference),
        "lower_95": float(low),
        "upper_95": float(high),
        "block_length": block_length,
        "draws": draws,
        "seed": seed,
        "method": "paired circular moving-block bootstrap; negative favors candidate",
    }


def _build_dataset(config, calendar):
    import qlib
    from qlib.data.cache import H
    from qlib.utils import init_instance_by_config

    H.clear()
    qlib.init(
        provider_uri=config["provider_uri"],
        region="us",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    task, segments = build_task(config, calendar)
    return init_instance_by_config(task["dataset"]), segments


def _labels(dataset, segment):
    from qlib.data.dataset.handler import DataHandlerLP

    sampler = dataset.prepare(
        segment, col_set="label", data_key=DataHandlerLP.DK_L, dtype=np.float64
    )
    sampler.config(fillna_type="none")
    index = sampler.get_index()
    result = pd.Series(
        [np.asarray(sampler[i])[-1, 0] for i in range(len(sampler))], index=index, name="label"
    )
    if not result.index.is_unique:
        raise ValueError("Duplicate label observations.")
    _finite_vector(result.to_numpy(), "labels")
    return result


def _fit_trial(config, calendar, dataset):
    from qlib.utils import init_instance_by_config

    task, _ = build_task(config, calendar)
    model = init_instance_by_config(task["model"])
    model.fit(dataset)
    return model


def _aligned_predict(model, dataset, segment, index):
    scores = model.predict(dataset, segment=segment)
    if not scores.index.equals(index) or not scores.index.is_unique:
        raise ValueError("Prediction indices must exactly match the declared label rows.")
    _finite_vector(scores.to_numpy(), "scores")
    return scores.to_numpy(dtype=np.float64)


def _source_files():
    root = Path(__file__).parent
    result = []
    for path in (
        Path(__file__),
        root / "model.py",
        root / "workflow.py",
        root / "catalog.py",
        root / "architectures/financial.py",
        root / "architectures/timesfm.py",
    ):
        content = path.read_bytes()
        result.append(
            {
                "path": str(path),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return result


def run_scratch_research(provider, output, *, seeds=(7, 42, 2026), epochs=12, device="cpu"):
    """Fit the fixed three-architecture roster, lock validation choices, then test."""
    import torch

    seeds = list(seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("seeds must be distinct integers in [0, 2**32).")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    provider, output = Path(provider).expanduser().resolve(), Path(output).expanduser().resolve()
    if output == provider or output in provider.parents or provider in output.parents:
        raise ValueError("Provider and output must be separate directories.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Research output must be new or empty.")
    calendar = _read_calendar(provider)
    configs = [
        benchmark_config(name, provider, output / name / f"seed-{seed}", seed, epochs, device)
        for name in RESEARCH_MODELS
        for seed in seeds
    ]
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = {
        "status": "registered",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "models": list(RESEARCH_MODELS),
        "seeds": seeds,
        "configs": configs,
        "label": LABEL,
        "provider_fingerprint": _provider_fingerprint(provider),
        "source_files": _source_files(),
        "selection": "all model epochs and all forecast shrinkage coefficients selected on validation; selection.json saved before first test prediction",
        "ensembles": "equal weights across every declared seed; equal weights across three architecture ensembles",
        "shrinkage": {
            "grid": list(SHRINKAGE_GRID),
            "anchor": "training-label mean",
            "tie_break": "smaller gamma",
        },
        "holdout": "previously inspected historical interval; not a fresh final holdout",
        "backtest_policy": "fixed score>0 long/cash; next-session close execution; SPY adjusted-close returns; costs 0,5,10,20bps one-way",
        "bootstrap": {"block_length": 20, "draws": 2000, "seed": 913},
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("financial-ts-qlib", "torch", "numpy", "pandas", "pyqlib")
        },
    }
    _save_json(output / "protocol.json", protocol)
    try:
        dataset, segments = _build_dataset(configs[0], calendar)
        training_labels, validation_labels = _labels(dataset, "train"), _labels(dataset, "valid")
        training_mean = float(training_labels.mean())
        validation = validation_labels.to_frame()
        validation["zero_return"], validation["training_mean"] = 0.0, training_mean
        fitted, trials = {}, []
        for config in configs:
            name, seed = config["model"]["name"], config["seed"]
            key = f"{name}__seed_{seed}"
            print(f"Research fit: {key}", flush=True)
            model = _fit_trial(config, calendar, dataset)
            if not np.isclose(model.target_mean_, training_mean, rtol=0, atol=1e-15):
                raise ValueError("Model target normalization differs from the training labels.")
            validation[key] = _aligned_predict(model, dataset, "valid", validation.index)
            fitted[key] = model
            folder = Path(config["output_dir"])
            folder.mkdir(parents=True)
            _save_json(folder / "config.json", config)
            _save_json(folder / "learning_curves.json", model.evals_result_)
            torch.save(
                {
                    "state_dict": model.model.state_dict(),
                    "target_mean": model.target_mean_,
                    "target_scale": model.target_scale_,
                },
                folder / "state.pt",
            )
            trials.append(
                {
                    "model": name,
                    "seed": seed,
                    "column": key,
                    "parameter_count": model.parameter_count_,
                    "best_epoch": model.best_epoch_,
                    "epochs_run": len(model.evals_result_["valid"]),
                    "fit_seconds": model.fit_seconds_,
                    "validation_rmse": rmse(validation_labels, validation[key]),
                    "learning_curves": model.evals_result_,
                }
            )
        calibrations = {}
        for name in RESEARCH_MODELS:
            columns = [f"{name}__seed_{seed}" for seed in seeds]
            validation[f"{name}__ensemble"] = validation[columns].mean(axis=1)
        ensemble_columns = [f"{name}__ensemble" for name in RESEARCH_MODELS]
        validation["three_architecture_blend"] = validation[ensemble_columns].mean(axis=1)
        for key in [*ensemble_columns, "three_architecture_blend"]:
            calibrated = select_shrinkage(validation_labels, validation[key], training_mean)
            calibrations[key] = calibrated
            validation[key + "__shrunk"] = training_mean + calibrated["gamma"] * (
                validation[key] - training_mean
            )
        eligible = [
            "zero_return",
            "training_mean",
            *ensemble_columns,
            "three_architecture_blend",
            *[key + "__shrunk" for key in calibrations],
        ]
        validation_scores = {key: rmse(validation_labels, validation[key]) for key in eligible}
        selected = min(eligible, key=lambda key: (validation_scores[key], eligible.index(key)))
        selection = {
            "locked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_predictions_started": False,
            "training_mean": training_mean,
            "calibrations": calibrations,
            "validation_rmse": validation_scores,
            "selected_by_validation": selected,
            "segments": segments,
            "train_count": len(training_labels),
            "valid_count": len(validation_labels),
        }
        _save_json(output / "selection.json", selection)
        validation.to_csv(output / "validation_predictions.csv")
        # No candidate architecture, epoch, mixture weight or calibration is changed below.
        test_labels = _labels(dataset, "test")
        test = test_labels.to_frame()
        test["zero_return"], test["training_mean"] = 0.0, training_mean
        for key, model in fitted.items():
            test[key] = _aligned_predict(model, dataset, "test", test.index)
        for name in RESEARCH_MODELS:
            test[f"{name}__ensemble"] = test[[f"{name}__seed_{seed}" for seed in seeds]].mean(
                axis=1
            )
        test["three_architecture_blend"] = test[ensemble_columns].mean(axis=1)
        for key, calibrated in calibrations.items():
            test[key + "__shrunk"] = training_mean + calibrated["gamma"] * (
                test[key] - training_mean
            )
        test.to_csv(output / "test_predictions.csv")
        for trial in trials:
            trial["test_rmse"] = rmse(test_labels, test[trial["column"]])
        metrics = [
            {
                "model": key,
                "validation_rmse": validation_scores[key],
                "test_rmse": rmse(test_labels, test[key]),
                "selected_by_validation": key == selected,
            }
            for key in eligible
        ]
        reference = "timesfm3_mini__ensemble"
        intervals = {
            key: paired_block_interval(
                test_labels, test[key], test[reference], block_length=min(20, len(test))
            )
            for key in eligible
            if key != reference
        }
        result = {
            "status": "completed",
            "selected_by_validation": selected,
            "metrics": metrics,
            "trials": trials,
            "paired_intervals": {"reference": reference, "comparisons": intervals},
            "test_count": len(test_labels),
            "validation_count": len(validation_labels),
            "segments": segments,
        }
        _save_json(output / "metrics.json", result)
        pd.DataFrame(metrics).to_csv(output / "comparison.csv", index=False)
        protocol.update(
            status="completed",
            segments=segments,
            finished_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(output / "protocol.json", protocol)
        return result
    except Exception as error:
        protocol.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(output / "protocol.json", protocol)
        raise
