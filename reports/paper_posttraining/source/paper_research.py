"""A preregistered historical screen of ES and directional weight editing.

The incumbent is the previously validation-selected three-seed TimesFM mini
ensemble. This is a forecasting adaptation of two papers, not a replication of
LLM reasoning or self-rewriting. Its historical test period is already known.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from financial_ts.backtest import run_backtest
from financial_ts.catalog import build_mini_encoder
from financial_ts.comparison import _file_record, _read_predictions, _require_same_rows
from financial_ts.model import _Windows
from financial_ts.posttraining import (
    ESConfig,
    apply_directional_edit,
    diagnose_error_direction,
    evolution_strategy,
)
from financial_ts.research import _build_dataset, _labels, paired_block_interval, rmse
from financial_ts.workflow import LABEL, _provider_fingerprint, _read_calendar, _save_json

BASE_SEEDS = (7, 42, 2026)
ES_SEEDS = (7, 42, 2026)
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10)
PRIMARY_ES_SEED = 42
EDIT_LAYER = "layers.1.ff.3"


class _MiniEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, features, padding):
        # Preserve the existing convention: average individual float64 scores.
        return torch.stack([m(features, padding).double() for m in self.members]).mean(0)


def _load_base(base_runs, provider):
    selection = json.loads((base_runs / "selection.json").read_text())
    protocol = json.loads((base_runs / "protocol.json").read_text())
    if selection["selected_by_validation"] != "timesfm3_mini__ensemble":
        raise ValueError(
            "This fixed screen requires the validation-selected TimesFM mini ensemble."
        )
    if protocol["seeds"] != list(BASE_SEEDS):
        raise ValueError("The incumbent must retain all three declared base seeds.")
    if _provider_fingerprint(provider) != protocol["provider_fingerprint"]:
        raise ValueError("Provider bytes differ from the incumbent's registered training data.")
    members, records, first_config = [], [], None
    mean = scale = None
    for seed in BASE_SEEDS:
        directory = base_runs / "timesfm3_mini" / f"seed-{seed}"
        config = json.loads((directory / "config.json").read_text())
        checkpoint = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
        current_mean, current_scale = checkpoint["target_mean"], checkpoint["target_scale"]
        if not np.isfinite([current_mean, current_scale]).all() or current_scale <= 0:
            raise ValueError("Invalid saved training target normalization.")
        if mean is not None and (mean != current_mean or scale != current_scale):
            raise ValueError("All ensemble members must use identical target normalization.")
        mean, scale = current_mean, current_scale
        parameters = config["model"]["params"]
        dimensions = {k: parameters[k] for k in ("d_model", "nhead", "num_layers", "dropout")}
        dimensions["d_feat"] = 157
        # Isolate initialization RNG; these temporary random weights are then replaced.
        with torch.random.fork_rng(devices=[]):
            member = build_mini_encoder("timesfm3_mini", **dimensions)
        member.load_state_dict(checkpoint["state_dict"], strict=True)
        members.append(member)
        records.extend(_file_record(directory / name) for name in ("config.json", "state.pt"))
        if first_config is None:
            first_config = config
        elif any(config[k] != first_config[k] for k in ("sequence_length", "model", "segments")):
            raise ValueError("Ensemble input settings do not match.")
    first_config["provider_uri"] = str(provider)
    first_config["model"]["params"]["device"] = "cpu"
    return _MiniEnsemble(members).eval(), mean, scale, first_config, selection, records


def _windows(dataset, segment, indices=None):
    from qlib.data.dataset.handler import DataHandlerLP

    sampler = dataset.prepare(
        segment, col_set="feature", data_key=DataHandlerLP.DK_I, dtype=np.float32
    )
    sampler.config(fillna_type="none")
    windows = _Windows(sampler, 157)
    positions = np.arange(len(windows)) if indices is None else np.asarray(indices)
    tensors = [windows[int(i)] for i in positions]
    if not tensors:
        raise ValueError("A research partition cannot be empty.")
    return (
        torch.stack([v[0] for v in tensors]),
        torch.stack([v[1] for v in tensors]),
        sampler.get_index()[positions],
    )


def _normalized_scores(model, features, padding):
    model.eval()
    with torch.inference_mode():
        values = (
            torch.cat(
                [
                    model(features[i : i + 128], padding[i : i + 128])
                    for i in range(0, len(features), 128)
                ]
            )
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Post-training produced invalid forecast scores.")
    return values


def _loss(model, data, targets):
    predictions = _normalized_scores(model, data[0], data[1])
    value = float(np.mean(np.square(predictions - targets)))
    if not np.isfinite(value):
        raise ValueError("Post-training loss is not finite.")
    return value


def _fit_positions(labels, count):
    if type(count) is not int or count < 32:
        raise ValueError("fit_samples must be an integer of at least 32.")
    # y[t-2] is known at close t. The preceding 20 such realized returns supply
    # a causal volatility denominator for diagnosis, never a future return.
    volatility = labels.shift(2).rolling(20, min_periods=20).std(ddof=0)
    eligible = np.flatnonzero(np.isfinite(volatility.to_numpy()))
    if len(eligible) < 32:
        raise ValueError("At least 32 eligible training observations are required.")
    indices = eligible[np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)]
    return indices, volatility.iloc[indices].to_numpy().clip(min=0.001)


def _diagnose(base, data, normalized_targets, volatility, scale):
    diagnosed, random, details = [], [], []
    for member_index, member in enumerate(base.members):
        hidden = []

        def capture(_module, _inputs, output):
            hidden.append(output[:, :, -member.horizon_patches :, :].mean((1, 2)).detach().cpu())

        handle = member.get_submodule(EDIT_LAYER).register_forward_hook(capture)
        try:
            predicted = _normalized_scores(member, data[0], data[1])
        finally:
            handle.remove()
        # Volatility-normalize only the failure diagnosis; all ES and selection
        # losses retain the same mean-squared return forecasting objective.
        errors = np.square((predicted - normalized_targets) * scale / volatility)
        direction = diagnose_error_direction(torch.cat(hidden), torch.from_numpy(errors))
        generator = torch.Generator().manual_seed(90210 + member_index)
        random_direction = torch.randn(direction.shape, generator=generator, dtype=direction.dtype)
        random_direction /= random_direction.norm()
        diagnosed.append(direction)
        random.append(random_direction)
        details.append(
            {
                "base_seed": BASE_SEEDS[member_index],
                "layer": EDIT_LAYER,
                "direction": direction.tolist(),
                "random_direction": random_direction.tolist(),
                "random_seed": 90210 + member_index,
                "diagnosis": "mean hidden output of highest minus lowest residual-loss quartile",
                "loss_denominator": "20 prior realized returns; volatility floor 0.001",
                "direction_norm": float(direction.norm()),
            }
        )
    return diagnosed, random, details


def _select_edit(base, directions, validation_data, normalized_targets):
    choices, selected = [], None
    for alpha in ALPHAS:
        candidate = copy.deepcopy(base)
        edits = [
            apply_directional_edit(candidate, f"members.{i}.{EDIT_LAYER}", direction, alpha)
            for i, direction in enumerate(directions)
        ]
        loss = _loss(candidate, validation_data, normalized_targets)
        choices.append({"alpha": alpha, "validation_normalized_mse": loss, "edits": edits})
        if selected is None or loss < selected[0]:
            selected = loss, alpha, candidate
    return selected[2], {"selected_alpha": selected[1], "candidates": choices}


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_paper_research(
    provider,
    base_runs,
    spy_csv,
    output,
    *,
    fit_samples=512,
    population=12,
    generations=8,
):
    """Run one fixed two-factor screen; never select a model or seed on test."""
    started = time.perf_counter()
    provider, base_runs, output = (
        Path(p).expanduser().resolve() for p in (provider, base_runs, output)
    )
    spy_csv = Path(spy_csv).expanduser().resolve()
    for source in (provider, base_runs):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(
                "Output must be separate from input provider and checkpoint directories."
            )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Research output must be new or empty.")
    es_config = ESConfig(
        population=population, generations=generations, sigma=0.001, learning_rate=1e-6
    )
    base, mean, scale, config, base_selection, sources = _load_base(base_runs, provider)
    torch.set_num_threads(2)
    roster = ["baseline", "directional_edit", "random_edit"]
    roster += [f"{method}_seed_{seed}" for seed in ES_SEEDS for method in ("es", "combined")]
    sources += [_file_record(base_runs / n) for n in ("selection.json", "protocol.json")]
    sources += [_file_record(spy_csv)]
    output.mkdir(parents=True)
    protocol = {
        "status": "registered",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "papers": ["https://arxiv.org/html/2509.24372v3", "https://arxiv.org/html/2611.21885v2"],
        "method_scope": "forecasting adaptation; externally diagnosed edits, not LLM self-rewriting",
        "roster": roster,
        "base_seeds": list(BASE_SEEDS),
        "es_seeds": list(ES_SEEDS),
        "primary_es_seed": PRIMARY_ES_SEED,
        "primary_objective": "common-date return RMSE; portfolio results are separate outcomes",
        "es": asdict(es_config),
        "es_objective": "negative normalized return MSE on fixed training-only examples",
        "selection": "generation zero and generation means eligible; full validation MSE only",
        "fit_samples_requested": fit_samples,
        "fit_sampling": "evenly spaced training indices with 20 prior realized return observations",
        "edit_layer": EDIT_LAYER,
        "alpha_grid": list(ALPHAS),
        "edit_selection": "one shared alpha across base members, selected on full validation",
        "combined_order": "directional edit then ES; same ES budget and paired noise seeds",
        "random_control": "random direction per base member; identical alpha grid and validation selection",
        "holdout": "2013-2018 interval previously inspected; exploratory historical follow-up, not fresh holdout",
        "development_reuse": "base checkpoints and post-training both selected on the same development validation set",
        "trading_rule": "fixed forecast>0 long/cash, next-session close execution, no trading-rule tuning",
        "costs_bps": [0, 5, 10, 20],
        "bootstrap": {"block_length": 20, "draws": 2000, "seed": 913},
        "target": LABEL,
        "provider_fingerprint": _provider_fingerprint(provider),
        "sources": sources,
        "source_files": [
            _file_record(Path(__file__)),
            _file_record(Path(__file__).with_name("posttraining.py")),
        ],
        "packages": {
            n: importlib.metadata.version(n) for n in ("torch", "numpy", "pandas", "pyqlib")
        },
        "parameter_count": sum(p.numel() for p in base.parameters()),
    }
    _save_json(output / "protocol.json", protocol)
    try:
        dataset, segments = _build_dataset(config, _read_calendar(provider))
        segments = {name: list(bounds) for name, bounds in segments.items()}
        if segments != base_selection["segments"]:
            raise ValueError("Evaluation partitions differ from the saved incumbent selection.")
        train_labels, validation_labels = _labels(dataset, "train"), _labels(dataset, "valid")
        if not np.isclose(train_labels.mean(), mean, rtol=0, atol=1e-15):
            raise ValueError("Training mean differs from saved checkpoints.")
        positions, volatility = _fit_positions(train_labels, fit_samples)
        train_data, validation_data = (
            _windows(dataset, "train", positions),
            _windows(dataset, "valid"),
        )
        if not validation_data[2].equals(validation_labels.index):
            raise ValueError("Validation feature and label rows differ.")
        train_target = (train_labels.iloc[positions].to_numpy() - mean) / scale
        valid_target = (validation_labels.to_numpy() - mean) / scale
        validation = validation_labels.to_frame()
        validation["baseline"] = _normalized_scores(base, *validation_data[:2]) * scale + mean
        archived_validation = _read_predictions(base_runs / "validation_predictions.csv")
        _require_same_rows(archived_validation, validation, "incumbent validation")
        replay_error = float(
            np.max(np.abs(validation["baseline"] - archived_validation["timesfm3_mini__ensemble"]))
        )
        if replay_error > 1e-10:
            raise ValueError(f"Base checkpoint validation replay differs: {replay_error}.")
        _save_json(
            output / "fit_samples.json",
            {
                "indices": positions.tolist(),
                "dates": [str(v[0].date()) for v in train_data[2]],
                "training_rows": len(train_labels),
                "sample_count": len(positions),
                "volatility": volatility.tolist(),
                "validation_replay_max_error": replay_error,
            },
        )
        directions, random_directions, diagnosis = _diagnose(
            base, train_data, train_target, volatility, scale
        )
        _save_json(output / "directions.json", diagnosis)
        models = {"baseline": base}
        models["directional_edit"], directional_selection = _select_edit(
            base, directions, validation_data, valid_target
        )
        models["random_edit"], random_selection = _select_edit(
            base, random_directions, validation_data, valid_target
        )
        print(
            f"Directional alpha={directional_selection['selected_alpha']}; random alpha={random_selection['selected_alpha']}",
            flush=True,
        )
        searches = {}
        for seed in ES_SEEDS:
            for method, initial in (("es", base), ("combined", models["directional_edit"])):
                key = f"{method}_seed_{seed}"
                candidate = copy.deepcopy(initial)
                print(
                    f"Post-training {key}: {population} proposals x {generations} generations",
                    flush=True,
                )
                result = evolution_strategy(
                    candidate,
                    lambda m: -_loss(m, train_data, train_target),
                    lambda m: _loss(m, validation_data, valid_target),
                    config=ESConfig(
                        population=population,
                        generations=generations,
                        sigma=0.001,
                        learning_rate=1e-6,
                        seed=seed,
                    ),
                )
                models[key] = candidate
                searches[key] = {
                    "seed": seed,
                    "selected_generation": result.selected_generation,
                    "validation_normalized_mse": result.selection_loss,
                    "candidate_evaluations": result.candidate_evaluations,
                    "history": result.history,
                }
                _save_json(output / "search_history.json", searches)
                print(
                    f"Selected {key}: generation {result.selected_generation}, validation RMSE {np.sqrt(result.selection_loss) * scale:.10f}",
                    flush=True,
                )
        for key, model in models.items():
            validation[key] = _normalized_scores(model, *validation_data[:2]) * scale + mean
        validation["zero_return"], validation["training_mean"] = 0.0, mean
        validation_rmse = {
            key: rmse(validation_labels, validation[key]) for key in validation if key != "label"
        }
        # The main four-arm comparison always uses predeclared seed 42. Seeds
        # 7 and 2026 diagnose optimizer stability and never replace seed 42.
        primary_roster = ["baseline", "directional_edit", "es_seed_42", "combined_seed_42"]
        chosen = min(primary_roster, key=lambda k: (validation_rmse[k], primary_roster.index(k)))
        selection = {
            "locked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "test_predictions_started": False,
            "primary_es_seed": PRIMARY_ES_SEED,
            "primary_roster": primary_roster,
            "selected_by_validation": chosen,
            "directional_edit": directional_selection,
            "random_edit": random_selection,
            "validation_rmse": validation_rmse,
            "states_sha256": {key: _state_digest(model) for key, model in models.items()},
            "segments": segments,
            "train_count": len(train_labels),
            "valid_count": len(validation_labels),
            "fit_count": len(positions),
            "training_mean": mean,
            "training_scale": scale,
            "optimizer_seeds_are_replicates": True,
        }
        _save_json(output / "selection.json", selection)
        selection_hash = _file_record(output / "selection.json")["sha256"]
        validation.to_csv(output / "validation_predictions.csv")
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir()
        for key, model in models.items():
            torch.save(
                {"state_dict": model.state_dict(), "target_mean": mean, "target_scale": scale},
                checkpoint_dir / f"{key}.pt",
            )
        # Selection is now durable. No updates, candidate choices, or threshold
        # fitting occur below this point.
        test_labels = _labels(dataset, "test")
        test_data = _windows(dataset, "test")
        if not test_data[2].equals(test_labels.index):
            raise ValueError("Test feature and label rows differ.")
        predictions = test_labels.to_frame()
        for key, model in models.items():
            predictions[key] = _normalized_scores(model, *test_data[:2]) * scale + mean
        predictions["zero_return"], predictions["training_mean"] = 0.0, mean
        predictions.to_csv(output / "predictions.csv")
        archived_test = _read_predictions(base_runs / "test_predictions.csv")
        _require_same_rows(archived_test, predictions, "incumbent test")
        test_replay_error = float(
            np.max(np.abs(predictions["baseline"] - archived_test["timesfm3_mini__ensemble"]))
        )
        if test_replay_error > 1e-10:
            raise ValueError(f"Base checkpoint test replay differs: {test_replay_error}.")
        metrics = []
        intervals = {}
        base_rmse = rmse(test_labels, predictions["baseline"])
        for key in validation_rmse:
            test_rmse = rmse(test_labels, predictions[key])
            metrics.append(
                {
                    "model": key,
                    "validation_rmse": validation_rmse[key],
                    "test_rmse": test_rmse,
                    "rmse_delta_vs_base": test_rmse - base_rmse,
                    "test_mae": float(np.mean(np.abs(test_labels - predictions[key]))),
                    "directional_accuracy": float(
                        np.mean((predictions[key] > 0) == (test_labels > 0))
                    ),
                    "selected_by_validation": key == chosen,
                    "test_count": len(test_labels),
                }
            )
            if key != "baseline":
                intervals[key] = paired_block_interval(
                    test_labels, predictions[key], predictions["baseline"]
                )
        pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
        _save_json(
            output / "paired_intervals.json",
            {
                "reference": "baseline",
                "qualification": "conditional historical diagnostics; no correction for prior or current research selection",
                "comparisons": intervals,
            },
        )
        prices = pd.read_csv(spy_csv, parse_dates=["date"], float_precision="round_trip")
        if set(prices["symbol"]) != {"SPY"}:
            raise ValueError("Execution source must explicitly identify SPY.")
        portfolio = run_backtest(
            predictions.drop(columns="label"), prices.set_index("date")["adj_close"]
        )
        portfolio["metrics"].to_csv(output / "backtest_metrics.csv", index=False)
        primary_ledger = portfolio["ledger"].loc[portfolio["ledger"]["cost_bps"].eq(5.0)]
        primary_ledger.to_csv(output / "equity.csv", index=False)
        prices.to_csv(output / "spy_prices.csv", index=False)
        _save_json(output / "backtest_settings.json", portfolio["settings"])
        # Stable calendar-block diagnostics do not refit or reselect a model.
        subperiods = []
        dates = predictions.index.get_level_values("datetime")
        for start, end in (
            ("2013-01-01", "2014-12-31"),
            ("2015-01-01", "2016-12-31"),
            ("2017-01-01", "2018-12-31"),
        ):
            mask = (dates >= start) & (dates <= end)
            for key in validation_rmse:
                subperiods.append(
                    {
                        "start": start,
                        "end": end,
                        "model": key,
                        "observations": int(mask.sum()),
                        "rmse": rmse(test_labels[mask], predictions.loc[mask, key]),
                    }
                )
        pd.DataFrame(subperiods).to_csv(output / "subperiod_metrics.csv", index=False)
        if _file_record(output / "selection.json")["sha256"] != selection_hash:
            raise RuntimeError("Selection changed after test scoring began.")
        summary = {
            "status": "completed",
            "selected_by_validation": chosen,
            "primary_es_seed": PRIMARY_ES_SEED,
            "test_count": len(test_labels),
            "validation_replay_max_error": replay_error,
            "test_replay_max_error": test_replay_error,
            "selection_sha256_before_test": selection_hash,
            "selection_sha256_after_test": selection_hash,
            "candidate_evaluations": sum(s["candidate_evaluations"] for s in searches.values()),
            "metrics": metrics,
            "backtest_5bps": portfolio["metrics"]
            .loc[portfolio["metrics"]["cost_bps"].eq(5)]
            .to_dict(orient="records"),
            "elapsed_seconds": time.perf_counter() - started,
            "interpretation": "exploratory historical screen; prospective superiority requires a fresh interval and nested walk-forward evaluation",
        }
        _save_json(output / "summary.json", summary)
        protocol.update(
            status="completed",
            finished_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(output / "protocol.json", protocol)
        return summary
    except Exception as error:
        protocol.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            elapsed_seconds=time.perf_counter() - started,
        )
        _save_json(output / "protocol.json", protocol)
        raise
