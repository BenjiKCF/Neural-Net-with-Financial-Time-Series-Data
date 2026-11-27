"""Pinned, resumable CSI300 comparison on the chenditc daily data release.

The protocol is saved before fitting. Every selected model uses the same
historical membership, feature/label dates and China-market execution rules.
The seed and training budget are deliberately explicit, rather than inheriting
the old SPX experiment's unrelated splits and performance claims.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from financial_ts.benchmark import benchmark_config
from financial_ts.catalog import TRAINABLE_MODELS
from financial_ts.workflow import (
    LABEL,
    _provider_fingerprint,
    _read_calendar,
    _save_json,
    resolve_segments,
    run_workflow,
)

OFFICIAL_VARIANTS = (
    "timesfm3_frozen_128",
    "timesfm3_frozen_256",
    "timesfm3_loop_128",
    "timesfm3_adapter_128",
)
CN_MODELS = (*TRAINABLE_MODELS, *OFFICIAL_VARIANTS)


def cn_segments(calendar: pd.DatetimeIndex) -> dict[str, tuple[str, str]]:
    """Fixed recent periods, with two label sessions purged at each boundary."""
    cal = pd.DatetimeIndex(calendar)
    valid_first = int(cal.searchsorted(pd.Timestamp("2024-01-01")))
    test_first = int(cal.searchsorted(pd.Timestamp("2025-01-01")))
    train_first = int(cal.searchsorted(pd.Timestamp("2020-01-01")))
    if train_first >= valid_first - 3 or test_first >= len(cal) - 3:
        raise ValueError("The CSI300 protocol needs 2020–2025 and later observed sessions.")
    segments = {
        "train": (str(cal[train_first].date()), str(cal[valid_first - 3].date())),
        "valid": (str(cal[valid_first].date()), str(cal[test_first - 3].date())),
        "test": (str(cal[test_first].date()), str(cal[-4].date())),
    }
    # Qlib backtesting reserves one real session after the final label's close.
    return resolve_segments(cal[:-1], segments)


def prepare_cn_benchmark(
    provider,
    output,
    *,
    epochs=6,
    seed=42,
    device="cpu",
    num_threads=2,
    checkpoint=None,
) -> dict:
    """Create or exactly recover a predeclared experiment, without running it."""
    for name, value in (("epochs", epochs), ("num_threads", num_threads)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    provider, output = Path(provider).expanduser().resolve(), Path(output).expanduser().resolve()
    if provider == output or provider in output.parents or output in provider.parents:
        raise ValueError("Provider and benchmark output must be separate directories.")
    metadata_path = provider / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("source") != "chenditc/investment_data" or metadata.get("synthetic"):
        raise ValueError("This protocol requires a validated real chenditc provider.")
    observed = _read_calendar(provider)
    cutoff = pd.Timestamp(metadata["target_trade_date"])
    if observed[-1] != cutoff:
        raise ValueError("Provider observed calendar and source manifest cutoff disagree.")
    segments = cn_segments(observed)
    models = list(TRAINABLE_MODELS) + (list(OFFICIAL_VARIANTS) if checkpoint else [])
    configs = {}
    for name in models:
        scratch_name = name if name in TRAINABLE_MODELS else "lightgbm"
        config = benchmark_config(scratch_name, provider, output / name, seed, epochs, device)
        config.update(
            region="cn",
            instruments="csi300",
            data_start_time="2019-01-01",
            data_end_time=str(cutoff.date()),
            segments=segments,
            experiment_name="chenditc_csi300",
            handler_cache_path=str(output / "cache" / "csi300_handler.pkl"),
        )
        config["backtest"] = {
            "enabled": True,
            "topk": 50,
            "n_drop": 5,
            "account": 100_000_000,
            "risk_degree": 0.95,
            "benchmark": "SH000300",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
        if name in TRAINABLE_MODELS:
            config["model"]["params"]["num_threads"] = num_threads
            if name != "lightgbm":
                config["model"]["params"]["batch_size"] = 512
                config["shared_windows"] = True
        else:
            mode = "frozen" if "frozen" in name else "loop" if "loop" in name else "adapter"
            config["handler_cache_path"] = str(output / "cache" / "csi300_prices.pkl")
            config["model"] = {
                "name": "timesfm3_panel",
                "params": {
                    "mode": mode,
                    "checkpoint_path": str(Path(checkpoint).expanduser().resolve()),
                    "context_length": 256 if name.endswith("256") else 128,
                    "batch_size": 64,
                    "num_threads": num_threads,
                    "device": device,
                    "train_samples": 1024,
                    "valid_samples": 512,
                    "epochs": 3,
                    "seed": seed,
                    "lr": 0.001,
                    "prediction_cache_dir": str(output / name / "inference_cache"),
                },
            }
        configs[name] = config
    identity = {
        "schema_version": 1,
        "source": metadata,
        "data_manifest_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "provider_fingerprint": _provider_fingerprint(provider),
        "models": models,
        "seed": seed,
        "epochs": epochs,
        "segments": segments,
        "label": LABEL,
        "configs": configs,
        "selection": "Each fitted model selects checkpoints on validation only. The report's primary model is selected by highest full-validation daily Rank IC among the eight scratch models; ties use model name.",
        "comparability": f"Eight scratch models fit all available training rows. Seven neural models share 32-session Alpha157 inputs, width 32, {epochs}-epoch maximum and AdamW settings; boosting has its own fixed budget. Official TimesFM uses raw adjusted-close contexts; loop/control fit 1,024 train and 512 validation samples and then score the complete panel.",
        "execution": "Qlib TopkDropoutStrategy(50,5), delayed close, SH000300, 100-share lots, 9.5% limit threshold, 5/15 bps entry/exit costs, CNY5 minimum, 95% invested target; final holdings marked to market.",
        "limitations": "Single declared seed; exploratory historical replay. Upstream revised data and historical membership are not certified publication-time snapshots. Official pretraining overlap is unknown. No market impact, borrow, or deployment claim.",
    }
    identity = json.loads(json.dumps(identity))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    with FileLock(str(output / ".protocol.lock")):
        if path.exists():
            existing = json.loads(path.read_text())
            if {key: existing.get(key) for key in identity} != identity:
                raise ValueError(
                    "Existing experiment protocol differs; choose a new output directory."
                )
            return existing
        _save_json(path, {**identity, "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    return json.loads(path.read_text())


def run_cn_models(protocol: dict, *, models=None, retry_failed=False) -> dict:
    """Run all requested declared models, reusing only completed matching runs."""
    names = protocol["models"] if models is None else list(models)
    if not names or len(set(names)) != len(names) or set(names) - set(protocol["models"]):
        raise ValueError("Requested models must be distinct members of the registered protocol.")
    provider = Path(protocol["configs"][names[0]]["provider_uri"])
    if _provider_fingerprint(provider) != protocol.get("provider_fingerprint"):
        raise ValueError(
            "Provider bytes changed since protocol registration; refuse mixed-data results."
        )
    results = {}
    for name in names:
        config = copy.deepcopy(protocol["configs"][name])
        output = Path(config["output_dir"])
        with FileLock(str(output.parent / f".{name}.lock")):
            if (output / "run.json").exists():
                run = json.loads((output / "run.json").read_text())
                resolved = json.loads((output / "resolved_config.json").read_text())
                expected = json.loads(json.dumps(config))
                if resolved != expected:
                    raise ValueError(f"{name} has a different resolved configuration.")
                if run.get("provider") != protocol["provider_fingerprint"]:
                    raise ValueError(f"{name} was run on different provider bytes.")
                if run.get("status") != "completed":
                    if not retry_failed:
                        raise ValueError(
                            f"{name} is incomplete; use --retry-failed to preserve and retry it."
                        )
                    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
                    archive = output.parent / "failed_attempts" / f"{name}-{stamp}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    output.rename(archive)
                    if (archive / "inference_cache").is_dir():
                        output.mkdir()
                        (archive / "inference_cache").rename(output / "inference_cache")
                    result = run_workflow(config)
                    _save_json(
                        output / "retry.json",
                        {"previous_attempt": str(archive), "configuration_unchanged": True},
                    )
                else:
                    result = {
                        "metrics": json.loads((output / "metrics.json").read_text()),
                        "run": run,
                    }
            else:
                print(f"CSI300 benchmark: {name}", flush=True)
                result = run_workflow(config)
            results[name] = result["metrics"]
    return results


def summarize_cn_benchmark(output) -> dict:
    """Export all completed metrics and pick the figure model from validation."""
    root = Path(output).expanduser().resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    rows, pending = [], []
    for name in protocol["models"]:
        path = root / name
        if (
            not (path / "metrics.json").is_file()
            or json.loads((path / "run.json").read_text()).get("status") != "completed"
        ):
            pending.append(name)
            continue
        metrics = json.loads((path / "metrics.json").read_text())
        valid = metrics["validation"]
        row = {
            "model": name,
            "validation_rmse": valid["rmse"],
            "validation_ic": valid.get("ic"),
            "validation_rank_ic": valid.get("rank_ic"),
            "test_rmse": metrics["rmse"],
            "zero_return_rmse": metrics["zero_return_rmse"],
            "test_ic": metrics.get("ic"),
            "test_rank_ic": metrics.get("rank_ic"),
            "test_rows": metrics["evaluated_count"],
            **metrics["model_details"],
        }
        report = pd.read_csv(path / "portfolio.csv", index_col=0, parse_dates=True)
        net = report["return"] - report["cost"]
        equity = (1 + net).cumprod()
        peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
        row.update(
            net_return=float(equity.iloc[-1] - 1),
            benchmark_return=float((1 + report["bench"]).prod() - 1),
            annualized_return_252=float(equity.iloc[-1] ** (252 / len(equity)) - 1),
            max_drawdown=float(np.min(equity.to_numpy() / peak - 1)),
            mean_daily_turnover=float(report["turnover"].mean()),
            trading_days=len(report),
        )
        rows.append(row)
    candidates = [
        r for r in rows if r["model"] in TRAINABLE_MODELS and r["validation_rank_ic"] is not None
    ]
    selected = (
        sorted(candidates, key=lambda r: (-r["validation_rank_ic"], r["model"]))[0]["model"]
        if candidates
        else None
    )
    result = {
        "status": "completed" if not pending else "running",
        "pending": pending,
        "primary_model": selected,
        "models": rows,
    }
    _save_json(root / "comparison.json", result)
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    return result
