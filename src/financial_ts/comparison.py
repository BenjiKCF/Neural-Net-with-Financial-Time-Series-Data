"""Join recorded forecasts on exact dates and simulate a fixed SPY policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from financial_ts.backtest import run_backtest
from financial_ts.research import RESEARCH_MODELS, paired_block_interval, rmse
from financial_ts.workflow import LABEL, _save_json

ARCHIVED_MODELS = (
    "lightgbm",
    "transformer",
    "deepseek_v41_mini",
    "glm54_mini",
    "qwen4_mini",
    "timesfm3_mini",
)
OFFICIAL_MODELS = ("timesfm3_frozen_128", "timesfm3_loop_128", "timesfm3_adapter_only_128")
DISPLAY_NAMES = {
    "zero_return": "Zero return",
    "training_mean": "Training mean",
    "lightgbm": "LightGBM",
    "transformer": "Transformer",
    "deepseek_v41_mini": "DeepSeek mini",
    "glm54_mini": "GLM loop mini",
    "qwen4_mini": "Qwen mini",
    "timesfm3_mini": "TimesFM mini (archived)",
    "timesfm3_mini_retrained": "TimesFM mini (retrained)",
    "timesfm3_pretrained_256": "Official TimesFM (256 closes)",
    "timesfm3_frozen_128": "Official TimesFM (128 closes)",
    "timesfm3_loop_128": "TimesFM loop + adapter (retrained)",
    "timesfm3_adapter_only_128": "TimesFM gated adapter only",
    "multiscale_residual_mixer": "Multiscale residual mixer",
    "regime_linear_moe": "Regime linear experts",
    "timesfm3_mini_shrunk": "TimesFM mini + shrinkage",
    "multiscale_residual_mixer_shrunk": "Multiscale mixer + shrinkage",
    "regime_linear_moe_shrunk": "Regime experts + shrinkage",
    "three_architecture_blend": "Three-architecture blend",
    "three_architecture_blend_shrunk": "Architecture blend + shrinkage",
    "cash": "Cash",
    "buy_and_hold": "SPY buy and hold",
}


def run_saved_backtest(predictions, spy_csv, output, *, costs_bps=(0, 5, 10, 20)):
    """Replay a saved one-instrument forecast CSV without any model fitting."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Backtest output must be new or empty.")
    frame = _read_predictions(predictions)
    prices = pd.read_csv(spy_csv, parse_dates=["date"], float_precision="round_trip")
    if set(prices["symbol"]) != {"SPY"}:
        raise ValueError("Execution data must explicitly identify SPY.")
    result = run_backtest(
        frame.drop(columns=["label"]), prices.set_index("date")["adj_close"], costs_bps=costs_bps
    )
    output.mkdir(parents=True, exist_ok=True)
    result["ledger"].to_csv(output / "ledger.csv", index=False)
    result["metrics"].to_csv(output / "metrics.csv", index=False)
    _save_json(
        output / "settings.json",
        {**result["settings"], "sources": [_file_record(predictions), _file_record(spy_csv)]},
    )
    return {
        "output_dir": str(output.resolve()),
        "metrics": result["metrics"].to_dict(orient="records"),
    }


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_predictions(path):
    frame = pd.read_csv(path, parse_dates=["datetime"], float_precision="round_trip")
    if not {"datetime", "instrument", "label"}.issubset(frame.columns):
        raise ValueError(f"Missing prediction identity/label columns: {path}")
    frame = frame.set_index(["datetime", "instrument"]).sort_index()
    if (
        frame.empty
        or not frame.index.is_unique
        or frame.index.get_level_values("instrument").nunique() != 1
    ):
        raise ValueError("Common evaluation requires unique rows for one signal instrument.")
    if not np.isfinite(frame["label"].to_numpy()).all():
        raise ValueError("Every evaluation row needs a finite target.")
    return frame


def _require_same_rows(reference, candidate, name):
    if not reference.index.equals(candidate.index):
        raise ValueError(f"{name} does not cover the exact common prediction rows.")
    delta = np.abs(reference["label"].to_numpy() - candidate["label"].to_numpy())
    if not np.isfinite(delta).all() or np.max(delta) > 1e-12:
        raise ValueError(f"{name} uses different target values or units.")


def _file_record(path):
    path = Path(path)
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def build_financial_comparison(
    historical_predictions, historical_report, scratch_dir, timesfm_dir, spy_csv, output
):
    """Publish exact aligned inputs, metrics and primary-cost equity curves.

    All model/epoch/shrinkage choices are already saved by the training runners.
    This function performs no fitting, calibration, or test-based policy search.
    """
    scratch_dir, timesfm_dir, output = Path(scratch_dir), Path(timesfm_dir), Path(output)
    inputs = [
        Path(historical_predictions),
        Path(historical_report),
        Path(spy_csv),
        scratch_dir / "test_predictions.csv",
        scratch_dir / "validation_predictions.csv",
        scratch_dir / "selection.json",
        scratch_dir / "metrics.json",
        scratch_dir / "protocol.json",
        timesfm_dir / "all_period_predictions.csv",
        timesfm_dir / "comparison.json",
    ]
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Comparison output must be new or empty.")
    historical = _read_predictions(historical_predictions)
    scratch = _read_predictions(scratch_dir / "test_predictions.csv")
    valid = _read_predictions(scratch_dir / "validation_predictions.csv")
    official = _read_predictions(timesfm_dir / "all_period_predictions.csv")
    if "segment" not in official or set(official["segment"]) != {"valid", "test"}:
        raise ValueError("Official comparison must identify full valid and test segments.")
    official_test = official[official["segment"] == "test"]
    official_valid = official[official["segment"] == "valid"]
    _require_same_rows(historical, scratch, "Scratch research")
    _require_same_rows(historical, official_test, "Official TimesFM test comparison")
    _require_same_rows(valid, official_valid, "Official TimesFM validation comparison")
    selection = _read_json(scratch_dir / "selection.json")
    scratch_metrics = _read_json(scratch_dir / "metrics.json")
    official_report = _read_json(timesfm_dir / "comparison.json")
    if scratch_metrics["status"] != "completed" or official_report["status"] != "completed":
        raise ValueError("Only completed training/evaluation runs can be compared.")
    archived = _read_json(historical_report)
    scores = historical[["label"]].copy()
    descriptors = {}
    mapping = {"zero_return": "zero_return", "training_mean": "training_mean"}
    for key in mapping:
        scores[key] = scratch[key]
        descriptors[key] = {
            "category": "reference",
            "parameter_count": int(key == "training_mean"),
            "ensemble_members": 1,
        }
    for name in ARCHIVED_MODELS:
        columns = [f"{name}__seed_{seed}" for seed in (7, 42, 2026)]
        if not np.isfinite(historical[columns].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Every archived seed score must be finite for {name}.")
        scores[name] = historical[columns].mean(axis=1)
        matched = [r for r in archived["scratch"]["runs"] if r["config"]["model"]["name"] == name]
        if len(matched) != 3:
            raise ValueError(f"Archived evidence must preserve three trials for {name}.")
        count = matched[0]["metrics"]["model_details"]["parameter_count"]
        descriptors[name] = {
            "category": "archived_seed_ensemble",
            "parameter_count": count * 3 if count else None,
            "parameter_count_per_member": count,
            "ensemble_members": 3,
        }
    scores["timesfm3_pretrained_256"] = historical["timesfm3_pretrained"]
    descriptors["timesfm3_pretrained_256"] = {
        "category": "archived_pretrained",
        "parameter_count": 330710976,
        "ensemble_members": 1,
    }
    for name in RESEARCH_MODELS:
        source = f"{name}__ensemble"
        destination = "timesfm3_mini_retrained" if name == "timesfm3_mini" else name
        mapping[source], mapping[source + "__shrunk"] = destination, name + "_shrunk"
    mapping.update(
        three_architecture_blend="three_architecture_blend",
        three_architecture_blend__shrunk="three_architecture_blend_shrunk",
    )
    counts = {
        name: [r["parameter_count"] for r in scratch_metrics["trials"] if r["model"] == name]
        for name in RESEARCH_MODELS
    }
    for source, destination in mapping.items():
        scores[destination] = scratch[source]
        if destination in descriptors:
            continue
        members = (
            sum(len(v) for v in counts.values())
            if source.startswith("three_architecture")
            else len(counts[source.split("__")[0]])
        )
        count = (
            sum(sum(v) for v in counts.values())
            if source.startswith("three_architecture")
            else sum(counts[source.split("__")[0]])
        )
        descriptors[destination] = {
            "category": "validation_shrinkage"
            if source.endswith("__shrunk")
            else "new_seed_ensemble",
            "parameter_count": count,
            "ensemble_members": members,
        }
    for name in OFFICIAL_MODELS:
        scores[name] = official_test[name]
        descriptors[name] = {
            "category": "pretrained_replay"
            if name == "timesfm3_frozen_128"
            else "pretrained_adaptation",
            "parameter_count": 330710976 if name == "timesfm3_frozen_128" else 330732737,
            "ensemble_members": 1,
            "trainable_parameter_count": 0 if name == "timesfm3_frozen_128" else 21761,
        }
    forecast_columns = list(descriptors)
    if not np.isfinite(scores[["label", *forecast_columns]].to_numpy()).all():
        raise ValueError("All common forecast scores must be finite.")
    validation_scores = {mapping[key]: value for key, value in selection["validation_rmse"].items()}
    validation_scores.update(
        {name: rmse(official_valid["label"], official_valid[name]) for name in OFFICIAL_MODELS}
    )
    forecasts = []
    reference = "timesfm3_mini"
    paired = {}
    for name in forecast_columns:
        forecasts.append(
            {
                "model": name,
                "display_name": DISPLAY_NAMES[name],
                **descriptors[name],
                "validation_rmse": validation_scores.get(name),
                "test_rmse": rmse(scores["label"], scores[name]),
            }
        )
        if name != reference:
            paired[name] = paired_block_interval(
                scores["label"], scores[name], scores[reference], block_length=min(20, len(scores))
            )
    price_frame = pd.read_csv(spy_csv, parse_dates=["date"], float_precision="round_trip")
    if set(price_frame["symbol"]) != {"SPY"}:
        raise ValueError("Execution data must explicitly identify SPY.")
    prices = price_frame.set_index("date")["adj_close"]
    backtest = run_backtest(scores[forecast_columns], prices, costs_bps=(0, 5, 10, 20))
    forecast_metrics = pd.DataFrame(forecasts)
    comparison = (
        backtest["metrics"]
        .rename(columns={"cost_bps": "one_way_cost_bps"})
        .merge(forecast_metrics, how="left", on="model", validate="many_to_one")
    )
    for control in ("cash", "buy_and_hold"):
        comparison.loc[comparison["model"] == control, "category"] = "portfolio_control"
        comparison.loc[comparison["model"] == control, "display_name"] = DISPLAY_NAMES[control]
    primary = backtest["ledger"].loc[backtest["ledger"]["cost_bps"] == 5].copy()
    equity = primary.pivot(index="exit_date", columns="model", values="net_equity")
    positions = primary.pivot(index="signal_date", columns="model", values="position")
    initial_date = primary["entry_date"].min()
    equity.loc[initial_date] = 1.0
    equity = equity.sort_index()
    # Preserve sufficient raw inputs to recompute every transaction-cost scenario.
    unique_periods = primary[primary["model"] == "buy_and_hold"]
    execution = unique_periods[["signal_date", "entry_date", "exit_date", "asset_return"]].copy()
    execution["signal_adjusted_close"] = prices.reindex(
        pd.DatetimeIndex(execution["signal_date"])
    ).to_numpy()
    execution["entry_adjusted_close"] = prices.reindex(
        pd.DatetimeIndex(execution["entry_date"])
    ).to_numpy()
    execution["exit_adjusted_close"] = prices.reindex(
        pd.DatetimeIndex(execution["exit_date"])
    ).to_numpy()
    settings = {
        **backtest["settings"],
        "test_count": len(scores),
        "validation_count": len(valid),
        "selected_by_validation": mapping[selection["selected_by_validation"]],
        "strategy_model_mapping": mapping,
        "display_names": DISPLAY_NAMES,
        "descriptors": descriptors,
        "forecast_target": LABEL,
        "selection": "scratch winner/calibration chosen on full validation; official variants retain pilot-sample selection; all trading rules fixed",
        "comparability": "all test RMSE uses identical SPX rows; portfolio uses next-close SPY total-return proxy; seed ensembles are averaged forecasts, not mean seed RMSE",
        "holdout": "previously inspected historical interval; exploratory research",
        "pretraining_overlap": "unknown for all official TimesFM variants",
        "timesfm_replication": official_report.get(
            "recovery", official_report.get("recovery_check")
        ),
        "period_scale": {
            "full_validation_zero_rmse": rmse(valid["label"], np.zeros(len(valid))),
            "test_zero_rmse": rmse(scores["label"], np.zeros(len(scores))),
        },
        "sources": [_file_record(p) for p in inputs],
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    spy_manifest = Path(spy_csv).with_name("manifest.json")
    if spy_manifest.is_file():
        settings["spy_provenance"] = _read_json(spy_manifest)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "predictions.csv")
    comparison.to_csv(output / "comparison.csv", index=False)
    forecast_metrics.to_csv(output / "forecast_metrics.csv", index=False)
    equity.to_csv(output / "equity.csv")
    positions.to_csv(output / "positions.csv")
    execution.to_csv(output / "execution.csv", index=False)
    quote_window = price_frame[
        (price_frame["date"] >= execution["signal_date"].min())
        & (price_frame["date"] <= execution["exit_date"].max())
    ]
    quote_window[["date", "symbol", "adj_close"]].to_csv(output / "spy_prices.csv", index=False)
    _save_json(output / "settings.json", settings)
    _save_json(output / "paired_intervals.json", {"reference": reference, "comparisons": paired})
    _save_json(
        output / "summary.json",
        {
            "selected_by_validation": settings["selected_by_validation"],
            "test_count": len(scores),
            "primary_metrics": comparison[comparison["one_way_cost_bps"] == 5].to_dict(
                orient="records"
            ),
        },
    )
    return {"output_dir": str(output.resolve()), "settings": settings, "metrics": comparison}
