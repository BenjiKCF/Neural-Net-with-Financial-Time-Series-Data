"""Validation-only tree/neural blends with exact panel and prediction lineage.

Weights are picked on each fold's validation predictions. Evaluation scores
never affect weights. Report both constituent controls and every attempted
weight, including unfavorable choices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from financial_ts.bundles import file_hash, write_json
from financial_ts.workflow import _json_ready, _prediction_statistics


def _aligned(left: pd.DataFrame, right: pd.DataFrame):
    frames = []
    for frame in (left, right):
        if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {
            "datetime",
            "instrument",
        }:
            raise ValueError("Blends require datetime/instrument indexed predictions.")
        frame = frame.reorder_levels(["datetime", "instrument"]).sort_index()
        if frame.empty or not frame.index.is_unique or not {"score", "label"}.issubset(frame):
            raise ValueError("Predictions require nonempty unique keys, score and raw label.")
        if not np.isfinite(frame.score).all() or np.isinf(frame.label).any():
            raise ValueError("Scores must be finite and labels cannot be infinite.")
        frames.append(frame)
    left, right = frames
    if not left.index.equals(right.index):
        raise ValueError(
            "Constituent panels differ; do not silently intersect or drop observations."
        )
    if not np.array_equal(left.label, right.label, equal_nan=True):
        raise ValueError("Constituent labels differ.")
    return left, right


def blend_predictions(left, right, weight):
    left, right = _aligned(left, right)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not np.isfinite(weight)
        or not 0 <= weight <= 1
    ):
        raise ValueError("Blend weight must be finite in [0, 1].")
    return pd.DataFrame(
        {
            "score": float(weight) * left.score + (1 - float(weight)) * right.score,
            "label": left.label,
        },
        index=left.index,
    )


def select_blend(
    left_validation, right_validation, *, weights=(0.0, 0.25, 0.5, 0.75, 1.0), metric="rank_ic"
):
    """Select a declared grid by validation IC or RMSE; all trials retained."""
    if metric not in {"rank_ic", "ic", "rmse"}:
        raise ValueError("Unknown blend selection metric.")
    weights = list(weights)
    if not weights or len(set(weights)) != len(weights):
        raise ValueError("Declare a nonempty distinct blend grid.")
    history = []
    for weight in weights:
        prediction = blend_predictions(left_validation, right_validation, weight)
        stats, _ = _prediction_statistics(prediction)
        history.append({"left_weight": float(weight), "metrics": _json_ready(stats)})

    def key(row):
        value = row["metrics"][metric]
        primary = float("inf") if value is None else value if metric == "rmse" else -value
        return primary, row["metrics"]["rmse"], row["left_weight"]

    best = min(history, key=key)
    return {
        "left_weight": best["left_weight"],
        "selection_metric": metric,
        "selection_split": "validation",
        "history": history,
        "tie_break": "validation RMSE then smaller left weight",
        "evaluation_used": False,
    }


def _read_prediction(path):
    return (
        pd.read_csv(
            path,
            parse_dates=["datetime"],
            dtype={"instrument": str},
            float_precision="round_trip",
            keep_default_na=False,
            na_values={"score": ["", "NaN", "nan"], "label": ["", "NaN", "nan"]},
        )
        .set_index(["datetime", "instrument"])
        .sort_index()
    )


def _seed_mean(paths):
    frames = [_read_prediction(path) for path in paths]
    for frame in frames[1:]:
        _aligned(frames[0], frame)
    result = frames[0].copy()
    result["score"] = np.mean([frame.score.to_numpy() for frame in frames], axis=0)
    return result


def run_rolling_blend(
    campaign,
    output,
    *,
    left="lightgbm",
    right="master_style",
    weights=(0.0, 0.25, 0.5, 0.75, 1.0),
    metric="rank_ic",
    replay=True,
):
    """Register a grid, choose on validation and replay matching ensemble controls.

    All declared seeds of both constituents must exist before registration. The
    fixed 50/50 blend and both seed-mean constituents use the same native cost
    scenarios as the selected blend. Each fold retains a receipt, including a
    failed result, without erasing completed folds or touching the source run.
    """
    from financial_ts.rolling import (
        _code_identity,
        _load_protocol,
        _package_versions,
        _replay_scenarios,
        summarize_rolling_campaign,
    )
    from financial_ts.workflow import _provider_fingerprint

    protocol = _load_protocol(campaign)
    source = Path(protocol["output_dir"]).resolve()
    provider = Path(protocol["provider_uri"]).resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Blend output must be a new immutable research campaign.")
    for protected in (source, provider):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Blend output must be separate from source campaign and provider.")
    if left == right:
        raise ValueError("Choose two distinct registered constituent candidates.")
    for name in (left, right):
        if name not in {item["id"] for item in protocol["candidates"]}:
            raise ValueError(f"Unknown constituent {name}.")
    weights = list(weights)
    if (
        not weights
        or len(set(weights)) != len(weights)
        or any(
            isinstance(w, bool)
            or not isinstance(w, (int, float))
            or not np.isfinite(w)
            or not 0 <= w <= 1
            for w in weights
        )
    ):
        raise ValueError("Blend grid must contain distinct weights in [0,1].")
    if metric not in {"rmse", "ic", "rank_ic"}:
        raise ValueError("Unknown selection metric.")
    if type(replay) is not bool:
        raise ValueError("replay must be boolean.")
    if replay:
        if _provider_fingerprint(provider) != protocol["provider_fingerprint"]:
            raise ValueError("Replay provider bytes changed since the parent campaign.")
        metadata = provider / "metadata.json"
        if (file_hash(metadata) if metadata.exists() else None) != protocol["data_manifest_sha256"]:
            raise ValueError("Replay data manifest changed since the parent campaign.")
    # Reanalysis can use corrected reporting code while the parent retains its
    # frozen runtime's original summaries. Validate the destination first, and
    # never let the summary helper write into the source campaign by default.
    output.mkdir(parents=True)
    input_audit = output / "input_audit"
    summary = summarize_rolling_campaign(campaign, summary_output_dir=input_audit)
    # Validate completeness and freeze exact input hashes before reading scores
    # for selection. No partial seed subset may enter the blend search.
    sources, inputs = {}, {}
    for fold in protocol["folds"]:
        name = fold["id"]
        sources[name] = {}
        for candidate in (left, right):
            rows = [
                row
                for row in summary["runs"]
                if row["fold"] == name and row["candidate"] == candidate
            ]
            if sorted(row["seed"] for row in rows) != sorted(protocol["seeds"]):
                raise ValueError("All registered constituent seeds must complete before blending.")
            paths = [
                source / row["artifact_path"] for row in sorted(rows, key=lambda row: row["seed"])
            ]
            # rolling summary artifact_path already names the workflow directory.
            sources[name][candidate] = paths
            for path in paths:
                for filename in (
                    "predictions.csv",
                    "validation_predictions.csv",
                    "resolved_config.json",
                ):
                    inputs[str((path / filename).relative_to(source))] = file_hash(path / filename)
    registration = {
        "schema_version": 1,
        "parent_protocol_sha256": protocol["protocol_sha256"],
        "parent_status": summary["status"],
        "source_campaign": str(source),
        "input_audit": {
            "directory": "input_audit",
            "analysis_receipt_sha256": file_hash(input_audit / "analysis_receipt.json"),
            "comparison_sha256": file_hash(input_audit / "comparison.json"),
        },
        "source_code_sha256": _code_identity(),
        "packages": _package_versions(),
        "provider_fingerprint": protocol["provider_fingerprint"],
        "input_files": inputs,
        "left": left,
        "right": right,
        "weights": weights,
        "metric": metric,
        "seeds": protocol["seeds"],
        "folds": [fold["id"] for fold in protocol["folds"]],
        "replay": replay,
        "cost_scenarios": protocol["cost_scenarios"],
        "selection": "per-fold validation only; all registered constituent seeds averaged",
        "controls": [left, right, "fixed_50_50"],
        "registered_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "interpretation": "Exploratory follow-up on historical folds; no new holdout or production promotion.",
    }
    write_json(output / "protocol.json", registration)
    receipts = []
    for fold in protocol["folds"]:
        name, paths = fold["id"], sources[fold["id"]]
        destination = output / name
        destination.mkdir()
        try:
            for values in paths.values():
                for path in values:
                    for filename in (
                        "predictions.csv",
                        "validation_predictions.csv",
                        "resolved_config.json",
                    ):
                        relative = str((path / filename).relative_to(source))
                        if file_hash(path / filename) != inputs[relative]:
                            raise ValueError("Constituent input changed after blend registration.")
            validation = {
                candidate: _seed_mean([path / "validation_predictions.csv" for path in values])
                for candidate, values in paths.items()
            }
            selected = select_blend(
                validation[left], validation[right], weights=weights, metric=metric
            )
            # Evaluation files are read only after the validation weight is fixed.
            write_json(destination / "selection.json", selected)
            evaluation = {
                candidate: _seed_mean([path / "predictions.csv" for path in values])
                for candidate, values in paths.items()
            }
            predictions = blend_predictions(
                evaluation[left], evaluation[right], selected["left_weight"]
            )
            valid = blend_predictions(validation[left], validation[right], selected["left_weight"])
            metrics, daily = _prediction_statistics(predictions)
            metrics["validation"], valid_daily = _prediction_statistics(valid)
            predictions.to_csv(destination / "predictions.csv")
            valid.to_csv(destination / "validation_predictions.csv")
            daily.to_csv(destination / "daily_ic.csv")
            valid_daily.to_csv(destination / "validation_daily_ic.csv")
            write_json(destination / "metrics.json", _json_ready(metrics))
            lineage = {
                str(path.relative_to(source) / filename): file_hash(path / filename)
                for values in paths.values()
                for path in values
                for filename in ("predictions.csv", "validation_predictions.csv")
            }
            config = json.loads((paths[left][0] / "resolved_config.json").read_text())
            config["output_dir"] = str(destination)
            config["model"] = {
                "name": "registered_prediction_blend",
                "params": {"left": left, "right": right, "left_weight": selected["left_weight"]},
            }
            config["export_inference_bundle"] = False
            write_json(destination / "resolved_config.json", config)
            run = {
                "status": "forecasts_completed",
                "provider": protocol["provider_fingerprint"],
                "origin": "registered validation blend of all-seed constituent means",
                "lineage": lineage,
            }
            write_json(destination / "run.json", run)
            portfolio = (
                _replay_scenarios(config, protocol, destination)
                if replay and config.get("backtest", {}).get("enabled")
                else None
            )
            controls = {}
            control_predictions = {
                left: evaluation[left],
                right: evaluation[right],
                "fixed_50_50": blend_predictions(evaluation[left], evaluation[right], 0.5),
            }
            for candidate, frame in control_predictions.items():
                target = destination / "controls" / candidate
                target.mkdir(parents=True)
                frame.to_csv(target / "predictions.csv")
                control_metrics, _ = _prediction_statistics(frame)
                control_portfolio = (
                    _replay_scenarios(config, protocol, target)
                    if replay and config.get("backtest", {}).get("enabled")
                    else None
                )
                controls[candidate] = {
                    "metrics": _json_ready(control_metrics),
                    "portfolio": control_portfolio,
                    "score_source": "all declared seed forecasts averaged before portfolio construction",
                }
                write_json(target / "receipt.json", controls[candidate])
            receipt = {
                "status": "completed",
                "fold": name,
                "selection": selected,
                "metrics": _json_ready(metrics),
                "controls": controls,
                "prediction_lineage": lineage,
                "portfolio": portfolio,
            }
            run["status"] = "completed"
            write_json(destination / "run.json", run)
        except Exception as error:
            receipt = {
                "status": "failed",
                "fold": name,
                "error": f"{type(error).__name__}: {error}",
                "failed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        write_json(destination / "receipt.json", _json_ready(receipt))
        receipts.append(receipt)
        # Retain progress/failures even if a later fold or process fails.
        write_json(
            output / "summary.json",
            {
                "status": "running",
                "registration": registration,
                "folds": receipts,
                "promotion_eligible": False,
            },
        )
    result = {
        "status": "completed"
        if all(row["status"] == "completed" for row in receipts)
        else "failed",
        "registration": registration,
        "folds": receipts,
        "promotion_eligible": False,
    }
    write_json(output / "summary.json", _json_ready(result))
    return _json_ready(result)
