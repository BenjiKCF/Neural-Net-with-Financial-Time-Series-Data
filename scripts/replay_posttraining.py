#!/usr/bin/env python3
"""Verify saved small-model forecasts offline, without neural-model training.

The provider must be the imported bundled SPX history. Feature normalization is
rebuilt on the recorded training dates; saved learned weights are never updated.
This does not load or distribute any official Google TimesFM checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from financial_ts.catalog import build_mini_encoder
from financial_ts.comparison import _read_predictions, _require_same_rows
from financial_ts.paper_research import _MiniEnsemble, _normalized_scores, _state_digest, _windows
from financial_ts.research import _build_dataset, _labels
from financial_ts.workflow import _provider_fingerprint, _read_calendar

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return path.read_text(encoding="utf-8")


def replay(provider, report, base_report):
    protocol = json.loads(read_json(report / "protocol.json"))
    selection = json.loads(read_json(report / "selection.json"))
    require(protocol["status"] == "completed", "The source campaign is not completed.")
    require(
        _provider_fingerprint(provider) == protocol["provider_fingerprint"],
        "Provider fingerprint differs from the registered input bytes.",
    )
    roster = protocol["roster"]
    require(set(roster) == set(selection["states_sha256"]), "Checkpoint roster mismatch.")
    configurations, members = [], []
    for seed in protocol["base_seeds"]:
        suffix = f"timesfm3_mini/seed-{seed}/config.json"
        path = base_report / suffix
        recorded = [item for item in protocol["sources"] if item["path"].endswith(suffix)]
        require(len(recorded) == 1, f"Missing unique registered configuration: {suffix}")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == recorded[0]["sha256"],
            f"Configuration hash mismatch: {path}",
        )
        config = json.loads(read_json(path))
        require(config["model"]["name"] == "timesfm3_mini", "Expected the original mini model.")
        if configurations:
            require(
                all(
                    config[key] == configurations[0][key]
                    for key in ("model", "sequence_length", "segments")
                ),
                "Ensemble members have different model or input settings.",
            )
        configurations.append(config)
        parameters = config["model"]["params"]
        dimensions = {key: parameters[key] for key in ("d_model", "nhead", "num_layers", "dropout")}
        with torch.random.fork_rng(devices=[]):
            members.append(build_mini_encoder("timesfm3_mini", d_feat=157, **dimensions))
    require(len(members) == 3, "The archived ensemble requires three members.")
    model = _MiniEnsemble(members).eval()
    torch.set_num_threads(2)
    config = copy.deepcopy(configurations[0])
    config.update(provider_uri=str(provider), segments=selection["segments"])
    config["model"]["params"]["device"] = "cpu"
    dataset, segments = _build_dataset(config, _read_calendar(provider))
    require(
        {key: list(value) for key, value in segments.items()} == selection["segments"],
        "Partition mismatch.",
    )
    training = _labels(dataset, "train")
    mean, scale = selection["training_mean"], selection["training_scale"]
    actual_scale = float(training.std(ddof=0))
    actual_scale = actual_scale if actual_scale > 1e-12 else 1.0
    require(
        np.isfinite([mean, scale]).all()
        and scale > 0
        and np.isclose(training.mean(), mean, rtol=0, atol=1e-15)
        and np.isclose(actual_scale, scale, rtol=0, atol=1e-15),
        "Training target normalization does not match the recorded values.",
    )
    frames, windows = {}, {}
    for segment, filename in (("valid", "validation_predictions.csv"), ("test", "predictions.csv")):
        frames[segment] = _read_predictions(report / filename)
        labels = _labels(dataset, segment)
        _require_same_rows(frames[segment], labels.to_frame(), f"{segment} labels")
        windows[segment] = _windows(dataset, segment)
        require(
            windows[segment][2].equals(labels.index), f"{segment} feature/label identities differ."
        )
        for column, value in (("zero_return", 0.0), ("training_mean", mean)):
            require(
                np.allclose(frames[segment][column], value, rtol=0, atol=TOLERANCE),
                f"Invalid {column} reference.",
            )
    cache, first_model, results = {}, {}, {}
    for name in roster:
        checkpoint = torch.load(
            report / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=True
        )
        require(
            checkpoint["target_mean"] == mean and checkpoint["target_scale"] == scale,
            f"{name}: target normalization changed.",
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        digest = _state_digest(model)
        require(digest == selection["states_sha256"][name], f"{name}: tensor hash mismatch.")
        if digest not in cache:
            cache[digest] = {
                segment: _normalized_scores(model, *data[:2]) * scale + mean
                for segment, data in windows.items()
            }
            first_model[digest] = name
        errors = {}
        for segment, frame in frames.items():
            require(
                name in frame and np.isfinite(frame[name].to_numpy()).all(),
                f"{name}: invalid {segment} recorded scores.",
            )
            errors[segment] = float(np.max(np.abs(cache[digest][segment] - frame[name].to_numpy())))
            require(
                errors[segment] <= TOLERANCE,
                f"{name}: {segment} replay error {errors[segment]} exceeds {TOLERANCE}.",
            )
        results[name] = {
            "state_sha256": digest,
            "inference_reused_from": first_model[digest],
            "max_absolute_error": errors,
        }
    return {
        "status": "passed",
        "model_training_performed": False,
        "preprocessing": "reconstructed using only the recorded training partition",
        "provider_fingerprint": protocol["provider_fingerprint"],
        "absolute_tolerance": TOLERANCE,
        "checkpoints_verified": len(results),
        "unique_states_evaluated": len(cache),
        "rows": {segment: len(frame) for segment, frame in frames.items()},
        "max_absolute_error": max(
            error for row in results.values() for error in row["max_absolute_error"].values()
        ),
        "models": results,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "pyqlib")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", type=Path, required=True, help="Imported bundled SPX provider directory"
    )
    parser.add_argument("--report", type=Path, default=ROOT / "reports/paper_posttraining")
    parser.add_argument("--base-report", type=Path, default=ROOT / "reports/model_research")
    parser.add_argument(
        "--output", type=Path, help="Optional new audit JSON; never overwrite an existing file"
    )
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.exists(), "Audit output already exists; choose a new file.")
    result = replay(
        *(path.expanduser().resolve() for path in (args.provider, args.report, args.base_report))
    )
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
