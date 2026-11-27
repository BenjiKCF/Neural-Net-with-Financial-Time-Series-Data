"""Batch-preserving parallel inference with exact trained-state agreement.

Each worker fits the same declared model locally. Only numerical predictions and
state hashes leave a worker. Assembly refuses differing model states, data,
source, indices, missing shards, overlapping rows, or altered evidence files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from financial_ts.workflow import _prediction_statistics, _save_json


def batch_bounds(rows: int, batch_size: int, shard: int, count: int) -> tuple[int, int]:
    """Partition complete batches; only the final original batch can be short."""
    if any(type(value) is not int for value in (rows, batch_size, shard, count)):
        raise ValueError("Shard dimensions must be integers.")
    if rows < 1 or batch_size < 1 or count < 1 or not 0 <= shard < count:
        raise ValueError("Invalid shard dimensions.")
    batches = (rows + batch_size - 1) // batch_size
    if count > batches:
        raise ValueError("Every shard must contain at least one complete original batch.")
    return (
        min(rows, (batches * shard // count) * batch_size),
        min(rows, (batches * (shard + 1) // count) * batch_size),
    )


def index_sha256(index: pd.MultiIndex) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(index).to_numpy().tobytes()).hexdigest()


def state_sha256(state: dict | None) -> str:
    digest = hashlib.sha256()
    for name, value in sorted((state or {}).items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(str(array.dtype).encode() + b"\0")
        digest.update(json.dumps(list(array.shape)).encode() + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


class BatchAlignedDatasetView:
    """An inference-only view; training and checkpoint selection use full data."""

    def __init__(self, dataset, shard: int, count: int, batch_size: int):
        self.dataset = dataset
        self.shard, self.count, self.batch_size = shard, count, batch_size
        self.segments = {}

    def prepare(self, segment, *, col_set, data_key):
        if segment not in {"valid", "test"} or col_set != "feature" or data_key != "infer":
            raise ValueError("Shard views expose validation/test inference indices only.")
        frame = self.dataset.prepare(segment, col_set=col_set, data_key=data_key)
        start, stop = batch_bounds(len(frame), self.batch_size, self.shard, self.count)
        self.segments[segment] = {
            "start": start,
            "stop": stop,
            "total": len(frame),
            "full_index_sha256": index_sha256(frame.index),
            "index_sha256": index_sha256(frame.index[start:stop]),
        }
        return frame.iloc[start:stop]


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble_shards(paths: list[Path], output: Path) -> dict:
    """Assemble complete forecasts only after byte and identity verification."""
    if not paths:
        raise ValueError("No inference shards supplied.")
    records = []
    for path in map(Path, paths):
        record = json.loads((path / "shard.json").read_text())
        if record.get("status") != "completed":
            raise ValueError("Cannot assemble an incomplete inference shard.")
        for item in record["files"]:
            relative = Path(item["path"])
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError("Shard evidence paths must be simple filenames.")
            artifact = path / relative
            if artifact.stat().st_size != item["bytes"] or _file_sha(artifact) != item["sha256"]:
                raise ValueError("Shard evidence failed its checksum.")
        records.append((record, path))
    records.sort(key=lambda pair: pair[0]["shard"])
    first = records[0][0]
    count = first["count"]
    if [record["shard"] for record, _ in records] != list(range(count)):
        raise ValueError("Assembly requires each declared shard exactly once.")
    if any(
        record["count"] != count or record["identity"] != first["identity"] for record, _ in records
    ):
        raise ValueError("Shards disagree on model state, data, configuration, or source.")
    frames, ranges = {}, {}
    for segment, filename in (
        ("valid", "validation_predictions.csv.gz"),
        ("test", "predictions.csv.gz"),
    ):
        parts, cursor = [], 0
        reference = first["segments"][segment]
        for record, path in records:
            spec = record["segments"][segment]
            expected = batch_bounds(
                reference["total"], first["identity"]["batch_size"], record["shard"], count
            )
            if (
                spec["start"] != cursor
                or (spec["start"], spec["stop"]) != expected
                or spec["total"] != reference["total"]
                or spec["full_index_sha256"] != reference["full_index_sha256"]
            ):
                raise ValueError("Shard boundaries or full inference indices disagree.")
            frame = pd.read_csv(
                path / filename, parse_dates=["datetime"], float_precision="round_trip"
            )
            frame = frame.set_index(["datetime", "instrument"])
            if (
                len(frame) != spec["stop"] - spec["start"]
                or not frame.index.is_unique
                or index_sha256(frame.index) != spec["index_sha256"]
                or not np.isfinite(frame["score"]).all()
                or np.isinf(frame["label"]).any()
            ):
                raise ValueError("Shard prediction rows fail their declared contract.")
            parts.append(frame)
            cursor = spec["stop"]
        combined = pd.concat(parts)
        if (
            cursor != reference["total"]
            or not combined.index.is_unique
            or index_sha256(combined.index) != reference["full_index_sha256"]
        ):
            raise ValueError("Assembled inference is incomplete or overlapping.")
        frames[segment] = combined
        ranges[segment] = reference["total"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh assembly directory.")
    output.mkdir(parents=True, exist_ok=True)
    metrics, test_ic = _prediction_statistics(frames["test"])
    metrics["validation"], valid_ic = _prediction_statistics(frames["valid"])
    metrics["model_details"] = first["model_details"]
    metrics["model_details"]["validation_rmse"] = metrics["validation"]["rmse"]
    metrics["qlib_metrics"] = {}
    for segment, filename in (
        ("valid", "validation_predictions.csv.gz"),
        ("test", "predictions.csv.gz"),
    ):
        frames[segment].to_csv(output / filename, compression={"method": "gzip", "mtime": 0})
    test_ic.to_csv(output / "daily_ic.csv")
    valid_ic.to_csv(output / "validation_daily_ic.csv")
    for filename in (
        "resolved_config.json",
        "fit_manifest.json",
        "fit_samples.json",
        "training_curves.json",
        "data_source.json",
        "protocol.json",
        "source.json",
    ):
        shutil.copyfile(records[0][1] / filename, output / filename)
    _save_json(output / "metrics.json", metrics)
    receipt = {
        "schema_version": 1,
        "status": "complete_forecasts_pending_portfolio_replay",
        "identity": first["identity"],
        "shards": [record for record, _ in records],
        "rows": ranges,
        "trained_state_agreement": "Exact SHA256 agreement across all independently fitted workers.",
        "batch_policy": "All original batch boundaries preserved; no inference rows sampled or omitted.",
    }
    _save_json(output / "assembly.json", receipt)
    _save_json(
        output / "run.json",
        {
            "status": "forecasts_completed",
            "execution": "Complete batch-aligned inference assembled from identical model states",
            "provider": first["identity"].get("provider_fingerprint"),
            "segments": first["identity"].get("segments"),
            "model": first["identity"]["model"],
            "portfolio_status": "pending deterministic native Qlib replay",
        },
    )
    return receipt
