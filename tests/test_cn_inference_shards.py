"""Prove parallel inference keeps all rows, original batches, and one model."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from financial_ts.cn_inference_shards import (
    BatchAlignedDatasetView,
    assemble_shards,
    batch_bounds,
    index_sha256,
    state_sha256,
)


@pytest.mark.parametrize(
    "rows,batch,count", [(137002, 64, 4), (72000, 64, 4), (100, 7, 3), (9, 8, 2)]
)
def test_partition_preserves_every_original_batch(rows, batch, count):
    serial = [list(range(start, min(rows, start + batch))) for start in range(0, rows, batch)]
    parallel = []
    for shard in range(count):
        lo, hi = batch_bounds(rows, batch, shard, count)
        parallel.extend(
            [list(range(start, min(hi, start + batch))) for start in range(lo, hi, batch)]
        )
    assert parallel == serial


def test_empty_or_malformed_shards_rejected():
    for values in [(1, 64, 0, 2), (0, 64, 0, 1), (10, 0, 0, 1), (10, 1, 2, 2), (10, 1, False, 1)]:
        with pytest.raises(ValueError):
            batch_bounds(*values)


def test_view_cannot_read_labels_and_retains_input_order():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-02", periods=5), ["AAA", "BBB"]], names=["datetime", "instrument"]
    )

    class Dataset:
        def prepare(self, *args, **kwargs):
            return pd.DataFrame({"close": np.arange(10)}, index=index)

    view = BatchAlignedDatasetView(Dataset(), shard=1, count=2, batch_size=3)
    selected = view.prepare("test", col_set="feature", data_key="infer")
    pd.testing.assert_index_equal(selected.index, index[6:])
    assert view.segments["test"]["full_index_sha256"] == index_sha256(index)
    with pytest.raises(ValueError, match="inference indices"):
        view.prepare("test", col_set="label", data_key="raw")


def test_state_hash_detects_exact_tensor_changes_without_exporting_tensors():
    state = {"b": torch.tensor([1.0]), "a": torch.tensor([2.0])}
    assert state_sha256(state) == state_sha256(dict(reversed(list(state.items()))))
    assert state_sha256(state) != state_sha256({**state, "a": torch.tensor([2.000001])})


@pytest.fixture
def shards(tmp_path):
    paths = []
    index = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-02", periods=5), ["AAA", "BBB"]], names=["datetime", "instrument"]
    )
    full = pd.DataFrame(
        {"score": np.linspace(-0.03, 0.02, 10), "label": np.linspace(0.01, 0.04, 10)}, index=index
    )
    full.iloc[3, 1] = np.nan
    for shard in range(2):
        path = tmp_path / f"shard-{shard}"
        path.mkdir()
        lo, hi = batch_bounds(len(full), 3, shard, 2)
        for name in ("predictions.csv.gz", "validation_predictions.csv.gz"):
            full.iloc[lo:hi].to_csv(path / name)
        for name in (
            "resolved_config.json",
            "fit_manifest.json",
            "fit_samples.json",
            "training_curves.json",
            "data_source.json",
            "protocol.json",
            "source.json",
        ):
            (path / name).write_text("{}")
        segment = {
            "start": lo,
            "stop": hi,
            "total": len(full),
            "full_index_sha256": index_sha256(index),
            "index_sha256": index_sha256(index[lo:hi]),
        }
        record = {
            "status": "completed",
            "shard": shard,
            "count": 2,
            "identity": {
                "model": "example",
                "batch_size": 3,
                "trained_state_sha256": "exact-state",
            },
            "segments": {"valid": segment, "test": segment},
            "model_details": {"name": "example", "best_epoch": 1},
            "files": [
                {
                    "path": p.name,
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sorted(path.iterdir())
            ],
        }
        (path / "shard.json").write_text(json.dumps(record))
        paths.append(path)
    return paths, full


def test_assemble_all_rows_and_recompute_metrics(shards, tmp_path):
    paths, full = shards
    receipt = assemble_shards(list(reversed(paths)), tmp_path / "assembled")
    actual = pd.read_csv(
        tmp_path / "assembled/predictions.csv.gz",
        parse_dates=["datetime"],
        float_precision="round_trip",
    ).set_index(["datetime", "instrument"])
    pd.testing.assert_frame_equal(actual, full)
    assert receipt["rows"] == {"valid": 10, "test": 10}
    metrics = json.loads((tmp_path / "assembled/metrics.json").read_text())
    finite = full.dropna()
    assert metrics["rmse"] == pytest.approx(np.sqrt(((finite.score - finite.label) ** 2).mean()))


@pytest.mark.parametrize(
    "failure", ["missing", "duplicate", "different_state", "corrupt_file", "gap"]
)
def test_assembly_rejects_incomplete_or_incompatible_evidence(shards, tmp_path, failure):
    paths, _ = shards
    if failure == "missing":
        paths = paths[:1]
    elif failure == "duplicate":
        paths = [paths[0], paths[0]]
    elif failure == "corrupt_file":
        with (paths[1] / "predictions.csv.gz").open("ab") as stream:
            stream.write(b"changed")
    else:
        record_path = paths[1] / "shard.json"
        record = json.loads(record_path.read_text())
        if failure == "different_state":
            record["identity"]["trained_state_sha256"] = "another-state"
        else:
            record["segments"]["test"]["start"] += 1
        record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        assemble_shards(paths, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
