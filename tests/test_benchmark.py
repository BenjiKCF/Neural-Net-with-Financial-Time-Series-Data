"""Comparison protocol comparability, early validation, and aggregate arithmetic."""

import json

import pandas as pd
import pytest

from financial_ts.benchmark import benchmark_config, run_benchmark
from financial_ts.catalog import NEURAL_MODELS, TRAINABLE_MODELS
from financial_ts.workflow import build_task


def test_model_comparison_keeps_label_splits_and_neural_budget_fixed(tmp_path):
    calendar = pd.bdate_range("2023-01-03", periods=240)
    tasks = {}
    configs = {}
    segments = {}
    for name in TRAINABLE_MODELS:
        config = benchmark_config(name, tmp_path / "provider", tmp_path / name, seed=7, epochs=12)
        task, resolved = build_task(config, calendar)
        configs[name], tasks[name], segments[name] = config, task, resolved
    reference = tasks["transformer"]["dataset"]["kwargs"]["handler"]
    for name in TRAINABLE_MODELS:
        assert segments[name] == segments["transformer"]
        assert tasks[name]["dataset"]["kwargs"]["handler"] == reference
        assert tasks[name]["model"]["kwargs"]["seed"] == 7
        assert configs[name]["backtest"]["enabled"] is False
    # Historical windows and AdamW budgets are matched among neural models;
    # LightGBM deliberately remains a separate tabular baseline.
    optimizer_fields = (
        "n_epochs",
        "early_stop",
        "batch_size",
        "lr",
        "weight_decay",
        "aux_loss_weight",
    )
    expected_budget = {
        key: configs["transformer"]["model"]["params"][key] for key in optimizer_fields
    }
    for name in NEURAL_MODELS:
        assert tasks[name]["dataset"]["class"] == "TSDatasetH"
        assert tasks[name]["dataset"]["kwargs"]["step_len"] == 32
        assert {
            key: configs[name]["model"]["params"][key] for key in optimizer_fields
        } == expected_budget
        assert tasks[name]["model"]["kwargs"]["architecture"] == name
    assert tasks["lightgbm"]["dataset"]["class"] == "DatasetH"
    assert expected_budget["aux_loss_weight"] == 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"models": ["qwen4_mini", "qwen4_mini"]},
        {"models": ["unpublished_architecture"]},
        {"seeds": [7, 7]},
        {"seeds": [True]},
        {"epochs": 0},
        {"models": ["timesfm3_pretrained"]},
    ],
)
def test_invalid_protocol_is_rejected_before_creating_outputs(tmp_path, monkeypatch, kwargs):
    def forbidden(*args, **kw):
        pytest.fail("Invalid comparison must fail before any training starts")

    monkeypatch.setattr("financial_ts.benchmark.run_workflow", forbidden)
    output = tmp_path / "comparison"
    with pytest.raises(ValueError):
        run_benchmark(tmp_path / "provider", output, **kwargs)
    assert not output.exists()


def test_aggregate_retains_each_run_and_reports_seed_dispersion(tmp_path, monkeypatch):
    # Known, differing seed errors catch accidental best-seed selection and
    # confusion between seed dispersion, standard error, and confidence intervals.
    errors = {
        ("transformer", 7): 0.2,
        ("transformer", 42): 0.4,
        ("qwen4_mini", 7): 0.3,
        ("qwen4_mini", 42): 0.5,
    }
    calls = []

    def completed(config):
        key = (config["model"]["name"], config["seed"])
        calls.append(key)
        return {
            "run": {"elapsed_seconds": 1.2},
            "metrics": {
                "rmse": errors[key],
                "zero_return_rmse": 0.6,
                "mae": errors[key] / 2,
                "directional_accuracy": 0.5,
                "prediction_count": 100,
                "model_details": {
                    "pretrained": False,
                    "parameter_count": 1000,
                    "best_epoch": 2,
                    "epochs_run": 3,
                    "validation_rmse": errors[key] - 0.01,
                    "fit_seconds": 1.0,
                },
            },
        }

    monkeypatch.setattr("financial_ts.benchmark.run_workflow", completed)
    output = tmp_path / "comparison"
    result = run_benchmark(
        tmp_path / "provider", output, models=["transformer", "qwen4_mini"], seeds=[7, 42], epochs=3
    )
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert calls == list(errors)
    assert len(result["runs"]) == 4
    aggregate = {row["model"]: row for row in result["aggregate"]}
    assert aggregate["transformer"]["test_rmse_mean"] == pytest.approx(0.3)
    assert aggregate["qwen4_mini"]["test_rmse_mean"] == pytest.approx(0.4)
    for row in aggregate.values():
        assert row["runs"] == 2
        assert row["test_rmse_std_across_seeds"] == pytest.approx(0.1)
    assert result["status"] == "completed"
    assert json.loads((output / "comparison.json").read_text()) == result
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["status"] == "completed" and len(protocol["configs"]) == 4
    assert len(pd.read_csv(output / "comparison.csv")) == 4


def test_pretrained_only_aggregate_is_json_ready_without_validation_score(tmp_path, monkeypatch):
    calls = []

    def completed(config):
        calls.append(config)
        return {
            "run": {"elapsed_seconds": 1.2},
            "metrics": {
                "rmse": 0.2,
                "zero_return_rmse": 0.3,
                "mae": 0.1,
                "directional_accuracy": 0.5,
                "prediction_count": 100,
                "model_details": {
                    "pretrained": True,
                    "parameter_count": 1000,
                    "best_epoch": None,
                    "epochs_run": None,
                    "validation_rmse": None,
                    "fit_seconds": 1.0,
                },
            },
        }

    monkeypatch.setattr("financial_ts.benchmark.run_workflow", completed)
    output = tmp_path / "comparison"
    result = run_benchmark(
        tmp_path / "provider",
        output,
        models=["timesfm3_pretrained"],
        seeds=[7, 42],
        checkpoint=tmp_path / "checkpoint",
    )
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert len(calls) == 1
    assert result["runs"][0]["seed"] is None
    assert result["aggregate"][0]["parameter_count"] == 1000
    assert result["aggregate"][0]["validation_rmse_mean"] is None
    assert json.loads((output / "comparison.json").read_text()) == result
