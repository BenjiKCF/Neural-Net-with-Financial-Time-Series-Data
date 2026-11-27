"""Selection boundaries and paired evidence for the fixed candidate campaign."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from financial_ts import research


def test_shrinkage_uses_validation_error_and_has_a_defined_tie_break():
    labels = np.array([0.01, -0.01, 0.02, -0.02])
    selected = research.select_shrinkage(labels, labels * 2, 0)
    assert selected["gamma"] == 0.5
    assert selected["validation_rmse"] == 0
    tied = research.select_shrinkage(np.zeros(4), np.zeros(4), 0)
    assert tied["gamma"] == 0


def test_paired_interval_retains_pairing_for_a_known_constant_difference():
    result = research.paired_block_interval(np.zeros(100), np.ones(100), np.full(100, 2), draws=100)
    assert result["rmse_difference"] == -1
    assert result["lower_95"] == result["upper_95"] == -1
    identical = research.paired_block_interval(
        np.arange(100), np.zeros(100), np.zeros(100), draws=100
    )
    assert identical["lower_95"] == identical["upper_95"] == 0


@pytest.mark.parametrize("scores", [[1, np.nan], [1, np.inf], [], [[1, 2]]])
def test_research_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):
        research.rmse([1, 2], scores)


def test_all_choices_are_saved_before_test_prediction_and_ignore_test_labels(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    provider = tmp_path / "provider"
    provider.mkdir()
    dates = pd.bdate_range("2000-01-03", periods=180)
    index = pd.MultiIndex.from_product([dates, ["SPX"]], names=["datetime", "instrument"])
    dataset = {
        "train": pd.Series(0.001, index=index[:80], name="label"),
        "valid": pd.Series(np.linspace(-0.01, 0.01, 50), index=index[80:130], name="label"),
        "test": pd.Series(np.linspace(-0.02, 0.02, 50), index=index[130:], name="label"),
    }
    segments = {
        key: [str(value.index[0][0].date()), str(value.index[-1][0].date())]
        for key, value in dataset.items()
    }
    monkeypatch.setattr(research, "_read_calendar", lambda _: dates)
    monkeypatch.setattr(research, "_provider_fingerprint", lambda _: {"sha256": "test"})
    monkeypatch.setattr(research, "_source_files", lambda: [])
    monkeypatch.setattr(research, "_build_dataset", lambda *_: (dataset, segments))
    monkeypatch.setattr(research, "_labels", lambda _, segment: dataset[segment])
    fitted = []
    active_output = None

    def fake_fit(config, calendar, data):
        fitted.append((config["model"]["name"], config["seed"]))

        def predict(_, segment):
            if segment == "test":
                assert len(fitted) == len(research.RESEARCH_MODELS)
                locked = json.loads((active_output / "selection.json").read_text())
                assert locked["test_predictions_started"] is False
                assert locked["selected_by_validation"]
            # A forecast determined by position, never by this partition's label values.
            return pd.Series(
                np.linspace(-0.02, 0.02, len(dataset[segment])), index=dataset[segment].index
            )

        return SimpleNamespace(
            target_mean_=float(dataset["train"].mean()),
            target_scale_=1.0,
            model=torch.nn.Linear(1, 1),
            evals_result_={"train": [1.0], "valid": [1.0]},
            parameter_count_=2,
            best_epoch_=1,
            fit_seconds_=0.01,
            predict=predict,
        )

    monkeypatch.setattr(research, "_fit_trial", fake_fit)
    selections = []
    results = []
    for run in (0, 1):
        fitted.clear()
        active_output = tmp_path / f"run-{run}"
        if run:
            dataset["test"] = dataset["test"] * 100
        results.append(research.run_scratch_research(provider, active_output, seeds=[42], epochs=1))
        selections.append(json.loads((active_output / "selection.json").read_text()))
    assert selections[0]["calibrations"] == selections[1]["calibrations"]
    assert selections[0]["selected_by_validation"] == selections[1]["selected_by_validation"]
    assert results[0]["metrics"] != results[1]["metrics"]


def test_research_refuses_to_overwrite_existing_work(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "important.json").write_text("{}")
    with pytest.raises(ValueError, match="new or empty"):
        research.run_scratch_research(tmp_path / "provider", out)
