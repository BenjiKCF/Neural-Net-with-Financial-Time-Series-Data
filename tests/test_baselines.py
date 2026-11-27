"""Baseline correctness and train-only state, independent of native Qlib I/O."""

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset.handler import DataHandlerLP

from financial_ts.baselines import TabularBaseline
from financial_ts.workflow import build_task


class Dataset:
    def __init__(self, x, y, *, validation_offset=0):
        self.calls = []
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=len(x)), ["A"]],
            names=["datetime", "instrument"],
        )
        self.x = pd.DataFrame(x, index=index, columns=["x", "z"], dtype=float)
        self.y = pd.DataFrame(y, index=index, columns=["return"])
        self.validation_offset = validation_offset

    def prepare(self, segment, col_set, data_key):
        self.calls.append((segment, col_set, data_key))
        if col_set == ["feature", "label"]:
            assert segment == "train" and data_key == DataHandlerLP.DK_L
            return pd.concat({"feature": self.x, "label": self.y}, axis=1)
        assert col_set == "feature" and data_key == DataHandlerLP.DK_I
        return self.x + (self.validation_offset if segment == "valid" else 0)


def test_ridge_matches_augmented_least_squares_and_never_reads_validation():
    x = np.array([[1, 2], [2, -1], [4, 3], [-1, 1]], dtype=float)
    y = x @ np.array([0.2, -0.1]) + 0.4
    data = Dataset(x, y, validation_offset=1e9)
    model = TabularBaseline(alpha=0.5)
    model.fit(data)
    centered = x - x.mean(axis=0)
    expected = np.linalg.lstsq(
        np.vstack([centered, np.sqrt(0.5) * np.eye(2)]),
        np.r_[y - y.mean(), [0, 0]],
        rcond=None,
    )[0]
    np.testing.assert_allclose(model.coefficient_, expected)
    assert data.calls == [("train", ["feature", "label"], DataHandlerLP.DK_L)]
    np.testing.assert_allclose(model.predict(data), centered @ expected + y.mean())


def test_future_perturbation_cannot_change_baseline_fitted_state():
    x = np.array([[1, 2], [2, 4], [3, -1]], dtype=float)
    original, changed = (
        Dataset(x, [0.1, -0.2, 0.3]),
        Dataset(x, [0.1, -0.2, 0.3], validation_offset=1e12),
    )
    a, b = TabularBaseline(), TabularBaseline()
    a.fit(original)
    b.fit(changed)
    np.testing.assert_array_equal(a.coefficient_, b.coefficient_)
    assert a.intercept_ == b.intercept_


def test_zero_return_is_exact_and_requires_fitted_schema():
    data = Dataset([[1, 2], [3, 4]], [0.1, -0.4])
    model = TabularBaseline(kind="zero")
    with pytest.raises(ValueError, match="Fit"):
        model.predict(data)
    model.fit(data)
    np.testing.assert_array_equal(model.predict(data), np.zeros(2))
    assert model.parameter_count_ == 0
    data.x.columns = ["z", "x"]
    with pytest.raises(ValueError, match="schema"):
        model.predict(data)


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, True])
def test_invalid_regularization_rejected(alpha):
    with pytest.raises(ValueError):
        TabularBaseline(alpha=alpha)


def test_nonfinite_training_and_prediction_rejected():
    data = Dataset([[1, 2], [3, 4]], [0.1, -0.4])
    model = TabularBaseline()
    model.fit(data)
    data.x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        model.predict(data)
    with pytest.raises(ValueError, match="finite"):
        TabularBaseline().fit(data)


@pytest.mark.parametrize("name", ["zero", "ridge"])
def test_workflow_dispatch_preserves_tabular_dataset(name):
    task, _ = build_task({"model": {"name": name}}, pd.bdate_range("2020-01-01", periods=180))
    assert task["model"]["class"] == "TabularBaseline"
    assert task["model"]["kwargs"]["kind"] == name
    assert task["dataset"]["class"] == "DatasetH"
