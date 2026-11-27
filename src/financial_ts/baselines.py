"""Small Qlib controls with the same feature and raw-return contract as DL runs."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model


class TabularBaseline(Model):
    """Zero-return or ridge control; every fitted quantity uses training rows only.

    Ridge minimizes sum-squared error plus ``alpha * ||coefficient||**2``.
    The intercept is unpenalized. Qlib's shared train-fitted feature processor
    provides normalization, so this adapter never fits a second preprocessing
    transform using validation or evaluation rows.
    """

    def __init__(self, kind="ridge", alpha=1.0):
        if kind not in {"zero", "ridge"}:
            raise ValueError("kind must be zero or ridge.")
        if isinstance(alpha, bool) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and strictly positive.")
        self.kind, self.alpha = kind, float(alpha)
        self.fitted_ = False

    def fit(self, dataset, **kwargs):
        started = time.perf_counter()
        frame = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        features, labels = frame["feature"], frame["label"].iloc[:, 0]
        if not frame.index.is_unique or frame.empty:
            raise ValueError("Training rows must be nonempty and unique.")
        x = features.to_numpy(dtype=np.float64)
        y = labels.to_numpy(dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Baseline training features and labels must be finite.")
        self.feature_names_ = list(features.columns)
        self.training_rows_ = len(frame)
        self.coefficient_ = np.zeros(x.shape[1], dtype=np.float64)
        self.intercept_ = 0.0
        if self.kind == "ridge":
            mean_x, mean_y = x.mean(axis=0), float(y.mean())
            centered_x, centered_y = x - mean_x, y - mean_y
            gram = centered_x.T @ centered_x
            gram.flat[:: len(gram) + 1] += self.alpha
            self.coefficient_ = np.linalg.solve(gram, centered_x.T @ centered_y)
            self.intercept_ = mean_y - float(mean_x @ self.coefficient_)
        if not np.isfinite(self.coefficient_).all() or not np.isfinite(self.intercept_):
            raise ValueError("Baseline fitting produced nonfinite coefficients.")
        self.parameter_count_ = 0 if self.kind == "zero" else len(self.coefficient_) + 1
        self.fit_seconds_ = time.perf_counter() - started
        self.fitted_ = True

    def predict(self, dataset, segment="test"):
        if not self.fitted_:
            raise ValueError("Fit the baseline before prediction.")
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if list(features.columns) != self.feature_names_:
            raise ValueError("Prediction feature schema differs from the fitted baseline.")
        if not features.index.is_unique or features.empty:
            raise ValueError("Prediction rows must be nonempty and unique.")
        x = features.to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Prediction features must be finite.")
        prediction = x @ self.coefficient_ + self.intercept_
        if not np.isfinite(prediction).all():
            raise ValueError("Baseline prediction overflowed.")
        return pd.Series(prediction, index=features.index, name="score")
