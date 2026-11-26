"""Optional official TimesFM 3 zero-shot forecasting through Qlib's model API.

Weights load only from an explicitly supplied local safetensors directory.
They are never downloaded by this adapter or embedded in Qlib's saved model.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model


class OfficialTimesFM3Model(Model):
    """Forecast two provider close prices and return forecast[t+2]/forecast[t+1]-1.

    This uses the official pretrained 330M model, not the small supervised
    architecture. fit loads fixed parameters without reading any labels.
    Inputs are provider $close prices in price units, including any upstream
    adjustments; they are not normalized factor features or unadjusted tape.
    Pretraining overlap with old market history is unknown.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        context_length: int = 256,
        batch_size: int = 8,
        device: str = "cpu",
        num_threads: int = 2,
    ):
        for name, value in (
            ("context_length", context_length),
            ("batch_size", batch_size),
            ("num_threads", num_threads),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 2 <= context_length <= 15360:
            raise ValueError("context_length must be between 2 and 15360 sessions.")
        if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
            raise ValueError("Official TimesFM3 supports an explicit cpu or cuda device here.")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir() or not all(
            (checkpoint / name).is_file() for name in ("config.json", "model.safetensors")
        ):
            raise ValueError(
                "checkpoint_path must be a local TimesFM3 directory containing config.json "
                "and model.safetensors. Download the official research checkpoint explicitly; "
                "this adapter does not download weights."
            )
        self.checkpoint_path = str(checkpoint)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.num_threads = num_threads
        self.fitted = False
        self.parameter_count_: int | None = None
        self.fit_seconds_: float | None = None
        self.checkpoint_config_sha256_: str | None = None
        self.checkpoint_size_bytes_: int | None = None
        # Private fields are omitted by Qlib's Serializable: no duplicated weights.
        self._forecaster = None

    def _get_forecaster(self):
        forecaster = getattr(self, "_forecaster", None)
        if forecaster is None:
            try:
                import torch
                from timesfm3 import ModelConfig, TimesFM3Forecaster
            except (ImportError, AttributeError) as exc:
                raise ImportError(
                    "Official TimesFM3 requires the optional dependency timesfm[torch]==3.0.2."
                ) from exc
            torch.set_num_threads(self.num_threads)
            forecaster = TimesFM3Forecaster(
                ModelConfig(
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    per_core_batch_size=self.batch_size,
                    local_files_only=True,
                )
            )
            self._forecaster = forecaster
        return forecaster

    def fit(self, dataset, **kwargs):
        """Load the fixed model without inspecting labels or fitting a scaler."""
        started = time.perf_counter()
        forecaster = self._get_forecaster()
        loaded_model = getattr(forecaster, "model", None)
        # Count the actual loaded parameters, never a hard-coded model-card size.
        # Test doubles may omit the model; then its parameter count is unknown.
        self.parameter_count_ = (
            sum(parameter.numel() for parameter in loaded_model.parameters())
            if loaded_model is not None
            else None
        )
        checkpoint = Path(self.checkpoint_path)
        self.checkpoint_config_sha256_ = hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest()
        self.checkpoint_size_bytes_ = (checkpoint / "model.safetensors").stat().st_size
        self.fit_seconds_ = time.perf_counter() - started
        self.fitted = True
        return self

    def predict(self, dataset, segment="test") -> pd.Series:
        if not self.fitted:
            raise ValueError("Call fit before predict to load the official pretrained model.")
        prepared = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = prepared.get_index() if hasattr(prepared, "get_index") else prepared.index
        if not isinstance(index, pd.MultiIndex) or not {"datetime", "instrument"}.issubset(
            index.names
        ):
            raise ValueError("Predictions require a Qlib datetime/instrument MultiIndex.")
        if not index.is_unique:
            raise ValueError("Prediction samples must have unique datetime/instrument indices.")
        if len(index) == 0:
            return pd.Series([], index=index, dtype=float, name="score")
        dates = pd.DatetimeIndex(index.get_level_values("datetime"))
        instruments = index.get_level_values("instrument")
        calendar = pd.DatetimeIndex(D.calendar(end_time=dates.max(), freq="day"))
        locations = calendar.get_indexer(dates)
        if (locations < 0).any():
            raise ValueError("Prediction dates must be actual provider calendar sessions.")
        first = max(0, int(locations.min()) - self.context_length + 1)
        close_frame = D.features(
            sorted(set(instruments)),
            ["$close"],
            start_time=calendar[first],
            end_time=dates.max(),
            freq="day",
        )
        closes = close_frame["$close"]
        if not closes.index.is_unique:
            raise ValueError("Provider close prices contain duplicate instrument/date rows.")
        histories = {
            instrument: closes.xs(instrument, level="instrument").sort_index()
            for instrument in set(instruments)
        }
        contexts = []
        for instrument, position in zip(instruments, locations):
            # Each context stops at this prediction's date, even in a mixed-date batch.
            context_dates = calendar[max(0, position - self.context_length + 1) : position + 1]
            values = histories[instrument].reindex(context_dates).to_numpy(dtype=np.float32)
            valid = np.flatnonzero(np.isfinite(values))
            values = values[valid[0] :] if len(valid) else values[:0]
            if len(values) < 2 or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(
                    f"{instrument} at {calendar[position].date()} needs at least two positive "
                    "observed closes and no internal missing sessions in its context."
                )
            contexts.append(values)
        forecaster = self._get_forecaster()
        scores = []
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start : start + self.batch_size]
            forecasts = list(
                forecaster.predict_batch(
                    contexts=batch,
                    horizon=2,
                    return_quantiles=False,
                    use_symmetric_averaging=False,
                )
            )
            if len(forecasts) != len(batch):
                raise ValueError("Official TimesFM3 returned a different number of forecasts.")
            for output in forecasts:
                forecast = np.asarray(output.forecast, dtype=np.float64)
                if (
                    forecast.shape != (2,)
                    or not np.isfinite(forecast).all()
                    or (forecast <= 0).any()
                ):
                    raise ValueError(
                        "Official TimesFM3 must return two finite positive forecast closes."
                    )
                score = forecast[1] / forecast[0] - 1
                if not np.isfinite(score):
                    raise ValueError("Forecast close ratio overflowed.")
                scores.append(score)
        return pd.Series(scores, index=index, name="score", dtype=float)
