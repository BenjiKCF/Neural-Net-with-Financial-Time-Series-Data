"""Shared CPU/GPU Qlib training for small time-series architectures.

Each sample contains only the historical window ending on its prediction date.
Labels are fetched separately during fitting and are never requested at inference.
Feature normalization belongs to the external, training-fitted Qlib handler.
"""

from __future__ import annotations

import copy
import math
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from torch import nn
from torch.utils.data import DataLoader, Dataset

from financial_ts.catalog import NEURAL_MODELS, build_mini_encoder


class _Windows(Dataset):
    """Turn Qlib's feature-only sampler into tensors without filling from the future."""

    def __init__(self, sampler: Any, d_feat: int, targets: np.ndarray | None = None):
        self.sampler = sampler
        self.d_feat = d_feat
        self.targets = targets

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int):
        features = np.asarray(self.sampler[index], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.d_feat or features.shape[0] == 0:
            raise ValueError(
                f"Expected a nonempty [time, {self.d_feat}] feature window; got {features.shape}. "
                "Use TSDatasetH and match d_feat to the handler's feature count."
            )
        padding = ~np.isfinite(features).any(axis=1)
        # The current-date readout token remains available even if all its inputs
        # are missing. This also prevents undefined attention for all-missing windows.
        padding[-1] = False
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        result = (torch.from_numpy(features), torch.from_numpy(padding))
        if self.targets is not None:
            return (*result, torch.tensor(self.targets[index], dtype=torch.float32))
        return result


class _TemporalEncoder(nn.Module):
    def __init__(self, d_feat: int, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(d_feat, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        embedded = self.projection(features)
        length = embedded.shape[1]
        positions = torch.arange(length, device=embedded.device, dtype=embedded.dtype).unsqueeze(1)
        rates = torch.exp(
            torch.arange(0, self.d_model, 2, device=embedded.device, dtype=embedded.dtype)
            * (-math.log(10_000.0) / self.d_model)
        )
        positional = torch.zeros(length, self.d_model, device=embedded.device, dtype=embedded.dtype)
        positional[:, 0::2] = torch.sin(positions * rates)
        positional[:, 1::2] = torch.cos(positions * rates[: self.d_model // 2])
        # Every token is <= the sample's prediction date; no future token is present.
        encoded = self.encoder(embedded + positional, src_key_padding_mask=padding)
        return self.head(encoded[:, -1, :]).squeeze(-1)


class SequenceForecastModel(Model):
    """Regress one future return from a Qlib ``TSDatasetH`` historical window.

    ``device='auto'`` chooses CUDA when available, otherwise CPU. Target mean and
    scale are fitted on the training segment only; predictions are returned in
    the original label units. ``best_epoch_`` is one-based, and ``evals_result_``
    contains sample-weighted train/validation MSE in original label units.
    """

    def __init__(
        self,
        d_feat: int = 157,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 30,
        early_stop: int = 5,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        seed: int = 42,
        device: str = "auto",
        n_jobs: int = 0,
        num_threads: int = 2,
        architecture: str = "transformer",
        architecture_kwargs: dict | None = None,
        aux_loss_weight: float = 0.01,
    ):
        for name, value in {
            "d_feat": d_feat,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "n_epochs": n_epochs,
            "early_stop": early_stop,
            "batch_size": batch_size,
            "num_threads": num_threads,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not isinstance(n_jobs, int) or n_jobs < 0:
            raise ValueError("n_jobs must be a nonnegative integer")
        if not np.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive)")
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay nonnegative, both finite")
        if architecture not in NEURAL_MODELS:
            raise ValueError(f"architecture must be one of {NEURAL_MODELS}")
        if architecture_kwargs is not None and not isinstance(architecture_kwargs, dict):
            raise ValueError("architecture_kwargs must be a mapping")
        if not np.isfinite(aux_loss_weight) or aux_loss_weight < 0:
            raise ValueError("aux_loss_weight must be finite and nonnegative")
        if set(architecture_kwargs or {}) & {"d_feat", "d_model", "nhead", "num_layers", "dropout"}:
            raise ValueError("Set common dimensions directly, not inside architecture_kwargs")
        if architecture == "transformer" and architecture_kwargs:
            raise ValueError("The plain Transformer has no additional architecture_kwargs")

        self.d_feat, self.d_model, self.nhead = d_feat, d_model, nhead
        self.num_layers, self.dropout = num_layers, dropout
        self.n_epochs, self.early_stop, self.batch_size = n_epochs, early_stop, batch_size
        self.lr, self.weight_decay, self.seed = lr, weight_decay, seed
        self.n_jobs, self.num_threads = n_jobs, num_threads
        self.device = self._resolve_device(device)
        self.architecture = architecture
        self.architecture_kwargs = copy.deepcopy(architecture_kwargs or {})
        self.aux_loss_weight = aux_loss_weight
        # These attributes are public so Qlib's Serializable retains them.
        self.model: nn.Module | None = None
        self.fitted = False
        self.target_mean_ = 0.0
        self.target_scale_ = 1.0
        self.evals_result_: dict[str, list[float]] = {"train": [], "valid": []}
        self.best_epoch_: int | None = None
        self.parameter_count_: int | None = None
        self.fit_seconds_: float | None = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda" or device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA is unavailable. Install a CUDA-enabled PyTorch build or use device='cpu'."
                )
            try:
                resolved = torch.device(device)
            except (RuntimeError, ValueError) as error:
                raise ValueError(
                    "Use device='auto', 'cpu', 'cuda', or an available 'cuda:N'."
                ) from error
            if resolved.index is not None and resolved.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {resolved.index} is unavailable; check torch.cuda.device_count()."
                )
            return resolved
        raise ValueError("Use device='auto', 'cpu', 'cuda', or an available 'cuda:N'.")

    def _features(self, dataset: Any, segment: str | slice, data_key: str):
        sampler = dataset.prepare(segment, col_set="feature", data_key=data_key, dtype=np.float32)
        if not hasattr(sampler, "get_index") or not hasattr(sampler, "config"):
            raise TypeError(
                "SequenceForecastModel requires TSDatasetH, which returns a TSDataSampler."
            )
        sampler.config(fillna_type="none")
        index = sampler.get_index()
        if len(index) != len(sampler) or index.has_duplicates:
            raise ValueError(f"{segment!r} must have one unique index entry per sample")
        if len(sampler):
            _Windows(sampler, self.d_feat)[0]  # Validate dimensions before constructing a loader.
        return sampler

    def _supervised(self, dataset: Any, segment: str):
        features = self._features(dataset, segment, DataHandlerLP.DK_L)
        if len(features) == 0:
            raise ValueError(f"The {segment} segment is empty")
        labels = dataset.prepare(
            segment, col_set="label", data_key=DataHandlerLP.DK_L, dtype=np.float64
        )
        if not hasattr(labels, "get_index") or not features.get_index().equals(labels.get_index()):
            raise ValueError(f"Feature and label indices do not match for {segment}")
        labels.config(fillna_type="none")
        targets = np.empty(len(labels), dtype=np.float64)
        for index in range(len(labels)):
            window = np.asarray(labels[index])
            if window.ndim != 2 or window.shape[1] != 1 or not window.shape[0]:
                raise ValueError("SequenceForecastModel supports exactly one label column")
            targets[index] = window[-1, 0]
        if not np.isfinite(targets).all():
            raise ValueError(
                f"The {segment} segment contains non-finite labels. Remove unavailable targets "
                "with a learning-only DropnaLabel processor and check label-horizon coverage."
            )
        return features, targets

    def _loader(self, windows: _Windows, *, shuffle: bool = False) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            windows,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.n_jobs,
            drop_last=False,
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    def _validation_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for features, padding, targets in loader:
                predicted = self.model(features.to(self.device), padding.to(self.device))
                loss = nn.functional.mse_loss(predicted, targets.to(self.device), reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Validation loss became non-finite; check features and learning rate"
                    )
                total += loss.item()
                count += len(targets)
        return total / count

    def fit(self, dataset: Any, evals_result: dict | None = None):
        """Train using learning data, select the best epoch on validation only."""
        started = time.perf_counter()
        self.fitted = False
        self.best_epoch_ = None
        self.evals_result_ = {"train": [], "valid": []}
        train_features, train_targets = self._supervised(dataset, "train")
        valid_features, valid_targets = self._supervised(dataset, "valid")
        self.target_mean_ = float(train_targets.mean())
        scale = float(train_targets.std(ddof=0))
        if not np.isfinite(self.target_mean_) or not np.isfinite(scale):
            raise ValueError("Target statistics overflowed; check the magnitude of training labels")
        self.target_scale_ = scale if scale > 1e-12 else 1.0
        normalized_train = (train_targets - self.target_mean_) / self.target_scale_
        normalized_valid = (valid_targets - self.target_mean_) / self.target_scale_
        if not np.isfinite(normalized_train).all() or not np.isfinite(normalized_valid).all():
            raise ValueError(
                "Target normalization produced non-finite values; check target magnitudes"
            )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.num_threads)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        dimensions = dict(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        if self.architecture == "transformer":
            self.model = _TemporalEncoder(**dimensions)
        else:
            self.model = build_mini_encoder(
                self.architecture, **dimensions, **self.architecture_kwargs
            )
        self.model = self.model.to(self.device)
        self.parameter_count_ = sum(parameter.numel() for parameter in self.model.parameters())
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        train_loader = self._loader(
            _Windows(train_features, self.d_feat, normalized_train), shuffle=True
        )
        valid_loader = self._loader(_Windows(valid_features, self.d_feat, normalized_valid))
        best_loss, stale_epochs, best_state = float("inf"), 0, None

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            total, count = 0.0, 0
            for features, padding, targets in train_loader:
                optimizer.zero_grad(set_to_none=True)
                predicted = self.model(features.to(self.device), padding.to(self.device))
                mse = nn.functional.mse_loss(predicted, targets.to(self.device))
                auxiliary = getattr(self.model, "aux_loss", 0.0)
                loss = mse + self.aux_loss_weight * auxiliary
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Training loss became non-finite; check features and learning rate"
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0, error_if_nonfinite=True
                )
                optimizer.step()
                # Compare forecasting error, not architecture-specific regularizers.
                total += mse.item() * len(targets)
                count += len(targets)

            valid_loss = self._validation_loss(valid_loader)
            self.evals_result_["train"].append(total / count * self.target_scale_**2)
            self.evals_result_["valid"].append(valid_loss * self.target_scale_**2)
            get_module_logger("SequenceForecastModel").info(
                "%s epoch %d/%d: train RMSE %.8f, validation RMSE %.8f",
                self.architecture,
                epoch,
                self.n_epochs,
                np.sqrt(self.evals_result_["train"][-1]),
                np.sqrt(self.evals_result_["valid"][-1]),
            )
            if valid_loss < best_loss:
                best_loss, stale_epochs, self.best_epoch_ = valid_loss, 0, epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= self.early_stop:
                    break

        if best_state is None:
            raise RuntimeError("No finite validation checkpoint was produced")
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.fitted = True
        self.fit_seconds_ = time.perf_counter() - started
        if evals_result is not None:
            evals_result.update(
                {name: values.copy() for name, values in self.evals_result_.items()}
            )
        return self

    def predict(self, dataset: Any, segment: str | slice = "test") -> pd.Series:
        """Return original-scale predictions, requiring no label columns."""
        if not self.fitted or self.model is None:
            raise ValueError("The model is not fitted. Call fit(dataset) before predict().")
        sampler = self._features(dataset, segment, DataHandlerLP.DK_I)
        index = sampler.get_index()
        if len(sampler) == 0:
            return pd.Series(index=index, dtype=np.float64, name="score")
        torch.set_num_threads(self.num_threads)
        self.model.eval()
        batches = []
        with torch.inference_mode():
            for features, padding in self._loader(_Windows(sampler, self.d_feat)):
                predicted = self.model(features.to(self.device), padding.to(self.device))
                batches.append(predicted.cpu().numpy().reshape(-1))
        values = np.concatenate(batches).astype(np.float64) * self.target_scale_ + self.target_mean_
        if not np.isfinite(values).all():
            raise ValueError("Inference produced non-finite predictions")
        return pd.Series(values, index=index, name="score")


class TemporalTransformerModel(SequenceForecastModel):
    """Backward-compatible name for the original plain-Transformer baseline."""
