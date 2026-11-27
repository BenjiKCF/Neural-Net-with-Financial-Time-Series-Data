"""Qlib daily-panel training with raw-label masks and validation-only selection.

Every optimizer step receives the complete eligible cross section for one date.
Missing labels remove loss terms, never stocks from the attention context. No
date subsampling or cross-date stock mixing is hidden inside ``batch_size``.
"""

from __future__ import annotations

import copy
import random
import time
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from qlib.data.dataset.handler import DataHandlerLP
from torch import nn

from financial_ts.architectures.cross_stock import CrossStockEncoder
from financial_ts.catalog import CROSS_SECTIONAL_MODELS
from financial_ts.model import SequenceForecastModel, _Windows
from financial_ts.objectives import daily_objective, portfolio_turnover_surrogate


class DailyPanel:
    """Stable, complete same-date groups over Qlib's historical-window sampler."""

    def __init__(self, sampler, d_feat: int, targets: np.ndarray | None = None):
        self.sampler = sampler
        self.index = sampler.get_index()
        if (
            not isinstance(self.index, pd.MultiIndex)
            or self.index.nlevels != 2
            or set(self.index.names) != {"datetime", "instrument"}
            or not self.index.is_unique
            or len(self.index) != len(sampler)
        ):
            raise ValueError("Daily panels need unique datetime/instrument sample indexes")
        dates = pd.DatetimeIndex(self.index.get_level_values("datetime"))
        instruments = self.index.get_level_values("instrument")
        if dates.hasnans or dates.tz is not None or not (dates == dates.normalize()).all():
            raise ValueError("Panel dates must be finite timezone-naive daily dates")
        if any(not isinstance(x, str) or not x for x in instruments):
            raise ValueError("Instrument identifiers must be nonempty strings")
        if targets is not None and (
            np.asarray(targets).ndim != 1 or len(targets) != len(self.index)
        ):
            raise ValueError("Target array must align to every inference-panel anchor")
        self.windows = _Windows(sampler, d_feat)
        self.targets = None if targets is None else np.asarray(targets, dtype=np.float64)
        self.groups = []
        for date in dates.unique().sort_values():
            positions = np.flatnonzero(dates == date)
            positions = positions[np.argsort(instruments.take(positions).to_numpy(), kind="stable")]
            self.groups.append((date, positions))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        date, positions = self.groups[item]
        windows = [self.windows[int(position)] for position in positions]
        result = {
            "date": date,
            "positions": positions,
            "instruments": tuple(self.index.get_level_values("instrument").take(positions)),
            "features": torch.stack([x[0] for x in windows]),
            "padding": torch.stack([x[1] for x in windows]),
        }
        if self.targets is not None:
            result["targets"] = torch.tensor(self.targets[positions], dtype=torch.float32)
        return result


class CrossSectionalForecastModel(SequenceForecastModel):
    """Daily Transformer, MASTER-style comparator, or regime-gated expert model.

    ``batch_size`` is accepted for workflow compatibility but never splits a
    cross section. Every date receives equal weight. Standard curves retain
    sample-weighted raw MSE for compatibility; ``objective_history_`` records
    the actual daily optimization/selection loss. ``selection_metric`` is
    declared as daily validation loss or sample-weighted validation MSE.
    """

    def __init__(
        self,
        d_feat=157,
        d_model=32,
        nhead=4,
        num_layers=1,
        dropout=0.0,
        n_epochs=6,
        early_stop=3,
        batch_size=512,
        lr=0.001,
        weight_decay=0.0001,
        seed=42,
        device="auto",
        n_jobs=0,
        num_threads=2,
        architecture="master_style",
        architecture_kwargs=None,
        aux_loss_weight=0.01,
        objective="huber",
        objective_kwargs=None,
        selection_metric="objective",
        pretrain_epochs=0,
        mask_probability=0.15,
        turnover_weight=0.0,
        precision="float32",
    ):
        if architecture not in CROSS_SECTIONAL_MODELS:
            raise ValueError(f"architecture must be one of {CROSS_SECTIONAL_MODELS}")
        if architecture_kwargs is not None and not isinstance(architecture_kwargs, dict):
            raise ValueError("architecture_kwargs must be a mapping")
        if objective_kwargs is not None and not isinstance(objective_kwargs, dict):
            raise ValueError("objective_kwargs must be a mapping")
        if selection_metric not in {"objective", "mse"}:
            raise ValueError("selection_metric must be objective or mse")
        if type(pretrain_epochs) is not int or pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be a nonnegative integer")
        if not 0 < mask_probability < 1 or not 0 <= turnover_weight < float("inf"):
            raise ValueError(
                "mask_probability must be in (0,1); turnover_weight must be finite >=0"
            )
        if precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        # Validate shared training settings without changing the historical adapter.
        super().__init__(
            d_feat=d_feat,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            n_epochs=n_epochs,
            early_stop=early_stop,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
            num_threads=num_threads,
            architecture="transformer",
            aux_loss_weight=aux_loss_weight,
        )
        self.architecture = architecture
        self.architecture_kwargs = copy.deepcopy(architecture_kwargs or {})
        self.objective, self.objective_kwargs = objective, copy.deepcopy(objective_kwargs or {})
        # Validate objectives now, including unknown settings, rather than after fitting.
        daily_objective(
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]),
            objective=objective,
            **self.objective_kwargs,
        )
        self.selection_metric = selection_metric
        self.pretrain_epochs, self.mask_probability = pretrain_epochs, mask_probability
        self.turnover_weight, self.precision = turnover_weight, precision
        self.network_config_ = self._network_config()
        self.objective_history_ = {
            "train": [],
            "valid": [],
            "base_forecast_loss": [],
            "routing_penalty": [],
            "turnover_penalty": [],
            "total_optimized_loss": [],
        }
        self.training_receipt_ = {}
        self.pretraining_receipt_ = {}
        self.routing_history_ = []

    def _network_config(self):
        options = {
            "cross_stock_attention": self.architecture != "daily_transformer",
            "market_context": self.architecture != "daily_transformer",
            "num_experts": 4 if self.architecture == "quantformer_moe" else 1,
        }
        options.update(self.architecture_kwargs)
        allowed = {"cross_stock_attention", "market_context", "num_experts", "uniform_routing"}
        if set(options) - allowed:
            raise ValueError(f"Unknown architecture options: {sorted(set(options) - allowed)}")
        if self.architecture == "daily_transformer" and (
            options["cross_stock_attention"]
            or options["market_context"]
            or options["num_experts"] != 1
        ):
            raise ValueError("daily_transformer is the temporal-only, one-expert control")
        return dict(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **options,
        )

    def constructor_config(self):
        names = (
            "d_feat",
            "d_model",
            "nhead",
            "num_layers",
            "dropout",
            "n_epochs",
            "early_stop",
            "batch_size",
            "lr",
            "weight_decay",
            "seed",
            "n_jobs",
            "num_threads",
            "architecture",
            "architecture_kwargs",
            "aux_loss_weight",
            "objective",
            "objective_kwargs",
            "selection_metric",
            "pretrain_epochs",
            "mask_probability",
            "turnover_weight",
            "precision",
        )
        return {
            **{name: copy.deepcopy(getattr(self, name)) for name in names},
            "device": str(self.device),
        }

    def _build_network(self):
        return CrossStockEncoder(**self.network_config_)

    def _autocast(self):
        return (
            torch.autocast(self.device.type, dtype=torch.bfloat16)
            if self.precision == "bfloat16"
            else nullcontext()
        )

    def _panel(self, dataset, segment, *, supervised):
        sampler = self._features(dataset, segment, DataHandlerLP.DK_I)
        if not len(sampler):
            raise ValueError(f"The {segment} segment is empty")
        targets = None
        if supervised:
            labels = dataset.prepare(
                segment, col_set="label", data_key=DataHandlerLP.DK_L, dtype=np.float64
            )
            if not hasattr(labels, "get_index") or not hasattr(labels, "config"):
                raise ValueError("Daily training requires a Qlib label sampler")
            if labels.get_index().has_duplicates:
                raise ValueError("Duplicate supervised label endpoints")
            labels.config(fillna_type="none")
            positions = sampler.get_index().get_indexer(labels.get_index())
            if (positions < 0).any():
                raise ValueError("Feature panel is missing supervised label endpoints")
            targets = np.full(len(sampler), np.nan)
            for position, row in zip(positions, labels):
                row = np.asarray(row)
                if row.ndim != 2 or row.shape[1] != 1 or not len(row):
                    raise ValueError("Native daily workflow requires exactly one raw-return label")
                targets[position] = row[-1, 0]
            if np.isinf(targets).any() or not np.isfinite(targets).any():
                raise ValueError("Training needs finite raw targets; infinite labels are invalid")
        return DailyPanel(sampler, self.d_feat, targets)

    def _pretrain(self, panel):
        self.pretraining_receipt_ = {
            "epochs": self.pretrain_epochs,
            "data_segment": "train",
            "first_anchor": str(panel.groups[0][0].date()),
            "last_anchor": str(panel.groups[-1][0].date()),
            "mask_probability": self.mask_probability,
            "losses": [],
            "uses_validation_or_test_features": False,
        }
        if not self.pretrain_epochs:
            return
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        generator = torch.Generator().manual_seed(self.seed + 1000)
        started = time.perf_counter()
        for _ in range(self.pretrain_epochs):
            self.model.train()
            losses = []
            for day in range(len(panel)):
                batch = panel[day]
                features, padding = (
                    batch["features"].to(self.device),
                    batch["padding"].to(self.device),
                )
                mask = (torch.rand(features.shape, generator=generator) < self.mask_probability).to(
                    self.device
                )
                mask &= ~padding[..., None]
                if not mask.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    reconstructed = self.model.reconstruct(features.masked_fill(mask, 0), padding)
                loss = nn.functional.mse_loss(reconstructed.float()[mask], features[mask])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite masked-reconstruction loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.pretraining_receipt_["losses"].append(float(np.mean(losses)))
        self.pretraining_receipt_["seconds"] = time.perf_counter() - started

    def _evaluate(self, panel):
        self.model.eval()
        losses, square_error, count = [], 0.0, 0
        undefined = 0
        with torch.inference_mode():
            for i in range(len(panel)):
                batch = panel[i]
                targets = batch["targets"].to(self.device)
                mask = torch.isfinite(targets)
                if not mask.any():
                    continue
                with self._autocast():
                    prediction = self.model(
                        batch["features"].to(self.device), batch["padding"].to(self.device)
                    )
                prediction = prediction.float()
                normalized = (targets - self.target_mean_) / self.target_scale_
                result = daily_objective(
                    prediction, normalized, objective=self.objective, **self.objective_kwargs
                )
                losses.append(float(result.loss))
                undefined += result.counts["undefined_ic_dates"]
                square_error += float(((prediction[mask] - normalized[mask]) ** 2).sum())
                count += int(mask.sum())
        if not losses:
            raise ValueError("No mature validation targets")
        return float(np.mean(losses)), square_error / count * self.target_scale_**2, undefined

    def fit(self, dataset, evals_result=None):
        started = time.perf_counter()
        self.fitted = False
        self.best_epoch_ = None
        self.evals_result_ = {"train": [], "valid": []}
        self.objective_history_ = {
            "train": [],
            "valid": [],
            "base_forecast_loss": [],
            "routing_penalty": [],
            "turnover_penalty": [],
            "total_optimized_loss": [],
        }
        self.routing_history_ = []
        train = self._panel(dataset, "train", supervised=True)
        valid = self._panel(dataset, "valid", supervised=True)
        finite_train = train.targets[np.isfinite(train.targets)]
        self.target_mean_ = float(finite_train.mean())
        scale = float(finite_train.std())
        if not np.isfinite([self.target_mean_, scale]).all():
            raise ValueError("Training target scale overflowed")
        self.target_scale_ = scale if scale > 1e-12 else 1.0
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.num_threads)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.model = self._build_network().to(self.device)
        self.parameter_count_ = sum(x.numel() for x in self.model.parameters())
        self._pretrain(train)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        best, stale, best_state = float("inf"), 0, None
        self.training_receipt_ = {
            "batching": "one complete decision-date cross section per update",
            "date_weighting": "equal",
            "stock_sampling": "none",
            "train_dates": len(train),
            "valid_dates": len(valid),
            "train_anchors": len(train.index),
            "finite_train_targets": len(finite_train),
            "objective": self.objective,
            "objective_kwargs": self.objective_kwargs,
            "selection_metric": self.selection_metric,
            "precision": self.precision,
            "turnover_weight": self.turnover_weight,
            "optimized_loss": "base forecast objective + weighted routing penalty + weighted turnover penalty",
            "validation_selection_includes_regularizers": False,
            "turnover_assumption": "softmax long-only weights; all current names tradable; absent names frozen; proportional turnover surrogate, not executed cost",
            "epochs": [],
        }
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            losses, square_error, count, skipped, undefined = [], 0.0, 0, 0, 0
            base_losses, routing_penalties, turnover_penalties = [], [], []
            previous = {}
            routing = []
            for i in range(len(train)):
                batch = train[i]
                targets = batch["targets"].to(self.device)
                finite = torch.isfinite(targets)
                if not finite.any():
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    prediction = self.model(
                        batch["features"].to(self.device), batch["padding"].to(self.device)
                    )
                prediction = prediction.float()
                normalized = (targets - self.target_mean_) / self.target_scale_
                result = daily_objective(
                    prediction, normalized, objective=self.objective, **self.objective_kwargs
                )
                loss = result.loss + self.aux_loss_weight * self.model.aux_loss
                routing_penalty = float((self.aux_loss_weight * self.model.aux_loss).detach())
                turnover_penalty = 0.0
                undefined += result.counts["undefined_ic_dates"]
                if self.turnover_weight:
                    names = batch["instruments"]
                    absent = sorted(set(previous) - set(names))
                    union = [*names, *absent]
                    union_scores = torch.cat([prediction, prediction.new_zeros(len(absent))])
                    holdings = prediction.new_tensor([previous.get(name, 0.0) for name in union])
                    tradable = torch.arange(len(union), device=self.device) < len(names)
                    turnover, weights = portfolio_turnover_surrogate(
                        union_scores, holdings, tradable=tradable
                    )
                    loss = loss + self.turnover_weight * turnover
                    turnover_penalty = float((self.turnover_weight * turnover).detach())
                    previous = dict(zip(union, weights.detach().cpu().tolist()))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite daily training objective")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                base_losses.append(float(result.loss.detach()))
                routing_penalties.append(routing_penalty)
                turnover_penalties.append(turnover_penalty)
                square_error += float(
                    ((prediction[finite] - normalized[finite]) ** 2).sum().detach()
                )
                count += int(finite.sum())
                routing.append(self.model.routing_diagnostics_["weights"])
            valid_objective, valid_mse, valid_undefined = self._evaluate(valid)
            self.evals_result_["train"].append(square_error / count * self.target_scale_**2)
            self.evals_result_["valid"].append(valid_mse)
            self.objective_history_["train"].append(float(np.mean(losses)))
            self.objective_history_["valid"].append(valid_objective)
            self.objective_history_["base_forecast_loss"].append(float(np.mean(base_losses)))
            self.objective_history_["routing_penalty"].append(float(np.mean(routing_penalties)))
            self.objective_history_["turnover_penalty"].append(float(np.mean(turnover_penalties)))
            self.objective_history_["total_optimized_loss"].append(float(np.mean(losses)))
            self.training_receipt_["epochs"].append(
                {
                    "epoch": epoch,
                    "skipped_no_target_dates": skipped,
                    "train_undefined_ic_dates": undefined,
                    "valid_undefined_ic_dates": valid_undefined,
                }
            )
            self.routing_history_.append(
                {
                    "epoch": epoch,
                    "mean_expert_weights": np.mean(routing, axis=0).tolist(),
                    "daily_expert_weights": routing,
                }
            )
            selected = valid_mse if self.selection_metric == "mse" else valid_objective
            if selected < best:
                best, stale, self.best_epoch_ = selected, 0, epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                stale += 1
                if stale >= self.early_stop:
                    break
        if best_state is None:
            raise ValueError("No finite validation checkpoint")
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.fitted = True
        self.fit_seconds_ = time.perf_counter() - started
        self.training_receipt_["selected_epoch"] = self.best_epoch_
        self.training_receipt_["selected_validation_metric"] = best
        if evals_result is not None:
            evals_result.update(copy.deepcopy(self.evals_result_))
        return self

    def predict(self, dataset, segment="test"):
        if not self.fitted or self.model is None:
            raise ValueError("Call fit before predicting")
        sampler = self._features(dataset, segment, DataHandlerLP.DK_I)
        if not len(sampler):
            return pd.Series(index=sampler.get_index(), dtype=np.float64, name="score")
        panel = DailyPanel(sampler, self.d_feat)
        result = np.empty(len(panel.index), dtype=np.float64)
        self.model.eval()
        torch.set_num_threads(self.num_threads)
        with torch.inference_mode():
            for i in range(len(panel)):
                batch = panel[i]
                with self._autocast():
                    prediction = self.model(
                        batch["features"].to(self.device), batch["padding"].to(self.device)
                    )
                result[batch["positions"]] = (
                    prediction.float().cpu().numpy() * self.target_scale_ + self.target_mean_
                )
        if not np.isfinite(result).all():
            raise ValueError("Inference produced nonfinite daily-panel scores")
        return pd.Series(result, index=panel.index, name="score")
