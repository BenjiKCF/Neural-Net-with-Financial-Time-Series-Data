"""Independent DoubleAdapt paper-method comparator with mature-label gates.

Implements the data/model adaptation mechanism in Zhao et al., KDD 2023,
arXiv:2306.09862v3, equations 6--21. No upstream source is vendored: the inspected
official API repository at e20b6e46ccbd2b6b8180ff3db62f68b16721684c has no license
file. This implementation uses exact differentiable inner gradients, a compact
MLP on this repository's current-date Qlib factors, and positive label scales.
It is not a reproduction of the authors' GRU/Alpha360 experiment or scores.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import functional_call

PAPER = "https://arxiv.org/abs/2306.09862v3"
UPSTREAM = "https://github.com/SJTU-DMTai/DoubleAdapt/tree/e20b6e46ccbd2b6b8180ff3db62f68b16721684c"


def _day(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tz is not None or result != result.normalize():
        raise ValueError("Expected finite timezone-naive session dates")
    return result


@dataclass
class AdaptationBatch:
    index: pd.MultiIndex
    features: torch.Tensor
    targets: torch.Tensor
    label_endpoints: np.ndarray

    def validate(self, *, as_of):
        cutoff = _day(as_of)
        if (not isinstance(self.index, pd.MultiIndex) or list(self.index.names) != ["datetime", "instrument"]
                or not self.index.is_unique or len(self.index) == 0):
            raise ValueError("Adaptation batches need unique nonempty date/instrument identities")
        dates = pd.DatetimeIndex(self.index.get_level_values("datetime"))
        for date in dates:
            _day(date)
        if dates.max() > cutoff:
            raise ValueError("Feature dates exceed the fitting cutoff")
        if (self.features.ndim != 2 or len(self.features) != len(self.index)
                or self.targets.shape != (len(self.index),)
                or np.asarray(self.label_endpoints).shape != (len(self.index),)):
            raise ValueError("Adaptation feature, target and endpoint shapes do not align")
        if not torch.isfinite(self.features).all() or torch.isinf(self.targets).any():
            raise ValueError("Features must be finite and labels cannot be infinite")
        finite = torch.isfinite(self.targets).detach().cpu().numpy()
        endpoints = np.asarray(self.label_endpoints, dtype="datetime64[ns]")
        if not finite.any():
            raise ValueError("No finite mature labels in adaptation batch")
        if (np.isnat(endpoints[finite]).any() or (endpoints[finite] > cutoff.to_datetime64()).any()
                or (endpoints[finite] <= dates.to_numpy()[finite]).any()):
            raise ValueError("Only matured labels with valid future information endpoints may be used")
        return self

    def finite(self):
        mask = torch.isfinite(self.targets)
        return self.features[mask], self.targets[mask]


@dataclass
class AdaptationTask:
    name: str
    support: AdaptationBatch
    query: AdaptationBatch

    def validate(self, *, as_of):
        if not self.name:
            raise ValueError("Task identity is required")
        self.support.validate(as_of=as_of)
        self.query.validate(as_of=as_of)
        first_query = min(self.query.index.get_level_values("datetime"))
        if max(self.support.index.get_level_values("datetime")) >= first_query:
            raise ValueError("Support observations must precede query observations")
        finite = torch.isfinite(self.support.targets).detach().cpu().numpy()
        if (np.asarray(self.support.label_endpoints, dtype="datetime64[ns]")[finite] >= first_query.to_datetime64()).any():
            raise ValueError("Support labels must mature strictly before query inference")
        return self


class PrototypeDataAdapter(nn.Module):
    """Residual affine feature heads and input-conditioned affine label heads."""

    def __init__(self, d_feat, heads=4, temperature=1.0):
        super().__init__()
        if type(d_feat) is not int or d_feat < 1 or type(heads) is not int or heads < 1:
            raise ValueError("Feature count and head count must be positive integers")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        self.d_feat, self.heads, self.temperature = d_feat, heads, temperature
        self.prototypes = nn.Parameter(torch.randn(heads, d_feat))
        self.feature_weight = nn.Parameter(torch.zeros(heads, d_feat, d_feat))
        self.feature_bias = nn.Parameter(torch.zeros(heads, d_feat))
        self.label_log_scale = nn.Parameter(torch.zeros(heads))
        self.label_bias = nn.Parameter(torch.zeros(heads))

    def weights(self, features):
        features = nn.functional.normalize(features, dim=-1, eps=1e-8)
        prototypes = nn.functional.normalize(self.prototypes, dim=-1, eps=1e-8)
        return torch.softmax(features @ prototypes.T / self.temperature, dim=-1)

    def forward(self, features):
        weights = self.weights(features)
        transformed = torch.einsum("nf,hof->nho", features, self.feature_weight) + self.feature_bias
        return features + torch.einsum("nh,nho->no", weights, transformed)

    def transform_labels(self, features, labels):
        weights = self.weights(features)
        values = labels[:, None] * self.label_log_scale.exp() + self.label_bias
        return (weights * values).sum(-1)

    def restore_labels(self, features, predictions):
        # Equation 12 is a weighted mixture of headwise inverses. It is not
        # generally the algebraic inverse of the mixture in equation 11.
        weights = self.weights(features)
        values = (predictions[:, None] - self.label_bias) / self.label_log_scale.exp()
        return (weights * values).sum(-1)


def _network(d_feat, hidden):
    return nn.Sequential(nn.Linear(d_feat, hidden), nn.Tanh(), nn.Linear(hidden, 1))


class DoubleAdaptModel(nn.Module):
    """Meta-learn model initialization and data adapters through an inner update."""

    def __init__(self, d_feat, *, hidden=16, heads=4, temperature=1.0, inner_lr=0.01,
                 model_meta_lr=0.001, data_meta_lr=0.001, label_regularization=0.1, seed=42):
        super().__init__()
        if type(hidden) is not int or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        for value in (inner_lr, model_meta_lr, data_meta_lr):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Learning rates must be finite and positive")
        if not np.isfinite(label_regularization) or label_regularization < 0:
            raise ValueError("Regularization must be finite and nonnegative")
        torch.manual_seed(seed)
        self.forecaster = _network(d_feat, hidden)
        self.adapter = PrototypeDataAdapter(d_feat, heads, temperature)
        self.inner_lr, self.model_meta_lr, self.data_meta_lr = inner_lr, model_meta_lr, data_meta_lr
        self.label_regularization = label_regularization
        self.register_buffer("label_mean", torch.tensor(0.0))
        self.register_buffer("label_scale", torch.tensor(1.0))
        self.normalization_fitted = False
        self.pending = None
        self.receipts = []
        self.last_causal_update = None

    def fit_normalization(self, tasks, *, as_of):
        if self.normalization_fitted:
            raise ValueError("The training-only target normalizer is immutable")
        rows = {}
        for task in tasks:
            task.validate(as_of=as_of)
            for batch in (task.support, task.query):
                for key, target in zip(batch.index, batch.targets.detach().cpu().tolist()):
                    if np.isfinite(target):
                        if key in rows and rows[key] != target:
                            raise ValueError("A training label has inconsistent duplicate values")
                        rows[key] = target
        if not rows:
            raise ValueError("No finite training labels for normalization")
        values = torch.tensor(list(rows.values()), dtype=self.label_mean.dtype)
        self.label_mean.copy_(values.mean())
        self.label_scale.copy_(values.std(unbiased=False).clamp_min(1e-6))
        self.normalization_fitted = True

    def _adapt(self, support, *, differentiable):
        if not self.normalization_fitted:
            raise ValueError("Fit the training-only normalizer first")
        x, raw_y = support.finite()
        y = (raw_y - self.label_mean) / self.label_scale
        transformed_y = self.adapter.transform_labels(x, y)
        params = dict(self.forecaster.named_parameters())
        prediction = functional_call(self.forecaster, params, (self.adapter(x),)).squeeze(-1)
        loss = nn.functional.mse_loss(prediction, transformed_y)
        gradients = torch.autograd.grad(loss, tuple(params.values()), create_graph=differentiable)
        adapted = {name: value - self.inner_lr * gradient for (name, value), gradient in zip(params.items(), gradients)}
        return adapted, loss, x, y, transformed_y

    def _forecast(self, parameters, features):
        intermediate = functional_call(self.forecaster, parameters, (self.adapter(features),)).squeeze(-1)
        return self.adapter.restore_labels(features, intermediate)

    def _state_hash(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def begin_task(self, support: AdaptationBatch, *, inference_start, as_of):
        if self.pending is not None:
            raise ValueError("Finish or discard the pending task before another update")
        support.validate(as_of=as_of)
        start = _day(inference_start)
        if self.last_causal_update is not None and _day(as_of) < self.last_causal_update:
            raise ValueError("A model updated on later labels cannot be used for historical inference")
        if _day(as_of) >= start:
            raise ValueError("Support cutoff must precede the inference period")
        finite = torch.isfinite(support.targets).detach().cpu().numpy()
        if (np.asarray(support.label_endpoints, dtype="datetime64[ns]")[finite] >= start.to_datetime64()).any():
            raise ValueError("Support labels have not matured before inference")
        adapted, loss, _, _, _ = self._adapt(support, differentiable=False)
        self.pending = {"support": copy.deepcopy(support), "parameters": {k: v.detach() for k, v in adapted.items()},
                        "inference_start": start, "state_hash": self._state_hash(), "emitted": {}, "features": {},
                        "support_loss": float(loss.detach())}
        return {"status": "adapted", "support_loss": float(loss.detach()), "state_sha256": self.pending["state_hash"]}

    def predict(self, features, index):
        if self.pending is None:
            raise ValueError("Begin a mature support task before inference")
        if self._state_hash() != self.pending["state_hash"]:
            raise ValueError("Model state changed before the pending task was finalized")
        if (not isinstance(index, pd.MultiIndex) or not index.is_unique or list(index.names) != ["datetime", "instrument"]
                or features.ndim != 2 or len(features) != len(index) or not torch.isfinite(features).all()):
            raise ValueError("Prediction needs finite features and unique date/instrument identities")
        if min(index.get_level_values("datetime")) < self.pending["inference_start"]:
            raise ValueError("Prediction dates precede the registered inference period")
        with torch.no_grad():
            scores = self._forecast(self.pending["parameters"], features) * self.label_scale + self.label_mean
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite DoubleAdapt predictions")
        for key, value, row in zip(index, scores.tolist(), features):
            if key in self.pending["emitted"]:
                if not torch.equal(row, self.pending["features"][key]):
                    raise ValueError("Published prediction features cannot change within a task")
                if self.pending["emitted"][key] != value:
                    raise ValueError("A published prediction cannot change within a task")
        for key, value, row in zip(index, scores.tolist(), features):
            self.pending["emitted"][key] = value
            self.pending["features"][key] = row.detach().clone()
        return scores.detach().clone()

    def finish_task(self, query: AdaptationBatch, *, as_of, task_name="online"):
        if self.pending is None:
            raise ValueError("No pending task to finish")
        if self._state_hash() != self.pending["state_hash"]:
            raise ValueError("Pending task model state changed")
        task = AdaptationTask(task_name, self.pending["support"], query).validate(as_of=as_of)
        if set(query.index) != set(self.pending["emitted"]):
            raise ValueError("Query labels must correspond exactly to already emitted predictions")
        if any(not torch.equal(row, self.pending["features"][key]) for key, row in zip(query.index, query.features)):
            raise ValueError("Query features changed after their predictions were emitted")
        receipt = self._meta_step(task, as_of=as_of)
        receipt["predictions_locked_before_labels"] = True
        self.pending = None
        self.last_causal_update = _day(as_of)
        return receipt

    def _meta_step(self, task: AdaptationTask, *, as_of):
        task.validate(as_of=as_of)
        adapted, inner_loss, support_x, support_y, transformed_y = self._adapt(task.support, differentiable=True)
        query_x, raw_y = task.query.finite()
        prediction = self._forecast(adapted, query_x)
        query_loss = nn.functional.mse_loss(prediction, (raw_y - self.label_mean) / self.label_scale)
        regularization = nn.functional.mse_loss(transformed_y, support_y)
        outer = query_loss + self.label_regularization * regularization
        named = [("forecaster." + name, p) for name, p in self.forecaster.named_parameters()]
        named += [("adapter." + name, p) for name, p in self.adapter.named_parameters()]
        gradients = torch.autograd.grad(outer, tuple(p for _, p in named))
        if not torch.isfinite(outer) or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("Nonfinite meta update; original weights retained")
        changes = []
        for (name, parameter), gradient in zip(named, gradients):
            rate = self.model_meta_lr if name.startswith("forecaster.") else self.data_meta_lr
            change = parameter.detach() - rate * gradient.detach()
            if not torch.isfinite(change).all():
                raise ValueError("Nonfinite meta parameters; original weights retained")
            if name.endswith("label_log_scale") and (change.abs() > 20).any():
                raise ValueError("Unsafe label inverse scale; original weights retained")
            changes.append(change)
        with torch.no_grad():
            for (_, parameter), change in zip(named, changes):
                parameter.copy_(change)
        receipt = {"task": task.name, "as_of": str(_day(as_of).date()), "support_loss": float(inner_loss.detach()),
                   "query_loss": float(query_loss.detach()), "label_regularization": float(regularization.detach()),
                   "gradient_mode": "exact_second_order_through_inner_update",
                   "model_gradient_norm": float(torch.stack([g.square().sum() for (n, _), g in zip(named, gradients) if n.startswith("forecaster.")]).sum().sqrt()),
                   "feature_adapter_gradient_norm": float(torch.stack([g.square().sum() for (n, _), g in zip(named, gradients) if "adapter." in n and "label_" not in n]).sum().sqrt()),
                   "label_adapter_gradient_norm": float(torch.stack([g.square().sum() for (n, _), g in zip(named, gradients) if "label_" in n]).sum().sqrt()),
                   "state_sha256": self._state_hash()}
        self.receipts.append(receipt)
        return receipt


def _query_cutoff(batch):
    finite = torch.isfinite(batch.targets).detach().cpu().numpy()
    return pd.Timestamp(np.asarray(batch.label_endpoints, dtype="datetime64[ns]")[finite].max())


def _simulate(model, tasks, *, update, on_prediction=None):
    frames = []
    for task in tasks:
        start = min(task.query.index.get_level_values("datetime"))
        support_cutoff = _query_cutoff(task.support)
        model.begin_task(task.support, inference_start=start, as_of=support_cutoff)
        scores = model.predict(task.query.features, task.query.index)
        frame = pd.DataFrame({"score": scores.numpy(), "label": task.query.targets.numpy()}, index=task.query.index)
        frames.append(frame)
        if on_prediction is not None:
            on_prediction(task.name, frame[["score"]].copy())
        if update:
            model.finish_task(task.query, as_of=_query_cutoff(task.query), task_name=task.name)
        else:
            model.pending = None
    return pd.concat(frames).sort_index()


def fit_doubleadapt(train_tasks, valid_tasks, *, as_of, epochs=2, **model_kwargs):
    """Select offline state on chronological validation; then mature that history."""
    if not train_tasks or not valid_tasks or type(epochs) is not int or epochs < 1:
        raise ValueError("Nonempty train/validation task sets and positive epochs are required")
    for task in train_tasks + valid_tasks:
        task.validate(as_of=as_of)
    train_end = max(max(t.query.index.get_level_values("datetime")) for t in train_tasks)
    valid_start = min(min(t.support.index.get_level_values("datetime")) for t in valid_tasks)
    if train_end >= valid_start or max(_query_cutoff(t.query) for t in train_tasks) >= valid_start:
        raise ValueError("Training tasks and label endpoints must precede validation")
    model = DoubleAdaptModel(train_tasks[0].support.features.shape[1], **model_kwargs)
    model.fit_normalization(train_tasks, as_of=as_of)
    best_state, best_metric, best_epoch, history = None, float("inf"), 0, []
    for epoch in range(epochs + 1):
        if epoch:
            for task in train_tasks:
                model._meta_step(task, as_of=as_of)
        copy_for_validation = copy.deepcopy(model)
        predictions = _simulate(copy_for_validation, valid_tasks, update=True)
        finite = predictions.dropna()
        metric = float(np.mean((finite["score"] - finite["label"]) ** 2))
        history.append({"epoch": epoch, "validation_mse": metric})
        if np.isfinite(metric) and metric < best_metric:
            best_metric, best_epoch, best_state = metric, epoch, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("No finite offline DoubleAdapt checkpoint")
    model.load_state_dict(best_state)
    # Like the paper's online initialization, the now-mature validation tasks
    # become history after checkpoint selection. Their forecasts are not final OOS.
    _simulate(model, valid_tasks, update=True)
    return model, {"selected_epoch": best_epoch, "validation_mse": best_metric, "history": history,
                   "normalizer_fit": "meta-training labels only", "validation_updated_after_selection": True,
                   "method": "independent_DoubleAdapt_paper_method", "upstream_reproduction": False}


def _merge_batches(batches):
    seen, indices, features, targets, endpoints = set(), [], [], [], []
    for batch in batches:
        take = np.array([key not in seen for key in batch.index])
        seen.update(batch.index)
        if take.any():
            indices.append(batch.index[take])
            features.append(batch.features[take])
            targets.append(batch.targets[take])
            endpoints.append(np.asarray(batch.label_endpoints)[take])
    return AdaptationBatch(indices[0].append(indices[1:]), torch.cat(features), torch.cat(targets), np.concatenate(endpoints))


def _supervised_fit(batch, normalizer, *, hidden, seed, epochs, lr, initial=None):
    torch.manual_seed(seed)
    model = _network(batch.features.shape[1], hidden)
    if initial is not None:
        model.load_state_dict(initial)
    x, raw_y = batch.finite()
    y = (raw_y - normalizer.label_mean) / normalizer.label_scale
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x).squeeze(-1), y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite supervised adaptation control")
        loss.backward()
        optimizer.step()
    return model


def run_doubleadapt_research(provider, output, spec):
    """Run a registered, source-pinned Qlib/raw-return comparator campaign."""
    import qlib
    from qlib.data import D
    from qlib.utils import init_instance_by_config

    from financial_ts.advanced_research import multi_horizon_targets
    from financial_ts.bundles import export_preprocessing
    from financial_ts.rolling import paired_date_block_intervals
    from financial_ts.workflow import (
        _prediction_statistics,
        _provider_fingerprint,
        _read_calendar,
        _save_json,
        build_task,
    )

    provider, output = Path(provider).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("DoubleAdapt campaign requires a new output directory")
    allowed = {"name", "purpose", "start", "region", "universe", "max_instruments", "warmup_sessions",
               "support_sessions", "query_sessions", "train_tasks", "valid_tasks", "online_tasks",
               "seed", "epochs", "num_threads", "model", "bootstrap_repetitions", "bootstrap_block_sessions"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Unsupported DoubleAdapt campaign fields")
    counts = {"support_sessions": 5, "query_sessions": 5, "train_tasks": 3,
              "valid_tasks": 1, "online_tasks": 3, "max_instruments": 20,
              "warmup_sessions": 60, "epochs": 2, "num_threads": 2}
    counts.update({key: spec[key] for key in counts if key in spec})
    if any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError("Task, data and optimization budgets must be positive integers")
    calendar = _read_calendar(provider)
    first = int(calendar.searchsorted(_day(spec.get("start", "2023-01-03"))))
    if first < counts["warmup_sessions"]:
        raise ValueError("Insufficient declared feature warmup history")
    task_ranges, cursor = [], first
    groups = ["train"] * counts["train_tasks"] + ["valid"] * counts["valid_tasks"] + ["online"] * counts["online_tasks"]
    for ordinal, group in enumerate(groups):
        support_start, support_end = cursor, cursor + counts["support_sessions"] - 1
        query_start, query_end = support_end + 3, support_end + 3 + counts["query_sessions"] - 1
        if query_end + 2 >= len(calendar):
            raise ValueError("Observed calendar is too short for mature task labels")
        task_ranges.append({"name": f"{group}-{ordinal:03d}", "group": group,
                            "support": [str(calendar[support_start].date()), str(calendar[support_end].date())],
                            "query": [str(calendar[query_start].date()), str(calendar[query_end].date())],
                            "support_cutoff": str(calendar[support_end + 2].date()),
                            "query_mature_at": str(calendar[query_end + 2].date())})
        cursor = query_end + 3
    region = spec.get("region", "cn")
    if region not in {"cn", "us"}:
        raise ValueError("Region must be cn or us")
    universe = spec.get("universe", "csi300" if region == "cn" else "all")
    qlib.init(provider_uri=str(provider), region=region, kernels=1, expression_cache=None, dataset_cache=None)
    final = pd.Timestamp(task_ranges[-1]["query_mature_at"])
    membership = D.list_instruments(D.instruments(universe), start_time=calendar[first - counts["warmup_sessions"]], end_time=final, as_list=False)
    selected = sorted(symbol for symbol, intervals in membership.items()
                      if any(pd.Timestamp(start) <= calendar[first] <= pd.Timestamp(end) for start, end in intervals))[:counts["max_instruments"]]
    if len(selected) < 2:
        raise ValueError("Need at least two initially eligible instruments")
    segments = {group: [next(row["support"][0] for row in task_ranges if row["group"] == group),
                        next(row["query"][1] for row in reversed(task_ranges) if row["group"] == group)]
                for group in ("train", "valid", "online")}
    model_options = {"hidden": 16, "heads": 4, "temperature": 1.0, "inner_lr": 0.01,
                     "model_meta_lr": 0.001, "data_meta_lr": 0.001, "label_regularization": 0.1,
                     **spec.get("model", {}), "seed": spec.get("seed", 42)}
    # Constructor validation before the protocol is published.
    DoubleAdaptModel(157, **model_options)
    fingerprint = _provider_fingerprint(provider)
    protocol = {"version": 1, "spec": spec, "status": "registered", "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "provider": fingerprint, "tasks": task_ranges, "initial_universe": selected,
                "universe_selection": "alphabetical budget among members eligible on initial training date; historical membership respected afterwards",
                "source_metadata": json.loads((provider / "metadata.json").read_text()) if (provider / "metadata.json").exists() else {},
                "feature_normalizer_fit": segments["train"], "target": "close[t+2] / close[t+1] - 1",
                "label_information_endpoint_sessions": 2, "model": model_options,
                "models": ["doubleadapt_paper", "frozen", "naive_incremental", "full_retrain"],
                "selection": "minimum offline validation raw-return MSE; fixed registered budget thereafter; no online test-score model selection",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "references": {"paper": PAPER, "official_source_revision": UPSTREAM, "upstream_license": "No license file found at the inspected revision; no source vendored"},
                "method_differences": ["independent equations-based implementation", "exact second-order gradients instead of first-order approximation", "compact MLP on 157 current-date factors instead of authors' GRU/Alpha360 setting", "positive affine label scales using exp(log_scale)", "finite chronological support/query gaps for mature delayed-return labels", "training-only label normalizer"],
                "limitations": ["exploratory bounded historical forecast screen", "not an official-code or paper-score reproduction", "no portfolio return or production approval", "historical publication-time data vintages remain unqualified"],
                "software": {name: importlib.metadata.version(name) for name in ("torch", "pyqlib", "numpy", "pandas")}}
    output.mkdir(parents=True)
    _save_json(output / "protocol.json", protocol)
    _save_json(output / "status.json", {"status": "running"})
    torch.set_num_threads(counts["num_threads"])
    try:
        qlib_task, _ = build_task({"provider_uri": str(provider), "region": region, "instruments": selected,
                                  "data_start_time": str(calendar[first - counts["warmup_sessions"]].date()),
                                  "data_end_time": str(final.date()), "warmup_sessions": counts["warmup_sessions"],
                                  "sequence_length": 8, "shared_windows": True,
                                  "segments": {"train": segments["train"], "valid": segments["valid"], "test": segments["online"]},
                                  "model": {"name": "transformer"}, "backtest": {"enabled": False}}, calendar)
        dataset = init_instance_by_config(qlib_task["dataset"])
        names, processing = export_preprocessing(dataset.handler)
        _save_json(output / "preprocessing.json", {"features": names, "recipe": processing})
        features = dataset.handler.fetch(selector=(calendar[first], final), col_set="feature", data_key="infer")
        features.index = features.index.reorder_levels(["datetime", "instrument"])
        features = features.sort_index()
        prices = D.features(selected, ["$close"], start_time=calendar[first - counts["warmup_sessions"]], end_time=final).iloc[:, 0]
        prices.index = prices.index.reorder_levels(["datetime", "instrument"])
        close = prices.unstack("instrument").reindex(columns=selected)
        data_calendar = calendar[(calendar >= calendar[first - counts["warmup_sessions"]]) & (calendar <= final)]
        eligibility = pd.DataFrame(False, index=data_calendar, columns=selected)
        for symbol in selected:
            for start, end in membership[symbol]:
                eligibility.loc[(eligibility.index >= start) & (eligibility.index <= end), symbol] = True
        labels, endpoints = multi_horizon_targets(close, data_calendar, horizons=(1,), eligible=eligibility)

        def batch(interval):
            dates = features.index.get_level_values("datetime")
            frame = features.loc[(dates >= interval[0]) & (dates <= interval[1])]
            active = [bool(eligibility.loc[date, instrument]) for date, instrument in frame.index]
            frame = frame.loc[active]
            result = AdaptationBatch(frame.index, torch.tensor(frame.to_numpy(), dtype=torch.float32),
                                     torch.tensor(labels.loc[frame.index, "return_1"].to_numpy(), dtype=torch.float32),
                                     endpoints.loc[frame.index, "return_1"].to_numpy())
            return result

        tasks = {group: [] for group in ("train", "valid", "online")}
        for row in task_ranges:
            item = AdaptationTask(row["name"], batch(row["support"]), batch(row["query"]))
            item.validate(as_of=row["query_mature_at"])
            tasks[row["group"]].append(item)
        train_all = _merge_batches([batch for task in tasks["train"] for batch in (task.support, task.query)])
        valid_all = _merge_batches([batch for task in tasks["valid"] for batch in (task.support, task.query)])
        fit_asof = max(_query_cutoff(t.query) for t in tasks["valid"])
        model, selection = fit_doubleadapt(tasks["train"], tasks["valid"], as_of=fit_asof,
                                           epochs=counts["epochs"], **model_options)
        _save_json(output / "selection.json", selection)
        _save_json(output / "normalization.json", {"mean": float(model.label_mean), "scale": float(model.label_scale), "fit": "meta-training labels only"})
        before_online = {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}
        np.savez_compressed(output / "offline_weights.npz", **before_online)
        locked = output / "locked_predictions"
        locked.mkdir()
        locked_receipts = []

        def lock_predictions(method, task_name, frame):
            path = locked / f"{method}-{task_name}.csv"
            frame.to_csv(path)
            locked_receipts.append({"method": method, "task": task_name,
                                    "file": str(path.relative_to(output)),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "query_labels_in_file": False})

        doubleadapt_predictions = _simulate(model, tasks["online"], update=True,
                                             on_prediction=lambda name, frame: lock_predictions("doubleadapt_paper", name, frame))
        doubleadapt_predictions["model"] = "doubleadapt_paper"
        forecasts = [doubleadapt_predictions]
        _save_json(output / "meta_update_receipts.json", model.receipts)
        np.savez_compressed(output / "online_weights.npz", **{name: value.detach().numpy() for name, value in model.state_dict().items()})
        hidden, seed, lr = model_options["hidden"], model_options["seed"], model_options["inner_lr"]
        # Simple controls select their supervised epoch on the same allowed
        # validation history, then fit that fixed number of passes before OOS.
        control_history, best, selected_epoch = [], float("inf"), 1
        for epoch in range(1, counts["epochs"] + 1):
            control = _supervised_fit(train_all, model, hidden=hidden, seed=seed, epochs=epoch, lr=lr)
            with torch.no_grad():
                x, y = valid_all.finite()
                pred = control(x).squeeze(-1) * model.label_scale + model.label_mean
                mse = float(((pred - y) ** 2).mean())
            control_history.append({"epoch": epoch, "validation_mse": mse})
            if mse < best:
                best, selected_epoch = mse, epoch
        initial_history = _merge_batches([train_all, valid_all])
        initial = _supervised_fit(initial_history, model, hidden=hidden, seed=seed, epochs=selected_epoch, lr=lr)
        control_receipts = {"selected_epoch": selected_epoch, "validation_history": control_history,
                            "initial_fit_includes_mature_validation_after_selection": True, "updates": []}
        for method in ("frozen", "naive_incremental", "full_retrain"):
            predictor = copy.deepcopy(initial)
            history = [initial_history]
            for task in tasks["online"]:
                support_cutoff = _query_cutoff(task.support)
                task.support.validate(as_of=support_cutoff)
                if method == "naive_incremental":
                    predictor = _supervised_fit(task.support, model, hidden=hidden, seed=seed, epochs=1, lr=lr,
                                                initial=predictor.state_dict())
                elif method == "full_retrain":
                    training = _merge_batches(history + [task.support])
                    training.validate(as_of=support_cutoff)
                    predictor = _supervised_fit(training, model, hidden=hidden, seed=seed, epochs=selected_epoch, lr=lr)
                with torch.no_grad():
                    scores = predictor(task.query.features).squeeze(-1) * model.label_scale + model.label_mean
                forecast = pd.DataFrame({"score": scores.numpy(), "label": task.query.targets.numpy(), "model": method}, index=task.query.index)
                forecasts.append(forecast)
                lock_predictions(method, task.name, forecast[["score"]])
                # The current query enters history only after its emitted scores
                # are retained; these labels cannot alter that task's forecast.
                task.query.validate(as_of=_query_cutoff(task.query))
                if method == "naive_incremental":
                    predictor = _supervised_fit(task.query, model, hidden=hidden, seed=seed, epochs=1, lr=lr,
                                                initial=predictor.state_dict())
                history.extend([task.support, task.query])
                control_receipts["updates"].append({"method": method, "task": task.name, "support_cutoff": str(support_cutoff.date()),
                                                     "query_mature_at": str(_query_cutoff(task.query).date()), "predictions_locked_before_labels": True})
            np.savez_compressed(output / f"{method}_weights.npz", **{name: value.detach().numpy() for name, value in predictor.state_dict().items()})
        _save_json(output / "control_receipts.json", control_receipts)
        _save_json(output / "prediction_locks.json", locked_receipts)
        predictions = pd.concat(forecasts).reset_index()
        predictions.to_csv(output / "predictions.csv", index=False)
        metrics, paired = {}, {}
        for method in protocol["models"]:
            panel = predictions.loc[predictions["model"] == method].set_index(["datetime", "instrument"])[["score", "label"]].sort_index()
            metrics[method], daily = _prediction_statistics(panel)
            daily.to_csv(output / f"{method}_daily_ic.csv")
            if method != "doubleadapt_paper":
                candidate = doubleadapt_predictions[["score", "label"]]
                paired[method] = paired_date_block_intervals(
                    candidate, panel, block_length=spec.get("bootstrap_block_sessions", 5),
                    draws=spec.get("bootstrap_repetitions", 200), seed=seed,
                )
        result = {"status": "completed", "models": metrics, "paired_date_block_comparison": paired,
                  "selection": selection, "deployment_approved": False,
                  "online_dates": int(doubleadapt_predictions.index.get_level_values("datetime").nunique()),
                  "online_rows_per_model": len(doubleadapt_predictions),
                  "method": "independent_DoubleAdapt_paper_method", "upstream_reproduction": False,
                  "paired_interpretation": "DoubleAdapt minus control; negative RMSE and positive IC favor DoubleAdapt", "limitations": protocol["limitations"]}
        _save_json(output / "results.json", result)
        manifest = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.rglob("*")) if path.is_file() and path.name != "status.json"}
        _save_json(output / "artifacts.json", manifest)
        _save_json(output / "status.json", {"status": "completed", "artifacts": "artifacts.json"})
        return json.loads((output / "results.json").read_text())
    except Exception as exc:
        _save_json(output / "status.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
