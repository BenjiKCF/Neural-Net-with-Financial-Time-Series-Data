"""Explicit research helpers for multiple horizons, calibration and adaptation.

These array-level experiments do not reuse the native one-session portfolio
replay. Their target endpoints and update cutoffs are explicit and independently
validated. Empirical interval calibration does not assert financial returns are
exchangeable or guarantee future coverage.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from financial_ts.architectures.cross_stock import CrossStockEncoder
from financial_ts.objectives import quantile_loss


def _day(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tz is not None or value != value.normalize():
        raise ValueError("Dates must be finite timezone-naive session dates")
    return value


def multi_horizon_targets(
    close: pd.DataFrame,
    calendar,
    *,
    horizons=(1, 5, 20),
    entry_delay=1,
    eligible: pd.DataFrame | None = None,
):
    """Return raw labels and their actual exchange-session information endpoints.

    ``close`` has calendar dates as index and instrument IDs as columns. Missing
    entry/exit prices remain missing; there is no per-stock shift, fill, or
    hindsight sale. Eligibility is evaluated at the signal date. Endpoint ``d+h``
    therefore controls fold purging for a holding horizon ``h`` and entry delay
    ``d``, irrespective of holidays or suspensions.
    """
    calendar = pd.DatetimeIndex(calendar)
    if (
        calendar.empty
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
        or calendar.hasnans
        or calendar.tz is not None
        or not (calendar == calendar.normalize()).all()
    ):
        raise ValueError("calendar must be unique increasing observed daily sessions")
    if type(entry_delay) is not int or entry_delay < 1:
        raise ValueError("entry_delay must be a positive session count")
    if (
        not horizons
        or any(type(h) is not int or h < 1 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must be distinct positive session counts")
    if (
        not isinstance(close.index, pd.DatetimeIndex)
        or not close.index.is_unique
        or not close.columns.is_unique
        or close.columns.empty
        or any(not isinstance(x, str) or not x for x in close.columns)
        or not close.index.isin(calendar).all()
    ):
        raise ValueError("Prices need unique calendar dates and nonempty instrument names")
    prices = close.reindex(calendar).astype(float)
    if ((prices <= 0) & prices.notna()).any().any() or np.isinf(prices.to_numpy()).any():
        raise ValueError("Observed prices must be finite and positive")
    if eligible is None:
        eligibility = prices.notna()
    else:
        if not eligible.index.equals(prices.index) or not eligible.columns.equals(prices.columns):
            raise ValueError("Eligibility must exactly align to calendar and instruments")
        if eligible.isna().any().any() or any(
            not pd.api.types.is_bool_dtype(dtype) for dtype in eligible.dtypes
        ):
            raise ValueError("Eligibility values must be explicit booleans")
        eligibility = eligible
    index = pd.MultiIndex.from_product([calendar, prices.columns], names=["datetime", "instrument"])
    labels, endpoints = {}, {}
    entry = prices.shift(-entry_delay)
    for horizon in horizons:
        name = f"return_{horizon}"
        values = (prices.shift(-(entry_delay + horizon)) / entry - 1).where(eligibility)
        labels[name] = values.to_numpy().reshape(-1)
        dates = pd.Series(calendar, index=calendar).shift(-(entry_delay + horizon))
        endpoints[name] = np.repeat(dates.to_numpy(), len(prices.columns))
    return pd.DataFrame(labels, index=index), pd.DataFrame(endpoints, index=index)


@dataclass
class HorizonBatch:
    date: pd.Timestamp
    features: torch.Tensor
    padding: torch.Tensor
    targets: torch.Tensor
    label_endpoints: np.ndarray

    def validate(self, cutoff, *, horizons: int):
        date, cutoff = _day(self.date), _day(cutoff)
        if date > cutoff:
            raise ValueError("Feature anchor occurs after the registered data cutoff")
        if (
            self.features.ndim != 3
            or self.padding.shape != self.features.shape[:2]
            or self.padding.dtype != torch.bool
            or self.targets.shape != (self.features.shape[0], horizons)
            or np.asarray(self.label_endpoints).shape != tuple(self.targets.shape)
        ):
            raise ValueError("Horizon batch dimensions do not align")
        if not self.features.shape[0] or not torch.isfinite(self.features[~self.padding]).all():
            raise ValueError("Horizon batches need finite observed features and nonempty stocks")
        endpoints = np.asarray(self.label_endpoints, dtype="datetime64[ns]")
        finite = torch.isfinite(self.targets).cpu().numpy()
        if np.isinf(self.targets.detach().cpu().numpy()).any():
            raise ValueError("Infinite returns are invalid")
        if (
            np.isnat(endpoints[finite]).any()
            or (endpoints[finite] > cutoff.to_datetime64()).any()
            or (endpoints[finite] <= date.to_datetime64()).any()
        ):
            raise ValueError("Only labels matured by the cutoff may enter fitting or selection")
        return date


class HorizonResearchModel:
    """Train actual return/quantile heads on explicit chronological array panels.

    The normalizer is fitted on training labels only. A new fit trains from
    scratch; ``incremental_update`` retains that normalizer and can roll back a
    harmful update using an explicitly later, mature validation interval.
    """

    def __init__(
        self,
        *,
        horizons=(1, 5, 20),
        quantiles=(0.1, 0.5, 0.9),
        seed=42,
        network_kwargs=None,
        epochs=3,
        lr=0.001,
        quantile_weight=0.1,
    ):
        if type(epochs) is not int or epochs < 1 or not 0 < lr < float("inf"):
            raise ValueError("epochs and lr must be positive")
        if not 0 <= quantile_weight < float("inf"):
            raise ValueError("quantile_weight must be finite and nonnegative")
        self.horizons, self.quantiles = tuple(horizons), tuple(quantiles)
        self.seed, self.network_kwargs = seed, copy.deepcopy(network_kwargs or {})
        if set(self.network_kwargs) & {"horizons", "quantiles"}:
            raise ValueError("Set horizons and quantiles directly")
        self.epochs, self.lr, self.quantile_weight = epochs, lr, quantile_weight
        self.network = None
        self.fitted = False
        self.target_mean_ = self.target_scale_ = None
        self.receipt_ = {}

    def _validate_segments(self, train, valid, as_of):
        if not train or not valid:
            raise ValueError("Both chronological train and validation panels are required")
        train_dates = [b.validate(as_of, horizons=len(self.horizons)) for b in train]
        valid_dates = [b.validate(as_of, horizons=len(self.horizons)) for b in valid]
        if train_dates != sorted(set(train_dates)) or valid_dates != sorted(set(valid_dates)):
            raise ValueError("Each segment needs unique increasing decision dates")
        if max(train_dates) >= min(valid_dates):
            raise ValueError("Validation must be strictly after the training panel")
        for batch in train:
            finite = torch.isfinite(batch.targets).cpu().numpy()
            if (
                np.asarray(batch.label_endpoints, dtype="datetime64[ns]")[finite]
                >= min(valid_dates).to_datetime64()
            ).any():
                raise ValueError("Training label endpoints cross the validation boundary")

    def _loss(self, batch):
        output = self.network.forward_heads(batch.features, batch.padding)
        target = (batch.targets - self.target_mean_) / self.target_scale_
        finite = torch.isfinite(target)
        if not finite.any():
            return None
        # Per-horizon means prevent the abundant short horizon from dominating.
        terms = [
            nn.functional.huber_loss(
                output["returns"][:, i][finite[:, i]], target[:, i][finite[:, i]]
            )
            for i in range(len(self.horizons))
            if finite[:, i].any()
        ]
        loss = torch.stack(terms).mean()
        if self.quantiles and self.quantile_weight:
            loss = loss + self.quantile_weight * quantile_loss(
                output["quantiles"], target, self.quantiles
            )
        return loss

    def _validation(self, batches):
        self.network.eval()
        losses = []
        with torch.inference_mode():
            for batch in batches:
                loss = self._loss(batch)
                if loss is not None:
                    losses.append(float(loss))
        if not losses or not np.isfinite(losses).all():
            raise ValueError("Validation needs finite, mature observations")
        return float(np.mean(losses))

    def _optimize(self, train, valid, *, retain_initial):
        initial = self._validation(valid) if retain_initial else float("inf")
        best, selected = initial, 0
        best_state = copy.deepcopy(self.network.state_dict()) if retain_initial else None
        history = []
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr)
        for epoch in range(1, self.epochs + 1):
            self.network.train()
            for batch in train:
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(batch)
                if loss is None:
                    continue
                if not torch.isfinite(loss):
                    raise ValueError("Multi-horizon optimization became nonfinite")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            metric = self._validation(valid)
            history.append(metric)
            if metric < best:
                best, selected = metric, epoch
                best_state = copy.deepcopy(self.network.state_dict())
        if best_state is None:
            raise ValueError("No finite multi-horizon checkpoint")
        self.network.load_state_dict(best_state)
        self.network.eval()
        return {
            "selected_epoch": selected,
            "validation_loss": best,
            "validation_history": history,
            "retained_epoch_zero": selected == 0,
            "initial_validation_loss": initial if np.isfinite(initial) else None,
        }

    def fit(self, train, valid, *, as_of):
        self.fitted = False
        self._validate_segments(train, valid, as_of)
        torch.manual_seed(self.seed)
        options = {
            "d_feat": train[0].features.shape[-1],
            **self.network_kwargs,
            "horizons": self.horizons,
            "quantiles": self.quantiles,
        }
        self.network = CrossStockEncoder(**options)
        stacked = torch.cat([batch.targets for batch in train])
        means, scales = [], []
        for i in range(len(self.horizons)):
            values = stacked[:, i][torch.isfinite(stacked[:, i])]
            if not len(values):
                raise ValueError(f"No mature training labels for horizon {self.horizons[i]}")
            means.append(values.mean())
            scale = values.std(unbiased=False)
            scales.append(scale if scale > 1e-12 else values.new_tensor(1.0))
        self.target_mean_, self.target_scale_ = torch.stack(means), torch.stack(scales)
        result = self._optimize(train, valid, retain_initial=False)
        self.receipt_ = {
            "horizons": list(self.horizons),
            "quantiles": list(self.quantiles),
            "seed": self.seed,
            "as_of": str(_day(as_of).date()),
            "fit_as_of": str(_day(as_of).date()),
            "maximum_label_endpoint_sessions": 1 + max(self.horizons),
            "normalizer_fit": "train labels only",
            "portfolio_replay": "not run",
            "fit": result,
            "updates": [],
        }
        self.fitted = True
        return self

    def predict(self, features, padding):
        if not self.fitted or self.network is None or self.target_mean_ is None:
            raise ValueError("Fit the multi-horizon model first")
        self.network.eval()
        with torch.inference_mode():
            outputs = self.network.forward_heads(features, padding)
        result = {"returns": outputs["returns"] * self.target_scale_ + self.target_mean_}
        if "quantiles" in outputs:
            result["quantiles"] = (
                outputs["quantiles"] * self.target_scale_[None, :, None]
                + self.target_mean_[None, :, None]
            )
        return result

    def incremental_update(self, train, valid, *, as_of):
        if not self.fitted or self.network is None:
            raise ValueError("Fit a reference model before an incremental update")
        if _day(as_of) <= _day(self.receipt_["as_of"]):
            raise ValueError("Incremental update cutoff must advance")
        self._validate_segments(train, valid, as_of)
        original = copy.deepcopy(self.network.state_dict())
        try:
            result = self._optimize(train, valid, retain_initial=True)
        except Exception:
            self.network.load_state_dict(original)
            self.network.eval()
            raise
        self.receipt_["updates"].append(
            {"as_of": str(_day(as_of).date()), **result, "normalizer_refitted": False}
        )
        self.receipt_["as_of"] = str(_day(as_of).date())
        return result


def calibrate_intervals(
    lower, upper, target, *, dates, label_endpoints, cutoff, inference_start, coverage=0.8
):
    """Chronological empirical interval expansion, aggregated by decision date.

    Each date contributes its worst noncoverage across that date's stocks. The
    finite-sample quantile uses a conservative order statistic. This is an
    empirical calibration artifact; temporal dependence can invalidate formal
    conformal guarantees, and future coverage must still be measured.
    """
    lower, upper, target = [np.asarray(x, dtype=float) for x in (lower, upper, target)]
    if lower.shape != upper.shape or lower.shape != target.shape or target.ndim != 2:
        raise ValueError("Intervals and targets must share [sample, horizon] shape")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie in (0,1)")
    dates = pd.DatetimeIndex(dates)
    cutoff, start = _day(cutoff), _day(inference_start)
    if (
        len(dates) != len(target)
        or dates.hasnans
        or dates.tz is not None
        or not (dates == dates.normalize()).all()
        or cutoff >= start
        or (dates > cutoff).any()
    ):
        raise ValueError("Calibration must precede the declared inference start")
    endpoints = np.asarray(label_endpoints, dtype="datetime64[ns]")
    if endpoints.shape != target.shape:
        raise ValueError("Calibration label endpoints must match targets")
    valid = np.isfinite(target)
    if (
        not np.isfinite(lower[valid]).all()
        or not np.isfinite(upper[valid]).all()
        or (lower[valid] > upper[valid]).any()
    ):
        raise ValueError("Calibration needs finite ordered intervals")
    if (
        np.isnat(endpoints[valid]).any()
        or (endpoints[valid] > cutoff.to_datetime64()).any()
        or (
            endpoints[valid] <= np.repeat(dates.to_numpy()[:, None], target.shape[1], axis=1)[valid]
        ).any()
    ):
        raise ValueError("Calibration labels must be future outcomes matured by its cutoff")
    expansion, samples = [], []
    for h in range(target.shape[1]):
        residual = np.maximum.reduce(
            [lower[:, h] - target[:, h], target[:, h] - upper[:, h], np.zeros(len(target))]
        )
        frame = pd.DataFrame({"date": dates[valid[:, h]], "score": residual[valid[:, h]]})
        scores = frame.groupby("date")["score"].max().to_numpy()
        if not len(scores):
            raise ValueError("Each horizon needs mature calibration dates")
        rank = int(np.ceil((len(scores) + 1) * coverage))
        if rank > len(scores):
            raise ValueError(
                "Insufficient calibration dates for the requested finite-sample coverage"
            )
        expansion.append(float(np.sort(scores)[rank - 1]))
        samples.append(len(scores))
    return {
        "expansion": expansion,
        "coverage_target": coverage,
        "calibration_dates_per_horizon": samples,
        "cutoff": str(cutoff.date()),
        "inference_start": str(start.date()),
        "method": "date-wise maximum residual; finite-sample empirical quantile",
        "guarantee": "No exchangeability or future financial coverage guarantee",
    }


def interval_metrics(lower, upper, target):
    lower, upper, target = [np.asarray(x, dtype=float) for x in (lower, upper, target)]
    if lower.shape != upper.shape or lower.shape != target.shape or target.ndim != 2:
        raise ValueError("Intervals and labels must share [sample, horizon] shape")
    results = []
    for h in range(target.shape[1]):
        valid = np.isfinite(target[:, h])
        if (
            not np.isfinite(lower[valid, h]).all()
            or not np.isfinite(upper[valid, h]).all()
            or (upper[valid, h] < lower[valid, h]).any()
        ):
            raise ValueError("Evaluation intervals must be finite and ordered")
        count = int(valid.sum())
        results.append(
            {
                "observations": count,
                "coverage": float(
                    (
                        (target[valid, h] >= lower[valid, h])
                        & (target[valid, h] <= upper[valid, h])
                    ).mean()
                )
                if count
                else None,
                "mean_width": float((upper[valid, h] - lower[valid, h]).mean()) if count else None,
            }
        )
    return results


def run_advanced_research(provider, output, spec):
    """Run a registered, bounded real-provider panel experiment end to end.

    This is an exploratory forecast experiment, with no portfolio or promotion
    claim. The initial universe is frozen from membership on the first training
    day, then membership exits/reentries are respected at every later date.
    Every target horizon uses the same delayed entry convention and has its own
    information endpoint. Output directories are immutable campaign identities.
    """
    import qlib
    from qlib.data import D
    from qlib.utils import init_instance_by_config

    from financial_ts.bundles import export_preprocessing
    from financial_ts.cross_sectional import DailyPanel
    from financial_ts.workflow import (
        _prediction_statistics,
        _provider_fingerprint,
        _read_calendar,
        _save_json,
        build_task,
    )

    provider, output = Path(provider).expanduser().resolve(), Path(output).expanduser().resolve()
    if provider == output or provider in output.parents or output in provider.parents:
        raise ValueError("Research output must be separate from the provider")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Advanced research needs a new or empty output directory")
    spec = copy.deepcopy(spec)
    allowed = {
        "name",
        "start",
        "region",
        "universe",
        "max_instruments",
        "horizons",
        "sequence_length",
        "warmup_sessions",
        "segment_sessions",
        "seed",
        "epochs",
        "num_threads",
        "network",
        "coverage",
        "purpose",
    }
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError(f"Unknown advanced research spec fields: {sorted(set(spec) - allowed)}")
    horizons = tuple(spec.get("horizons", (1, 5, 20)))
    if (
        not horizons
        or any(type(h) is not int or h < 1 for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("horizons must be distinct positive integers")
    counts = {
        "train": 40,
        "valid": 10,
        "calibration": 10,
        "test": 10,
        "update_train": 20,
        "update_valid": 10,
        "post_update_test": 10,
    }
    provided_counts = spec.get("segment_sessions", {})
    if set(provided_counts) - set(counts):
        raise ValueError("Unknown chronological research segment")
    counts.update(provided_counts)
    if any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError("Segment lengths must be positive session counts")
    warmup, sequence = spec.get("warmup_sessions", 60), spec.get("sequence_length", 8)
    limit, threads = spec.get("max_instruments", 20), spec.get("num_threads", 2)
    if any(type(x) is not int or x < 1 for x in (warmup, sequence, limit, threads)):
        raise ValueError("Warmup, sequence length, instrument budget and threads must be positive")
    calendar = _read_calendar(provider)
    first = int(calendar.searchsorted(_day(spec.get("start", "2023-01-03"))))
    if first < warmup or first >= len(calendar):
        raise ValueError("Start date needs the declared source history")
    endpoint = 1 + max(horizons)
    cursor, segments = first, {}
    for name, count in counts.items():
        end = cursor + count - 1
        if end + endpoint >= len(calendar):
            raise ValueError("Source calendar is too short for segments and maximum label endpoint")
        segments[name] = [str(calendar[cursor].date()), str(calendar[end].date())]
        cursor = end + endpoint + 1
    final_position = int(calendar.get_loc(pd.Timestamp(segments["post_update_test"][1]))) + endpoint
    final_date = calendar[final_position]
    as_of = calendar[int(calendar.get_loc(pd.Timestamp(segments["calibration"][0]))) - 1]
    calibration_cutoff = calendar[int(calendar.get_loc(pd.Timestamp(segments["test"][0]))) - 1]
    update_as_of = calendar[
        int(calendar.get_loc(pd.Timestamp(segments["post_update_test"][0]))) - 1
    ]
    region = spec.get("region", "cn")
    if region not in {"cn", "us"}:
        raise ValueError("region must be cn or us")
    qlib.init(
        provider_uri=str(provider),
        region=region,
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    universe = spec.get("universe", "csi300" if region == "cn" else "all")
    membership = D.list_instruments(
        D.instruments(universe),
        start_time=calendar[first - warmup],
        end_time=final_date,
        as_list=False,
    )
    selected = sorted(
        symbol
        for symbol, intervals in membership.items()
        if any(
            pd.Timestamp(start) <= calendar[first] <= pd.Timestamp(end) for start, end in intervals
        )
    )[:limit]
    if len(selected) < 2:
        raise ValueError("At least two eligible stocks are needed on the first training date")
    source_metadata = (
        json.loads((provider / "metadata.json").read_text())
        if (provider / "metadata.json").is_file()
        else {}
    )
    fingerprint = _provider_fingerprint(provider)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "status": "registered",
        "spec": spec,
        "registered_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "provider": fingerprint,
        "source_metadata": source_metadata,
        "segments": segments,
        "horizons": list(horizons),
        "entry_delay_sessions": 1,
        "maximum_label_endpoint_sessions": endpoint,
        "initial_universe": selected,
        "universe_selection": "alphabetical budget among eligible stocks at first training date; subsequent membership respected",
        "feature_normalizer_fit": segments["train"],
        "fit_as_of": str(as_of.date()),
        "calibration_cutoff": str(calibration_cutoff.date()),
        "update_as_of": str(update_as_of.date()),
        "models": ["joint_point", "joint_quantile", *[f"single_{h}" for h in horizons]],
        "adaptation_controls": ["frozen", "incremental", "full_retrain"],
        "full_retrain_definition": "all model weights and target scaler fitted again on original + update train panels; initial feature scaler remains fixed for all adaptation controls",
        "selection": "minimum equally weighted daily validation Huber/pinball objective; retain incremental epoch zero if better",
        "software": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "pyqlib", "numpy", "pandas")
            },
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "exploratory bounded forecast screen",
            "no portfolio replay or production qualification",
            "historical source revisions are not verified publication-time vintages",
            "simple incremental control is not a DoubleAdapt reproduction",
        ],
    }
    _save_json(output / "protocol.json", protocol)
    _save_json(output / "status.json", {"status": "running"})
    torch.set_num_threads(threads)
    try:
        task, _ = build_task(
            {
                "provider_uri": str(provider),
                "region": region,
                "instruments": selected,
                "data_start_time": str(calendar[first - warmup].date()),
                "data_end_time": str(final_date.date()),
                "warmup_sessions": warmup,
                "sequence_length": sequence,
                "shared_windows": True,
                "segments": {name: segments[name] for name in ("train", "valid", "test")},
                "model": {"name": "master_style"},
                "backtest": {"enabled": False},
            },
            calendar,
        )
        task["dataset"]["kwargs"]["segments"] = segments
        dataset = init_instance_by_config(task["dataset"])
        feature_names, preprocessing = export_preprocessing(dataset.handler)
        _save_json(
            output / "preprocessing.json", {"features": feature_names, "recipe": preprocessing}
        )
        prices = D.features(
            selected, ["$close"], start_time=calendar[first - warmup], end_time=final_date
        ).iloc[:, 0]
        prices.index = prices.index.reorder_levels(["datetime", "instrument"])
        prices = prices.unstack("instrument").reindex(columns=selected)
        data_calendar = calendar[first - warmup : final_position + 1]
        eligibility = pd.DataFrame(False, index=data_calendar, columns=selected)
        for symbol in selected:
            for start, end in membership[symbol]:
                eligibility.loc[
                    (eligibility.index >= start) & (eligibility.index <= end), symbol
                ] = True
        labels, endpoints = multi_horizon_targets(
            prices, data_calendar, horizons=horizons, eligible=eligibility
        )
        panels = {}
        identities = {}
        for name in segments:
            sampler = dataset.prepare(name, col_set="feature", data_key="infer", dtype=np.float32)
            panel = DailyPanel(sampler, 157)
            batches, keys = [], []
            for i in range(len(panel)):
                batch = panel[i]
                active = np.asarray(
                    [eligibility.loc[batch["date"], symbol] for symbol in batch["instruments"]],
                    dtype=bool,
                )
                if not active.any():
                    continue
                stocks = np.asarray(batch["instruments"])[active]
                row_index = pd.MultiIndex.from_tuples(
                    [(batch["date"], s) for s in stocks], names=["datetime", "instrument"]
                )
                batches.append(
                    HorizonBatch(
                        batch["date"],
                        batch["features"][active],
                        batch["padding"][active],
                        torch.tensor(labels.reindex(row_index).to_numpy(), dtype=torch.float32),
                        endpoints.reindex(row_index).to_numpy(),
                    )
                )
                keys.append(row_index)
            if not batches:
                raise ValueError(f"No eligible daily panels in segment {name}")
            panels[name], identities[name] = batches, keys
        _save_json(
            output / "panel_receipt.json",
            {
                "segments": {
                    name: {
                        "dates": len(batches),
                        "samples": sum(len(b.features) for b in batches),
                        "finite_labels_by_horizon": torch.isfinite(
                            torch.cat([b.targets for b in batches])
                        )
                        .sum(0)
                        .tolist(),
                    }
                    for name, batches in panels.items()
                },
                "processor_fit": segments["train"],
                "provider": fingerprint,
            },
        )
        network = {
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "dropout": 0.0,
            **spec.get("network", {}),
        }
        models, trial_results, forecasts = {}, {}, []

        def checkpoint(name, model):
            path = output / name
            path.mkdir(exist_ok=False)
            np.savez_compressed(
                path / "weights.npz",
                **{
                    key: value.detach().cpu().numpy()
                    for key, value in model.network.state_dict().items()
                },
            )
            _save_json(path / "receipt.json", model.receipt_)
            _save_json(
                path / "network_config.json",
                {
                    "d_feat": 157,
                    **network,
                    "horizons": list(model.horizons),
                    "quantiles": list(model.quantiles),
                },
            )
            _save_json(
                path / "normalization.json",
                {"mean": model.target_mean_.tolist(), "scale": model.target_scale_.tolist()},
            )
            return path

        def evaluate(name, model, segment, *, horizon_positions=None, calibration=None):
            positions = (
                list(range(len(horizons))) if horizon_positions is None else horizon_positions
            )
            frames = []
            for batch, index in zip(panels[segment], identities[segment]):
                output_heads = model.predict(batch.features, batch.padding)
                for j, h_position in enumerate(positions):
                    frame = pd.DataFrame(
                        {
                            "score": output_heads["returns"][:, j].numpy().astype(np.float64),
                            "label": labels.loc[index, f"return_{horizons[h_position]}"].to_numpy(
                                dtype=np.float64
                            ),
                        },
                        index=index,
                    )
                    frame["model"], frame["segment"], frame["horizon"] = (
                        name,
                        segment,
                        horizons[h_position],
                    )
                    if "quantiles" in output_heads:
                        frame["lower"] = output_heads["quantiles"][:, j, 0].numpy()
                        frame["upper"] = output_heads["quantiles"][:, j, -1].numpy()
                        frame["uncalibrated_lower"] = frame["lower"]
                        frame["uncalibrated_upper"] = frame["upper"]
                        if calibration is not None:
                            frame["lower"] -= calibration["expansion"][j]
                            frame["upper"] += calibration["expansion"][j]
                    frames.append(frame)
            frame = pd.concat(frames).sort_index()
            forecasts.append(frame)
            pd.concat(forecasts).reset_index().to_csv(output / "predictions.csv", index=False)
            results = {}
            for horizon, group in frame.groupby("horizon"):
                metrics, _ = _prediction_statistics(group[["score", "label"]])
                if "lower" in group:
                    metrics["interval"] = interval_metrics(
                        group[["lower"]], group[["upper"]], group[["label"]]
                    )[0]
                    metrics["uncalibrated_interval"] = interval_metrics(
                        group[["uncalibrated_lower"]],
                        group[["uncalibrated_upper"]],
                        group[["label"]],
                    )[0]
                results[str(horizon)] = metrics
            return results

        for name in protocol["models"]:
            h_positions = (
                list(range(len(horizons)))
                if name.startswith("joint_")
                else [horizons.index(int(name.split("_")[1]))]
            )
            selected_horizons = tuple(horizons[i] for i in h_positions)
            selected_quantiles = (0.1, 0.5, 0.9) if name == "joint_quantile" else ()
            model = HorizonResearchModel(
                horizons=selected_horizons,
                quantiles=selected_quantiles,
                seed=spec.get("seed", 42),
                epochs=spec.get("epochs", 2),
                network_kwargs=network,
            )

            def selected_batches(segment):
                return [
                    HorizonBatch(
                        batch.date,
                        batch.features,
                        batch.padding,
                        batch.targets[:, h_positions],
                        batch.label_endpoints[:, h_positions],
                    )
                    for batch in panels[segment]
                ]

            model.fit(selected_batches("train"), selected_batches("valid"), as_of=as_of)
            calibration = None
            if name == "joint_quantile":
                intervals = [
                    model.predict(b.features, b.padding)["quantiles"].numpy()
                    for b in panels["calibration"]
                ]
                intervals = np.concatenate(intervals)
                calibration = calibrate_intervals(
                    intervals[..., 0],
                    intervals[..., -1],
                    labels.reindex(
                        identities["calibration"][0].append(identities["calibration"][1:])
                    ).to_numpy(),
                    dates=np.concatenate(
                        [
                            np.repeat(b.date.to_datetime64(), len(b.features))
                            for b in panels["calibration"]
                        ]
                    ),
                    label_endpoints=np.concatenate(
                        [b.label_endpoints for b in panels["calibration"]]
                    ),
                    cutoff=calibration_cutoff,
                    inference_start=segments["test"][0],
                    coverage=spec.get("coverage", 0.8),
                )
                _save_json(output / "calibration.json", calibration)
            trial_results[name] = {
                "test": evaluate(
                    name, model, "test", horizon_positions=h_positions, calibration=calibration
                )
            }
            checkpoint(name, model)
            models[name] = model
            _save_json(output / "results.json", {"status": "running", "trials": trial_results})

        reference = models["joint_point"]
        incremental = copy.deepcopy(reference)
        incremental.incremental_update(
            panels["update_train"], panels["update_valid"], as_of=update_as_of
        )
        retrained = HorizonResearchModel(
            horizons=horizons,
            quantiles=(),
            seed=spec.get("seed", 42),
            epochs=spec.get("epochs", 2),
            network_kwargs=network,
        )
        retrained.fit(
            panels["train"] + panels["update_train"], panels["update_valid"], as_of=update_as_of
        )
        for name, model in {
            "frozen": reference,
            "incremental": incremental,
            "full_retrain": retrained,
        }.items():
            trial_results[name] = {"post_update_test": evaluate(name, model, "post_update_test")}
            checkpoint(name, model)
        pd.concat(forecasts).reset_index().to_csv(output / "predictions.csv", index=False)
        result = {
            "status": "completed",
            "trials": trial_results,
            "evidence_status": "exploratory forecast screen; no promotion decision",
            "doubleadapt": "not implemented or reproduced; incremental is a simple controlled update",
            "portfolio_replay": "not run; native portfolio target remains one session",
        }
        _save_json(output / "results.json", result)
        _save_json(output / "status.json", {"status": "completed"})
        _save_json(
            output / "artifacts.json",
            {
                "files": {
                    path.relative_to(output).as_posix(): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(output.rglob("*"))
                    if path.is_file()
                },
                "code": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("advanced_research.py", "cross_sectional.py", "objectives.py")
                },
            },
        )
        # The public runner result is itself JSON-safe, as are the disk receipts.
        return json.loads((output / "results.json").read_text())
    except Exception as error:
        _save_json(
            output / "status.json",
            {"status": "failed", "exception": type(error).__name__, "message": str(error)},
        )
        raise
