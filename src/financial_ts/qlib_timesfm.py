"""Qlib panel forecasting and bounded adapter fitting for official TimesFM 3.

All inference dates are evaluated. Only the adapter-fitting sample budgets are
bounded. Suspensions are forward-filled within each causal price context; an
initial missing prefix is trimmed, then padded only for fixed-length adapter
inputs. No test labels are requested during fitting or prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model

from financial_ts.timesfm_pretrained import OfficialTimesFM3Model


class CloseOnlyHandler(DataHandlerLP):
    """Small, unnormalised price/label panel with normal Qlib membership rules."""

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        fit_start_time=None,
        fit_end_time=None,
        label=None,
    ):
        # Fit dates are accepted for the standard handler configuration contract;
        # no price or label normalization is fitted by this handler.
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader={
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (["$close"], ["CLOSE"]),
                        "label": label or (["Ref($close, -2)/Ref($close, -1)-1"], ["LABEL0"]),
                    },
                },
            },
            infer_processors=[],
            learn_processors=[{"class": "DropnaLabel"}],
        )


def _index(prepared) -> pd.MultiIndex:
    index = prepared.get_index() if hasattr(prepared, "get_index") else prepared.index
    if (
        not isinstance(index, pd.MultiIndex)
        or not {"datetime", "instrument"}.issubset(index.names)
        or not index.is_unique
    ):
        raise ValueError("Samples require a unique datetime/instrument MultiIndex.")
    return index


class _CloseContextBank:
    """Read closes only through requested dates and retain actual calendar gaps."""

    def __init__(self, index: pd.MultiIndex, context_length: int):
        self.context_length = context_length
        dates = pd.DatetimeIndex(index.get_level_values("datetime"))
        self.calendar = pd.DatetimeIndex(D.calendar(end_time=dates.max(), freq="day"))
        locations = self.calendar.get_indexer(dates)
        if (locations < 0).any():
            raise ValueError("Every sample date must be an observed provider session.")
        self.first = max(0, int(locations.min()) - context_length + 1)
        instruments = sorted(set(index.get_level_values("instrument")))
        frame = D.features(
            instruments,
            ["$close"],
            start_time=self.calendar[self.first],
            end_time=dates.max(),
            freq="day",
        )
        if frame.empty or not frame.index.is_unique:
            raise ValueError("Provider closes must be nonempty and uniquely indexed.")
        closes = frame["$close"]
        observed = set(closes.index.get_level_values("instrument"))
        self.prices = {
            instrument: closes.xs(instrument, level="instrument")
            .reindex(self.calendar[self.first :])
            .to_numpy(dtype=np.float32)
            for instrument in instruments
            if instrument in observed
        }

    def _window(self, instrument, date):
        if instrument not in self.prices:
            return np.empty(0, dtype=np.float32)
        position = int(self.calendar.get_loc(date)) - self.first
        return self.prices[instrument][max(0, position - self.context_length + 1) : position + 1]

    def eligible(self, index: pd.MultiIndex) -> np.ndarray:
        result = []
        for instrument, date in zip(
            index.get_level_values("instrument"), index.get_level_values("datetime")
        ):
            values = self._window(instrument, date)
            finite = np.isfinite(values)
            result.append(
                finite.sum() >= 2 and not np.isinf(values).any() and (values[finite] > 0).all()
            )
        return np.asarray(result, dtype=bool)

    def contexts(self, index: pd.MultiIndex, *, pad: bool) -> tuple[list[np.ndarray], dict]:
        contexts, filled, padded = [], 0, 0
        for instrument, date in zip(
            index.get_level_values("instrument"), index.get_level_values("datetime")
        ):
            values = self._window(instrument, date).copy()
            observed = np.flatnonzero(np.isfinite(values))
            if len(observed) < 2 or np.isinf(values).any() or (values[observed] <= 0).any():
                raise ValueError(
                    f"{instrument} at {pd.Timestamp(date).date()} needs two positive observed "
                    "closes inside its causal context. No future-price substitution is allowed."
                )
            values = values[observed[0] :]
            missing = ~np.isfinite(values)
            filled += int(missing.sum())
            # The first remaining price is observed; indices only refer backward.
            previous = np.maximum.accumulate(np.where(missing, 0, np.arange(len(values))))
            values = values[previous]
            if pad and len(values) < self.context_length:
                padded += self.context_length - len(values)
                values = np.pad(values, (self.context_length - len(values), 0), mode="edge")
            contexts.append(np.ascontiguousarray(values, dtype=np.float32))
        return contexts, {"forward_filled_values": filled, "leading_padded_values": padded}


def _balanced_rows(index: pd.MultiIndex, budget: int) -> np.ndarray:
    """Balance historical member quotas and space dates within each member."""
    instruments = index.get_level_values("instrument")
    dates = pd.DatetimeIndex(index.get_level_values("datetime"))
    groups = {}
    for instrument in sorted(set(instruments)):
        rows = np.flatnonzero(instruments == instrument)
        groups[instrument] = rows[np.argsort(dates[rows].asi8, kind="stable")]
    names = list(groups)
    if len(names) > budget:
        names = [names[i] for i in np.linspace(0, len(names) - 1, budget, dtype=int)]
    quota = dict.fromkeys(names, 0)
    remaining = min(budget, sum(len(groups[name]) for name in names))
    while remaining:
        for name in names:
            if quota[name] < len(groups[name]):
                quota[name] += 1
                remaining -= 1
                if not remaining:
                    break
    selected = []
    for name, count in quota.items():
        selected.extend(groups[name][np.linspace(0, len(groups[name]) - 1, count, dtype=int)])
    return np.asarray(sorted(selected), dtype=int)


def _file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class QlibTimesFM3Model(Model):
    """Official frozen checkpoint, shared-loop adapter, or gated adapter control.

    The two fitted variants share the same bounded, member-balanced fitting
    samples and epoch-zero-eligible validation selection. ``adapter_state_``
    retains only the small local trainable tensors. Qlib serialization excludes
    the private loaded 330M backbone; saved models reload the explicit checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        mode: str = "frozen",
        context_length: int = 128,
        batch_size: int = 64,
        num_threads: int = 2,
        device: str = "cpu",
        train_samples: int = 1024,
        valid_samples: int = 512,
        epochs: int = 3,
        seed: int = 42,
        lr: float = 0.001,
        prediction_cache_dir: str | Path | None = None,
    ):
        if mode not in {"frozen", "loop", "adapter"}:
            raise ValueError("mode must be frozen, loop, or adapter.")
        checked = OfficialTimesFM3Model(
            checkpoint_path, context_length, batch_size, device, num_threads
        )
        for name, value in {
            "train_samples": train_samples,
            "valid_samples": valid_samples,
            "epochs": epochs,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (float, int))
            or not math.isfinite(lr)
            or lr <= 0
        ):
            raise ValueError("lr must be finite and positive.")
        self.checkpoint_path = checked.checkpoint_path
        self.mode, self.context_length, self.batch_size = mode, context_length, batch_size
        self.num_threads, self.device = num_threads, device
        self.train_samples, self.valid_samples, self.epochs = train_samples, valid_samples, epochs
        self.seed, self.lr = seed, float(lr)
        self.prediction_cache_dir = (
            str(Path(prediction_cache_dir).expanduser().absolute())
            if prediction_cache_dir is not None
            else None
        )
        if self.prediction_cache_dir is not None:
            cache = Path(self.prediction_cache_dir)
            checkpoint = Path(self.checkpoint_path)
            if cache == checkpoint or checkpoint in cache.parents:
                raise ValueError("Prediction cache must be outside the official checkpoint.")
        self.fitted = False
        self.adapter_state_ = None
        self.best_epoch_ = None
        self.parameter_count_ = None
        self.trainable_parameter_count_ = 0
        self.target_mean_, self.target_scale_ = 0.0, 1.0
        self.evals_result_ = {"train": [], "valid": []}
        self.training_curves_ = []
        self.manifest_ = {}
        self._backend = None

    def _new_backend(self):
        if self.mode == "frozen":
            backend = OfficialTimesFM3Model(
                self.checkpoint_path,
                self.context_length,
                self.batch_size,
                self.device,
                self.num_threads,
            )
            backend.fit(None)
            return backend
        from financial_ts.timesfm_comparison import GatedAdapterTimesFM3
        from financial_ts.timesfm_loop import LoopTimesFM3

        cls = LoopTimesFM3 if self.mode == "loop" else GatedAdapterTimesFM3
        return cls(self.checkpoint_path, device=self.device, num_threads=self.num_threads)

    def _get_backend(self):
        backend = getattr(self, "_backend", None)
        if backend is None:
            self._verify_checkpoint_identity()
            backend = self._new_backend()
            if self.mode != "frozen" and self.adapter_state_ is not None:
                backend.load_adapter_state_dict(self.adapter_state_)
            self._backend = backend
        return backend

    def _verify_checkpoint_identity(self):
        """Reject replaced weights before lazy reload or any prediction-cache reuse."""
        if not self.fitted:
            return
        expected = self.manifest_.get("checkpoint_files", [])
        if {item.get("name") for item in expected} != {"config.json", "model.safetensors"}:
            raise ValueError("The fitted model has no complete checkpoint identity.")
        current = [_file_identity(Path(self.checkpoint_path) / item["name"]) for item in expected]
        if current != expected:
            raise ValueError("Local checkpoint bytes differ from the fitted model identity.")

    def _predict_contexts(self, contexts: list[np.ndarray]) -> np.ndarray:
        import torch

        backend = self._get_backend()
        if self.mode == "frozen":
            outputs = list(
                backend._get_forecaster().predict_batch(
                    contexts=contexts,
                    horizon=2,
                    return_quantiles=False,
                    use_symmetric_averaging=False,
                )
            )
            if len(outputs) != len(contexts):
                raise ValueError("Official backend returned the wrong number of forecasts.")
            closes = np.asarray([output.forecast for output in outputs], dtype=np.float64)
            if (
                closes.shape != (len(contexts), 2)
                or not np.isfinite(closes).all()
                or (closes <= 0).any()
            ):
                raise ValueError("Official backend must return two finite positive closes.")
            return closes[:, 1] / closes[:, 0] - 1
        backend.eval()
        with torch.no_grad():
            scores = backend.return_scores(torch.as_tensor(np.stack(contexts), device=self.device))
        result = scores.detach().cpu().numpy().astype(np.float64)
        if result.shape != (len(contexts),) or not np.isfinite(result).all():
            raise ValueError("Adapter backend must return one finite score per context.")
        return result

    def _supervised(self, dataset, segment: str, budget: int):
        # These raw labels retain return units and Qlib's historical membership.
        frame = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_R)
        index = _index(frame)
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 1:
            raise ValueError("Adapter training requires one raw-return label column.")
        labels = frame.iloc[:, 0].to_numpy(dtype=np.float64)
        finite = np.isfinite(labels)
        index, labels = index[finite], labels[finite]
        if len(index) == 0:
            raise ValueError(f"No finite {segment} labels.")
        bank = _CloseContextBank(index, self.context_length)
        eligible = bank.eligible(index)
        index, labels = index[eligible], labels[eligible]
        if len(index) == 0:
            raise ValueError(f"No causal positive close contexts in {segment}.")
        chosen = _balanced_rows(index, budget)
        index, labels = index[chosen], labels[chosen]
        contexts, fills = bank.contexts(index, pad=True)
        records = [
            {
                "datetime": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "instrument": str(instrument),
                "label": float(label),
                "context_sha256": hashlib.sha256(context.astype("<f4").tobytes()).hexdigest(),
            }
            for date, instrument, label, context in zip(
                index.get_level_values("datetime"),
                index.get_level_values("instrument"),
                labels,
                contexts,
            )
        ]
        return np.stack(contexts), labels, records, fills

    def _rmse(self, contexts, labels):
        scores = np.concatenate(
            [
                self._predict_contexts(list(contexts[start : start + self.batch_size]))
                for start in range(0, len(contexts), self.batch_size)
            ]
        )
        return float(np.sqrt(np.mean((scores - labels) ** 2)))

    def fit(self, dataset, evals_result: dict | None = None):
        import torch

        started = time.perf_counter()
        self.fitted = False
        self.adapter_state_ = None
        self.training_curves_ = []
        self.evals_result_ = {"train": [], "valid": []}
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.num_threads)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self._backend = self._new_backend()
        backend = self._backend
        checkpoint = Path(self.checkpoint_path)
        self.manifest_ = {
            "mode": self.mode,
            "context_length": self.context_length,
            "seed": self.seed,
            "selection": "minimum sampled validation RMSE, including unchanged epoch zero",
            "sampling": "balanced member quotas; evenly spaced eligible dates within member",
            "inference_sample_cap": None,
            "context_policy": "causal forward-fill inside context; trim leading missing prefix",
            "adapter_padding": "left-pad shortened contexts using their earliest observed close",
            "test_labels_read_during_fit": False,
            "checkpoint_files": [
                _file_identity(checkpoint / name) for name in ("config.json", "model.safetensors")
            ],
            "source_files": [
                _file_identity(Path(__file__).with_name(name))
                for name in (
                    "qlib_timesfm.py",
                    "timesfm_pretrained.py",
                    "timesfm_loop.py",
                    "timesfm_comparison.py",
                )
            ],
        }
        self.checkpoint_config_sha256_ = self.manifest_["checkpoint_files"][0]["sha256"]
        self.checkpoint_size_bytes_ = self.manifest_["checkpoint_files"][1]["size_bytes"]
        if self.mode == "frozen":
            self.parameter_count_ = backend.parameter_count_
            self.best_epoch_ = 0
            self.manifest_["selection"] = "fixed pretrained checkpoint; no fitting or labels"
        else:
            train_x, train_y, train_records, train_fills = self._supervised(
                dataset, "train", self.train_samples
            )
            valid_x, valid_y, valid_records, valid_fills = self._supervised(
                dataset, "valid", self.valid_samples
            )
            if max(row["datetime"] for row in train_records) >= min(
                row["datetime"] for row in valid_records
            ):
                raise ValueError("Training and validation samples must be chronological.")
            base = list(backend.base_parameters())
            trainable = {name: p for name, p in backend.named_parameters() if p.requires_grad}
            initial = backend.adapter_state_dict()
            if (
                not base
                or any(parameter.requires_grad for parameter in base)
                or not trainable
                or set(trainable) != set(initial)
                or {id(p) for p in base} & {id(p) for p in trainable.values()}
            ):
                raise ValueError("Only saved adapter tensors may be trainable.")
            self.parameter_count_ = sum(p.numel() for p in backend.parameters())
            self.trainable_parameter_count_ = sum(p.numel() for p in trainable.values())
            self.target_mean_ = float(train_y.mean())
            scale = float(train_y.std(ddof=0))
            self.target_scale_ = scale if scale > 1e-12 else 1.0
            self.samples_ = {"train": train_records, "valid": valid_records}
            self.manifest_.update(
                train_samples=len(train_y),
                valid_samples=len(valid_y),
                requested_train_samples=self.train_samples,
                requested_valid_samples=self.valid_samples,
                context_fills={"train": train_fills, "valid": valid_fills},
                target_mean=self.target_mean_,
                target_scale=self.target_scale_,
                base_frozen=True,
                optimizer_names=list(trainable),
            )
            train_rmse, best_rmse = self._rmse(train_x, train_y), self._rmse(valid_x, valid_y)
            best_state, self.best_epoch_ = initial, 0
            self.training_curves_.append(
                {"epoch": 0, "train_rmse": train_rmse, "validation_rmse": best_rmse, "updates": 0}
            )
            optimizer = torch.optim.AdamW(trainable.values(), lr=self.lr, weight_decay=0.0)
            generator = np.random.default_rng(self.seed)
            updates = 0
            for epoch in range(1, self.epochs + 1):
                backend.train()
                squared_error, count = 0.0, 0
                order = generator.permutation(len(train_y))
                for start in range(0, len(order), self.batch_size):
                    batch = order[start : start + self.batch_size]
                    x = torch.as_tensor(train_x[batch], device=self.device)
                    y = torch.as_tensor(train_y[batch], device=self.device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = backend.return_scores(x)
                    loss = ((prediction - y) / self.target_scale_).square().mean()
                    if prediction.shape != y.shape or not torch.isfinite(loss):
                        raise ValueError("Invalid adapter training scores or loss.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(trainable.values()), 1.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    squared_error += float((prediction.detach().double() - y).square().sum())
                    count += len(batch)
                    updates += 1
                if any(p.grad is not None for p in base):
                    raise ValueError("Frozen official parameters received a gradient.")
                valid_rmse = self._rmse(valid_x, valid_y)
                self.training_curves_.append(
                    {
                        "epoch": epoch,
                        "train_rmse": math.sqrt(squared_error / count),
                        "validation_rmse": valid_rmse,
                        "updates": updates,
                    }
                )
                if valid_rmse < best_rmse:
                    best_rmse, best_state, self.best_epoch_ = (
                        valid_rmse,
                        backend.adapter_state_dict(),
                        epoch,
                    )
            backend.load_adapter_state_dict(best_state)
            self.adapter_state_ = backend.adapter_state_dict()
            self.manifest_.update(
                best_epoch=self.best_epoch_,
                selected_sampled_validation_rmse=best_rmse,
                epochs_run=self.epochs,
                optimizer_updates=updates,
                training_examples_seen=len(train_y) * self.epochs,
            )
        self.evals_result_ = {
            "train": [row["train_rmse"] ** 2 for row in self.training_curves_],
            "valid": [row["validation_rmse"] ** 2 for row in self.training_curves_],
        }
        self.manifest_.update(
            parameter_count=self.parameter_count_,
            trainable_parameter_count=self.trainable_parameter_count_,
            training_curves=self.training_curves_,
            target_statistics_fit_partition="train" if self.mode != "frozen" else None,
        )
        if evals_result is not None:
            evals_result.update(self.evals_result_)
        self.fit_seconds_ = time.perf_counter() - started
        self.fitted = True
        return self

    def predict(self, dataset, segment="test") -> pd.Series:
        if not self.fitted:
            raise ValueError("Fit the model before prediction.")
        self._verify_checkpoint_identity()
        prepared = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = _index(prepared)
        del prepared
        if len(index) == 0:
            return pd.Series([], index=index, dtype=float, name="score")
        bank = _CloseContextBank(index, self.context_length)
        scores = np.full(len(index), np.nan, dtype=np.float64)
        fills = {"forward_filled_values": 0, "leading_padded_values": 0}
        cache, done = self._prediction_cache(index, bank, segment, scores)
        chunk_start = done
        started = time.perf_counter()
        for start in range(done, len(index), self.batch_size):
            batch_index = index[start : start + self.batch_size]
            contexts, batch_fills = bank.contexts(batch_index, pad=self.mode != "frozen")
            batch_scores = self._predict_contexts(contexts)
            if batch_scores.shape != (len(batch_index),) or not np.isfinite(batch_scores).all():
                raise ValueError("Predictions must contain one finite score per requested row.")
            scores[start : start + len(batch_scores)] = batch_scores
            for name in fills:
                fills[name] += batch_fills[name]
            completed = start + len(batch_scores)
            batches = (completed - done + self.batch_size - 1) // self.batch_size
            if batches % 100 == 0 or completed == len(index):
                if cache is not None:
                    temporary = cache / "pending.npz"
                    values = scores[chunk_start:completed]
                    np.savez(
                        temporary,
                        start=chunk_start,
                        scores=values,
                        sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    )
                    temporary.replace(cache / f"part-{chunk_start:09d}.npz")
                    chunk_start = completed
                progress = {
                    "mode": self.mode,
                    "segment": str(segment),
                    "completed": completed,
                    "total": len(index),
                    "resumed_rows": done,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if cache is not None:
                    temporary = cache / "progress.json.tmp"
                    temporary.write_text(json.dumps(progress) + "\n")
                    temporary.replace(cache / "progress.json")
                print(json.dumps({"timesfm_progress": progress}), flush=True)
        self.last_prediction_ = {
            "segment": str(segment),
            "rows": len(index),
            "resumed_rows": done,
            "context_fill_counts_scope": "newly computed rows only",
            **fills,
        }
        return pd.Series(scores, index=index, name="score", dtype=float)

    def _prediction_cache(self, index, bank, segment, scores):
        if self.prediction_cache_dir is None:
            return None, 0
        digest = hashlib.sha256()
        settings = {
            "mode": self.mode,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "device": self.device,
            "num_threads": self.num_threads,
            "checkpoint": self.manifest_.get("checkpoint_files"),
            "source": [
                _file_identity(Path(__file__).with_name(name))
                for name in (
                    "qlib_timesfm.py",
                    "timesfm_pretrained.py",
                    "timesfm_loop.py",
                    "timesfm_comparison.py",
                )
            ],
        }
        digest.update(json.dumps(settings, sort_keys=True).encode())
        for name, value in sorted((self.adapter_state_ or {}).items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        digest.update(pd.util.hash_pandas_object(index).to_numpy().tobytes())
        digest.update(bank.calendar.asi8.tobytes())
        for instrument, values in sorted(bank.prices.items()):
            digest.update(instrument.encode())
            digest.update(values.astype("<f4").tobytes())
        identity = digest.hexdigest()
        segment_name = str(segment)
        if segment_name not in {"train", "valid", "test"}:
            raise ValueError("Prediction caching supports train, valid, or test segments.")
        directory = Path(self.prediction_cache_dir) / f"{segment_name}-{identity}"
        directory.mkdir(parents=True, exist_ok=True)
        identity_file = directory / "identity.json"
        expected = {"fingerprint": identity, "rows": len(index), "settings": settings}
        if identity_file.exists():
            if json.loads(identity_file.read_text()) != expected:
                raise ValueError("Prediction cache identity is inconsistent.")
        else:
            identity_file.write_text(json.dumps(expected, indent=2) + "\n")
        done = 0
        for path in sorted(directory.glob("part-*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                start, values = int(saved["start"]), saved["scores"]
                checksum = str(saved["sha256"]) if "sha256" in saved else None
            if (
                start != done
                or values.ndim != 1
                or len(values) == 0
                or start + len(values) > len(index)
                or not np.isfinite(values).all()
                or checksum != hashlib.sha256(values.tobytes()).hexdigest()
            ):
                raise ValueError("Prediction cache has a gap or an invalid score chunk.")
            scores[start : start + len(values)] = values
            done += len(values)
        return directory, done
