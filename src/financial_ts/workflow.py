"""Validated chronological research workflows and inspectable run artifacts."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from financial_ts.catalog import MODEL_NAMES, NEURAL_MODELS

LABEL = "Ref($close, -2) / Ref($close, -1) - 1"
LABEL_HORIZON = 2
FEATURE_COUNT = 157


def resolve_segments(
    calendar: pd.DatetimeIndex,
    segments: dict | None = None,
    horizon: int = LABEL_HORIZON,
    warmup: int = 60,
) -> dict[str, tuple[str, str]]:
    """Resolve dates and reject labels crossing chronological split boundaries.

    Calendar positions, rather than weekdays or calendar-day offsets, determine
    label availability. Auto splits reserve historical feature warmup, two
    sessions at each boundary, and two sessions after the final test prediction.
    """
    cal = pd.DatetimeIndex(calendar)
    if cal.empty or cal.hasnans or not cal.is_unique or not cal.is_monotonic_increasing:
        raise ValueError("The provider calendar must be nonempty, sorted, and unique.")
    if cal.tz is not None or not (cal == cal.normalize()).all():
        raise ValueError("Expected timezone-naive daily trading dates.")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions.")
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative number of trading sessions.")
    names = ("train", "valid", "test")
    positions: dict[str, tuple[int, int]] = {}
    if segments is None:
        usable = len(cal) - horizon - warmup
        if usable < max(20, 5 * (horizon + 1)):
            raise ValueError("Not enough trading sessions after feature warmup and label purging.")
        valid_start = warmup + int(usable * 0.6)
        test_start = warmup + int(usable * 0.8)
        positions = {
            "train": (warmup, valid_start - horizon - 1),
            "valid": (valid_start, test_start - horizon - 1),
            "test": (test_start, len(cal) - horizon - 1),
        }
    else:
        if set(segments) != set(names):
            raise ValueError("segments must contain exactly train, valid, and test.")
        for name in names:
            bounds = segments[name]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"{name} requires a [start, end] date pair.")
            start, end = (pd.Timestamp(value) for value in bounds)
            if pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None:
                raise ValueError(f"{name} dates must be finite and timezone-naive.")
            if start != start.normalize() or end != end.normalize():
                raise ValueError(f"{name} bounds must be daily dates without time components.")
            if start > end or start < cal[0] or end > cal[-1]:
                raise ValueError(f"{name} bounds are outside the provider calendar or reversed.")
            first = int(cal.searchsorted(start, side="left"))
            last = int(cal.searchsorted(end, side="right")) - 1
            positions[name] = (first, last)
    for name, (start, end) in positions.items():
        if start > end:
            raise ValueError(f"{name} contains no trading sessions after purging.")
        if end + horizon >= len(cal):
            raise ValueError(f"{name} needs {horizon} more provider sessions for its labels.")
    for left, right in zip(names, names[1:]):
        if positions[left][1] + horizon >= positions[right][0]:
            raise ValueError(
                f"{left} labels overlap {right}: leave at least {horizon} provider "
                "trading sessions between the feature segments."
            )
    return {
        name: (cal[first].strftime("%Y-%m-%d"), cal[last].strftime("%Y-%m-%d"))
        for name, (first, last) in positions.items()
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Load data-only YAML; paths are relative to the working directory."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The workflow configuration must be a YAML mapping.")
    allowed = {
        "provider_uri",
        "region",
        "instruments",
        "output_dir",
        "experiment_name",
        "seed",
        "sequence_length",
        "warmup_sessions",
        "data_start_time",
        "data_end_time",
        "handler_cache_path",
        "shared_windows",
        "segments",
        "model",
        "backtest",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return config


def _read_calendar(provider: Path) -> pd.DatetimeIndex:
    path = provider / "calendars" / "day.txt"
    if not path.is_file():
        raise ValueError(f"No daily Qlib calendar at {path}; run fts import-csv first.")
    return pd.DatetimeIndex(pd.to_datetime(path.read_text().splitlines(), errors="raise"))


def _window_calendar(config: dict, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Bound feature calculation to observed dates without modifying the provider."""
    cal = pd.DatetimeIndex(calendar)
    if cal.empty:
        raise ValueError("The provider calendar is empty.")
    bounds = []
    for name, default in (("data_start_time", cal[0]), ("data_end_time", cal[-1])):
        date = pd.Timestamp(config.get(name, default))
        if pd.isna(date) or date.tz is not None or date != date.normalize():
            raise ValueError(f"{name} must be a finite, timezone-naive daily date.")
        if date < cal[0] or date > cal[-1]:
            raise ValueError(f"{name} is outside the provider calendar.")
        bounds.append(date)
    start, end = bounds
    if start > end:
        raise ValueError("data_start_time must not follow data_end_time.")
    cal = cal[(cal >= start) & (cal <= end)]
    if cal.empty:
        raise ValueError("The requested data window contains no trading sessions.")
    return cal


class ValidationSignalRecord:
    """Record validation forecasts while the fitted task's live dataset is available."""

    def __init__(self, model, dataset, recorder):
        self.model, self.dataset, self.recorder = model, dataset, recorder

    def generate(self):
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
        from qlib.utils import class_casting

        prediction = self.model.predict(self.dataset, segment="valid")
        if isinstance(prediction, pd.Series):
            prediction = prediction.to_frame("score")
        # TSDatasetH's normal prepare returns a window sampler. Raw labels use
        # DatasetH's tabular view, exactly as Qlib's native SignalRecord does.
        with class_casting(self.dataset, DatasetH):
            label = self.dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R)
        self.recorder.save_objects(
            **{"validation_pred.pkl": prediction, "validation_label.pkl": label}
        )


def build_task(config: dict, calendar: pd.DatetimeIndex) -> tuple[dict, dict]:
    """Build a native Qlib task, with immutable label semantics and fitted dates."""
    calendar = _window_calendar(config, calendar)
    # Qlib's executor uses the following real calendar session to close a step.
    # Reserve it in addition to the two future closes needed for the label.
    split_calendar = calendar[:-1] if config.get("backtest", {}).get("enabled", False) else calendar
    segments = resolve_segments(
        split_calendar, config.get("segments"), warmup=config.get("warmup_sessions", 60)
    )
    seed = int(config.get("seed", 42))
    model_spec = config.get("model", {"name": "lightgbm"})
    if not isinstance(model_spec, dict) or set(model_spec) - {"name", "params"}:
        raise ValueError("model accepts name and params only.")
    name = model_spec.get("name", "lightgbm")
    params = copy.deepcopy(model_spec.get("params", {}))
    if not isinstance(params, dict):
        raise ValueError("model.params must be a mapping.")
    if name == "lightgbm":
        defaults = {
            "loss": "mse",
            "num_boost_round": 300,
            "early_stopping_rounds": 30,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 30,
            "num_threads": 2,
            "seed": seed,
            "deterministic": True,
            "force_col_wise": True,
        }
        defaults.update(params)
        model = {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt", "kwargs": defaults}
    elif name in NEURAL_MODELS:
        if "d_feat" in params and params["d_feat"] != FEATURE_COUNT:
            raise ValueError(f"This OHLCV handler requires d_feat={FEATURE_COUNT}.")
        if "architecture" in params:
            raise ValueError(
                "Choose the architecture with model.name, not model.params.architecture."
            )
        defaults = {"d_feat": FEATURE_COUNT, "seed": seed, "architecture": name}
        defaults.update(params)
        model = {
            "class": "SequenceForecastModel",
            "module_path": "financial_ts.model",
            "kwargs": defaults,
        }
    elif name == "timesfm3_panel":
        model = {
            "class": "QlibTimesFM3Model",
            "module_path": "financial_ts.qlib_timesfm",
            "kwargs": params,
        }
    elif name == "timesfm3_pretrained":
        model = {
            "class": "OfficialTimesFM3Model",
            "module_path": "financial_ts.timesfm_pretrained",
            "kwargs": params,
        }
    else:
        raise ValueError(f"model.name must be one of {MODEL_NAMES}.")
    handler = {
        "class": "OHLCVAlpha158",
        "module_path": "financial_ts.handler",
        "kwargs": {
            "instruments": config.get("instruments", "all"),
            "start_time": calendar[0].strftime("%Y-%m-%d"),
            "end_time": calendar[-1].strftime("%Y-%m-%d"),
            "fit_start_time": segments["train"][0],
            "fit_end_time": segments["train"][1],
            "label": ([LABEL], ["LABEL0"]),
            "infer_processors": [
                {"class": "FiniteFeatures", "module_path": "financial_ts.handler"},
                {
                    "class": "RobustZScoreNorm",
                    "kwargs": {"fields_group": "feature", "clip_outlier": True},
                },
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            # Raw returns are meaningful for both a single index and a stock universe.
            "learn_processors": [{"class": "DropnaLabel"}],
        },
    }
    if name == "timesfm3_panel":
        # The official checkpoint consumes prices, not the normalized factors.
        # Keep exactly the same membership/date/label panel with a smaller cache.
        handler = {
            "class": "CloseOnlyHandler",
            "module_path": "financial_ts.qlib_timesfm",
            "kwargs": {
                key: handler["kwargs"][key]
                for key in (
                    "instruments",
                    "start_time",
                    "end_time",
                    "fit_start_time",
                    "fit_end_time",
                    "label",
                )
            },
        }
    dataset_kwargs = {"handler": handler, "segments": segments}
    shared_windows = config.get("shared_windows", False)
    if type(shared_windows) is not bool or (shared_windows and name not in NEURAL_MODELS):
        raise ValueError("shared_windows must be boolean and is supported only by neural models.")
    if name in NEURAL_MODELS:
        step_len = config.get("sequence_length", 20)
        if not isinstance(step_len, int) or step_len < 1:
            raise ValueError("sequence_length must be a positive integer.")
        dataset_kwargs["step_len"] = step_len
    task = {
        "model": model,
        "dataset": {
            "class": "SharedTSDatasetH"
            if shared_windows
            else "TSDatasetH"
            if name in NEURAL_MODELS
            else "DatasetH",
            "module_path": "financial_ts.shared_panel" if shared_windows else "qlib.data.dataset",
            "kwargs": dataset_kwargs,
        },
        "record": [
            {"class": "ValidationSignalRecord", "module_path": "financial_ts.workflow"},
            {"class": "SignalRecord", "module_path": "qlib.workflow.record_temp"},
        ],
    }
    return task, segments


def _portfolio_config(config: dict, segments: dict, calendar: pd.DatetimeIndex, count: int) -> dict:
    bt = config.get("backtest", {})
    cn = config.get("region", "us") == "cn"
    topk, n_drop = bt.get("topk", 50 if cn else 5), bt.get("n_drop", 5 if cn else 1)
    if type(topk) is not int or type(n_drop) is not int or not 1 <= topk <= count:
        raise ValueError(f"backtest.topk must be an integer between 1 and {count} instruments.")
    if not 0 <= n_drop <= topk:
        raise ValueError("backtest.n_drop must be between 0 and topk.")
    costs = {
        key: float(bt.get(key, default))
        for key, default in (
            ("open_cost", 0.0005 if cn else 0.001),
            ("close_cost", 0.0015 if cn else 0.001),
            ("min_cost", 5.0 if cn else 0.0),
        )
    }
    if any(not math.isfinite(value) or value < 0 for value in costs.values()):
        raise ValueError("Backtest costs must be finite and nonnegative.")
    account = float(bt.get("account", 100000000 if cn else 100000))
    risk = float(bt.get("risk_degree", 0.95))
    if not math.isfinite(account) or account <= 0 or not 0 < risk <= 1:
        raise ValueError("Backtest account must be positive and risk_degree must be in (0,1].")
    limit = bt.get("limit_threshold", 0.095 if cn else None)
    if limit is not None:
        limit = float(limit)
        if not math.isfinite(limit) or not 0 < limit < 1:
            raise ValueError("backtest.limit_threshold must be None or a finite value in (0,1).")
    trade_unit = bt.get("trade_unit", 100 if cn else None)
    if trade_unit is not None and (type(trade_unit) is not int or trade_unit < 1):
        raise ValueError("backtest.trade_unit must be None or a positive integer.")
    first = int(calendar.get_loc(pd.Timestamp(segments["test"][0])))
    last = int(calendar.get_loc(pd.Timestamp(segments["test"][1])))
    return {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": {
                    "class": "DeterministicRankSignal",
                    "module_path": "financial_ts.cn_portfolio",
                    "kwargs": {"signal": "<PRED>"},
                }
                if cn
                else "<PRED>",
                "topk": topk,
                "n_drop": n_drop,
                "risk_degree": risk,
            },
        },
        "backtest": {
            # A score at close t enters at close t+1; t+2 values its label return.
            "start_time": calendar[first + 1].strftime("%Y-%m-%d"),
            "end_time": calendar[last + LABEL_HORIZON].strftime("%Y-%m-%d"),
            "account": account,
            "benchmark": bt.get("benchmark", "SH000300" if cn else None),
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": limit,
                "deal_price": "close",
                "trade_unit": trade_unit,
                **costs,
            },
        },
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _save_json(path: Path, value: Any):
    path.write_text(
        json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _provider_fingerprint(provider: Path) -> dict:
    """Hash the actual consumed daily provider, not only its filename or date."""
    digest = hashlib.sha256()
    paths = sorted(
        set(provider.glob("calendars/*.txt"))
        | set(provider.glob("instruments/*.txt"))
        | set(provider.glob("features/*/*.day.bin"))
    )
    nbytes = 0
    for path in paths:
        digest.update(path.relative_to(provider).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                nbytes += len(chunk)
    return {"sha256": digest.hexdigest(), "file_count": len(paths), "bytes": nbytes}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_handler(
    path: str | Path, handler_config: dict, provider_fingerprint: dict, region: str
) -> tuple[str, dict]:
    """Create/reuse a trusted local handler; reject stale or corrupted cache files.

    Qlib serializes handlers with pickle. This cache is only for files created
    locally by this workflow, never downloaded model/data artifacts. The
    sidecar binds the exact provider bytes, processing recipe and fit interval
    to the serialized bytes before Qlib is allowed to deserialize them.
    """
    from filelock import FileLock
    from qlib.utils import init_instance_by_config

    cache = Path(path).expanduser().resolve()
    sidecar = cache.with_suffix(cache.suffix + ".json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    expected = _json_ready(
        {
            "schema_version": 1,
            "origin": "financial_ts.workflow locally generated handler",
            "raw_storage": "labels_only; processed infer and learn features retained",
            "provider": provider_fingerprint,
            "region": region,
            "handler": handler_config,
            "handler_source_sha256": _file_sha256(Path(__file__).with_name("handler.py")),
            "packages": {
                name: importlib.metadata.version(name) for name in ("pyqlib", "pandas", "numpy")
            },
        }
    )
    with FileLock(str(cache) + ".lock", timeout=3600):
        if cache.exists() or sidecar.exists():
            if not cache.is_file() or not sidecar.is_file():
                raise ValueError("Handler cache and its provenance sidecar must both exist.")
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if saved.get("identity") != expected:
                raise ValueError(
                    "Handler cache does not match the provider, data/fit dates, universe, "
                    "or processing recipe; choose a new handler_cache_path."
                )
            if saved.get("sha256") != _file_sha256(cache):
                raise ValueError("Handler cache SHA256 differs from its provenance sidecar.")
            return cache.as_uri(), {**saved, "reused": True}
        handler = init_instance_by_config(copy.deepcopy(handler_config))
        # Native SignalRecord needs DK_R labels, but no shipped model consumes
        # the unprocessed factors. Avoid a third full factor copy in the cache.
        handler._data = handler._data.loc[:, ["label"]].copy()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".pkl", delete=False) as f:
                temporary = Path(f.name)
            handler.to_pickle(temporary, dump_all=True)
            saved = {
                "identity": expected,
                "sha256": _file_sha256(temporary),
                "bytes": temporary.stat().st_size,
            }
            temporary.replace(cache)
            _save_json(sidecar, saved)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return cache.as_uri(), {**saved, "reused": False}


def _prediction_statistics(joined: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Score raw returns and daily cross-sectional correlations using Qlib."""
    from qlib.contrib.eva.alpha import calc_ic

    if joined.empty or not joined.index.is_unique or not np.isfinite(joined["score"]).all():
        raise ValueError("Model predictions are empty, duplicated, or nonfinite.")
    if np.isinf(joined["label"].to_numpy()).any():
        raise ValueError("Labels contain infinite returns; check price magnitudes/adjustments.")
    finite = joined.dropna(subset=["score", "label"])
    if finite.empty:
        raise ValueError("No evaluable labels; check prices and provider coverage.")
    errors = finite["score"] - finite["label"]
    # A constant forecast or a one-instrument day has undefined correlation.
    # Retain those dates as NaN rather than interpreting them as zero skill.
    if finite.groupby(level="datetime").size().max() > 1:
        ic, rank_ic = calc_ic(finite["score"], finite["label"])
    else:
        dates = finite.index.get_level_values("datetime").unique()
        ic = pd.Series(np.nan, index=dates, dtype=float)
        rank_ic = ic.copy()
    daily = pd.DataFrame({"ic": ic, "rank_ic": rank_ic})
    stats = {
        "prediction_count": len(joined),
        "evaluated_count": len(finite),
        "unavailable_label_count": len(joined) - len(finite),
        "rmse": np.sqrt(np.mean(errors**2)),
        "mae": np.mean(np.abs(errors)),
        "zero_return_rmse": np.sqrt(np.mean(finite["label"] ** 2)),
        "directional_accuracy": np.mean(np.sign(finite["score"]) == np.sign(finite["label"])),
        "pooled_pearson_correlation": finite["score"].corr(finite["label"]),
        "ic": ic.mean(),
        "rank_ic": rank_ic.mean(),
        "ic_ir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "rank_ic_ir": rank_ic.mean() / rank_ic.std() if rank_ic.std() > 0 else np.nan,
        "ic_days": int(ic.notna().sum()),
        "rank_ic_days": int(rank_ic.notna().sum()),
    }
    return stats, daily


def run_workflow(config: dict) -> dict:
    """Fit once, evaluate the untouched test split, and export a native Qlib run.

    Qlib maintains global provider state: call separate workflows sequentially,
    or use separate processes for parallel experiments.
    """
    import qlib
    from qlib.data import D
    from qlib.model.trainer import task_train

    started = time.perf_counter()
    config = copy.deepcopy(config)
    provider = Path(config.get("provider_uri", ".data/qlib")).expanduser().resolve()
    output = Path(config.get("output_dir", "runs/research")).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        model_spec = config.get("model", {})
        cache = model_spec.get("params", {}).get("prediction_cache_dir")
        cache_only = (
            model_spec.get("name") == "timesfm3_panel"
            and cache is not None
            and Path(cache).expanduser().resolve() == output / "inference_cache"
            and list(output.iterdir()) == [output / "inference_cache"]
            and (output / "inference_cache").is_dir()
        )
        if not cache_only:
            raise ValueError(f"Run directory is not empty: {output}. Choose a new output_dir.")
    if output == provider or provider in output.parents or output in provider.parents:
        raise ValueError("Run output and the Qlib data provider must be separate directories.")
    region = config.get("region", "us")
    if region not in {"us", "cn"}:
        raise ValueError("region must be us or cn.")
    calendar = _read_calendar(provider)
    data_manifest = {}
    manifest = provider / "metadata.json"
    if manifest.exists():
        data_manifest = json.loads(manifest.read_text())
        observed_end = data_manifest.get("target_trade_date")
        if observed_end is not None:
            observed_end = pd.Timestamp(observed_end)
            requested_end = pd.Timestamp(config.get("data_end_time", observed_end))
            if requested_end > observed_end:
                raise ValueError("data_end_time exceeds the provider's observed target_trade_date.")
            config["data_end_time"] = requested_end.strftime("%Y-%m-%d")
    task, segments = build_task(config, calendar)
    calendar = _window_calendar(config, calendar)
    output.mkdir(parents=True, exist_ok=True)
    config.update(provider_uri=str(provider), output_dir=str(output), segments=segments)
    run_init = {
        "provider_uri": str(provider),
        "region": region,
        "kernels": 1,
        "expression_cache": None,
        "dataset_cache": None,
        "exp_manager": {
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": (output / "mlruns").as_uri(), "default_exp_name": "financial_ts"},
        },
    }
    qlib.init(**run_init)
    instruments = config.get("instruments", "all")
    if not isinstance(instruments, (str, list)) or not instruments:
        raise ValueError("instruments must be a market name or a nonempty list of symbols.")
    instrument_config = D.instruments(instruments if isinstance(instruments, str) else "all")
    available = D.list_instruments(
        instrument_config,
        start_time=segments["train"][0],
        end_time=segments["test"][1],
        as_list=True,
    )
    if isinstance(instruments, list):
        if any(not isinstance(symbol, str) for symbol in instruments):
            raise ValueError("All explicit instruments must be strings.")
        missing = set(instruments) - set(available)
        if missing:
            raise ValueError(
                f"Instruments not available in the requested date range: {sorted(missing)}"
            )
        available = sorted(set(instruments))
    if not available:
        raise ValueError("No instruments in the requested universe/date range.")
    if len(available) > 1:
        task["record"].append(
            {
                "class": "SigAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"ana_long_short": False, "ann_scaler": 252},
            }
        )
    backtest_enabled = bool(config.get("backtest", {}).get("enabled", False))
    if backtest_enabled:
        portfolio = _portfolio_config(config, segments, calendar, len(available))
        task["record"].append(
            {
                "class": "PortfolioRecord",
                "module_path": "financial_ts.record",
                "kwargs": {"config": portfolio},
            }
        )
    _save_json(output / "resolved_config.json", config)
    (output / "qlib_task.yaml").write_text(
        yaml.safe_dump(_json_ready({"qlib_init": run_init, "task": task}), sort_keys=False),
        encoding="utf-8",
    )
    metadata = {
        "status": "running",
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
        "provider": _provider_fingerprint(provider),
        "label": LABEL,
        "feature_count": 1
        if config.get("model", {}).get("name") == "timesfm3_panel"
        else FEATURE_COUNT,
        "segments": segments,
        "seed": task["model"]["kwargs"].get("seed"),
        "config_file": "resolved_config.json",
        "task_file": "qlib_task.yaml",
        "instruments": sorted(available),
        "synthetic": False,
    }
    if data_manifest:
        metadata["data_manifest"] = data_manifest
        metadata["synthetic"] = metadata["data_manifest"].get("synthetic", False)
    for package in (
        "financial-ts-qlib",
        "pyqlib",
        "numpy",
        "pandas",
        "lightgbm",
        "mlflow",
        "torch",
        "filelock",
        "timesfm",
    ):
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    # A command may be invoked from a different checkout; identify this package's
    # source repository only, rather than attributing an unrelated caller's HEAD.
    code_root = Path(__file__).resolve().parents[2]
    try:
        if not (code_root / ".git").exists():
            raise FileNotFoundError("Installed package has no source checkout metadata.")
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=code_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        metadata["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=code_root,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        metadata["git_commit"] = None
    _save_json(output / "run.json", metadata)
    try:
        if config.get("handler_cache_path"):
            cache_path = Path(config["handler_cache_path"]).expanduser().resolve()
            if cache_path == provider or provider in cache_path.parents:
                raise ValueError("handler_cache_path must be outside the market data provider.")
            handler, cache_metadata = _cached_handler(
                cache_path,
                task["dataset"]["kwargs"]["handler"],
                metadata["provider"],
                region,
            )
            task["dataset"]["kwargs"]["handler"] = handler
            metadata["handler_cache"] = cache_metadata
            _save_json(output / "run.json", metadata)
            (output / "qlib_task.yaml").write_text(
                yaml.safe_dump(_json_ready({"qlib_init": run_init, "task": task}), sort_keys=False),
                encoding="utf-8",
            )
        recorder = task_train(
            task,
            experiment_name=config.get("experiment_name", "financial_ts"),
            recorder_name=task["model"]["class"],
        )
        # These artifacts were created in this process, never obtained from an untrusted download.
        pred = recorder.load_object("pred.pkl").iloc[:, 0].rename("score")
        label = recorder.load_object("label.pkl").iloc[:, 0].rename("label")
        joined = pred.to_frame().join(label)
        metrics, daily_ic = _prediction_statistics(joined)
        joined.to_csv(output / "predictions.csv")
        daily_ic.to_csv(output / "daily_ic.csv")
        valid_pred = recorder.load_object("validation_pred.pkl").iloc[:, 0].rename("score")
        valid_label = recorder.load_object("validation_label.pkl").iloc[:, 0].rename("label")
        validation = valid_pred.to_frame().join(valid_label)
        metrics["validation"], valid_ic = _prediction_statistics(validation)
        validation.to_csv(output / "validation_predictions.csv")
        valid_ic.to_csv(output / "validation_daily_ic.csv")
        metrics["qlib_metrics"] = recorder.list_metrics()
        fitted_model = recorder.load_object("params.pkl")
        history = getattr(fitted_model, "evals_result_", {})
        valid_history = history.get("valid", [])
        metrics["model_details"] = {
            "name": config.get("model", {}).get("name", "lightgbm"),
            "parameter_count": getattr(fitted_model, "parameter_count_", None),
            "best_epoch": getattr(fitted_model, "best_epoch_", None),
            "epochs_run": len(valid_history) if valid_history else None,
            "validation_rmse": float(np.sqrt(min(valid_history))) if valid_history else None,
            "fit_seconds": getattr(fitted_model, "fit_seconds_", None),
            "pretrained": config.get("model", {}).get("name")
            in {"timesfm3_pretrained", "timesfm3_panel"},
        }
        if config.get("model", {}).get("name") == "timesfm3_panel":
            metrics["model_details"].update(
                mode=fitted_model.mode,
                trainable_parameter_count=fitted_model.trainable_parameter_count_,
                epochs_run=fitted_model.manifest_.get("epochs_run", 0),
            )
            _save_json(output / "fit_manifest.json", fitted_model.manifest_)
            _save_json(output / "fit_samples.json", getattr(fitted_model, "samples_", {}))
            _save_json(output / "training_curves.json", fitted_model.training_curves_)
        if config.get("model", {}).get("name", "lightgbm") == "lightgbm":
            booster = fitted_model.model
            valid_l2 = booster.best_score.get("valid", {}).get("l2")
            metrics["model_details"]["validation_rmse"] = (
                float(np.sqrt(valid_l2)) if valid_l2 is not None else None
            )
            metrics["model_details"]["best_boosting_iteration"] = booster.best_iteration
        metrics["model_details"]["validation_rmse"] = metrics["validation"]["rmse"]
        if config.get("model", {}).get("name") in {"timesfm3_pretrained", "timesfm3_panel"}:
            metadata["checkpoint_config_sha256"] = fitted_model.checkpoint_config_sha256_
            metadata["checkpoint_size_bytes"] = fitted_model.checkpoint_size_bytes_
            metadata["pretraining_overlap"] = (
                "unknown; historical replay is not a clean unseen holdout"
            )
        if history:
            _save_json(output / "learning_curves.json", history)
        del fitted_model
        if backtest_enabled:
            from financial_ts.cn_portfolio import portfolio_metrics

            report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
            report.to_csv(output / "portfolio.csv")
            metrics["portfolio"] = portfolio_metrics(report, portfolio, region)
        _save_json(output / "metrics.json", metrics)
        metadata.update(
            status="completed",
            elapsed_seconds=time.perf_counter() - started,
            recorder_id=recorder.info["id"],
            experiment_id=recorder.info["experiment_id"],
            finished_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        )
        _save_json(output / "run.json", metadata)
        return _json_ready({"output_dir": str(output), "run": metadata, "metrics": metrics})
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _save_json(output / "run.json", metadata)
        raise
