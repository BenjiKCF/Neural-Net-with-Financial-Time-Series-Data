"""Registered rolling research with causal folds and complete trial provenance.

The immutable protocol precedes every trial. Checkpoint fitting remains native
Qlib; selection uses validation only. Historical results are exploratory, and
portfolio scenarios retain the original simulator's accounting qualifications.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.metadata
import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from financial_ts.benchmark import benchmark_config
from financial_ts.catalog import CROSS_SECTIONAL_MODELS, NEURAL_MODELS, TRAINABLE_MODELS
from financial_ts.workflow import (
    LABEL,
    LABEL_HORIZON,
    _json_ready,
    _portfolio_config,
    _prediction_statistics,
    _provider_fingerprint,
    _read_calendar,
    resolve_segments,
    run_workflow,
)

ROLLING_MODELS = ("zero", "ridge", *TRAINABLE_MODELS, *CROSS_SECTIONAL_MODELS)
_SAFE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}\Z")


def _canonical(value):
    return json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(_json_ready(value), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def _positive(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _name(value, field):
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must be a safe, nonempty identifier.")
    return value


def _calendar(values):
    cal = pd.DatetimeIndex(values)
    if (
        cal.empty
        or cal.hasnans
        or not cal.is_unique
        or not cal.is_monotonic_increasing
        or cal.tz is not None
        or not (cal == cal.normalize()).all()
    ):
        raise ValueError("Calendar must contain sorted unique timezone-naive observed dates.")
    return cal


def make_rolling_folds(
    calendar,
    *,
    train_sessions=504,
    valid_sessions=126,
    test_sessions=63,
    step_sessions=None,
    warmup_sessions=60,
    label_endpoint_sessions=2,
    expanding=False,
    start=None,
    end=None,
    max_folds=None,
):
    """Construct disjoint evaluation windows from observed sessions.

    ``start`` is the first training feature date, not the first evaluation date.
    The validation label must be mature by close of the session preceding the
    first evaluation prediction. Feature warmup may precede training. One final
    observed session is reserved for the Qlib portfolio executor.
    """
    cal = _calendar(calendar)
    for key, value in (
        ("train_sessions", train_sessions),
        ("valid_sessions", valid_sessions),
        ("test_sessions", test_sessions),
        ("label_endpoint_sessions", label_endpoint_sessions),
    ):
        _positive(value, key)
    _positive(warmup_sessions, "warmup_sessions", minimum=0)
    step_sessions = test_sessions if step_sessions is None else step_sessions
    _positive(step_sessions, "step_sessions")
    if step_sessions < test_sessions:
        raise ValueError("Evaluation folds must not overlap: step_sessions >= test_sessions.")
    if type(expanding) is not bool:
        raise ValueError("expanding must be boolean.")
    if max_folds is not None:
        _positive(max_folds, "max_folds")
    for name, value in (("start", start), ("end", end)):
        if value is not None:
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp) or timestamp.tz is not None or timestamp != timestamp.normalize():
                raise ValueError(f"{name} must be a finite timezone-naive daily date.")
    first = warmup_sessions if start is None else int(cal.searchsorted(pd.Timestamp(start)))
    if first < warmup_sessions:
        raise ValueError("The first training date lacks the declared feature warmup.")
    last = len(cal) - label_endpoint_sessions - 2
    if end is not None:
        last = min(last, int(cal.searchsorted(pd.Timestamp(end), side="right")) - 1)
    folds = []
    shift = 0
    while max_folds is None or len(folds) < max_folds:
        train_first = first if expanding else first + shift
        train_last = first + train_sessions - 1 + shift
        valid_first = train_last + label_endpoint_sessions + 1
        valid_last = valid_first + valid_sessions - 1
        test_first = valid_last + label_endpoint_sessions + 1
        test_last = test_first + test_sessions - 1
        if test_last > last:
            break
        segments = {
            "train": (str(cal[train_first].date()), str(cal[train_last].date())),
            "valid": (str(cal[valid_first].date()), str(cal[valid_last].date())),
            "test": (str(cal[test_first].date()), str(cal[test_last].date())),
        }
        folds.append(
            {
                "id": f"fold-{len(folds):03d}",
                "segments": segments,
                "retrain_asof": str(cal[test_first - 1].date()),
                "training_label_endpoint": str(cal[train_last + label_endpoint_sessions].date()),
                "validation_label_endpoint": str(cal[valid_last + label_endpoint_sessions].date()),
                "data_start_time": str(cal[train_first - warmup_sessions].date()),
                "data_end_time": str(cal[test_last + label_endpoint_sessions + 1].date()),
            }
        )
        shift += step_sessions
    if not folds:
        raise ValueError(
            "No complete rolling folds fit the observed calendar and requested budget."
        )
    return json.loads(_canonical(folds))


def _validate_folds(cal, folds, warmup):
    if not isinstance(folds, list) or not folds:
        raise ValueError("folds must be a nonempty list.")
    result, names, previous_test_end = [], set(), None
    for number, source in enumerate(folds):
        if not isinstance(source, dict) or "segments" not in source:
            raise ValueError("Each explicit fold must be a mapping with segments.")
        name = _name(source.get("id", f"fold-{number:03d}"), "fold.id")
        if name in names:
            raise ValueError("Fold identifiers must be unique.")
        names.add(name)
        segments = resolve_segments(cal[:-1], source.get("segments"), horizon=LABEL_HORIZON)
        start, stop = map(pd.Timestamp, segments["test"])
        if previous_test_end is not None and start <= previous_test_end:
            raise ValueError("Evaluation folds must be chronological and nonoverlapping.")
        previous_test_end = stop
        train_start = int(cal.get_loc(pd.Timestamp(segments["train"][0])))
        train_end = int(cal.get_loc(pd.Timestamp(segments["train"][1])))
        valid_end = int(cal.get_loc(pd.Timestamp(segments["valid"][1])))
        test_start, test_end = int(cal.get_loc(start)), int(cal.get_loc(stop))
        if train_start < warmup:
            raise ValueError("A fold lacks the declared historical feature warmup.")
        cutoff = pd.Timestamp(source.get("retrain_asof", cal[test_start - 1]))
        if cutoff not in cal or cutoff >= start:
            raise ValueError("Retraining cutoff must be an observed session before evaluation.")
        if cal[train_end + LABEL_HORIZON] > cutoff or cal[valid_end + LABEL_HORIZON] > cutoff:
            raise ValueError("All training and validation labels must mature by retrain_asof.")
        result.append(
            {
                "id": name,
                "segments": segments,
                "retrain_asof": str(cutoff.date()),
                "training_label_endpoint": str(cal[train_end + LABEL_HORIZON].date()),
                "validation_label_endpoint": str(cal[valid_end + LABEL_HORIZON].date()),
                "data_start_time": str(cal[train_start - warmup].date()),
                "data_end_time": str(cal[test_end + LABEL_HORIZON + 1].date()),
                "maturity_checks": {
                    "training_labels_before_validation": True,
                    "training_labels_mature_at_retrain": True,
                    "validation_labels_mature_at_retrain": True,
                    "selection_before_first_evaluation_decision": True,
                    "label_endpoint_sessions": LABEL_HORIZON,
                },
            }
        )
    return json.loads(_canonical(result))


def _code_identity():
    root = Path(__file__).parent
    return _digest({str(p.relative_to(root)): _sha256(p) for p in sorted(root.rglob("*.py"))})


def _package_versions():
    versions = {}
    for package in ("numpy", "pandas", "pyqlib", "lightgbm", "torch", "mlflow", "filelock"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _candidates(spec):
    models = spec.get("models", ["zero", "ridge", "lightgbm", "transformer"])
    overrides = spec.get("model_params", {})
    if not isinstance(models, list) or not models or not isinstance(overrides, dict):
        raise ValueError("models must be nonempty and model_params must be a mapping.")
    candidates = []
    for entry in models:
        row = {"id": entry, "model": entry} if isinstance(entry, str) else copy.deepcopy(entry)
        if not isinstance(row, dict) or set(row) - {"id", "model", "params"}:
            raise ValueError("Each model is a name or an id/model/params mapping.")
        name = _name(row.get("id", row.get("model")), "candidate.id")
        model = row.get("model", name)
        if model not in ROLLING_MODELS:
            raise ValueError(f"Rolling models must be drawn from {ROLLING_MODELS}.")
        params = copy.deepcopy(overrides.get(model, {}))
        if not isinstance(params, dict) or not isinstance(row.get("params", {}), dict):
            raise ValueError("Model parameters must be mappings.")
        params.update(row.get("params", {}))
        if "seed" in params:
            raise ValueError("Register seeds centrally; do not override model.params.seed.")
        candidates.append({"id": name, "model": model, "params": params})
    if len({x["id"] for x in candidates}) != len(candidates):
        raise ValueError("Candidate identifiers must be unique.")
    if set(overrides) - {x["model"] for x in candidates}:
        raise ValueError("model_params contains an undeclared model.")
    return candidates


def _scenarios(spec):
    scenarios = spec.get(
        "cost_scenarios",
        [
            {"id": "base", "fee_multiplier": 1.0, "account_multiplier": 1.0},
            {"id": "fees_2x", "fee_multiplier": 2.0, "account_multiplier": 1.0},
            {"id": "account_0.1x", "fee_multiplier": 1.0, "account_multiplier": 0.1},
        ],
    )
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("cost_scenarios must be nonempty.")
    output = []
    for source in scenarios:
        if not isinstance(source, dict) or set(source) - {
            "id",
            "fee_multiplier",
            "account_multiplier",
            "slippage_bps",
        }:
            raise ValueError("Unsupported cost scenario fields.")
        row = {
            "id": _name(source.get("id"), "scenario.id"),
            "fee_multiplier": source.get("fee_multiplier", 1.0),
            "account_multiplier": source.get("account_multiplier", 1.0),
            "slippage_bps": source.get("slippage_bps", 0.0),
        }
        for name in ("fee_multiplier", "account_multiplier", "slippage_bps"):
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite.")
            if value < 0 or (name == "account_multiplier" and value == 0):
                raise ValueError(f"Invalid {name}.")
        output.append(row)
    if len({x["id"] for x in output}) != len(output):
        raise ValueError("Cost scenario identifiers must be unique.")
    return output


def prepare_rolling_campaign(provider, output, spec=None):
    """Register exact folds, seeds, budgets, costs and trial configs before fitting.

    ``spec`` is JSON/YAML data. Repeating an identical registration is safe;
    changing any scientific, source, code or output identity requires a new root.
    Official checkpoints are intentionally excluded from production-oriented
    candidate registration because their upstream overlap/rights differ.
    """
    spec = copy.deepcopy({} if spec is None else spec)
    allowed = {
        "name",
        "purpose",
        "folds",
        "rolling",
        "models",
        "model_params",
        "seeds",
        "epochs",
        "device",
        "num_threads",
        "region",
        "instruments",
        "warmup_sessions",
        "sequence_length",
        "backtest",
        "cost_scenarios",
        "selection_metric",
        "bootstrap",
        "reference",
        "holdout_status",
        "export_inference_bundle",
    }
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError(f"Unknown rolling specification fields: {sorted(set(spec) - allowed)}")
    provider, output = Path(provider).expanduser().resolve(), Path(output).expanduser().resolve()
    if provider == output or provider in output.parents or output in provider.parents:
        raise ValueError("Provider and campaign output must be separate directories.")
    cal = _calendar(_read_calendar(provider))
    metadata_path = provider / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("target_trade_date") and cal[-1] != pd.Timestamp(metadata["target_trade_date"]):
        raise ValueError("Observed calendar differs from the pinned source cutoff.")
    warmup = _positive(spec.get("warmup_sessions", 60), "warmup_sessions", minimum=0)
    if "folds" in spec and "rolling" in spec:
        raise ValueError("Supply explicit folds or a rolling recipe, not both.")
    recipe = copy.deepcopy(spec.get("rolling", {}))
    if recipe.get("label_endpoint_sessions", LABEL_HORIZON) != LABEL_HORIZON:
        raise ValueError("Executable Qlib rolling campaigns currently require label endpoint 2.")
    if "warmup_sessions" in recipe and recipe["warmup_sessions"] != warmup:
        raise ValueError("Register warmup_sessions once at campaign level.")
    recipe["warmup_sessions"] = warmup
    folds = spec.get("folds")
    if folds is None:
        folds = make_rolling_folds(cal, **recipe)
    folds = _validate_folds(cal, folds, warmup)
    seeds = spec.get("seeds", [7, 42, 2026])
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("seeds must be distinct integers in [0, 2**32).")
    epochs = _positive(spec.get("epochs", 6), "epochs")
    threads = _positive(spec.get("num_threads", 2), "num_threads")
    sequence_length = _positive(spec.get("sequence_length", 32), "sequence_length")
    region = spec.get("region", "cn")
    if region not in {"cn", "us"}:
        raise ValueError("region must be cn or us.")
    candidates = _candidates(spec)
    selection_metric = spec.get("selection_metric", "rank_ic")
    if selection_metric not in {"rank_ic", "ic", "rmse"}:
        raise ValueError("selection_metric must be rank_ic, ic, or rmse.")
    reference = spec.get("reference", "zero")
    if reference not in {x["id"] for x in candidates}:
        raise ValueError("reference must identify one registered candidate.")
    bootstrap = {"block_lengths": [5, 20, 60], "draws": 2000, "seed": 913}
    bootstrap.update(spec.get("bootstrap", {}))
    if set(bootstrap) != {"block_lengths", "draws", "seed"}:
        raise ValueError("Unsupported bootstrap fields.")
    _positive(bootstrap["draws"], "bootstrap.draws", minimum=100)
    _positive(bootstrap["seed"], "bootstrap.seed", minimum=0)
    if (
        not isinstance(bootstrap["block_lengths"], list)
        or not bootstrap["block_lengths"]
        or len(set(bootstrap["block_lengths"])) != len(bootstrap["block_lengths"])
    ):
        raise ValueError("Bootstrap block lengths must be a nonempty distinct list.")
    for length in bootstrap["block_lengths"]:
        _positive(length, "bootstrap.block_lengths")
    bt = {
        "enabled": True,
        "topk": 50 if region == "cn" else 5,
        "n_drop": 5 if region == "cn" else 1,
        "account": 100_000_000 if region == "cn" else 100_000,
        "risk_degree": 0.95,
        "benchmark": "SH000300" if region == "cn" else None,
        "open_cost": 0.0005 if region == "cn" else 0.001,
        "close_cost": 0.0015 if region == "cn" else 0.001,
        "min_cost": 5.0 if region == "cn" else 0,
    }
    bt.update(spec.get("backtest", {}))
    if type(bt["enabled"]) is not bool:
        raise ValueError("backtest.enabled must be boolean.")
    scenarios = _scenarios(spec)
    trials = []
    for fold in folds:
        for candidate in candidates:
            model = candidate["model"]
            template = (
                model
                if model in TRAINABLE_MODELS
                else "transformer"
                if model in CROSS_SECTIONAL_MODELS
                else "lightgbm"
            )
            for seed in seeds:
                trial_id = f"{fold['id']}/{candidate['id']}/seed-{seed}"
                config = benchmark_config(
                    template,
                    provider,
                    output / "trials" / trial_id,
                    seed,
                    epochs,
                    spec.get("device", "cpu"),
                )
                config.update(
                    region=region,
                    instruments=spec.get("instruments", "csi300" if region == "cn" else "all"),
                    experiment_name=spec.get("name", "rolling_research"),
                    warmup_sessions=warmup,
                    sequence_length=sequence_length,
                    segments=fold["segments"],
                    data_start_time=fold["data_start_time"],
                    data_end_time=fold["data_end_time"],
                    handler_cache_path=str(output / "cache" / f"{fold['id']}.pkl"),
                    backtest=copy.deepcopy(bt),
                    export_inference_bundle=spec.get("export_inference_bundle", True),
                )
                config["model"]["name"] = model
                if model in {"zero", "ridge"}:
                    config["model"]["params"] = {"alpha": 1.0}
                else:
                    config["model"]["params"]["num_threads"] = threads
                config["model"]["params"].update(candidate["params"])
                if model in (*NEURAL_MODELS, *CROSS_SECTIONAL_MODELS):
                    config["shared_windows"] = True
                trials.append(
                    {
                        "id": trial_id,
                        "fold": fold["id"],
                        "candidate": candidate["id"],
                        "model": model,
                        "seed": seed,
                        "config": config,
                    }
                )
    packages = _package_versions()
    identity = {
        "schema_version": 1,
        "output_dir": str(output),
        "provider_uri": str(provider),
        "provider_fingerprint": _provider_fingerprint(provider),
        "data_manifest": metadata,
        "data_manifest_sha256": _sha256(metadata_path) if metadata_path.exists() else None,
        "source_code_sha256": _code_identity(),
        "packages": packages,
        "specification": spec,
        "folds": folds,
        "candidates": candidates,
        "seeds": seeds,
        "trials": trials,
        "label": {
            "expression": LABEL,
            "entry_delay_sessions": 1,
            "holding_sessions": 1,
            "endpoint_sessions": 2,
        },
        "selection": {
            "metric": selection_metric,
            "aggregation": "mean validation metric across all declared seeds; no seed selection",
            "fallback": "validation RMSE only when no candidate has a finite registered primary metric",
            "incomplete_policy": "no fold selection until every registered trial completes",
            "tie_break": "candidate identifier ascending",
        },
        "bootstrap": bootstrap,
        "reference": reference,
        "cost_scenarios": scenarios,
        "interpretation": {
            "holdout": spec.get(
                "holdout_status",
                "exploratory historical rolling evaluation; previously inspected dates are not fresh confirmation",
            ),
            "portfolio": "Each fold starts an independent funded portfolio; fold returns must not be concatenated as a continuously traded strategy. Qlib lifecycle settlement remains qualified until independently reconciled.",
            "costs": "Native portfolio replays vary explicit fees, fixed slippage assumptions and capital; multipliers are not a market impact or capacity model.",
            "seeds": "Initialization dispersion is not uncertainty across market regimes; paired intervals resample dates.",
        },
    }
    identity = json.loads(_canonical(identity))
    protocol = {**identity, "protocol_sha256": _digest(identity), "registered_at_utc": _now()}
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".registration.lock")):
        path = output / "protocol.json"
        if path.exists():
            existing = _load_protocol(output)
            if existing["protocol_sha256"] != protocol["protocol_sha256"]:
                raise ValueError("Existing protocol differs; choose a new campaign directory.")
            return existing
        if any(p.name != ".registration.lock" for p in output.iterdir()):
            raise ValueError("A new campaign requires an empty output directory.")
        _save(path, protocol)
    return protocol


def _load_protocol(value):
    supplied = value if isinstance(value, dict) else None
    root = Path(value["output_dir"] if supplied is not None else value).expanduser().resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    identity = {
        k: v for k, v in protocol.items() if k not in {"protocol_sha256", "registered_at_utc"}
    }
    if protocol.get("protocol_sha256") != _digest(identity):
        raise ValueError("Campaign protocol integrity verification failed.")
    if Path(protocol["output_dir"]).resolve() != root:
        raise ValueError("Protocol output directory does not match its registered location.")
    if supplied is not None and supplied != protocol:
        raise ValueError("Supplied protocol differs from the registered immutable file.")
    return protocol


def _read_predictions(path):
    frame = pd.read_csv(
        path, parse_dates=["datetime"], dtype={"instrument": str}, float_precision="round_trip"
    )
    if not {"datetime", "instrument", "score", "label"}.issubset(frame):
        raise ValueError("Predictions require datetime, instrument, score and label columns.")
    frame = frame.set_index(["datetime", "instrument"])[["score", "label"]].sort_index()
    if frame.empty or not frame.index.is_unique or not np.isfinite(frame["score"]).all():
        raise ValueError("Prediction rows must be nonempty, unique and have finite scores.")
    return frame


def paired_date_block_intervals(candidate, reference, *, block_length=20, draws=2000, seed=913):
    """Paired circular date-block intervals retaining every daily cross section.

    Candidate/reference date-instrument keys and raw labels must match exactly.
    RMSE retains stock-row weights; IC and Rank IC give each usable date equal
    weight. Undefined paired correlation dates remain missing and are counted.
    Blocks must be shorter than the available period: a full-period circular
    block merely rotates every date and cannot estimate resampling uncertainty.
    """
    _positive(draws, "draws", minimum=100)
    _positive(seed, "seed", minimum=0)
    frames = []
    for source in (candidate, reference):
        if not isinstance(source, pd.DataFrame) or not {"score", "label"}.issubset(source):
            raise ValueError("Paired inputs require score and label DataFrame columns.")
        if not isinstance(source.index, pd.MultiIndex) or set(source.index.names) != {
            "datetime",
            "instrument",
        }:
            raise ValueError("Paired inputs require datetime/instrument keys.")
        frame = source.reorder_levels(["datetime", "instrument"]).sort_index()
        if frame.empty or not frame.index.is_unique or not np.isfinite(frame["score"]).all():
            raise ValueError("Paired inputs must have nonempty unique rows and finite scores.")
        if np.isinf(frame["label"]).any():
            raise ValueError("Paired labels contain infinite returns.")
        frames.append(frame)
    candidate, reference = frames
    if not candidate.index.equals(reference.index):
        raise ValueError("Paired candidate/reference panels must have exactly matching keys.")
    if not np.array_equal(candidate["label"], reference["label"], equal_nan=True):
        raise ValueError("Paired candidate/reference raw labels differ.")
    valid = candidate["label"].notna()
    candidate, reference = candidate.loc[valid], reference.loc[valid]
    if candidate.empty:
        raise ValueError("No mature finite labels in the paired panel.")
    daily = (
        pd.DataFrame(
            {
                "candidate_se": (candidate["score"] - candidate["label"]) ** 2,
                "reference_se": (reference["score"] - reference["label"]) ** 2,
            }
        )
        .groupby(level="datetime")
        .agg(["sum", "size"])
    )
    dates = pd.DatetimeIndex(daily.index)
    _positive(block_length, "block_length")
    if block_length >= len(dates):
        raise ValueError(
            "block_length must be smaller than the number of paired dates; "
            "a full-period circular block has no resampling variation."
        )
    se_c = daily[("candidate_se", "sum")].to_numpy()
    se_r = daily[("reference_se", "sum")].to_numpy()
    count = daily[("candidate_se", "size")].to_numpy()
    correlations = []
    for frame in (candidate, reference):
        rows = []
        for date, group in frame.groupby(level="datetime"):
            usable = (
                len(group) >= 2 and group["score"].nunique() > 1 and group["label"].nunique() > 1
            )
            rows.append(
                {
                    "datetime": date,
                    "ic": group["score"].corr(group["label"]) if usable else np.nan,
                    "rank_ic": group["score"].corr(group["label"], method="spearman")
                    if usable
                    else np.nan,
                }
            )
        correlations.append(pd.DataFrame(rows).set_index("datetime").reindex(dates))
    delta = correlations[0] - correlations[1]
    rng = np.random.default_rng(seed)
    blocks = (len(dates) + block_length - 1) // block_length
    samples = {"rmse": [], "ic": [], "rank_ic": []}
    for _ in range(draws):
        starts = rng.integers(0, len(dates), size=blocks)
        positions = ((starts[:, None] + np.arange(block_length)) % len(dates)).ravel()[: len(dates)]
        samples["rmse"].append(
            np.sqrt(se_c[positions].sum() / count[positions].sum())
            - np.sqrt(se_r[positions].sum() / count[positions].sum())
        )
        for metric in ("ic", "rank_ic"):
            values = delta[metric].to_numpy()[positions]
            finite = values[np.isfinite(values)]
            samples[metric].append(float(finite.mean()) if len(finite) else np.nan)
    output = {}
    for metric, values in samples.items():
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        estimate = (
            np.sqrt(se_c.sum() / count.sum()) - np.sqrt(se_r.sum() / count.sum())
            if metric == "rmse"
            else delta[metric].mean()
        )
        low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (None, None)
        output[metric] = {
            "difference": estimate,
            "lower_95": low,
            "upper_95": high,
            "valid_draws": len(finite),
            "paired_usable_dates": len(dates)
            if metric == "rmse"
            else int(delta[metric].notna().sum()),
        }
    return _json_ready(
        {
            "method": "paired circular moving blocks of dates; whole cross sections retained",
            "difference": "candidate minus reference; negative RMSE and positive IC favor candidate",
            "date_count": len(dates),
            "row_count": len(candidate),
            "unavailable_label_rows": int((~valid).sum()),
            "block_length": block_length,
            "effective_blocks": len(dates) / block_length,
            "uncertainty_qualification": (
                "Very limited resampling information: fewer than two full block lengths "
                "fit within the evaluation period; narrow intervals are not strong evidence."
                if len(dates) < 2 * block_length
                else None
            ),
            "draws": draws,
            "seed": seed,
            "metrics": output,
        }
    )


def paired_return_block_intervals(candidate, reference, *, block_length=20, draws=2000, seed=913):
    """Resample paired daily net returns for the same fold, seed and scenario."""
    _positive(draws, "draws", minimum=100)
    _positive(block_length, "block_length")
    _positive(seed, "seed", minimum=0)
    if not isinstance(candidate, pd.Series) or not isinstance(reference, pd.Series):
        raise ValueError("Paired portfolio returns must be dated Series.")
    candidate, reference = candidate.sort_index(), reference.sort_index()
    _calendar(candidate.index)
    _calendar(reference.index)
    if not candidate.index.equals(reference.index):
        raise ValueError("Paired portfolio returns must cover exactly matching dates.")
    c, r = candidate.to_numpy(dtype=float), reference.to_numpy(dtype=float)
    if not np.isfinite(c).all() or not np.isfinite(r).all() or (c < -1).any() or (r < -1).any():
        raise ValueError("Portfolio returns must be finite and no smaller than -1.")
    if block_length >= len(c):
        raise ValueError(
            "block_length must be smaller than the number of paired portfolio dates; "
            "a full-period circular block has no resampling variation."
        )
    rng = np.random.default_rng(seed)
    blocks = (len(c) + block_length - 1) // block_length
    starts = rng.integers(0, len(c), size=(draws, blocks))
    indices = ((starts[:, :, None] + np.arange(block_length)) % len(c)).reshape(draws, -1)[
        :, : len(c)
    ]
    mean_samples = (c[indices] - r[indices]).mean(axis=1)
    cumulative_samples = np.prod(1 + c[indices], axis=1) - np.prod(1 + r[indices], axis=1)
    output = {}
    for name, values, estimate in (
        ("mean_daily_net_return", mean_samples, float((c - r).mean())),
        ("compounded_net_return", cumulative_samples, float(np.prod(1 + c) - np.prod(1 + r))),
    ):
        lower, upper = np.quantile(values, [0.025, 0.975])
        output[name] = {"difference": estimate, "lower_95": float(lower), "upper_95": float(upper)}
    return {
        "method": "paired circular date-block bootstrap of same-seed native scenario returns",
        "difference": "candidate minus reference; positive favors candidate",
        "date_count": len(c),
        "block_length": block_length,
        "effective_blocks": len(c) / block_length,
        "uncertainty_qualification": (
            "Very limited resampling information: fewer than two full block lengths "
            "fit within the evaluation period; narrow intervals are not strong evidence."
            if len(c) < 2 * block_length
            else None
        ),
        "draws": draws,
        "seed": seed,
        "metrics": output,
    }


def _portfolio_summary(report):
    required = ["return", "cost", "bench", "turnover"]
    if (
        report.empty
        or not set(required).issubset(report)
        or not np.isfinite(report[required]).all().all()
    ):
        raise ValueError("Native portfolio reports must be complete and finite.")
    net = report["return"] - report["cost"]
    if (net <= -1).any():
        raise ValueError("Native portfolio contains a total loss or invalid return.")
    equity = (1 + net).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, equity.to_numpy()])[1:]
    return {
        "net_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(np.min(equity.to_numpy() / peak - 1)),
        "mean_daily_turnover": float(report["turnover"].mean()),
        "trading_days": len(report),
        "sum_daily_cost_ratios": float(report["cost"].sum()),
    }


def _replay_scenarios(config, protocol, workflow_root):
    if not config["backtest"]["enabled"]:
        return {"status": "disabled_by_registered_protocol", "scenarios": []}
    import qlib
    from qlib.backtest import backtest
    from qlib.data import D
    from qlib.data.cache import H
    from qlib.utils import fill_placeholder

    from financial_ts.cn_portfolio import numeric_positions, terminated_holdings_audit

    provider = Path(protocol["provider_uri"])
    scores = _read_predictions(workflow_root / "predictions.csv")[["score"]]
    calendar = _read_calendar(provider)
    H.clear()
    qlib.init(
        provider_uri=str(provider),
        region=config["region"],
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    output = []
    for scenario in protocol["cost_scenarios"]:
        varied = copy.deepcopy(config)
        bt = varied["backtest"]
        bt["account"] *= scenario["account_multiplier"]
        for fee in ("open_cost", "close_cost", "min_cost"):
            bt[fee] *= scenario["fee_multiplier"]
        for fee in ("open_cost", "close_cost"):
            bt[fee] += scenario["slippage_bps"] / 10000
        portfolio = _portfolio_config(
            varied,
            config["segments"],
            calendar,
            scores.index.get_level_values("instrument").nunique(),
        )
        native = copy.deepcopy(portfolio)
        native["strategy"] = fill_placeholder(native["strategy"], {"<PRED>": scores})
        if native["backtest"]["benchmark"] is None:
            dates = D.calendar(
                start_time=native["backtest"]["start_time"], end_time=native["backtest"]["end_time"]
            )
            native["backtest"]["benchmark"] = pd.Series(0.0, index=dates)
        reports, _ = backtest(
            executor={
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
            },
            strategy=native["strategy"],
            **native["backtest"],
        )
        report, positions = reports["1day"]
        target = workflow_root / "scenarios" / scenario["id"]
        target.mkdir(parents=True, exist_ok=True)
        report.to_csv(target / "portfolio.csv")
        numeric = numeric_positions(positions)
        with (target / "positions.json.gz").open("wb") as stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
                compressed.write(_canonical(numeric).encode())
        lifecycle = terminated_holdings_audit(positions, provider)
        _save(target / "source_gap_holdings.json", lifecycle)
        exposure = [
            sum(h.get("weight", 0) for h in row["holdings"]) for row in numeric["positions"]
        ]
        summary = {
            **scenario,
            **_portfolio_summary(report),
            "mean_gross_exposure": float(np.mean(exposure)),
            "maximum_gross_exposure": float(np.max(exposure)),
            "total_final_stale_weight": lifecycle["total_final_stale_weight"],
            "portfolio_config": portfolio,
        }
        _save(target / "summary.json", summary)
        output.append(summary)
    result = {
        "status": "completed",
        "qualification": protocol["interpretation"]["costs"],
        "scenarios": output,
    }
    _save(workflow_root / "cost_scenarios.json", result)
    return result


def _artifact_manifest(workflow_root):
    paths = [
        p
        for p in workflow_root.rglob("*")
        if p.is_file()
        and ("mlruns" not in p.parts or p.name == "params.pkl")
        and not p.name.endswith(".lock")
    ]
    return {
        str(p.relative_to(workflow_root)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
        for p in sorted(paths)
    }


def _verify_attempt(attempt, trial, protocol):
    receipt = json.loads((attempt / "receipt.json").read_text())
    if receipt.get("protocol_sha256") != protocol["protocol_sha256"] or receipt.get(
        "trial_sha256"
    ) != _digest(trial):
        raise ValueError("Trial receipt identity differs from its registered protocol.")
    ordinal = int(attempt.name.removeprefix("attempt-"))
    previous = attempt.parent / f"attempt-{ordinal - 1:04d}"
    expected_previous = (
        str(previous.relative_to(Path(protocol["output_dir"]))) if ordinal > 1 else None
    )
    if receipt.get("attempt") != ordinal or receipt.get("previous_attempt") != expected_previous:
        raise ValueError("Trial receipt attempt ordinal/previous_attempt chain is invalid.")
    if ordinal > 1 and not (previous / "receipt.json").is_file():
        raise ValueError("Trial attempt chain is missing its previous receipt.")
    if "interruption_recovery" in receipt:
        recovery = receipt["interruption_recovery"]
        original = attempt / "receipt.before-recovery.json"
        if (
            receipt.get("status") != "failed"
            or not original.is_file()
            or _sha256(original) != recovery.get("previous_receipt_sha256")
        ):
            raise ValueError("Interrupted attempt recovery evidence is invalid.")
        original_receipt = json.loads(original.read_text())
        if (
            original_receipt.get("status") != "running"
            or original_receipt.get("trial_sha256") != receipt["trial_sha256"]
            or original_receipt.get("protocol_sha256") != receipt["protocol_sha256"]
            or original_receipt.get("attempt") != receipt["attempt"]
        ):
            raise ValueError(
                "Interrupted attempt recovery does not match original running receipt."
            )
    if receipt.get("status") == "completed":
        for relative, identity in receipt["artifacts"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Invalid relative path in the trial artifact manifest.")
            path = attempt / "workflow" / relative
            if (
                not path.is_file()
                or path.stat().st_size != identity["bytes"]
                or _sha256(path) != identity["sha256"]
            ):
                raise ValueError(f"Completed trial artifact integrity failed: {relative}")
    return receipt


def _attempts(root, trial):
    attempts = sorted((root / "trials" / trial["id"]).glob("attempt-*"))
    if any(
        not path.is_dir() or path.name != f"attempt-{ordinal:04d}"
        for ordinal, path in enumerate(attempts, 1)
    ):
        raise ValueError("Trial attempt directories must form a contiguous numbered chain.")
    return attempts


def run_rolling_campaign(value, *, retry_failed=False, trial_ids=None, summarize=True):
    """Run/resume registered trials sequentially, preserving every failed attempt.

    A failed trial does not prevent the other registered trials from running.
    Retry creates a new attempt directory. Completed receipts are verified before
    reuse, including predictions, metrics, inference bundle and scenario ledgers.
    """
    if type(summarize) is not bool:
        raise ValueError("summarize must be boolean.")
    protocol = _load_protocol(value)
    root, provider = Path(protocol["output_dir"]), Path(protocol["provider_uri"])
    if _provider_fingerprint(provider) != protocol["provider_fingerprint"]:
        raise ValueError("Provider bytes changed after campaign registration.")
    metadata = provider / "metadata.json"
    if (_sha256(metadata) if metadata.exists() else None) != protocol["data_manifest_sha256"]:
        raise ValueError("Data manifest changed after campaign registration.")
    if _code_identity() != protocol["source_code_sha256"]:
        raise ValueError("Source code changed after registration; use a new campaign.")
    if _package_versions() != protocol["packages"]:
        raise ValueError("Runtime package versions changed after campaign registration.")
    requested = (
        [trial["id"] for trial in protocol["trials"]] if trial_ids is None else list(trial_ids)
    )
    if (
        not requested
        or len(set(requested)) != len(requested)
        or set(requested) - {x["id"] for x in protocol["trials"]}
    ):
        raise ValueError("trial_ids must be distinct registered trial identifiers.")
    for trial in protocol["trials"]:
        if trial["id"] not in requested:
            continue
        lock = root / "trials" / trial["id"] / ".trial.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock)):
            attempts = _attempts(root, trial)
            if attempts:
                # A terminated worker can leave the directory before its first
                # atomic receipt. Holding the trial lock proves it is no longer
                # owned; preserve the directory and record the missing evidence.
                if not (attempts[-1] / "receipt.json").exists():
                    _save(
                        attempts[-1] / "receipt.json",
                        {
                            "schema_version": 1,
                            "trial_id": trial["id"],
                            "trial_sha256": _digest(trial),
                            "protocol_sha256": protocol["protocol_sha256"],
                            "status": "failed",
                            "attempt": len(attempts),
                            "started_at_utc": None,
                            "finished_at_utc": _now(),
                            "elapsed_seconds": None,
                            "error": "Interrupted attempt has no receipt; no completed work is inferred.",
                            "previous_attempt": str(attempts[-2].relative_to(root))
                            if len(attempts) > 1
                            else None,
                        },
                    )
                previous = _verify_attempt(attempts[-1], trial, protocol)
                if previous["status"] == "running":
                    # The exclusive lock is free, so the prior owner is gone.
                    # Retain its exact bytes before recording the observed interruption.
                    receipt_path = attempts[-1] / "receipt.json"
                    archived_receipt = attempts[-1] / "receipt.before-recovery.json"
                    if (
                        archived_receipt.exists()
                        and archived_receipt.read_bytes() != receipt_path.read_bytes()
                    ):
                        raise ValueError("Conflicting preserved interruption receipt.")
                    archived_receipt.write_bytes(receipt_path.read_bytes())
                    observed = _now()
                    previous.update(
                        status="failed",
                        finished_at_utc=observed,
                        elapsed_seconds=None,
                        error="Worker disappeared while its attempt was running; completion unverified.",
                        interruption_recovery={
                            "observed_at_utc": observed,
                            "previous_receipt_sha256": _sha256(archived_receipt),
                        },
                    )
                    _save(receipt_path, previous)
                if previous["status"] == "completed" or not retry_failed:
                    continue
            attempt = lock.parent / f"attempt-{len(attempts) + 1:04d}"
            attempt.mkdir()
            receipt = {
                "schema_version": 1,
                "trial_id": trial["id"],
                "trial_sha256": _digest(trial),
                "protocol_sha256": protocol["protocol_sha256"],
                "status": "running",
                "attempt": len(attempts) + 1,
                "started_at_utc": _now(),
                "previous_attempt": str(attempts[-1].relative_to(root)) if attempts else None,
            }
            _save(attempt / "receipt.json", receipt)
            started = time.perf_counter()
            try:
                config = copy.deepcopy(trial["config"])
                config["output_dir"] = str(attempt / "workflow")
                from qlib.data.cache import H

                H.clear()
                result = run_workflow(config)
                if result["run"].get("status") != "completed":
                    raise ValueError("Workflow did not complete.")
                scenarios = _replay_scenarios(config, protocol, attempt / "workflow")
                receipt.update(
                    status="completed",
                    scenario_status=scenarios["status"],
                    artifacts=_artifact_manifest(attempt / "workflow"),
                )
            except Exception as exc:
                receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            receipt.update(finished_at_utc=_now(), elapsed_seconds=time.perf_counter() - started)
            _save(attempt / "receipt.json", receipt)
    if summarize:
        return summarize_rolling_campaign(root)
    return {"protocol_sha256": protocol["protocol_sha256"], "requested_trials": requested}


def _seed_ensemble(paths, filename):
    frames = [_read_predictions(path / filename) for path in paths]
    first = frames[0]
    for frame in frames[1:]:
        if not frame.index.equals(first.index) or not np.array_equal(
            frame["label"], first["label"], equal_nan=True
        ):
            raise ValueError("Seed ensemble panel keys or raw labels differ.")
    result = first.copy()
    result["score"] = np.mean([frame["score"].to_numpy() for frame in frames], axis=0)
    return result


def summarize_rolling_campaign(value, *, summary_output_dir=None):
    """Audit saved artifacts and select per-fold models using only validation.

    Complete seed ensembles are paired within each fold; no test-driven winner
    or cross-fold portfolio compounding is reported. Failure/search history is
    preserved even after a later retry succeeds. A separate summary_output_dir
    preserves the original campaign reports and records the analysis code used
    to regenerate derived artifacts from the verified, unchanged trial outputs.
    """
    protocol = _load_protocol(value)
    root = Path(protocol["output_dir"])
    report_root = root if summary_output_dir is None else Path(summary_output_dir).resolve()
    analysis_receipt = None
    if summary_output_dir is not None:
        if report_root == root or any(
            report_root.is_relative_to(root / name) for name in ("trials", "summary")
        ):
            raise ValueError("A separate summary directory must preserve the original campaign.")
        report_root.mkdir(parents=True, exist_ok=True)
        original = root / "comparison.json"
        analysis_receipt = {
            "schema_version": 1,
            "source_campaign": str(root),
            "summary_output_dir": str(report_root),
            "protocol_sha256": protocol["protocol_sha256"],
            "registered_source_code_sha256": protocol["source_code_sha256"],
            "analysis_source_code_sha256": _code_identity(),
            "analysis_module": {"path": str(Path(__file__).resolve()), "sha256": _sha256(__file__)},
            "original_comparison": {
                "path": str(original),
                "sha256": _sha256(original) if original.exists() else None,
            },
            "reason": (
                "Regenerated derived summary from verified archived trial artifacts with "
                "current analysis code; registered trials and experiment settings unchanged."
            ),
            "bootstrap_policy": (
                "Report insufficient_dates when block_length >= paired date count; "
                "a full-period circular block has no resampling variation. Report "
                "effective_blocks = date_count / block_length and qualify fewer than "
                "two full block lengths as very limited resampling information."
            ),
        }
    rows, history, completed, pending, failed = [], [], {}, [], []
    for trial in protocol["trials"]:
        attempts = _attempts(root, trial)
        if not attempts:
            pending.append(trial["id"])
            continue
        for attempt in attempts:
            receipt = _verify_attempt(attempt, trial, protocol)
            history.append(
                {"trial_id": trial["id"], "attempt_path": str(attempt.relative_to(root)), **receipt}
            )
        receipt = _verify_attempt(attempts[-1], trial, protocol)
        if receipt["status"] != "completed":
            (failed if receipt["status"] == "failed" else pending).append(trial["id"])
            continue
        path = attempts[-1] / "workflow"
        metric = json.loads((path / "metrics.json").read_text())
        row = {
            "trial_id": trial["id"],
            "fold": trial["fold"],
            "candidate": trial["candidate"],
            "model": trial["model"],
            "seed": trial["seed"],
            "attempt": receipt["attempt"],
            "artifact_path": str(path.relative_to(root)),
            "validation_rmse": metric["validation"]["rmse"],
            "validation_ic": metric["validation"].get("ic"),
            "validation_rank_ic": metric["validation"].get("rank_ic"),
            "test_rmse": metric["rmse"],
            "test_ic": metric.get("ic"),
            "test_rank_ic": metric.get("rank_ic"),
            "test_rows": metric["evaluated_count"],
            "validation_undefined_ic_days": None,
            "fit_seconds": metric.get("model_details", {}).get("fit_seconds"),
            "parameter_count": metric.get("model_details", {}).get("parameter_count"),
            "elapsed_seconds": receipt["elapsed_seconds"],
        }
        if (path / "validation_daily_ic.csv").exists():
            daily_valid = pd.read_csv(path / "validation_daily_ic.csv")
            row["validation_undefined_ic_days"] = int(daily_valid["ic"].isna().sum())
            row["validation_undefined_rank_ic_days"] = int(daily_valid["rank_ic"].isna().sum())
        if (path / "daily_ic.csv").exists():
            daily_test = pd.read_csv(path / "daily_ic.csv")
            row["test_undefined_ic_days"] = int(daily_test["ic"].isna().sum())
            row["test_undefined_rank_ic_days"] = int(daily_test["rank_ic"].isna().sum())
        if (path / "portfolio.csv").exists():
            row.update(_portfolio_summary(pd.read_csv(path / "portfolio.csv", index_col=0)))
        rows.append(row)
        completed[(trial["fold"], trial["candidate"], trial["seed"])] = path
    frame = pd.DataFrame(rows)
    selections, comparisons, ensemble_rows, portfolio_comparisons = [], [], [], []
    for fold in protocol["folds"]:
        fold_rows = frame.loc[frame["fold"] == fold["id"]] if len(frame) else frame
        required = len(protocol["candidates"]) * len(protocol["seeds"])
        if len(fold_rows) != required:
            selections.append(
                {
                    "fold": fold["id"],
                    "status": "incomplete",
                    "selected_candidate": None,
                    "completed_trials": len(fold_rows),
                    "required_trials": required,
                }
            )
            continue
        candidates = []
        metric_name = "validation_" + protocol["selection"]["metric"]
        for candidate in protocol["candidates"]:
            group = fold_rows.loc[fold_rows["candidate"] == candidate["id"]]
            values = pd.to_numeric(group[metric_name], errors="coerce")
            candidates.append(
                {
                    "candidate": candidate["id"],
                    "primary": float(values.mean()) if values.notna().all() else None,
                    "validation_rmse": float(group["validation_rmse"].mean()),
                }
            )
        usable = [x for x in candidates if x["primary"] is not None and np.isfinite(x["primary"])]
        fallback = not usable
        if fallback:
            best = min(candidates, key=lambda x: (x["validation_rmse"], x["candidate"]))
        else:
            direction = 1 if protocol["selection"]["metric"] == "rmse" else -1
            best = min(usable, key=lambda x: (direction * x["primary"], x["candidate"]))
        selections.append(
            {
                "fold": fold["id"],
                "status": "selected",
                "selected_candidate": best["candidate"],
                "fallback_to_validation_rmse": fallback,
                "selection_inputs": candidates,
                "selection_uses_validation_only": True,
                "decision_asof": fold["retrain_asof"],
                "note": "Historical reconstruction; summary computation occurs after trials but reads validation metrics only.",
            }
        )
        ensembles = {}
        for candidate in protocol["candidates"]:
            paths = [completed[(fold["id"], candidate["id"], seed)] for seed in protocol["seeds"]]
            ensemble = _seed_ensemble(paths, "predictions.csv")
            valid = _seed_ensemble(paths, "validation_predictions.csv")
            target = report_root / "summary" / fold["id"] / candidate["id"]
            target.mkdir(parents=True, exist_ok=True)
            ensemble.to_csv(target / "ensemble_predictions.csv")
            valid.to_csv(target / "ensemble_validation_predictions.csv")
            stats, daily = _prediction_statistics(ensemble)
            valid_stats, _ = _prediction_statistics(valid)
            daily.to_csv(target / "daily_ic.csv")
            ensemble_rows.append(
                {
                    "fold": fold["id"],
                    "candidate": candidate["id"],
                    "seed_count": len(paths),
                    "test": stats,
                    "validation": valid_stats,
                    "source_trials": [str(p.relative_to(root)) for p in paths],
                }
            )
            ensembles[candidate["id"]] = ensemble
        reference = ensembles[protocol["reference"]]
        for name, ensemble in ensembles.items():
            if name == protocol["reference"]:
                continue
            days = (
                ensemble.loc[ensemble["label"].notna()].index.get_level_values("datetime").nunique()
            )
            for length in protocol["bootstrap"]["block_lengths"]:
                result = {
                    "fold": fold["id"],
                    "candidate": name,
                    "reference": protocol["reference"],
                    "block_length": length,
                }
                if length >= days:
                    result.update(
                        status="insufficient_dates",
                        date_count=days,
                        effective_blocks=days / length,
                        reason=(
                            "Block length must be shorter than the paired date count; "
                            "a full-period circular block has no resampling variation."
                        ),
                    )
                else:
                    result.update(
                        status="completed",
                        result=paired_date_block_intervals(
                            ensemble,
                            reference,
                            block_length=length,
                            draws=protocol["bootstrap"]["draws"],
                            seed=protocol["bootstrap"]["seed"],
                        ),
                    )
                comparisons.append(result)
    # Compare each seed's actual native portfolio; never confuse averaging seed
    # forecasts with trading an ensemble or pool seeds as independent market days.
    for fold in protocol["folds"]:
        for candidate in protocol["candidates"]:
            name = candidate["id"]
            if name == protocol["reference"]:
                continue
            for seed in protocol["seeds"]:
                pair = [(fold["id"], name, seed), (fold["id"], protocol["reference"], seed)]
                if any(key not in completed for key in pair):
                    continue
                for scenario in protocol["cost_scenarios"]:
                    files = [
                        completed[key] / "scenarios" / scenario["id"] / "portfolio.csv"
                        for key in pair
                    ]
                    if not all(path.exists() for path in files):
                        continue
                    series = []
                    for path in files:
                        report = pd.read_csv(path, index_col=0, parse_dates=True)
                        series.append(report["return"] - report["cost"])
                    for length in protocol["bootstrap"]["block_lengths"]:
                        row = {
                            "fold": fold["id"],
                            "candidate": name,
                            "reference": protocol["reference"],
                            "seed": seed,
                            "scenario": scenario["id"],
                            "block_length": length,
                        }
                        if length >= len(series[0]):
                            row.update(
                                status="insufficient_dates",
                                date_count=len(series[0]),
                                effective_blocks=len(series[0]) / length,
                                reason=(
                                    "Block length must be shorter than the paired date count; "
                                    "a full-period circular block has no resampling variation."
                                ),
                            )
                        else:
                            row.update(
                                status="completed",
                                result=paired_return_block_intervals(
                                    *series,
                                    block_length=length,
                                    draws=protocol["bootstrap"]["draws"],
                                    seed=protocol["bootstrap"]["seed"],
                                ),
                            )
                        portfolio_comparisons.append(row)
    aggregates = []
    if len(frame):
        for (fold_id, candidate_id), group in frame.groupby(["fold", "candidate"], sort=True):
            aggregate = {
                "fold": fold_id,
                "candidate": candidate_id,
                "completed_seeds": len(group),
                "required_seeds": len(protocol["seeds"]),
                "status": "complete" if len(group) == len(protocol["seeds"]) else "incomplete",
            }
            for metric in (
                "validation_rmse",
                "validation_ic",
                "validation_rank_ic",
                "test_rmse",
                "test_ic",
                "test_rank_ic",
                "net_return",
                "max_drawdown",
                "mean_daily_turnover",
            ):
                if metric in group:
                    values = pd.to_numeric(group[metric], errors="coerce")
                    aggregate[metric] = {
                        "mean": values.mean(),
                        "std_across_seeds": values.std(ddof=0),
                        "finite_seed_count": int(values.notna().sum()),
                    }
            aggregates.append(aggregate)

    result = {
        "schema_version": 1,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "failed" if failed else "partial" if pending else "completed",
        "registered_trials": len(protocol["trials"]),
        "completed_trials": len(rows),
        "pending": pending,
        "failed": failed,
        "failed_attempt_count": sum(row["status"] == "failed" for row in history),
        "runs": rows,
        "selections": selections,
        "seed_ensembles": ensemble_rows,
        "paired_comparisons": comparisons,
        "portfolio_paired_comparisons": portfolio_comparisons,
        "fold_seed_aggregates": aggregates,
        "interpretation": protocol["interpretation"],
        "promotion": {
            "eligible": False,
            "reason": "Exploratory rolling research requires separate prospective, lifecycle, operational and usage-rights gates.",
        },
    }
    with FileLock(str(report_root / ".summary.lock")):
        _save(report_root / "comparison.json", result)
        _save(
            report_root / "trial_history.json",
            {"protocol_sha256": protocol["protocol_sha256"], "attempts": history},
        )
        frame.to_csv(report_root / "comparison.csv", index=False)
        if analysis_receipt is not None:
            _save(report_root / "analysis_receipt.json", analysis_receipt)
    return _json_ready(result)
