"""Independently check completed CSI300 predictions and Qlib portfolio accounting.

Raw binary closes and membership files are read directly, without loading the
large fitted handler or fitting any model. Native Qlib position/booster pickles
are read only with --trusted-local-pickles; use that flag solely for artifacts
produced locally by this project's own runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class Source:
    def __init__(self, provider: Path):
        self.provider = provider
        self.calendar = pd.DatetimeIndex(
            pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
        )
        self.membership = pd.read_csv(
            provider / "instruments/csi300.txt",
            sep="\t",
            header=None,
            names=["instrument", "start", "end"],
            parse_dates=["start", "end"],
        )
        self.closes = {}
        self.last_dates = {}

    def close(self, instrument):
        if instrument not in self.closes:
            path = self.provider / "features" / instrument.lower() / "close.day.bin"
            data = np.fromfile(path, dtype="<f4")
            offset = int(data[0])
            end = min(len(self.calendar), offset + len(data) - 1)
            values = np.full(len(self.calendar), np.nan, dtype=np.float32)
            values[offset:end] = data[1 : end - offset + 1]
            self.closes[instrument] = values
            valid = np.flatnonzero(np.isfinite(values))
            self.last_dates[instrument] = self.calendar[valid[-1]] if len(valid) else None
        return self.closes[instrument]

    def expected_index(self, bounds):
        dates = self.calendar[(self.calendar >= bounds[0]) & (self.calendar <= bounds[1])]
        pairs = []
        for row in self.membership.itertuples(index=False):
            active = dates[(dates >= row.start) & (dates <= row.end)]
            pairs.extend((date, row.instrument) for date in active)
        index = pd.MultiIndex.from_tuples(pairs, names=["datetime", "instrument"])
        if not index.is_unique:
            raise AssertionError(
                "Overlapping source membership intervals produce duplicate samples"
            )
        return index.sort_values()

    def labels(self, frame):
        expected = np.full(len(frame), np.nan, dtype=np.float64)
        for instrument, group in frame.groupby("instrument", sort=False):
            dates = self.calendar.get_indexer(group["datetime"])
            if (dates < 0).any() or (dates + 2 >= len(self.calendar)).any():
                raise AssertionError(
                    "Prediction date or its delayed label falls outside observed calendar"
                )
            close = self.close(instrument)
            expected[group.index] = close[dates + 2] / close[dates + 1] - 1
        return expected


def _correlations(frame):
    result = []
    for _, group in frame.groupby("datetime", sort=True):
        x = group["score"].to_numpy()
        y = group["label"].to_numpy()
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            result.append((np.nan, np.nan))
        else:
            rank_x, rank_y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
            result.append((np.corrcoef(x, y)[0, 1], np.corrcoef(rank_x, rank_y)[0, 1]))
    values = np.asarray(result)
    return {
        "ic": float(np.nanmean(values[:, 0])),
        "rank_ic": float(np.nanmean(values[:, 1])),
        "ic_days": int(np.isfinite(values[:, 0]).sum()),
        "rank_ic_days": int(np.isfinite(values[:, 1]).sum()),
    }


def _canonical_panel_labels(index, labels):
    """Compare native float32 labels by keys, independently of CSV row order."""
    return (
        pd.Series(np.asarray(labels, dtype=np.float32), index=index, name="label")
        .reorder_levels(["datetime", "instrument"])
        .sort_index()
    )


def _check_predictions(path, metrics, source, bounds):
    frame = pd.read_csv(path, parse_dates=["datetime"], float_precision="round_trip")
    if frame.empty or frame.duplicated(["datetime", "instrument"]).any():
        raise AssertionError("Empty or duplicated prediction rows")
    if not np.isfinite(frame["score"]).all():
        raise AssertionError("Nonfinite prediction scores")
    observed_index = pd.MultiIndex.from_frame(frame[["datetime", "instrument"]])
    expected_index = source.expected_index(bounds)
    extra = observed_index.difference(expected_index)
    if len(extra):
        raise AssertionError(f"{len(extra)} scores lie outside historical CSI300 membership")
    missing = expected_index.difference(observed_index)
    missing_by_instrument = []
    for instrument in missing.get_level_values("instrument").unique():
        dates = missing[missing.get_level_values("instrument") == instrument].get_level_values(
            "datetime"
        )
        close = source.close(instrument)
        finite_closes = int(np.isfinite(close[source.calendar.get_indexer(dates)]).sum())
        if finite_closes:
            raise AssertionError("A source membership row with an observed close has no prediction")
        missing_by_instrument.append(
            {
                "instrument": instrument,
                "rows": len(dates),
                "first": str(dates.min().date()),
                "last": str(dates.max().date()),
                "last_observed_close_date": str(source.last_dates[instrument].date()),
                "finite_close_rows": finite_closes,
            }
        )
    binary_labels = source.labels(frame)
    if not np.array_equal(np.isnan(binary_labels), frame["label"].isna().to_numpy()):
        raise AssertionError("Exported label missingness differs from raw binary close horizons")
    label_error = float(np.nanmax(np.abs(binary_labels - frame["label"].to_numpy())))
    if label_error > 1e-7:
        raise AssertionError(f"Raw binary delayed-return label error {label_error}")
    valid = frame.dropna(subset=["label"])
    residual = valid["score"] - valid["label"]
    calculated = {
        "prediction_count": len(frame),
        "evaluated_count": len(valid),
        "unavailable_label_count": len(frame) - len(valid),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "zero_return_rmse": float(np.sqrt(np.mean(valid["label"] ** 2))),
        "directional_accuracy": float((np.sign(valid["score"]) == np.sign(valid["label"])).mean()),
        **_correlations(valid),
    }
    errors = {}
    for key, value in calculated.items():
        recorded = metrics[key]
        if recorded is None and not math.isfinite(value):
            continue
        errors[key] = abs(value - recorded)
        # CSV serializes the original float32 labels to their decimal precision;
        # score/label calculations are independently repeated in float64 here.
        tolerance = 2e-8 if key in {"zero_return_rmse"} else 2e-10
        if errors[key] > tolerance:
            raise AssertionError(f"Prediction metric {key} differs by {errors[key]}")
    return (
        {
            "sha256": _sha(path),
            "membership_rows": len(expected_index),
            "prediction_rows": len(frame),
            "raw_binary_label_max_absolute_error": label_error,
            "metric_absolute_errors": errors,
            "unavailable_source_rows": missing_by_instrument,
            "calculated": calculated,
        },
        observed_index,
        frame["label"].to_numpy(),
    )


def _artifact_dir(run, metadata):
    return run / "mlruns" / metadata["experiment_id"] / metadata["recorder_id"] / "artifacts"


def _check_positions(run, metadata, source, bounds, trusted):
    path = _artifact_dir(run, metadata) / "portfolio_analysis/positions_normal_1day.pkl"
    if not trusted or not path.exists():
        return {
            "status": "not_checked",
            "reason": "requires available trusted local Qlib position artifact",
        }
    # This explicit branch is restricted to artifacts created by our local runs.
    with path.open("rb") as stream:
        positions = pickle.load(stream)
    candidate_symbols = source.expected_index(bounds).get_level_values("instrument").unique()
    terminated = {}
    final_date = max(pd.Timestamp(date) for date in positions)
    for instrument in candidate_symbols:
        source.close(instrument)
        end = source.last_dates[instrument]
        if end is not None and end < final_date:
            terminated[instrument] = end
    details = []
    for instrument, end in terminated.items():
        held = []
        for date, position in positions.items():
            date = pd.Timestamp(date)
            if date > end and instrument in position.position:
                stock = position.position[instrument]
                held.append((date, float(stock["weight"]), float(stock["price"])))
        if held:
            details.append(
                {
                    "instrument": instrument,
                    "last_observed_close_date": str(end.date()),
                    "stale_holding_days": len(held),
                    "first_stale_holding_date": str(held[0][0].date()),
                    "last_stale_holding_date": str(held[-1][0].date()),
                    "first_stale_weight": held[0][1],
                    "maximum_stale_weight": max(row[1] for row in held),
                    "final_stale_weight": held[-1][1] if held[-1][0] == final_date else 0.0,
                    "marked_price": held[-1][2],
                    "last_price_constant": all(row[2] == held[0][2] for row in held),
                }
            )
    return {
        "status": "checked",
        "artifact_sha256": _sha(path),
        "terminated_series_holdings": details,
        "total_final_stale_weight": sum(row["final_stale_weight"] for row in details),
        "interpretation": "These holdings have no later source close and remain at last observed price. Corporate-action settlement is unresolved; the return is a native Qlib simulation, not executable realized P&L.",
    }


def _check_booster(run, metadata, trusted):
    if not trusted:
        return {"status": "not_checked"}
    path = _artifact_dir(run, metadata) / "params.pkl"
    with path.open("rb") as stream:
        model = pickle.load(stream)
    booster = model.model
    log = _artifact_dir(run, metadata).parent / "metrics/l2.valid"
    history = [(int(row.split()[2]), float(row.split()[1])) for row in log.read_text().splitlines()]
    best = min(history, key=lambda row: row[1])
    if booster.best_iteration != best[0] + 1:
        raise AssertionError("Selected LightGBM iteration differs from minimum validation loss")
    selected_rmse = math.sqrt(booster.best_score["valid"]["l2"])
    exported = json.loads((run / "metrics.json").read_text())["validation"]["rmse"]
    if abs(selected_rmse - exported) > 1e-12:
        raise AssertionError(
            "LightGBM selected-booster validation error differs from exported score"
        )
    return {
        "status": "checked",
        "best_iteration": booster.best_iteration,
        "saved_tree_count": booster.num_trees(),
        "logged_rounds": len(history),
        "best_validation_l2": best[1],
        "last_logged_validation_l2": history[-1][1],
        "selected_validation_rmse": selected_rmse,
        "exported_rmse_absolute_error": abs(selected_rmse - exported),
        "interpretation": "recorder.list_metrics()['l2.valid'] is the last logged early-stopping round, while the saved booster and exported predictions use best_iteration.",
    }


def audit(runs: Path, provider: Path | None = None, *, trusted_local_pickles=False):
    protocol = json.loads((runs / "protocol.json").read_text())
    first_config = next(iter(protocol["configs"].values()))
    source = Source(provider or Path(first_config["provider_uri"]))
    segments = protocol["segments"]
    for previous, following in [("train", "valid"), ("valid", "test")]:
        last = source.calendar.get_loc(pd.Timestamp(segments[previous][1]))
        first = source.calendar.get_loc(pd.Timestamp(segments[following][0]))
        if last + 2 >= first:
            raise AssertionError("A label horizon crosses the next chronological partition")
    last = source.calendar.get_loc(pd.Timestamp(segments["test"][1]))
    if last + 3 >= len(source.calendar):
        raise AssertionError("Missing final observed label/executor reserve sessions")
    results, pending, failures, reference = {}, [], [], {}
    for name in protocol["models"]:
        run = runs / name
        if not (run / "run.json").exists():
            pending.append(name)
            continue
        metadata = json.loads((run / "run.json").read_text())
        if metadata.get("status") != "completed":
            pending.append(name)
            continue
        try:
            metrics = json.loads((run / "metrics.json").read_text())
            item = {}
            for period, filename, section in [
                ("valid", "validation_predictions.csv", metrics["validation"]),
                ("test", "predictions.csv", metrics),
            ]:
                checked, index, labels = _check_predictions(
                    run / filename, section, source, segments[period]
                )
                canonical = _canonical_panel_labels(index, labels)
                if period in reference:
                    if not canonical.equals(reference[period]):
                        raise AssertionError(
                            "Models do not share identical prediction indices and labels"
                        )
                else:
                    reference[period] = canonical
                item[period] = checked
            report_path = run / "portfolio.csv"
            report = pd.read_csv(
                report_path, index_col=0, parse_dates=True, float_precision="round_trip"
            )
            first = source.calendar.get_loc(pd.Timestamp(segments["test"][0]))
            expected_dates = source.calendar[first + 1 : last + 3]
            if not report.index.equals(expected_dates):
                raise AssertionError("Portfolio dates do not follow the declared delayed execution")
            initial = protocol["configs"][name]["backtest"]["account"]
            nav = initial * (1 + report["return"] - report["cost"]).cumprod()
            relative_error = float(((nav - report["account"]) / report["account"]).abs().max())
            if relative_error > 1e-10:
                raise AssertionError(
                    "Native portfolio account does not reconcile with daily net returns"
                )
            item["portfolio"] = {
                "sha256": _sha(report_path),
                "days": len(report),
                "start": str(report.index[0].date()),
                "end": str(report.index[-1].date()),
                "maximum_NAV_relative_error": relative_error,
                "simulated_net_return": float(nav.iloc[-1] / initial - 1),
                "source_gap_holdings": _check_positions(
                    run, metadata, source, segments["test"], trusted_local_pickles
                ),
            }
            if name == "lightgbm":
                item["booster_selection"] = _check_booster(run, metadata, trusted_local_pickles)
            results[name] = item
        except Exception as exc:
            failures.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
    gap_models = [
        name
        for name, item in results.items()
        if item["portfolio"]["source_gap_holdings"].get("terminated_series_holdings")
    ]
    return _clean(
        {
            "status": "passed" if not failures else "failed",
            "audited_models": list(results),
            "pending_models": pending,
            "failures": failures,
            "models_with_unresolved_source_gap_holdings": gap_models,
            "backtest_qualification": (
                "native Qlib simulation with unresolved terminated-series valuation"
                if gap_models
                else "native Qlib simulation; real execution and source corporate actions are not certified"
            ),
            "segments": segments,
            "scope": "Independent raw binary label/membership checks, CSV metric replay, selected-model iteration, and native NAV accounting. Source corporate-action gaps remain unresolved and are quantified, not repaired or excluded.",
            "models": results,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--trusted-local-pickles", action="store_true")
    args = parser.parse_args()
    result = audit(args.runs, args.provider, trusted_local_pickles=args.trusted_local_pickles)
    destination = args.report_dir or args.runs
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ["status", "audited_models", "pending_models", "failures"]
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
