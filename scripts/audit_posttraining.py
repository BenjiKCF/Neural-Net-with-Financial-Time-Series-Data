#!/usr/bin/env python3
"""Independently audit the saved paper-inspired post-training experiment.

Uses NumPy and pandas, and imports no project training or accounting code.
By default, reads reports/paper_posttraining and writes only JSON to stdout.
Pass --report runs/paper-posttraining for the original run. Optional --evidence
adds checkpoint tensor-hash checks (and requires PyTorch). An explicit --output
creates a new audit JSON file; existing files are never overwritten.

This verifies recorded selection, predictions, source snapshots and arithmetic.
It does not establish prospective performance or independently retrain a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (7, 42, 2026)
COSTS = (0.0, 5.0, 10.0, 20.0)
PRIMARY = ("baseline", "directional_edit", "es_seed_42", "combined_seed_42")
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.10)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_frame(path, dates=()):
    return pd.read_csv(path, parse_dates=list(dates), float_precision="round_trip")


def close(actual, expected, context, *, atol=3e-12, rtol=3e-12):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, f"Shape mismatch: {context}")
    require(
        np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True),
        f"Numerical mismatch: {context}",
    )
    finite = np.isfinite(actual) & np.isfinite(expected)
    return float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0


def file_record(path):
    content = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def predictions(path):
    frame = read_frame(path, ("datetime",))
    require({"datetime", "instrument", "label"}.issubset(frame), f"Missing identity: {path}")
    require(frame.instrument.eq("SPX").all(), f"Expected SPX signals: {path}")
    require(frame.datetime.is_monotonic_increasing, f"Unsorted dates: {path}")
    require(not frame.datetime.duplicated().any(), f"Duplicated dates: {path}")
    require(frame.datetime.notna().all(), f"Missing dates: {path}")
    frame = frame.set_index(["datetime", "instrument"])
    require(np.isfinite(frame.to_numpy(dtype=float)).all(), f"Nonfinite prediction: {path}")
    return frame


def forecast_metrics(test, valid, selection, saved, summary):
    columns = list(test.columns.drop("label"))
    require(set(columns) == set(valid.columns.drop("label")), "Partition model rosters differ")
    require(set(saved.model) == set(columns) and saved.model.is_unique, "Forecast metric roster")
    require(len(summary["metrics"]) == len(columns), "Summary forecast metric count")
    summary_rows = {row["model"]: row for row in summary["metrics"]}
    y, v = test.label.to_numpy(), valid.label.to_numpy()
    baseline_rmse = float(np.sqrt(np.mean((y - test.baseline.to_numpy()) ** 2)))
    largest = 0.0
    for name in columns:
        p, q = test[name].to_numpy(), valid[name].to_numpy()
        expected = {
            "validation_rmse": float(np.sqrt(np.mean((v - q) ** 2))),
            "test_rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "rmse_delta_vs_base": float(np.sqrt(np.mean((y - p) ** 2))) - baseline_rmse,
            "test_mae": float(np.mean(np.abs(y - p))),
            "directional_accuracy": float(np.mean((p > 0) == (y > 0))),
            "test_count": len(test),
        }
        row = saved.set_index("model").loc[name]
        for key, value in expected.items():
            largest = max(largest, close(row[key], value, f"metrics/{name}/{key}"))
            largest = max(largest, close(summary_rows[name][key], value, f"summary/{name}/{key}"))
        close(selection["validation_rmse"][name], expected["validation_rmse"], f"selection/{name}")
        selected = name == selection["selected_by_validation"]
        require(bool(row["selected_by_validation"]) == selected, f"Selection flag: {name}")
        require(summary_rows[name]["selected_by_validation"] is selected, f"Summary flag: {name}")
    return largest


def selection_checks(protocol, selection, searches, test, valid, summary):
    require(tuple(protocol["base_seeds"]) == SEEDS, "Base seeds changed")
    require(tuple(protocol["es_seeds"]) == SEEDS, "Optimizer seeds changed")
    require(tuple(protocol["alpha_grid"]) == ALPHAS, "Alpha grid changed")
    require(
        protocol["primary_es_seed"] == selection["primary_es_seed"] == 42, "Primary seed changed"
    )
    require(summary["primary_es_seed"] == 42, "Summary primary seed changed")
    require(tuple(selection["primary_roster"]) == PRIMARY, "Primary comparison roster changed")
    require(selection["test_predictions_started"] is False, "Selection is not recorded before test")
    require(selection["optimizer_seeds_are_replicates"] is True, "Seed selection policy changed")
    require(selection["valid_count"] == len(valid), "Validation count differs")
    require(summary["test_count"] == len(test), "Test count differs")
    require(
        pd.Timestamp(protocol["created_at_utc"])
        <= pd.Timestamp(selection["locked_at_utc"])
        <= pd.Timestamp(protocol["finished_at_utc"]),
        "Registered chronology is inconsistent",
    )
    selected = min(
        PRIMARY, key=lambda name: (selection["validation_rmse"][name], PRIMARY.index(name))
    )
    require(
        selection["selected_by_validation"] == summary["selected_by_validation"] == selected,
        "Primary method was not selected on validation with the declared tie rule",
    )
    mean, scale = selection["training_mean"], selection["training_scale"]
    require(math.isfinite(mean) and math.isfinite(scale) and scale > 0, "Invalid target scaling")
    for frame in (test, valid):
        require(
            np.array_equal(frame.zero_return.to_numpy(), np.zeros(len(frame))),
            "Zero control changed",
        )
        require(
            np.array_equal(frame.training_mean.to_numpy(), np.full(len(frame), mean)),
            "Mean control changed",
        )
    alpha_results = {}
    for name in ("directional_edit", "random_edit"):
        record = selection[name]
        choices = record["candidates"]
        require(tuple(c["alpha"] for c in choices) == ALPHAS, f"Alpha candidate roster: {name}")
        require(
            all(math.isfinite(c["validation_normalized_mse"]) for c in choices),
            f"Alpha loss: {name}",
        )
        best = min(
            enumerate(choices), key=lambda item: (item[1]["validation_normalized_mse"], item[0])
        )[1]
        require(record["selected_alpha"] == best["alpha"], f"Alpha validation selection: {name}")
        close(
            np.mean(((valid[name] - valid.label) / scale) ** 2),
            best["validation_normalized_mse"],
            f"Selected edit loss: {name}",
        )
        close(
            choices[0]["validation_normalized_mse"],
            np.mean(((valid.baseline - valid.label) / scale) ** 2),
            f"Alpha-zero baseline: {name}",
        )
        for choice in choices:
            require(len(choice["edits"]) == len(SEEDS), f"Edit member count: {name}")
            for member, edit in enumerate(choice["edits"]):
                require(edit["alpha"] == choice["alpha"], f"Edit strength metadata: {name}")
                require(
                    edit["layer_name"] == f"members.{member}.{protocol['edit_layer']}",
                    f"Edit layer metadata: {name}",
                )
                if choice["alpha"] == 0:
                    require(
                        edit["weight_change_norm"] == edit["bias_change_norm"] == 0,
                        f"Alpha zero changed weights: {name}",
                    )
        if best["alpha"] == 0:
            require(
                selection["states_sha256"][name] == selection["states_sha256"]["baseline"],
                f"Alpha zero state differs: {name}",
            )
            for frame in (test, valid):
                require(
                    np.array_equal(frame[name], frame.baseline), f"Alpha zero scores differ: {name}"
                )
        alpha_results[name] = best["alpha"]
    expected_searches = {f"{method}_seed_{seed}" for seed in SEEDS for method in ("es", "combined")}
    require(set(searches) == expected_searches, "ES search roster differs")
    population, generations = protocol["es"]["population"], protocol["es"]["generations"]
    total_evaluations, selected_generations = 0, {}
    for name, search in searches.items():
        seed = int(name.rsplit("_", 1)[1])
        require(search["seed"] == seed, f"Optimizer seed mismatch: {name}")
        history = search["history"]
        require(
            [r["generation"] for r in history] == list(range(generations + 1)), f"History: {name}"
        )
        require(
            history[0]["status"] == "initial" and history[0]["candidate_seeds"] == [],
            f"Generation zero not eligible: {name}",
        )
        require(
            all(math.isfinite(r["validation_loss"]) for r in history), f"Validation loss: {name}"
        )
        best = min(history, key=lambda row: (row["validation_loss"], row["generation"]))
        require(search["selected_generation"] == best["generation"], f"Selected generation: {name}")
        close(
            search["validation_normalized_mse"], best["validation_loss"], f"Selected loss: {name}"
        )
        close(
            search["validation_normalized_mse"],
            np.mean(((valid[name] - valid.label) / scale) ** 2),
            f"Saved forecast/search loss: {name}",
        )
        initial = "directional_edit" if name.startswith("combined") else "baseline"
        close(
            history[0]["validation_loss"],
            np.mean(((valid[initial] - valid.label) / scale) ** 2),
            f"Initial ES state: {name}",
        )
        rng, previous_best = random.Random(seed), math.inf
        for row in history:
            require(
                row["selected_on_generation"] is (row["validation_loss"] < previous_best),
                f"Strict-improvement selection flag: {name}/{row['generation']}",
            )
            previous_best = min(previous_best, row["validation_loss"])
            if row["generation"] == 0:
                continue
            expected_seeds = [rng.randrange(2**63) for _ in range(population)]
            require(row["candidate_seeds"] == expected_seeds, f"Gaussian noise seeds: {name}")
            require(len(row["candidate_rewards"]) == population, f"Reward roster: {name}")
            # This completed experiment has no rejected proposals. A future run
            # with rejection needs a separately documented audit of its budget.
            require(
                all(r is not None and math.isfinite(r) for r in row["candidate_rewards"]),
                f"This audit expects all-finite completed proposal rewards: {name}",
            )
            require(row["valid_candidates"] == population, f"Finite candidate count: {name}")
            close(row["reward_mean"], np.mean(row["candidate_rewards"]), f"Reward mean: {name}")
            close(row["reward_std"], np.std(row["candidate_rewards"]), f"Reward std: {name}")
        require(
            search["candidate_evaluations"] == population * generations, f"Search budget: {name}"
        )
        total_evaluations += search["candidate_evaluations"]
        selected_generations[name] = best["generation"]
    require(summary["candidate_evaluations"] == total_evaluations, "Total search budget mismatch")
    zero_identity = alpha_results["directional_edit"] == 0
    if zero_identity:
        for seed in SEEDS:
            es, combined = f"es_seed_{seed}", f"combined_seed_{seed}"
            require(
                searches[es] == searches[combined],
                f"Zero-edit paired search histories differ: {seed}",
            )
            require(
                selection["states_sha256"][es] == selection["states_sha256"][combined],
                f"Zero-edit paired states differ: {seed}",
            )
            for frame in (test, valid):
                require(
                    np.array_equal(frame[es], frame[combined]),
                    f"Zero-edit paired scores differ: {seed}",
                )
    return {
        "primary_seed": 42,
        "selected_by_validation": selected,
        "selected_alphas": alpha_results,
        "selected_generations": selected_generations,
        "candidate_evaluations": total_evaluations,
        "zero_directional_alpha_makes_combined_identical_to_es": zero_identity,
    }


def simulate(scores, cost_bps, dates, entries, exits, entry_prices, exit_prices):
    """Independent explicit cash-and-share accounting, including all ledger fields."""
    rate = cost_bps / 10000
    cash, shares, gross_nav, peak, previous = 1.0, 0.0, 1.0, 1.0, 0
    rows = []
    for i, score in enumerate(scores):
        target = int(score > 0)
        starting = cash + shares * entry_prices[i]
        buy = int(target == 1 and shares == 0)
        sell = int(target == 0 and shares > 0)
        notional = fee = 0.0
        if buy:
            shares = cash / ((1 + rate) * entry_prices[i])
            notional = shares * entry_prices[i]
            fee = notional * rate
            cash = 0.0
        elif sell:
            notional = shares * entry_prices[i]
            fee = notional * rate
            cash, shares = notional - fee, 0.0
        cost_factor = (cash + shares * entry_prices[i]) / starting
        liquidation = int(i == len(scores) - 1 and shares > 0)
        liquidation_notional = shares * exit_prices[i] if liquidation else 0.0
        liquidation_fee = liquidation_notional * rate
        if liquidation:
            cash, shares = liquidation_notional - liquidation_fee, 0.0
        nav = cash + shares * exit_prices[i]
        asset_return = exit_prices[i] / entry_prices[i] - 1
        gross_return = target * asset_return
        gross_nav *= 1 + gross_return
        peak = max(peak, nav)
        rows.append(
            {
                "signal_date": dates[i],
                "entry_date": entries[i],
                "exit_date": exits[i],
                "score": score,
                "previous_position": previous,
                "position": target,
                "asset_return": asset_return,
                "gross_return": gross_return,
                "entry_trade": buy,
                "exit_trade": sell,
                "liquidation_trade": liquidation,
                "exposure_turnover": buy + sell + liquidation,
                "traded_notional": notional,
                "liquidation_notional": liquidation_notional,
                "trade_cost": fee,
                "liquidation_cost": liquidation_fee,
                "cost_factor": cost_factor,
                "liquidation_factor": 1 - rate if liquidation else 1.0,
                "starting_equity": starting,
                "net_return": nav / starting - 1,
                "gross_equity": gross_nav,
                "net_equity": nav,
                "drawdown": nav / peak - 1,
            }
        )
        previous = target
    ledger = pd.DataFrame(rows)
    daily = ledger.net_return.to_numpy()
    elapsed = (exits[-1] - entries[0]).days
    ending = ledger.net_equity.iloc[-1]
    deviation = float(np.std(daily, ddof=1))
    metrics = {
        "periods": len(scores),
        "elapsed_days": elapsed,
        "ending_equity": ending,
        "total_return": ending - 1,
        "gross_total_return": gross_nav - 1,
        "cagr": math.expm1(math.log(ending) * 365.25 / elapsed),
        "annualized_volatility": deviation * math.sqrt(252),
        "sharpe_zero_cash": float(np.mean(daily)) / deviation * math.sqrt(252)
        if deviation > 0
        else np.nan,
        "max_drawdown": ledger.drawdown.min(),
        "mean_exposure": ledger.position.mean(),
        "entries": ledger.entry_trade.sum(),
        "exits": ledger.exit_trade.sum() + ledger.liquidation_trade.sum(),
        "trade_count": ledger.exposure_turnover.sum(),
        "completed_round_trips": ledger.entry_trade.sum(),
        "exposure_turnover": ledger.exposure_turnover.sum(),
        "fees_paid_per_initial_equity": ledger.trade_cost.sum() + ledger.liquidation_cost.sum(),
    }
    return ledger, metrics


def portfolio_checks(report, test, summary):
    settings = load_json(report / "backtest_settings.json")
    require(
        tuple(settings["costs_bps"]) == COSTS and settings["primary_cost_bps"] == 5,
        "Cost assumptions differ",
    )
    require(
        settings["initial_equity"] == 1
        and settings["cash_return"] == 0
        and settings["terminal_liquidation"]
        and not settings["shorting"]
        and not settings["leverage"],
        "Portfolio assumptions differ",
    )
    prices = read_frame(report / "spy_prices.csv", ("date",))
    require(prices.symbol.eq("SPY").all(), "Execution prices must identify SPY")
    calendar = pd.DatetimeIndex(prices.date)
    require(
        calendar.is_unique and calendar.is_monotonic_increasing and not calendar.hasnans,
        "Invalid price calendar",
    )
    require(
        np.isfinite(prices.adj_close).all() and prices.adj_close.gt(0).all(),
        "Invalid adjusted prices",
    )
    dates = pd.DatetimeIndex(test.index.get_level_values("datetime"))
    positions = calendar.get_indexer(dates)
    require(
        (positions >= 0).all() and (np.diff(positions) == 1).all(), "Nonconsecutive signal sessions"
    )
    require(positions[-1] + 2 < len(calendar), "Missing execution sessions")
    entries, exits = calendar[positions + 1], calendar[positions + 2]
    values = prices.adj_close.to_numpy()
    entry_prices, exit_prices = values[positions + 1], values[positions + 2]
    strategies = {name: test[name].to_numpy() for name in test.columns if name != "label"}
    strategies.update(cash=np.zeros(len(test)), buy_and_hold=np.ones(len(test)))
    saved = read_frame(report / "backtest_metrics.csv")
    require(not saved.duplicated(["model", "cost_bps"]).any(), "Duplicate portfolio metric rows")
    require(
        set(zip(saved.model, saved.cost_bps)) == {(m, c) for m in strategies for c in COSTS},
        "Portfolio scenario roster differs",
    )
    equity = read_frame(report / "equity.csv", ("signal_date", "entry_date", "exit_date"))
    require(
        set(equity.model) == set(strategies) and equity.cost_bps.eq(5).all(),
        "Primary ledger roster",
    )
    require(not equity.duplicated(["model", "signal_date"]).any(), "Duplicate ledger rows")
    require(len(equity) == len(test) * len(strategies), "Incomplete primary ledger")
    summary_rows = {r["model"]: r for r in summary["backtest_5bps"]}
    require(set(summary_rows) == set(strategies), "Summary portfolio roster")
    max_metric, max_ledger, checked_fields = 0.0, 0.0, set()
    for name, scores in strategies.items():
        for cost in COSTS:
            ledger, expected = simulate(
                scores, cost, dates, entries, exits, entry_prices, exit_prices
            )
            row = saved.loc[saved.model.eq(name) & saved.cost_bps.eq(cost)].iloc[0]
            for field, value in expected.items():
                max_metric = max(
                    max_metric, close(row[field], value, f"portfolio/{name}/{cost}/{field}")
                )
                if cost == 5:
                    summary_value = summary_rows[name][field]
                    summary_value = np.nan if summary_value is None else summary_value
                    max_metric = max(
                        max_metric, close(summary_value, value, f"portfolio summary/{name}/{field}")
                    )
            for field, value in (
                ("signal_start", dates[0]),
                ("signal_end", dates[-1]),
                ("entry_start", entries[0]),
                ("exit_end", exits[-1]),
            ):
                require(row[field] == value.strftime("%Y-%m-%d"), f"Portfolio date: {name}/{field}")
            if cost == 5:
                actual = equity.loc[equity.model.eq(name)].reset_index(drop=True)
                for field in ledger.columns:
                    if field.endswith("_date"):
                        require(
                            actual[field].equals(ledger[field]),
                            f"Ledger date mismatch: {name}/{field}",
                        )
                    else:
                        max_ledger = max(
                            max_ledger,
                            close(actual[field], ledger[field], f"ledger/{name}/{field}"),
                        )
                    checked_fields.add(field)
    return {
        "portfolio_scenarios": len(saved),
        "primary_ledger_rows": len(equity),
        "primary_ledger_fields_checked": sorted(checked_fields),
        "maximum_metric_absolute_difference": max_metric,
        "maximum_ledger_absolute_difference": max_ledger,
        "published_price_rows": len(prices),
    }


def source_checks(report, protocol, evidence):
    records = []
    for expected in protocol["source_files"]:
        name = Path(expected["path"]).name
        path = report / "source" / name
        if not path.is_file() and evidence is not None:
            path = evidence / "source" / name
        observed = file_record(path)
        require(
            observed["sha256"] == expected["sha256"] and observed["bytes"] == expected["bytes"],
            f"Registered source snapshot differs: {name}",
        )
        records.append(observed)
    return records


def checkpoint_checks(evidence, selection):
    if evidence is None:
        return {
            "status": "not_requested",
            "reason": "No --evidence directory supplied; model tensors are not needed for accounting replay.",
        }
    import torch

    checked = []
    for name, expected_hash in selection["states_sha256"].items():
        path = evidence / "checkpoints" / f"{name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        close(checkpoint["target_mean"], selection["training_mean"], f"Checkpoint mean: {name}")
        close(checkpoint["target_scale"], selection["training_scale"], f"Checkpoint scale: {name}")
        digest = hashlib.sha256()
        for key, tensor in checkpoint["state_dict"].items():
            require(
                torch.isfinite(tensor).all().item(), f"Nonfinite checkpoint tensor: {name}/{key}"
            )
            digest.update(key.encode() + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        require(digest.hexdigest() == expected_hash, f"Selected checkpoint tensor hash: {name}")
        checked.append({"model": name, "state_sha256": expected_hash, **file_record(path)})
    return {"status": "passed", "checkpoints": checked}


def audit(report, evidence=None):
    protocol, summary = (load_json(report / name) for name in ("protocol.json", "summary.json"))
    selection = load_json(report / "selection.json")
    require(protocol["status"] == summary["status"] == "completed", "Experiment not completed")
    test, valid = (
        predictions(report / "predictions.csv"),
        predictions(report / "validation_predictions.csv"),
    )
    expected_models = set(protocol["roster"]) | {"zero_return", "training_mean"}
    require(
        set(test.columns) == expected_models | {"label"},
        "Prediction roster differs from registration",
    )
    for segment, frame in (("valid", valid), ("test", test)):
        dates = frame.index.get_level_values("datetime")
        require(
            [dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")]
            == selection["segments"][segment],
            f"Partition boundaries: {segment}",
        )
    selection_digest = file_record(report / "selection.json")["sha256"]
    require(
        selection_digest
        == summary["selection_sha256_before_test"]
        == summary["selection_sha256_after_test"],
        "Selection hash changed or disagrees with before/after records",
    )
    source_records = source_checks(report, protocol, evidence)
    forecast_error = forecast_metrics(
        test, valid, selection, read_frame(report / "metrics.csv"), summary
    )
    selection_result = selection_checks(
        protocol, selection, load_json(report / "search_history.json"), test, valid, summary
    )
    fit = load_json(report / "fit_samples.json")
    require(
        fit["sample_count"] == selection["fit_count"] == len(fit["indices"]) == len(fit["dates"]),
        "Fitting sample count mismatch",
    )
    require(len(set(fit["indices"])) == len(fit["indices"]), "Duplicate fitting sample positions")
    require(
        all(0 <= i < selection["train_count"] for i in fit["indices"]),
        "Fitting indices outside train",
    )
    require(
        all(
            selection["segments"]["train"][0] <= d <= selection["segments"]["train"][1]
            for d in fit["dates"]
        ),
        "Fitting dates outside train",
    )
    directions = load_json(report / "directions.json")
    require([row["base_seed"] for row in directions] == list(SEEDS), "Direction member roster")
    for row in directions:
        close(
            np.linalg.norm(row["direction"]),
            row["direction_norm"],
            "Recorded direction norm",
            atol=1e-6,
        )
        close(np.linalg.norm(row["random_direction"]), 1.0, "Random direction norm", atol=1e-6)
    portfolio = portfolio_checks(report, test, summary)
    files = [
        "protocol.json",
        "summary.json",
        "selection.json",
        "search_history.json",
        "fit_samples.json",
        "directions.json",
        "predictions.csv",
        "validation_predictions.csv",
        "metrics.csv",
        "backtest_metrics.csv",
        "backtest_settings.json",
        "equity.csv",
        "spy_prices.csv",
    ]
    return {
        "status": "passed",
        "method": "Independent saved-forecast checks and explicit cash/share accounting; no project imports",
        "test_observations": len(test),
        "validation_observations": len(valid),
        "forecast_models": len(expected_models),
        "maximum_forecast_metric_absolute_difference": forecast_error,
        "selection": selection_result,
        "portfolio": portfolio,
        "registered_source_snapshots": source_records,
        "selection_sha256": selection_digest,
        "checkpoint_evidence": checkpoint_checks(evidence, selection),
        "files": [file_record(report / name) for name in files],
        "limitations": [
            "Verifies recorded chronology and selection hashes, not externally witnessed execution timestamps.",
            "Does not retrain models or independently regenerate every forecast from features.",
            "The historical test interval was previously inspected; this audit does not establish prospective superiority.",
            "Uses the supplied complete-session SPY adjusted-close series as a total-return proxy, not historical fills.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/paper_posttraining")
    parser.add_argument(
        "--evidence", type=Path, help="Run directory containing source/ and checkpoints/"
    )
    parser.add_argument(
        "--output", type=Path, help="Create a new audit JSON; existing paths are rejected"
    )
    args = parser.parse_args()
    result = audit(args.report.resolve(), args.evidence.resolve() if args.evidence else None)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content, end="")


if __name__ == "__main__":
    main()
