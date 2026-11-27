#!/usr/bin/env python3
"""Render paper-inspired post-training evidence without fitting or selecting models.

Requires Matplotlib and NumPy in the rendering environment, not in the package's
mandatory dependencies. Run from any directory, after the experiment finishes:

    python scripts/render_posttraining.py --preview-dir /tmp/posttraining-previews

Inputs in --report (default reports/paper_posttraining): metrics.csv,
backtest_metrics.csv, equity.csv and selection.json. The equity table is the
long-form, primary-5-bps subset of financial_ts.backtest.run_backtest's ledger.
Selection must record primary_es_seed: 42. No model is selected by this script.
--wait-seconds optionally waits for missing input files, without fabricating them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = ("metrics.csv", "backtest_metrics.csv", "equity.csv", "selection.json")
PRIMARY_COST = 5.0
PRIMARY_SEED = 42
SEEDS = (7, 42, 2026)
COSTS = (0.0, 5.0, 10.0, 20.0)
PAPER, INK, MUTED, GRID = "#f8fafc", "#17233a", "#64748b", "#e2e8f0"
TEAL, PURPLE = "#0f766e", "#6358c9"
FOCUS = (
    "baseline", "directional_edit", "es_seed_42", "combined_seed_42",
    "random_edit", "buy_and_hold",
)
FORECAST_ORDER = (
    "baseline", "directional_edit", "es_seed_42", "combined_seed_42",
    "random_edit", "zero_return", "training_mean",
)
NAMES = {
    "baseline": "Base checkpoint",
    "directional_edit": "Directional weight edit",
    "random_edit": "Random-direction control",
    "es_seed_42": "Evolution strategies · seed 42",
    "combined_seed_42": "Directional edit + ES · seed 42",
    "zero_return": "Zero-return reference",
    "training_mean": "Training-mean reference",
    "buy_and_hold": "SPY buy and hold",
    "cash": "Cash · zero interest",
}
COLORS = {
    "baseline": INK,
    "directional_edit": TEAL,
    "es_seed_42": PURPLE,
    "combined_seed_42": "#c16b25",
    "random_edit": "#8b91a0",
    "buy_and_hold": "#237ca5",
    "cash": "#a3acba",
}


def _numeric(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid numeric field: {field}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite field: {field}")
    return result


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty evidence table: {path}")
    return rows


def _wait_for_files(report, seconds):
    deadline = time.monotonic() + seconds
    while True:
        missing = [name for name in REQUIRED_FILES
                   if not (report / name).is_file() or (report / name).stat().st_size == 0]
        if not missing:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FileNotFoundError(
                f"Experiment artifacts are not ready in {report}: {', '.join(missing)}. "
                "Finish the experiment or use --wait-seconds. No figures were written."
            )
        time.sleep(min(1.0, remaining))


def _date(value, field):
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid date: {field}") from error
    if result.tzinfo is not None or result.time() != datetime.min.time():
        raise ValueError(f"Expected a timezone-naive daily date: {field}")
    return result


def _load(report):
    selection = json.loads((report / "selection.json").read_text(encoding="utf-8"))
    if selection.get("primary_es_seed") != PRIMARY_SEED:
        raise ValueError("selection.json must explicitly record primary_es_seed: 42.")

    metrics = {}
    for row in _read_csv(report / "metrics.csv"):
        model = row["model"]
        if model in metrics:
            raise ValueError(f"Duplicate forecast metric row: {model}")
        for field in ("validation_rmse", "test_rmse"):
            row[field] = _numeric(row.get(field), f"{model}/{field}")
            if row[field] <= 0:
                raise ValueError(f"RMSE must be positive for relative comparison: {model}")
        metrics[model] = row
    required_models = set(FORECAST_ORDER) | {
        f"{method}_seed_{seed}" for method in ("es", "combined") for seed in SEEDS
    }
    if not required_models <= metrics.keys():
        raise ValueError(f"Missing forecast models: {sorted(required_models - metrics.keys())}")

    backtests = {}
    for row in _read_csv(report / "backtest_metrics.csv"):
        model = row["model"]
        for field in ("cost_bps", "total_return", "max_drawdown"):
            row[field] = _numeric(row.get(field), f"{model}/{field}")
        key = (model, row["cost_bps"])
        if key in backtests:
            raise ValueError(f"Duplicate backtest metric row: {key}")
        backtests[key] = row
    for model in FOCUS:
        if any((model, cost) not in backtests for cost in COSTS):
            raise ValueError(f"Missing one of the four declared cost scenarios for {model}")

    groups = defaultdict(list)
    for row in _read_csv(report / "equity.csv"):
        if "cost_bps" in row and not np.isclose(
            _numeric(row["cost_bps"], "equity/cost_bps"), PRIMARY_COST,
            rtol=0, atol=1e-12,
        ):
            raise ValueError("equity.csv must contain only the primary 5-bps scenario.")
        groups[row["model"]].append(row)
    if not set(FOCUS) <= groups.keys():
        raise ValueError(f"Missing primary equity series: {sorted(set(FOCUS) - groups.keys())}")

    curves, dates = {}, None
    for model, rows in groups.items():
        signal_dates = [_date(r["signal_date"], f"{model}/signal_date") for r in rows]
        entry_dates = [_date(r["entry_date"], f"{model}/entry_date") for r in rows]
        exit_dates = [_date(r["exit_date"], f"{model}/exit_date") for r in rows]
        if any(not signal < entry < exit_ for signal, entry, exit_ in
               zip(signal_dates, entry_dates, exit_dates)):
            raise ValueError(f"Delayed signal/entry/exit ordering fails for {model}")
        if any(later <= earlier for earlier, later in zip(exit_dates, exit_dates[1:])):
            raise ValueError(f"Ledger dates must be increasing and unique for {model}")
        # The standard ledger starts after the first realized return. Anchor the
        # observed curve at the engine's documented initial, uninvested NAV.
        curve_dates = [entry_dates[0], *exit_dates]
        if dates is not None and curve_dates != dates:
            raise ValueError("All model curves must use identical daily observation dates.")
        dates = curve_dates
        equity = np.array([1.0, *[_numeric(r["net_equity"], "net_equity") for r in rows]])
        drawdown = np.array([0.0, *[_numeric(r["drawdown"], "drawdown") for r in rows]])
        returns = np.array([_numeric(r["net_return"], "net_return") for r in rows])
        if equity.min() <= 0 or not np.allclose(
            equity[1:], np.cumprod(1 + returns), rtol=1e-9, atol=1e-11,
        ):
            raise ValueError(f"Compounded ledger returns disagree with equity for {model}")
        if not np.allclose(
            drawdown, equity / np.maximum.accumulate(equity) - 1,
            rtol=1e-8, atol=1e-11,
        ):
            raise ValueError(f"Recorded drawdown disagrees with NAV for {model}")
        if (model, PRIMARY_COST) not in backtests:
            raise ValueError(f"No primary backtest metric for ledger model {model}")
        summary = backtests[model, PRIMARY_COST]
        if not np.isclose(equity[-1] - 1, summary["total_return"], rtol=1e-8, atol=1e-11):
            raise ValueError(f"Final NAV disagrees with total-return metric for {model}")
        if not np.isclose(drawdown.min(), summary["max_drawdown"], rtol=1e-8, atol=1e-11):
            raise ValueError(f"Drawdown disagrees with metric for {model}")
        curves[model] = {"equity": equity, "drawdown": drawdown}
    return metrics, backtests, dates, curves, selection


def _canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (1, 1), width - 2, height - 2, boxstyle="round,pad=0,rounding_size=16",
        facecolor=PAPER, edgecolor=GRID, linewidth=1,
    ))
    return fig, ax


def _text(ax, x, y, text, *, size=11, color=INK, weight="normal", **kwargs):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight, va="center", **kwargs)


def _style(ax, grid_axis="y"):
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd5e1")
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_axisbelow(True)


def _linestyle(model):
    return (0, (2, 3)) if model == "random_edit" else (0, (6, 3)) if model == "buy_and_hold" else "-"


def _curve_groups(curves, backtests, selection):
    """Collapse only verified identical curves; retain every model in the CSVs."""
    shown = list(FOCUS)
    names = {model: NAMES[model] for model in FOCUS}
    no_edit = selection.get("directional_edit", {}).get("selected_alpha") == 0
    for first, second in (("baseline", "directional_edit"), ("es_seed_42", "combined_seed_42")):
        same = all(np.array_equal(curves[first][field], curves[second][field])
                   for field in ("equity", "drawdown"))
        same = same and all(backtests[first, cost]["total_return"] ==
                            backtests[second, cost]["total_return"] for cost in COSTS)
        if no_edit and same:
            shown.remove(second)
            names[first] = ("Base / directional edit (α = 0)" if first == "baseline"
                            else "ES / combined · seed 42")
    return shown, names


def _legend(fig, shown, names, *, top, totals=None):
    labels = [names[model] for model in shown]
    if totals is not None:
        labels = [f"{label}   {totals[model, PRIMARY_COST]['total_return']:+.1%}"
                  for label, model in zip(labels, shown)]
    handles = [Line2D([], [], color=COLORS[model], linewidth=2.1,
                      linestyle=_linestyle(model)) for model in shown]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.066, top),
               ncol=2, frameon=False, fontsize=10.1, columnspacing=2.1,
               handlelength=2.4, labelspacing=0.8)


def _save(fig, output, name, width, height, title, description, preview):
    path = output / f"{name}.svg"
    fig.savefig(path, format="svg", facecolor=PAPER,
                metadata={"Date": None, "Creator": "render_posttraining.py", "Title": title})
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", path.read_text(encoding="utf-8"))
    match = re.search(r"<svg\b[^>]*>", svg)
    if match is None:
        raise ValueError("Matplotlib did not produce an SVG document.")
    header = re.sub(r'\bwidth="[^"]*"', f'width="{width}"', match.group())
    header = re.sub(r'\bheight="[^"]*"', f'height="{height}"', header)
    header = header[:-1] + f' role="img" aria-labelledby="{name}-title {name}-desc">'
    header += f'\n <title id="{name}-title">{escape(title)}</title>'
    header += f'\n <desc id="{name}-desc">{escape(description)}</desc>'
    svg = svg[:match.start()] + header + svg[match.end():]
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    if preview:
        fig.savefig(preview / f"{name}.png", dpi=100, facecolor=PAPER)
    plt.close(fig)
    print(path)


def _interval(dates):
    return f"{dates[0]:%d %b %Y}–{dates[-1]:%d %b %Y}"


def render_equity(backtests, dates, curves, selection, output, preview):
    width, height = 1200, 1000
    fig, canvas = _canvas(width, height)
    _text(canvas, 43, 48, "Paper-inspired edits: the trading outcome", size=25, weight="bold")
    _text(canvas, 43, 86, "SPY long/cash · 5 bps one-way cost · primary ES seed 42 fixed before evaluation",
          size=11.8, color=MUTED)
    shown, names = _curve_groups(curves, backtests, selection)
    _legend(fig, shown, names, top=0.885, totals=backtests)
    if len(shown) < len(FOCUS):
        _text(canvas, 90, 196, "Validation selected α = 0: directional edit kept the base; combined is identical to ES.",
              size=10.4, color=MUTED)
    nav_ax = fig.add_axes((0.095, 0.425, 0.86, 0.32))
    dd_ax = fig.add_axes((0.095, 0.16, 0.86, 0.195), sharex=nav_ax)
    for ax in (nav_ax, dd_ax):
        _style(ax)
        ax.set_xlim(dates[0], dates[-1])
    nav_ax.axhline(10_000, color=COLORS["cash"], linewidth=0.8, linestyle=(0, (2, 4)))
    for model in shown:
        nav_ax.plot(dates, curves[model]["equity"] * 10_000, color=COLORS[model],
                    linewidth=1.7, linestyle=_linestyle(model))
        dd_ax.plot(dates, curves[model]["drawdown"], color=COLORS[model],
                   linewidth=1.3, linestyle=_linestyle(model))
    nav_ax.set_ylabel("Value of an initial $10,000", color=INK, fontsize=11, labelpad=10)
    nav_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:,.0f}k"))
    nav_ax.tick_params(labelbottom=False)
    dd_ax.set_ylabel("Drawdown from prior peak", color=INK, fontsize=11, labelpad=10)
    dd_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    dd_ax.set_ylim(top=0.005)
    dd_ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    dd_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(dd_ax.xaxis.get_major_locator()))
    dd_ax.set_xlabel("Date", color=INK, fontsize=11)
    notes = (
        f"{_interval(dates)} · historical exploratory simulation on a previously inspected interval.",
        "Signals use SPX history; SPY adjusted closes proxy realized returns. Execution starts at the next session's close.",
        "Entry and final-liquidation costs are included. Cash earns zero interest; no leverage or shorting.",
        "Legend values are net total returns. Exactly identical methods share one curve and a combined legend entry.",
    )
    for y, note in zip((897, 925, 953, 980), notes):
        _text(canvas, 43, y, note, size=10.2, color=MUTED)
    _save(fig, output, "posttraining-equity", width, height,
          "Paper-inspired model edits: SPY equity and drawdown at five basis points",
          f"{_interval(dates)}. Historical exploratory simulation, not prospective verification. "
          + "; ".join(f"{NAMES[m]}: net total return {backtests[m, PRIMARY_COST]['total_return']:.6%}"
                      for m in FOCUS), preview)


def _delta(metrics, model, column):
    return 100 * (metrics[model][column] / metrics["baseline"][column] - 1)


def render_comparison(metrics, backtests, dates, selection, output, preview):
    width, height = 1200, 1040
    fig, canvas = _canvas(width, height)
    _text(canvas, 43, 48, "Do the edits improve the forecast?", size=26, weight="bold")
    _text(canvas, 43, 86, "Same SPX return target · lower RMSE is better · seed 42 is the declared primary run",
          size=11.8, color=MUTED)
    fig.legend(
        [Line2D([], [], color=PURPLE, marker="o", linestyle="none", markersize=6),
         Line2D([], [], color=TEAL, marker="D", linestyle="none", markersize=6)],
        ["Validation change vs. base", "Historical test change vs. base"],
        loc="upper left", bbox_to_anchor=(0.335, 0.88), ncol=2, frameon=False,
        fontsize=10, columnspacing=2.5,
    )
    _text(canvas, 43, 184, "MODEL / CONTROL", size=10, weight="bold", color=MUTED)
    _text(canvas, 1155, 184, "TEST RMSE", size=10, weight="bold", color=MUTED, ha="right")
    ax = fig.add_axes((0.34, 0.43, 0.45, 0.36))
    _style(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(len(FORECAST_ORDER) - 0.45, -0.55)
    values = [_delta(metrics, model, col) for model in FORECAST_ORDER
              for col in ("validation_rmse", "test_rmse")]
    low, high = min(0.0, min(values)), max(0.0, max(values))
    margin = max((high - low) * 0.10, 0.005)
    ax.set_xlim(low - margin, high + margin)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.2f}" if value else "0"))
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    for i, model in enumerate(FORECAST_ORDER):
        if model == "random_edit":
            ax.axhspan(i - 0.43, i + 0.43, color="#eef0f4", zorder=0)
        for col, offset, color, marker in (
            ("validation_rmse", -0.12, PURPLE, "o"),
            ("test_rmse", 0.12, TEAL, "D"),
        ):
            ax.scatter(_delta(metrics, model, col), i + offset, s=44, color=color,
                       marker=marker, edgecolors=PAPER, linewidths=0.7, zorder=3)
        y = height - ax.transData.transform((0, i))[1]
        label = NAMES[model]
        if selection.get("directional_edit", {}).get("selected_alpha") == 0:
            if model == "directional_edit":
                label = "Directional edit · α = 0"
            elif model == "combined_seed_42":
                label = "Combined · same as ES"
        _text(canvas, 43, y, label, size=10.8,
              color=MUTED if model == "random_edit" else INK)
        _text(canvas, 1155, y, f"{metrics[model]['test_rmse']:.8f}", size=11, ha="right")
    ax.set_xlabel("Relative change in RMSE against the base checkpoint (%)",
                  color=INK, fontsize=10.5, labelpad=12)

    _text(canvas, 43, 699, "Search-seed sensitivity: forecast error and trading return", size=17.5, weight="bold")
    _text(canvas, 43, 734, "ES and combined are shown together; identical values appear once. All changes are against the base checkpoint.",
          size=10.1, color=MUTED)
    for x, label, align in ((43, "ES SEARCH SEED", "left"), (425, "TEST RMSE", "right"),
                            (666, "RMSE CHANGE", "right"), (915, "NET RETURN · 5 BPS", "right"),
                            (1155, "NET CHANGE", "right")):
        _text(canvas, x, 773, label, size=9.7, color=MUTED, weight="bold", ha=align)

    def pair_text(es_value, combined_value, formatter):
        return formatter(es_value) if es_value == combined_value else (
            formatter(es_value) + " /\n" + formatter(combined_value))

    for seed, y in zip(SEEDS, (813, 866, 919)):
        if seed == PRIMARY_SEED:
            canvas.add_patch(FancyBboxPatch(
                (31, y - 23), 1138, 46, boxstyle="round,pad=0,rounding_size=6",
                facecolor="#e9edf8", edgecolor="none"))
        _text(canvas, 43, y, f"{seed}" + (" · primary" if seed == PRIMARY_SEED else ""),
              size=11.2, weight="bold" if seed == PRIMARY_SEED else "normal")
        models = (f"es_seed_{seed}", f"combined_seed_{seed}")
        rmse = [metrics[model]["test_rmse"] for model in models]
        rmse_delta = [_delta(metrics, model, "test_rmse") for model in models]
        returns = [backtests[model, PRIMARY_COST]["total_return"] for model in models]
        return_delta = [100 * (value - backtests["baseline", PRIMARY_COST]["total_return"])
                        for value in returns]
        for x, values_, formatter in (
            (425, rmse, lambda value: f"{value:.10f}"),
            (666, rmse_delta, lambda value: f"{value:+.4f}%"),
            (915, returns, lambda value: f"{value:+.2%}"),
            (1155, return_delta, lambda value: f"{value:+.2f} pp"),
        ):
            _text(canvas, x, y, pair_text(*values_, formatter), size=10.8, ha="right")
    _text(canvas, 43, 981, f"{_interval(dates)} trading interval; forecast rows use the matching signal dates. Historical, exploratory evidence.",
          size=10.1, color=MUTED)
    _text(canvas, 43, 1009, "Three ES perturbation seeds measure search sensitivity, not confidence intervals. pp means percentage points.",
          size=10.1, color=MUTED)
    _save(fig, output, "posttraining-comparison", width, height,
          "Forecast error and search-seed stability of paper-inspired model edits",
          "Each change is 100 times model RMSE / baseline RMSE minus one, calculated separately for each split. "
          "Seed 42 is declared in selection.json; other seeds show search sensitivity, not statistical confidence. "
          + "; ".join(f"{NAMES[m]}: test RMSE {metrics[m]['test_rmse']:.8f}" for m in FORECAST_ORDER),
          preview)


def render_costs(backtests, dates, curves, selection, output, preview):
    width, height = 1200, 770
    fig, canvas = _canvas(width, height)
    _text(canvas, 43, 48, "How much do costs change the outcome?", size=25, weight="bold")
    _text(canvas, 43, 86, "Identical signals and positions · four fixed cost scenarios · higher net return is better",
          size=11.8, color=MUTED)
    shown, names = _curve_groups(curves, backtests, selection)
    _legend(fig, shown, names, top=0.85)
    if len(shown) < len(FOCUS):
        _text(canvas, 90, 199, "Identical methods share one curve: base = directional edit (α = 0), and ES = combined.",
              size=10.3, color=MUTED)
    ax = fig.add_axes((0.095, 0.235, 0.86, 0.455))
    _style(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 4)))
    ax.axvline(PRIMARY_COST, color="#94a3b8", linewidth=0.9, linestyle=(0, (3, 4)))
    for model in shown:
        ax.plot(COSTS, [backtests[model, cost]["total_return"] for cost in COSTS],
                marker="s" if model == "random_edit" else "o", markersize=5,
                linewidth=1.8, color=COLORS[model], linestyle=_linestyle(model))
    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks(COSTS)
    ax.set_xlabel("One-way transaction cost (basis points)", color=INK, fontsize=11, labelpad=9)
    ax.set_ylabel("Total return after costs", color=INK, fontsize=11, labelpad=10)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    notes = (
        f"{_interval(dates)} · previously inspected historical interval · primary ES seed 42.",
        "Costs apply to traded notional, including entry and final liquidation. The vertical line marks the primary 5-bps case.",
        "Buy and hold pays entry and exit costs too. Rates are sensitivity assumptions; adjusted closes proxy execution prices.",
        "No candidate is selected from this plot. Historical gains do not establish a prospectively verified trading advantage.",
    )
    for y, note in zip((664, 692, 719, 746), notes):
        _text(canvas, 43, y, note, size=10.2, color=MUTED)
    _save(fig, output, "posttraining-costs", width, height,
          "Transaction-cost sensitivity of paper-inspired model edits",
          f"{_interval(dates)}. Total returns at one-way costs of zero, five, ten and twenty basis points. "
          + "; ".join(f"{NAMES[m]}: " + ", ".join(
              f"{cost:g} bps {backtests[m, cost]['total_return']:.6%}" for cost in COSTS)
              for m in FOCUS), preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "paper_posttraining")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets")
    parser.add_argument("--preview-dir", type=Path, help="Optional directory for PNG layout previews")
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="Bounded wait for missing input files; default fails immediately")
    args = parser.parse_args()
    if not math.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        parser.error("--wait-seconds must be a finite, nonnegative number")
    report, output = args.report.resolve(), args.output.resolve()
    preview = args.preview_dir.resolve() if args.preview_dir else None
    _wait_for_files(report, args.wait_seconds)
    metrics, backtests, dates, curves, selection = _load(report)
    output.mkdir(parents=True, exist_ok=True)
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans", "svg.fonttype": "path",
        "svg.hashsalt": "financial-ts-paper-posttraining", "axes.unicode_minus": True,
    }):
        render_equity(backtests, dates, curves, selection, output, preview)
        render_comparison(metrics, backtests, dates, selection, output, preview)
        render_costs(backtests, dates, curves, selection, output, preview)


if __name__ == "__main__":
    main()
