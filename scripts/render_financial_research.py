#!/usr/bin/env python3
"""Render the common-period forecast and SPY backtest evidence as standalone SVGs.

Requires Matplotlib and NumPy. Run from any directory:

    python /path/to/repo/scripts/render_financial_research.py \
        --preview-dir /tmp/financial-research-previews

Inputs are comparison.csv, equity.csv, and settings.json in the report directory.
The renderer does not fit models, select candidates, or construct missing results.
Every supplied equity observation is plotted. PNG previews stay outside Git.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
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
DEFAULT_REPORT = ROOT / "reports" / "financial_research"
OUTPUT = ROOT / "docs" / "assets"
PAPER, INK, MUTED, GRID = "#f8fafc", "#17233a", "#64748b", "#e2e8f0"
TEAL, PURPLE = "#0f766e", "#6358c9"
PRIMARY_COST = 5.0
COST_GRID = (0.0, 5.0, 10.0, 20.0)
FOCUS = (
    "buy_and_hold",
    "timesfm3_mini_retrained",
    "timesfm3_loop_128",
    "multiscale_residual_mixer",
    "regime_linear_moe",
)
COLORS = {
    "buy_and_hold": INK,
    "timesfm3_mini": PURPLE,
    "timesfm3_mini_retrained": PURPLE,
    "timesfm3_loop_128": "#c96a2b",
    "multiscale_residual_mixer": TEAL,
    "regime_linear_moe": "#197eab",
    "cash": "#94a3b8",
}
FALLBACK_NAMES = {
    "buy_and_hold": "SPY buy and hold",
    "timesfm3_mini": "TimesFM 3 mini · archived ensemble",
    "timesfm3_mini_retrained": "TimesFM 3 mini · rerun ensemble",
    "timesfm3_loop_128": "TimesFM 3 loop · 128 closes",
    "timesfm3_frozen_128": "TimesFM 3 frozen · 128 closes",
    "timesfm3_adapter_only_128": "TimesFM 3 gated adapter · 128 closes",
    "timesfm3_pretrained_256": "TimesFM 3 frozen · 256 closes",
    "multiscale_residual_mixer": "Multiscale residual mixer",
    "regime_linear_moe": "Regime-gated linear experts",
    "three_architecture_blend": "Three-architecture blend",
    "zero_return": "Zero-return forecast",
    "training_mean": "Training-mean forecast",
    "lightgbm": "LightGBM",
    "transformer": "Plain Transformer",
    "deepseek_v41_mini": "DeepSeek V4.1 Flash 1017 mini",
    "glm54_mini": "GLM-5.4 loop mini",
    "qwen4_mini": "Qwen4 mini",
    "cash": "Cash · zero interest",
}


def _number(value, name, *, optional=False):
    if value is None or str(value).strip().lower() in {"", "nan", "null", "none"}:
        if optional:
            return None
        raise ValueError(f"Missing numeric value: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite numeric value: {name}")
    return result


def _load(report):
    with (report / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        comparison = list(csv.DictReader(stream))
    if not comparison:
        raise ValueError("The comparison table is empty.")
    for row in comparison:
        for column in ("one_way_cost_bps", "total_return", "ending_equity"):
            row[column] = _number(row.get(column), f"{row.get('model')}/{column}")
        for column in ("test_rmse", "validation_rmse"):
            row[column] = _number(row.get(column), f"{row.get('model')}/{column}", optional=True)
    primary = {row["model"]: row for row in comparison if row["one_way_cost_bps"] == PRIMARY_COST}
    if len(primary) != sum(row["one_way_cost_bps"] == PRIMARY_COST for row in comparison):
        raise ValueError("Duplicate model rows at the primary cost assumption.")
    if not primary:
        raise ValueError("The report must include the declared 5-bps primary assumption.")
    with (report / "equity.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if "exit_date" not in fields:
            raise ValueError("equity.csv must have an exit_date column.")
        observations = list(reader)
    if len(observations) < 2:
        raise ValueError("Equity requires an explicit initial NAV and subsequent observations.")
    dates = [datetime.fromisoformat(row["exit_date"]) for row in observations]
    if any(later <= earlier for earlier, later in zip(dates, dates[1:])):
        raise ValueError("Equity dates must be unique and increasing.")
    equity = {
        field: np.asarray([_number(row[field], field) for row in observations])
        for field in fields
        if field != "exit_date"
    }
    for name, values in equity.items():
        if values.min() <= 0 or not np.isclose(values[0], 1.0, rtol=0, atol=1e-12):
            raise ValueError(f"{name} must have positive NAV and an explicit initial value of one.")
        if name in primary and not np.isclose(values[-1], primary[name]["ending_equity"]):
            raise ValueError(f"{name} ending NAV differs between comparison and equity tables.")
    settings = json.loads((report / "settings.json").read_text(encoding="utf-8"))
    selected = settings["selected_by_validation"]
    if not isinstance(selected, str) or selected not in primary:
        raise ValueError("settings.selected_by_validation must identify a comparison model.")
    focus = list(FOCUS)
    if selected not in focus:
        focus.append(selected)
    for name in focus:
        if name not in primary or name not in equity:
            raise ValueError(f"Required focus model {name} is absent from the evidence.")
    return comparison, primary, dates, equity, selected, focus


def _name(model, primary):
    label = primary.get(model, {}).get("display_name")
    if label:
        return label
    if model.endswith("_shrunk"):
        return _name(model.removesuffix("_shrunk"), primary) + " · shrunk"
    return FALLBACK_NAMES.get(model, model.replace("_", " "))


def _color(model):
    if model in COLORS:
        return COLORS[model]
    if model.endswith("_shrunk"):
        return _color(model.removesuffix("_shrunk"))
    return "#b14b7a"


def _canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (1, 1),
            width - 2,
            height - 2,
            boxstyle="round,pad=0,rounding_size=16",
            facecolor=PAPER,
            edgecolor=GRID,
            linewidth=1,
        )
    )
    return fig, ax


def _label(ax, x, y, value, size=11, color=INK, weight="normal", **kwargs):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight, va="center", **kwargs)


def _style_axes(ax):
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cbd5e1")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_axisbelow(True)


def _save(fig, name, width, height, title, description, preview):
    path = OUTPUT / f"{name}.svg"
    fig.savefig(
        path,
        format="svg",
        facecolor=PAPER,
        metadata={"Date": None, "Creator": "render_financial_research.py", "Title": title},
    )
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", path.read_text(encoding="utf-8"))
    match = re.search(r"<svg\b[^>]*>", svg)
    if match is None:
        raise ValueError("Matplotlib did not produce an SVG document.")
    header = re.sub(r'\bwidth="[^"]*"', f'width="{width}"', match.group())
    header = re.sub(r'\bheight="[^"]*"', f'height="{height}"', header)
    header = header[:-1] + f' role="img" aria-labelledby="{name}-title {name}-desc">'
    header += f'\n <title id="{name}-title">{escape(title)}</title>'
    header += f'\n <desc id="{name}-desc">{escape(description)}</desc>'
    svg = svg[: match.start()] + header + svg[match.end() :]
    path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    if preview:
        fig.savefig(preview / f"{name}.png", dpi=100, facecolor=PAPER)
    plt.close(fig)
    print(path.relative_to(ROOT))


def render_forecasts(primary, selected, preview):
    records = sorted(
        (row for row in primary.values() if row["test_rmse"] is not None),
        key=lambda row: (row["test_rmse"], row["model"]),
    )
    zero = primary.get("zero_return", {}).get("test_rmse")
    if zero is None or zero <= 0:
        raise ValueError("A positive, measured zero_return RMSE is required for the delta plot.")
    names = [textwrap.fill(_name(row["model"], primary), width=35) for row in records]
    heights = [42 if "\n" in name else 33 for name in names]
    height, width = 330 + sum(heights), 1200
    fig, ax = _canvas(width, height)
    _label(ax, 43, 49, "Forecast accuracy on the same dates", size=25, weight="bold")
    _label(
        ax,
        43,
        87,
        "Common SPX target · historical test interval · lower RMSE is better",
        size=12,
        color=MUTED,
    )
    _label(ax, 43, 132, "MODEL / FORECAST", size=10, color=MUTED, weight="bold")
    _label(ax, 1048, 132, "TEST RMSE", size=10, color=MUTED, weight="bold", ha="right")
    _label(ax, 1156, 132, "VS. ZERO", size=10, color=MUTED, weight="bold", ha="right")
    deltas = [100 * (row["test_rmse"] / zero - 1) for row in records]
    low, high = min(min(deltas), 0.0), max(max(deltas), 0.0)
    margin = max((high - low) * 0.08, 0.08)
    low, high = low - margin, high + margin
    left, right = 390, 925

    def x(value):
        return left + (value - low) / (high - low) * (right - left)

    bottom = 153 + sum(heights)
    for tick in MaxNLocator(nbins=6).tick_values(low, high):
        if low <= tick <= high:
            ax.plot((x(tick), x(tick)), (151, bottom), color=GRID, linewidth=0.8)
            _label(
                ax,
                x(tick),
                bottom + 23,
                f"{tick:+.1f}" if tick else "0",
                size=10,
                color=MUTED,
                ha="center",
            )
    ax.plot((x(0), x(0)), (151, bottom), color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    position = 153
    for row, name, row_height, delta in zip(records, names, heights, deltas):
        y = position + row_height / 2
        if row["model"] == selected:
            ax.add_patch(
                FancyBboxPatch(
                    (31, position + 1),
                    1138,
                    row_height - 2,
                    boxstyle="round,pad=0,rounding_size=5",
                    facecolor="#e4f2ed",
                    edgecolor="none",
                )
            )
        color = TEAL if delta < 0 else PURPLE
        marker = "D" if "shrunk" in row["model"] else "o"
        ax.scatter(
            [x(delta)],
            [y],
            s=48,
            color=color,
            marker=marker,
            edgecolor=PAPER,
            linewidth=0.8,
            zorder=4,
        )
        if row["model"] == selected:
            ax.scatter(
                [x(delta)], [y], s=128, facecolors="none", edgecolors=INK, linewidth=1.1, zorder=5
            )
        _label(ax, 43, y, name, size=10.3)
        _label(ax, 1048, y, f"{row['test_rmse']:.8f}", size=10.8, ha="right")
        _label(ax, 1156, y, f"{delta:+.3f}%", size=10.8, color=color, ha="right", weight="bold")
        position += row_height
    _label(
        ax,
        (left + right) / 2,
        bottom + 54,
        "Relative change in RMSE versus zero return (%)",
        size=11,
        ha="center",
    )
    _label(
        ax,
        43,
        height - 70,
        "Ring / tinted row: model selected on validation. Diamonds: validation-calibrated shrinkage.",
        size=10.3,
        color=MUTED,
    )
    _label(
        ax,
        43,
        height - 44,
        "Seed ensembles are scored after averaging forecasts; the earlier README table averages individual-seed RMSE.",
        size=10.1,
        color=MUTED,
    )
    _label(
        ax,
        43,
        height - 20,
        "Previously inspected historical interval; archived and rerun forecasts are labeled separately. No confidence intervals shown.",
        size=10,
        color=MUTED,
    )
    _save(
        fig,
        "research-forecast",
        width,
        height,
        "Common-period forecast RMSE comparison",
        "; ".join(f"{_name(r['model'], primary)}: RMSE {r['test_rmse']:.8f}" for r in records)
        + f". Validation selected {_name(selected, primary)}. Negative percentage changes favor the model over zero return.",
        preview,
    )


def _legend(fig, focus, primary, selected, *, top):
    handles, names = [], []
    for model in focus:
        handles.append(
            Line2D(
                [],
                [],
                color=_color(model),
                linewidth=2.1,
                linestyle="--" if model == "cash" else "-",
            )
        )
        name = _name(model, primary)
        names.append(name + (" [validation choice]" if model == selected else ""))
    fig.legend(
        handles,
        names,
        loc="upper left",
        bbox_to_anchor=(0.071, top),
        ncol=2,
        frameon=False,
        fontsize=10,
        columnspacing=2.1,
        handlelength=2.1,
        labelspacing=0.7,
    )


def render_backtest(primary, dates, equity, selected, focus, preview):
    width, height = 1200, 1000
    fig, canvas = _canvas(width, height)
    _label(canvas, 43, 48, "What happened to $10,000?", size=26, weight="bold")
    _label(
        canvas,
        43,
        87,
        "SPY long/cash backtest · 5 bps one-way cost · next-session close execution",
        size=12,
        color=MUTED,
    )
    shown = [*focus, *(["cash"] if "cash" in equity and "cash" not in focus else [])]
    _legend(fig, shown, primary, selected, top=0.885)
    nav_ax = fig.add_axes((0.095, 0.435, 0.86, 0.32))
    dd_ax = fig.add_axes((0.095, 0.16, 0.86, 0.205), sharex=nav_ax)
    for ax in (nav_ax, dd_ax):
        _style_axes(ax)
        ax.set_xlim(dates[0], dates[-1])
    for model in shown:
        values = equity[model]
        style = "--" if model == "cash" else "-"
        nav_ax.plot(dates, values * 10_000, color=_color(model), linewidth=1.7, linestyle=style)
        drawdown = values / np.maximum.accumulate(values) - 1
        dd_ax.plot(dates, drawdown, color=_color(model), linewidth=1.3, linestyle=style)
    nav_ax.set_ylabel("Portfolio value (USD)", color=INK, fontsize=11, labelpad=10)
    nav_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value / 1000:,.0f}k"))
    nav_ax.tick_params(labelbottom=False)
    dd_ax.set_ylabel("Drawdown from prior peak", color=INK, fontsize=11, labelpad=10)
    dd_ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    dd_ax.set_ylim(top=0.005)
    dd_ax.xaxis.set_major_locator(mdates.YearLocator())
    dd_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    dd_ax.set_xlabel("Date", color=INK, fontsize=11)
    _label(
        canvas,
        43,
        901,
        f"Observed interval: {dates[0]:%d %b %Y}–{dates[-1]:%d %b %Y}. All daily NAV observations are plotted.",
        size=10.5,
        color=MUTED,
    )
    _label(
        canvas,
        43,
        929,
        "Signals use SPX history; trading returns use SPY adjusted closes. Initial NAV and final liquidation are included.",
        size=10.3,
        color=MUTED,
    )
    _label(
        canvas,
        43,
        955,
        "Cash earns zero interest. No leverage or shorting. Curves can overlap when forecasts imply identical positions.",
        size=10.3,
        color=MUTED,
    )
    _label(
        canvas,
        43,
        980,
        "This historical simulation assumes closing fills; adjusted-close returns are not broker execution records.",
        size=10.2,
        color=MUTED,
    )
    _save(
        fig,
        "research-backtest",
        width,
        height,
        "SPY strategy equity and drawdown at five basis points",
        f"{dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}. "
        + "; ".join(f"{_name(m, primary)}: ending NAV {equity[m][-1]:.6f}" for m in shown)
        + ". Initial NAV one is displayed as ten thousand dollars. Drawdown is measured from each strategy's prior peak.",
        preview,
    )


def render_costs(comparison, primary, dates, selected, focus, preview):
    width, height = 1200, 760
    fig, canvas = _canvas(width, height)
    _label(canvas, 43, 48, "How much do trading costs change the result?", size=24, weight="bold")
    _label(
        canvas,
        43,
        87,
        "Same signals and positions · four declared cost assumptions · higher net return is better",
        size=12,
        color=MUTED,
    )
    _legend(fig, focus, primary, selected, top=0.85)
    ax = fig.add_axes((0.095, 0.23, 0.86, 0.475))
    _style_axes(ax)
    ax.axhline(0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 4)))
    ax.axvline(PRIMARY_COST, color="#94a3b8", linewidth=0.9, linestyle=(0, (3, 4)))
    for model in focus:
        rows = [row for row in comparison if row["model"] == model]
        values = {row["one_way_cost_bps"]: row["total_return"] for row in rows}
        if len(rows) != len(values) or set(values) != set(COST_GRID):
            raise ValueError(f"{model} must have exactly the four registered cost scenarios.")
        ax.plot(
            COST_GRID,
            [100 * values[cost] for cost in COST_GRID],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=_color(model),
        )
    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks(COST_GRID, ["0", "5", "10", "20"])
    ax.set_xlabel("One-way trading cost (basis points)", color=INK, fontsize=11, labelpad=9)
    ax.set_ylabel("Total return after costs (%)", color=INK, fontsize=11, labelpad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}%"))
    _label(
        canvas,
        43,
        662,
        f"SPY exposure, {dates[0]:%d %b %Y}–{dates[-1]:%d %b %Y}; score > 0 holds SPY, otherwise cash.",
        size=10.5,
        color=MUTED,
    )
    _label(
        canvas,
        43,
        690,
        "The dashed vertical line marks the primary 5-bps assumption. Costs apply to turnover, including entry and liquidation.",
        size=10.2,
        color=MUTED,
    )
    _label(
        canvas,
        43,
        716,
        "Buy and hold is charged for entry and exit too. Cash return is zero; rates are sensitivity assumptions, not broker quotes.",
        size=10.2,
        color=MUTED,
    )
    _label(
        canvas,
        43,
        741,
        "A lower forecast RMSE does not guarantee better timing, lower turnover, or higher net return.",
        size=10.4,
        color=MUTED,
    )
    _save(
        fig,
        "research-costs",
        width,
        height,
        "Strategy return sensitivity to transaction costs",
        "Total return at zero, five, ten, and twenty basis points one-way trading cost. "
        + "; ".join(_name(m, primary) for m in focus)
        + ". Identical trading signals are used across cost assumptions.",
        preview,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    preview = args.preview_dir.resolve() if args.preview_dir else None
    if preview:
        if not preview.is_relative_to(Path("/tmp").resolve()):
            parser.error("Keep preview PNGs in /tmp; committed figures are SVG files.")
        preview.mkdir(parents=True, exist_ok=True)
    comparison, primary, dates, equity, selected, focus = _load(args.report_dir.resolve())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "financial-research-evidence",
            "axes.unicode_minus": True,
        }
    ):
        render_forecasts(primary, selected, preview)
        render_backtest(primary, dates, equity, selected, focus, preview)
        render_costs(comparison, primary, dates, selected, focus, preview)


if __name__ == "__main__":
    main()
