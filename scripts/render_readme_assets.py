#!/usr/bin/env python3
"""Render the README's vector artwork from the recorded benchmark evidence.

Requires Matplotlib; no forecasting dependencies or training are needed. Run
from any directory, for example:

    python scripts/render_readme_assets.py --preview-dir /tmp/financial-ts-previews

The committed outputs are self-contained SVGs. Optional PNG previews remain in
/tmp. Fonts are outlined and SVG identifiers are deterministic; generated assets
contain no remote references, scripts, or embedded raster images.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from datetime import date
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "architecture_benchmark.json"
OUTPUT = ROOT / "docs" / "assets"
NAVY = "#0b1220"
MINT = "#5eead4"
VIOLET = "#818cf8"
INK = "#17233a"
MUTED = "#64748b"
PAPER = "#f8fafc"
GRID = "#e2e8f0"
TEAL = "#0f766e"
PURPLE = "#6358c9"
MODEL_NAMES = {
    "lightgbm": "LightGBM",
    "transformer": "Temporal Transformer",
    "deepseek_v41_mini": "DeepSeek V4.1 Flash 1017 mini",
    "glm54_mini": "GLM-5.4 mini",
    "qwen4_mini": "Qwen4 mini",
    "timesfm3_mini": "TimesFM 3 mini",
}


def canvas(width, height, background):
    """Return a pixel-coordinate canvas with its own opaque background."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=background)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, text, size=12, color=INK, weight="normal", **kwargs):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        va="center",
        fontfamily="DejaVu Sans",
        **kwargs,
    )


def rounded(ax, x, y, width, height, face, edge="none", radius=10, **kwargs):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color, connection="arc3", width=1.6, **kwargs):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            connectionstyle=connection,
            linewidth=width,
            color=color,
            **kwargs,
        )
    )


def save(fig, name, width, height, title, description, preview_dir):
    path = OUTPUT / f"{name}.svg"
    fig.savefig(
        path,
        format="svg",
        facecolor=fig.get_facecolor(),
        metadata={"Date": None, "Creator": "render_readme_assets.py", "Title": title},
    )
    # Explicit pixel dimensions and accessible descriptions survive GitHub's
    # image handling. The SVG paths include glyph outlines, not font downloads.
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    root = re.search(r"<svg\b[^>]*>", svg)
    if root is None:
        raise ValueError(f"Matplotlib did not produce an SVG root for {name}.")
    header = re.sub(r'\bwidth="[^"]*"', f'width="{width}"', root.group())
    header = re.sub(r'\bheight="[^"]*"', f'height="{height}"', header)
    header = header[:-1] + f' role="img" aria-labelledby="{name}-title {name}-desc">'
    header += (
        f'\n <title id="{name}-title">{escape(title)}</title>'
        f'\n <desc id="{name}-desc">{escape(description)}</desc>'
    )
    svg = svg[: root.start()] + header + svg[root.end() :]
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    path.write_text(svg, encoding="utf-8")
    if preview_dir is not None:
        fig.savefig(preview_dir / f"{name}.png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(path.relative_to(ROOT))


def render_hero(preview_dir):
    width, height = 1200, 320
    fig, ax = canvas(width, height, NAVY)
    rounded(ax, 1, 1, width - 2, height - 2, NAVY, "#263348", radius=18, linewidth=1)
    ax.plot((56, 98), (62, 62), color=MINT, linewidth=3, solid_capstyle="round")
    label(ax, 112, 62, "OPEN RESEARCH TOOLKIT", size=10.5, color="#a9b8cd", weight="bold")
    label(ax, 54, 133, "Financial Time-Series Lab", size=31.5, color="#f8fafc", weight="bold")
    label(
        ax,
        56,
        180,
        "Qlib workflows. Compact architectures. Reproducible results.",
        size=12.7,
        color="#c4cfdf",
    )
    for x, text, color in (
        (56, "CHRONOLOGICAL SPLITS", MINT),
        (282, "COMPACT MODELS", VIOLET),
        (477, "INSPECTABLE EVIDENCE", MINT),
    ):
        ax.scatter([x + 3], [247], s=17, color=color, zorder=3)
        label(ax, x + 17, 247, text, size=9, color="#9eafc6")

    # Abstract patches and a repeated computation path; this is a motif, not a
    # simulated market series or an asserted diagram of any one architecture.
    rounded(ax, 831, 37, 329, 246, "#101c30", "#25354c", radius=17, linewidth=1)
    label(ax, 855, 65, "LOOPS / PATCHES", size=9, color="#93a7c2", weight="bold")
    for row in range(3):
        for column in range(3):
            x, y = 855 + column * 16, 130 + row * 23
            rounded(
                ax,
                x,
                y,
                11,
                17,
                MINT if (row + column) % 3 == 0 else "#27585d",
                radius=2,
            )
    arrow(ax, (907, 161), (945, 161), MINT)
    rounded(ax, 950, 123, 78, 78, "#192643", "#6876b7", radius=12, linewidth=1.2)
    for shift, color in ((0, VIOLET), (12, "#a7afff"), (24, MINT)):
        rounded(ax, 969, 141 + shift, 40, 6, color, radius=3)
    arrow(ax, (1028, 161), (1072, 161), VIOLET)
    for row in range(2):
        for column in range(3):
            rounded(
                ax,
                1080 + column * 17,
                142 + row * 24,
                12,
                18,
                VIOLET if row == 0 else "#394574",
                radius=3,
            )
    arrow(ax, (1014, 118), (967, 118), VIOLET, "arc3,rad=1.35", width=1.7)
    arrow(ax, (970, 207), (1017, 207), MINT, "arc3,rad=1.2", width=1.5)
    label(ax, 994, 258, "STRUCTURE, THEN EVIDENCE", size=8.3, color="#9eafc6", ha="center")
    save(
        fig,
        "hero",
        width,
        height,
        "Financial Time-Series Lab",
        "Qlib workflows. Compact architectures. Reproducible results. "
        "Abstract patch and loop motif; it does not depict market observations.",
        preview_dir,
    )


def benchmark_rows(report):
    rows = []
    expected = set(report["protocol"]["scratch_models"])
    if expected != set(MODEL_NAMES):
        raise ValueError("Update display labels when the recorded model set changes.")
    for aggregate in report["scratch"]["aggregate"]:
        name = aggregate["model"]
        runs = [r["summary"] for r in report["scratch"]["runs"] if r["summary"]["model"] == name]
        values = [r["test_rmse"] for r in runs]
        mean, std = statistics.mean(values), statistics.pstdev(values)
        zero = aggregate["zero_return_rmse"]
        if not (
            zero > 0
            and len(runs) == len(report["protocol"]["seeds"])
            and math.isclose(mean, aggregate["test_rmse_mean"], abs_tol=1e-14)
            and math.isclose(std, aggregate["test_rmse_std_across_seeds"], abs_tol=1e-14)
            and all(math.isclose(r["zero_return_rmse"], zero, abs_tol=1e-14) for r in runs)
        ):
            raise ValueError(f"Inconsistent recorded benchmark values for {name}.")
        rows.append(
            {
                "name": name,
                "mean": 100 * (mean / zero - 1),
                "std": 100 * std / zero,
                "seeds": [100 * (value / zero - 1) for value in values],
            }
        )
    return sorted(rows, key=lambda row: row["mean"])


def render_benchmark(report, preview_dir):
    width, height = 1200, 660
    fig, card = canvas(width, height, PAPER)
    rounded(card, 1, 1, width - 2, height - 2, PAPER, GRID, radius=17, linewidth=1)
    label(card, 42, 45, "Historical forecast error", size=23, weight="bold")
    start, end = report["data"]["segments"]["test"]
    seed_count = len(report["protocol"]["seeds"])
    label(
        card,
        43,
        82,
        f"{seed_count} seeds · same {start[:4]}–{end[:4]} test split · lower is better",
        size=12.5,
        color=MUTED,
    )
    label(card, 43, 124, "MODEL", size=10, color=MUTED, weight="bold")
    label(card, 1137, 124, "MEAN CHANGE", size=10, color=MUTED, weight="bold", ha="right")

    rows = benchmark_rows(report)
    ax = fig.add_axes((0.298, 0.266, 0.563, 0.514), facecolor=PAPER)
    left = min(-0.5, min(r["mean"] - r["std"] for r in rows) - 0.22)
    right = max(r["mean"] + r["std"] for r in rows) + 0.35
    ax.set(xlim=(left, right), ylim=(len(rows) - 0.4, -0.6))
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", colors=MUTED, labelsize=11, length=0, pad=10)
    ticks = [i / 2 for i in range(math.ceil(left * 2), math.floor(right * 2) + 1)]
    ax.set_xticks(ticks, ["0" if v == 0 else f"{v:+.1f}" for v in ticks])
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.axvline(0, color="#52657e", linewidth=1.4, linestyle=(0, (4, 4)), zorder=1)
    ax.text(
        0,
        1.035,
        "ZERO RETURN",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=9,
        fontweight="bold",
    )
    for index, row in enumerate(rows):
        color = TEAL if row["mean"] < 0 else PURPLE
        offsets = (
            [0]
            if seed_count == 1
            else [0.14 * (i / (seed_count - 1) * 2 - 1) for i in range(seed_count)]
        )
        ax.scatter(
            row["seeds"],
            [index + offset for offset in offsets],
            s=28,
            facecolors=PAPER,
            edgecolors="#9aa8ba",
            linewidths=1.25,
            zorder=3,
        )
        ax.errorbar(
            row["mean"],
            index,
            xerr=row["std"],
            fmt="o",
            color=color,
            markersize=8.3,
            elinewidth=2.2,
            capsize=4,
            capthick=1.5,
            markeredgecolor=PAPER,
            markeredgewidth=1.2,
            zorder=4,
        )
        _, y_pixel = ax.transData.transform((0, index))
        y_card = height - y_pixel
        label(card, 43, y_card, MODEL_NAMES[row["name"]], size=11.3, color=INK)
        label(
            card,
            1138,
            y_card,
            f"{row['mean']:+.3f}%",
            size=12.3,
            color=color,
            weight="bold",
            ha="right",
        )

    label(
        card,
        702,
        536,
        "Relative change in test RMSE vs. zero-return forecast (%)",
        size=11.6,
        color=INK,
        ha="center",
    )
    best = rows[0]
    observation = (
        f"The observed gain is small: {MODEL_NAMES[best['name']]} has {abs(best['mean']):.3f}% lower RMSE than zero return."
        if best["mean"] < 0
        else "None of the recorded model means improves on the zero-return forecast."
    )
    rounded(card, 42, 562, 1116, 35, "#e9f5f1", radius=7)
    label(
        card,
        56,
        580,
        observation,
        size=11,
        color=TEAL,
    )
    label(
        card,
        43,
        618,
        "Open dots: individual seeds. Solid marks: mean ± 1 seed SD, not a confidence interval.",
        size=10.2,
        color=MUTED,
    )
    label(
        card,
        43,
        642,
        "Exploratory historical comparison. Official pretrained TimesFM 3 is evaluated separately and excluded here.",
        size=10.2,
        color=MUTED,
    )
    save(
        fig,
        "benchmark",
        width,
        height,
        "Relative historical RMSE across six trainable models",
        "Each mean is 100 times (model RMSE divided by zero-return RMSE minus one). "
        "Whiskers show one population standard deviation across seeds; open dots are individual seeds. "
        + "; ".join(f"{MODEL_NAMES[r['name']]}: {r['mean']:+.3f}%" for r in rows)
        + ". Lower is better. Official pretrained TimesFM 3 is excluded.",
        preview_dir,
    )


def render_splits(report, preview_dir):
    width, height = 1200, 400
    fig, ax = canvas(width, height, PAPER)
    rounded(ax, 1, 1, width - 2, height - 2, PAPER, GRID, radius=17, linewidth=1)
    label(ax, 43, 44, "A chronological split, with a clear cutoff", size=23, weight="bold")
    label(
        ax,
        43,
        80,
        "Historical SPX data · ordered partitions · actual calendar spacing",
        size=12.5,
        color=MUTED,
    )
    manifest = report["data"]["data_manifest"]
    first = date.fromisoformat(manifest["start_date"]).toordinal()
    last = date.fromisoformat(manifest["end_date"]).toordinal()
    start_x, end_x = 44, 1156

    def position(value):
        ordinal = date.fromisoformat(value).toordinal()
        return start_x + (ordinal - first) / (last - first) * (end_x - start_x)

    segments = report["data"]["segments"]
    colors = {
        "train": ("#bdece0", "#12594e"),
        "valid": ("#dcdaf8", "#49428a"),
        "test": ("#d8e5f3", "#305a7d"),
    }
    names = {"train": "TRAIN", "valid": "VALIDATION", "test": "TEST"}
    ax.add_patch(Rectangle((start_x, 160), end_x - start_x, 68, facecolor="#d9e2ec", linewidth=0))
    for name, (begin, end) in segments.items():
        left, right = position(begin), position(end)
        fill, text_color = colors[name]
        ax.add_patch(Rectangle((left, 160), right - left, 68, facecolor=fill, linewidth=0))
        midpoint = (left + right) / 2
        label(
            ax, midpoint, 182, names[name], size=11.5, color=text_color, weight="bold", ha="center"
        )
        label(ax, midpoint, 210, f"{begin} → {end}", size=10.2, color=text_color, ha="center")

    warmup = (start_x + position(segments["train"][0])) / 2
    label(ax, 44, 128, "History warmup", size=10.4, color=MUTED)
    arrow(ax, (109, 138), (warmup, 158), "#9aa8ba", width=1)
    for earlier, later in (("train", "valid"), ("valid", "test")):
        middle = (position(segments[earlier][1]) + position(segments[later][0])) / 2
        label(ax, middle, 127, "2 sessions purged", size=10.3, color=MUTED, ha="center")
        ax.plot((middle, middle), (141, 231), color="#71819a", linewidth=1, linestyle=(0, (2, 3)))

    for year in range(date.fromordinal(first).year, date.fromordinal(last).year + 1, 5):
        point = max(first, date(year, 1, 1).toordinal())
        x = start_x + (point - first) / (last - first) * (end_x - start_x)
        ax.plot((x, x), (231, 237), color="#94a3b8", linewidth=1)
        label(ax, x, 253, str(year), size=10, color=MUTED, ha="center" if x > 50 else "left")
    ax.plot((end_x, end_x), (231, 237), color="#94a3b8", linewidth=1)
    label(ax, end_x, 253, str(date.fromordinal(last).year), size=10, color=MUTED, ha="right")

    for begin, end, text, color in (
        (
            segments["train"][0],
            segments["train"][1],
            "Fit normalization on training data only",
            TEAL,
        ),
        (segments["valid"][0], segments["test"][1], "Reuse those fitted statistics", PURPLE),
    ):
        left, right = position(begin), position(end)
        ax.plot(
            (left + 6, left + 6, right - 6, right - 6),
            (278, 284, 284, 278),
            color=color,
            linewidth=1.35,
        )
        label(ax, (left + right) / 2, 306, text, size=11.3, color=color, ha="center")
    label(
        ax,
        43,
        350,
        "Inputs stop at close t. The label uses the return from close t+1 to close t+2.",
        size=11.4,
        color=INK,
    )
    label(
        ax,
        43,
        377,
        "Spacing is proportional to calendar time; purge markers identify the tiny gaps without widening them.",
        size=10.2,
        color=MUTED,
    )
    save(
        fig,
        "splits",
        width,
        height,
        "Chronological training, validation and test partitions",
        "; ".join(
            f"{names[name].title()}: {start} through {end}"
            for name, (start, end) in segments.items()
        )
        + ". Two sessions are purged at each partition boundary. "
        "Normalization is fitted on training data and reused unchanged. Dates are positioned on a proportional calendar scale.",
        preview_dir,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview-dir", type=Path, help="Optional PNG preview directory under /tmp"
    )
    args = parser.parse_args()
    preview_dir = args.preview_dir.resolve() if args.preview_dir else None
    if preview_dir is not None:
        if not preview_dir.is_relative_to(Path("/tmp").resolve()):
            parser.error("Keep preview PNGs in /tmp; only SVGs belong in the repository.")
        preview_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "path",
            "svg.hashsalt": "financial-ts-readme-assets",
            "axes.unicode_minus": True,
        }
    ):
        render_hero(preview_dir)
        render_benchmark(report, preview_dir)
        render_splits(report, preview_dir)


if __name__ == "__main__":
    main()
