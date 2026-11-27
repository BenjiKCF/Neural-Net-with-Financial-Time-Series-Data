"""Export Qlib's native diagnostic figures as README images and an offline report.

All plotted series are produced by ``qlib.contrib.report``.  This module only
validates the saved evidence, adds explanatory labels and exports the figures.
Qlib's default portfolio/risk plots accumulate daily returns by addition.  They
are deliberately not relabelled as compounded NAV, CAGR or compound drawdown.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

RETURN_CONVENTION = (
    "Native Qlib plots sum daily returns; the curves and drawdowns are additive, "
    "not compounded NAV. Native daily risk analysis annualizes with 238 sessions."
)
GROUP_CONVENTION = (
    "Five groups sorted by the model score. These are equal-weight label-return "
    "diagnostics before trading costs, not the executed portfolio. Group 1 has "
    "the highest scores. Long-short means Group 1 minus Group 5."
)
SOURCES = {
    "portfolio": "qlib.contrib.report.analysis_position.report_graph",
    "score-ic": "qlib.contrib.report.analysis_position.score_ic_graph",
    "risk-summary": "qlib.contrib.report.analysis_position.risk_analysis_graph",
    "monthly-annualized-return": "qlib.contrib.report.analysis_position.risk_analysis_graph",
    "monthly-max-drawdown": "qlib.contrib.report.analysis_position.risk_analysis_graph",
    "monthly-information-ratio": "qlib.contrib.report.analysis_position.risk_analysis_graph",
    "monthly-volatility": "qlib.contrib.report.analysis_position.risk_analysis_graph",
    "group-return": "qlib.contrib.report.analysis_model.model_performance_graph",
    "group-distribution": "qlib.contrib.report.analysis_model.model_performance_graph",
    "daily-ic": "qlib.contrib.report.analysis_model.model_performance_graph",
    "monthly-ic": "qlib.contrib.report.analysis_model.model_performance_graph",
    "ic-distribution": "qlib.contrib.report.analysis_model.model_performance_graph",
    "score-autocorrelation": "qlib.contrib.report.analysis_model.model_performance_graph",
}
TITLES = {
    "portfolio": "Portfolio report",
    "score-ic": "Daily IC and rank IC",
    "risk-summary": "Excess-return risk analysis",
    "monthly-annualized-return": "Monthly excess-return annualization",
    "monthly-max-drawdown": "Monthly excess-return drawdown",
    "monthly-information-ratio": "Monthly information ratio",
    "monthly-volatility": "Monthly daily excess-return volatility",
    "group-return": "Returns by score quintile",
    "group-distribution": "Quintile spread distributions",
    "daily-ic": "Daily information coefficient",
    "monthly-ic": "Monthly information coefficient",
    "ic-distribution": "IC distribution and normal Q-Q plot",
    "score-autocorrelation": "Score rank autocorrelation",
}


def _validate_frames(predictions: pd.DataFrame, portfolio: pd.DataFrame):
    if not isinstance(predictions.index, pd.MultiIndex) or set(predictions.index.names) != {
        "datetime",
        "instrument",
    }:
        raise ValueError("Predictions need a datetime/instrument MultiIndex.")
    if not predictions.index.is_unique or predictions.empty:
        raise ValueError("Predictions must be nonempty with unique date/instrument identities.")
    if not {"score", "label"}.issubset(predictions):
        raise ValueError("Predictions need score and label columns.")
    pred = predictions[["score", "label"]].copy().reorder_levels(["datetime", "instrument"])
    symbols = pred.index.get_level_values("instrument")
    if symbols.hasnans or any(not isinstance(value, str) or not value.strip() for value in symbols):
        raise ValueError("Prediction instruments must be nonempty strings.")
    dates = pd.DatetimeIndex(pred.index.get_level_values("datetime"))
    if dates.hasnans or dates.tz is not None or not (dates == dates.normalize()).all():
        raise ValueError("Prediction dates must be finite, timezone-naive daily dates.")
    if not np.isfinite(pred["score"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction scores must be finite.")
    if np.isinf(pred["label"].to_numpy(dtype=float)).any():
        raise ValueError("Prediction labels cannot contain infinite returns.")
    pred = pred.dropna(subset=["label"]).sort_index()
    if pred.empty:
        raise ValueError("There are no evaluable prediction labels.")
    required = ["return", "cost", "bench", "turnover"]
    if portfolio.empty or not set(required).issubset(portfolio):
        raise ValueError("Portfolio needs nonempty return, cost, bench and turnover columns.")
    report = portfolio[required].copy()
    if not isinstance(report.index, pd.DatetimeIndex):
        raise ValueError("Portfolio index must be a DatetimeIndex.")
    if (
        report.index.hasnans
        or report.index.tz is not None
        or not report.index.is_unique
        or not report.index.is_monotonic_increasing
        or not (report.index == report.index.normalize()).all()
    ):
        raise ValueError("Portfolio dates must be sorted, unique, timezone-naive daily dates.")
    if not np.isfinite(report.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio values must be finite.")
    if (report[["cost", "turnover"]] < 0).any().any():
        raise ValueError("Portfolio costs and turnover cannot be negative.")
    report.index.name = "date"
    return pred, report


def build_qlib_figures(
    predictions: pd.DataFrame,
    portfolio: pd.DataFrame,
    *,
    include_model_diagnostics: bool = True,
):
    """Return native figures, their input summary, and explicit diagnostic omissions.

    Qlib's distribution plots require nonconstant data. Degenerate statistical
    diagnostics are reported as unavailable rather than inventing a distribution.
    Neither input frame is modified, and no model is selected by this function.
    """
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.report import analysis_position

    pred, report = _validate_frames(predictions, portfolio)
    figures = {"portfolio": analysis_position.report_graph(report, show_notebook=False)[0]}
    omissions = []
    cross_sections = pred.groupby(level="datetime")
    informative_dates = cross_sections.apply(
        lambda frame: (
            len(frame) > 1 and frame["score"].nunique() > 1 and frame["label"].nunique() > 1
        )
    )
    if informative_dates.any():
        figures["score-ic"] = analysis_position.score_ic_graph(pred, show_notebook=False)[0]
    else:
        omissions.append({"chart": "score-ic", "reason": "No nonconstant cross-sectional pairs."})

    analysis = pd.concat(
        {
            "excess_return_without_cost": risk_analysis(report["return"] - report["bench"]),
            "excess_return_with_cost": risk_analysis(
                report["return"] - report["bench"] - report["cost"]
            ),
        }
    )
    # The native monthly renderer requires at least three observations in a month.
    month_sizes = report.groupby([report.index.year, report.index.month]).size()
    risk_figures = analysis_position.risk_analysis_graph(
        analysis_df=analysis,
        report_normal_df=report if (month_sizes >= 3).any() else None,
        show_notebook=False,
    )
    names = [
        "risk-summary",
        "monthly-annualized-return",
        "monthly-max-drawdown",
        "monthly-information-ratio",
        "monthly-volatility",
    ]
    figures.update(zip(names, risk_figures))
    if len(risk_figures) == 1:
        omissions.append({"chart": "monthly-risk", "reason": "No month has three observations."})

    if include_model_diagnostics:
        from qlib.contrib.report import analysis_model

        requests = [
            ("group_return", ["group-return", "group-distribution"]),
            ("pred_ic", ["daily-ic", "monthly-ic", "ic-distribution"]),
            ("pred_autocorr", ["score-autocorrelation"]),
        ]
        for graph, chart_names in requests:
            if graph == "group_return" and cross_sections.size().min() < 5:
                omissions.append(
                    {"chart": graph, "reason": "At least one date has fewer than five stocks."}
                )
                continue
            if graph == "pred_ic" and informative_dates.sum() < 3:
                omissions.append(
                    {"chart": graph, "reason": "Fewer than three informative cross-sections."}
                )
                continue
            try:
                native = analysis_model.model_performance_graph(
                    pred.copy(), graph_names=[graph], N=5, show_notebook=False
                )
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
                omissions.append({"chart": graph, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if len(native) != len(chart_names):
                raise RuntimeError(f"Unexpected native Qlib chart count for {graph}.")
            figures.update(zip(chart_names, native))

    summary = {
        "evaluated_prediction_frame_sha256": hashlib.sha256(
            pred.to_csv(float_format="%.17g", date_format="%Y-%m-%d").encode()
        ).hexdigest(),
        "portfolio_frame_sha256": hashlib.sha256(
            report.to_csv(float_format="%.17g", date_format="%Y-%m-%d").encode()
        ).hexdigest(),
        "prediction_rows": len(predictions),
        "evaluated_rows": len(pred),
        "unavailable_label_rows": len(predictions) - len(pred),
        "prediction_dates": pred.index.get_level_values("datetime").nunique(),
        "instruments": pred.index.get_level_values("instrument").nunique(),
        "informative_ic_dates": int(informative_dates.sum()),
        "signal_start": datestr(pred.index.get_level_values("datetime").min()),
        "signal_end": datestr(pred.index.get_level_values("datetime").max()),
        "portfolio_start": datestr(report.index.min()),
        "portfolio_end": datestr(report.index.max()),
        "portfolio_days": len(report),
        "return_convention": RETURN_CONVENTION,
        "group_return_convention": GROUP_CONVENTION,
        "native_risk_sessions_per_year": 238,
        "risk_metrics": json.loads(analysis.reset_index().to_json(orient="records")),
    }
    return figures, summary, omissions


def datestr(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _label_figure(figure, name, model_name, dataset_label):
    """Keep the native graph and series; add presentation labels only."""
    subtitle = dataset_label
    if name == "portfolio" or name.startswith("monthly-") and name != "monthly-ic":
        subtitle += " · Additive daily returns; native Qlib convention"
    elif name == "risk-summary":
        subtitle += " · Native Qlib: additive returns, 238 sessions/year"
    elif name.startswith("group-"):
        subtitle += " · Label-return diagnostic before trading costs"
    figure.update_layout(
        title={
            "text": f"Qlib {TITLES[name]} · {escape(model_name)}<br><sup>{escape(subtitle)}</sup>",
            "x": 0.04,
            "xanchor": "left",
            "font": {"size": 22},
        },
        width=1360,
        height=1420 if name == "portfolio" else 620,
        font={"family": "Arial, sans-serif", "size": 13},
        margin={"l": 85, "r": 280 if name == "portfolio" else 75, "t": 115, "b": 90},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if name == "portfolio":
        native_names = {
            "cum_bench": "Benchmark",
            "cum_return_wo_cost": "Portfolio before costs",
            "cum_return_w_cost": "Portfolio after costs",
            "return_wo_mdd": "Gross drawdown",
            "return_w_cost_mdd": "Net drawdown",
            "cum_ex_return_wo_cost": "Excess before costs",
            "cum_ex_return_w_cost": "Excess after costs",
            "turnover": "Turnover",
            "cum_ex_return_w_cost_mdd": "Net excess drawdown",
            "cum_ex_return_wo_cost_mdd": "Gross excess drawdown",
        }
        for trace in figure.data:
            trace.name = native_names.get(trace.name.replace(" ", "_"), trace.name)
        for number, label in enumerate(
            [
                "Return sum",
                "Gross DD",
                "Net DD",
                "Excess sum",
                "Turnover",
                "Net excess DD",
                "Gross excess DD",
            ],
            1,
        ):
            axis = "yaxis" if number == 1 else f"yaxis{number}"
            figure.layout[axis].update(title_text=label, tickformat=".0%")
        figure.update_layout(legend={"x": 1.015, "y": 1, "font": {"size": 12}})
        # Keep every observation; fewer labelled category ticks improve readability.
        figure.update_xaxes(nticks=10)
    elif name == "group-return":
        figure.update_yaxes(title_text="Sum of daily label returns", tickformat=".0%")
    elif name in {"score-ic", "daily-ic", "monthly-ic", "score-autocorrelation"}:
        figure.update_yaxes(title_text="Correlation" if name != "monthly-ic" else "Year")
    if name.startswith("monthly-") and name != "monthly-ic":
        figure.update_layout(legend={"orientation": "h", "y": -0.23, "x": 0})
    if name == "risk-summary":
        figure.update_layout(margin={"l": 85, "r": 45, "t": 130, "b": 100})
        figure.update_xaxes(
            tickangle=0,
            tickvals=["excess_return_without_cost", "excess_return_with_cost"],
            ticktext=["Before costs", "After costs"],
        )


def export_qlib_figures(
    predictions: pd.DataFrame,
    portfolio: pd.DataFrame,
    output_dir: str | Path,
    *,
    model_name: str,
    dataset_label: str = "",
    include_model_diagnostics: bool = True,
    export_png: bool = True,
) -> dict:
    """Write native PNGs, Plotly JSONs and one self-contained interactive HTML.

    The output directory must be new or empty. Plotly is embedded only once in
    the HTML, making the whole report usable offline without per-chart copies.
    Set ``export_png=False`` only when no static renderer is needed.
    """
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Chart output must be new or empty: {output}")
    figures, summary, omissions = build_qlib_figures(
        predictions, portfolio, include_model_diagnostics=include_model_diagnostics
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "model": model_name,
        "dataset": dataset_label,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("pyqlib", "plotly", *(["kaleido"] if export_png else []))
        },
        "inputs": summary,
        "omissions": omissions,
        "charts": [],
        "interactive_report": "report.html",
    }
    fragments = []
    for number, (name, figure) in enumerate(figures.items()):
        _label_figure(figure, name, model_name, dataset_label)
        json_path = output / f"{name}.json"
        figure.write_json(json_path, pretty=False)
        chart = {"name": name, "native_function": SOURCES[name], "json": json_path.name}
        if export_png:
            png_path = output / f"{name}.png"
            figure.write_image(png_path, format="png", scale=1.5)
            chart["png"] = png_path.name
        manifest["charts"].append(chart)
        fragments.append(f"<section id='{name}'><h2>{escape(TITLES[name])}</h2>")
        if name.startswith("group-"):
            fragments.append(f"<p>{escape(GROUP_CONVENTION)}</p>")
        fragments.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=number == 0,
                div_id=f"qlib-{name}",
                config={"responsive": True, "displaylogo": False},
            )
        )
        fragments.append("</section>")
    nav = " · ".join(f"<a href='#{name}'>{escape(TITLES[name])}</a>" for name in figures)
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Qlib diagnostics · {escape(model_name)}</title>"
        "<style>body{font:16px Arial,sans-serif;color:#17233a;margin:32px auto;"
        "padding:0 24px;max-width:1440px}p{max-width:1000px;line-height:1.6}"
        "section{margin:42px 0;overflow:auto}nav{line-height:1.8}a{color:#275d9c}"
        ".plotly-graph-div{max-width:100%}</style></head><body>"
        f"<h1>Qlib diagnostics · {escape(model_name)}</h1><p>{escape(dataset_label)}</p>"
        f"<p>{escape(RETURN_CONVENTION)}</p>"
        f"<p>{summary['evaluated_rows']:,} evaluated predictions across "
        f"{summary['instruments']:,} instruments; signals {summary['signal_start']} "
        f"to {summary['signal_end']}.</p><nav>{nav}</nav>" + "".join(fragments) + "</body></html>"
    )
    (output / "report.html").write_text(html, encoding="utf-8")
    manifest["status"] = "completed"
    manifest["files"] = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def export_qlib_run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    dataset_label: str = "",
    include_model_diagnostics: bool = True,
    export_png: bool = True,
) -> dict:
    """Load CSV or gzip CSV evidence and export its native diagnostics.

    Plain CSVs take precedence when present. Gzip support permits an offline
    chart replay directly from the compact evidence committed to the repository.
    """
    run = Path(run_dir).expanduser().resolve()
    prediction_path = run / "predictions.csv"
    if not prediction_path.is_file():
        prediction_path = run / "predictions.csv.gz"
    portfolio_path = run / "portfolio.csv"
    if not portfolio_path.is_file():
        portfolio_path = run / "portfolio.csv.gz"
    pred = pd.read_csv(
        prediction_path,
        dtype={"instrument": str},
        keep_default_na=False,
        na_values={"score": ["", "NaN", "nan"], "label": ["", "NaN", "nan"]},
        float_precision="round_trip",
    )
    if not {"datetime", "instrument", "score", "label"}.issubset(pred):
        raise ValueError("predictions.csv needs datetime, instrument, score and label columns.")
    pred["datetime"] = pd.to_datetime(pred["datetime"], errors="raise")
    pred = pred.set_index(["datetime", "instrument"])
    report = pd.read_csv(
        portfolio_path, index_col=0, parse_dates=True, float_precision="round_trip"
    )
    return export_qlib_figures(
        pred,
        report,
        output_dir,
        model_name=model_name or run.name,
        dataset_label=dataset_label,
        include_model_diagnostics=include_model_diagnostics,
        export_png=export_png,
    )
