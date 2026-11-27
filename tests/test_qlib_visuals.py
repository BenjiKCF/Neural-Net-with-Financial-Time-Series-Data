"""Check native chart arithmetic and inspectable exports from saved evidence."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from financial_ts.qlib_visuals import build_qlib_figures, export_qlib_figures, export_qlib_run


def frames():
    dates = pd.bdate_range("2025-01-02", periods=24)
    index = pd.MultiIndex.from_product(
        [dates, ["NA", *[f"SH{i:06d}" for i in range(9)]]], names=["datetime", "instrument"]
    )
    rng = np.random.default_rng(123)
    score = rng.normal(0, 0.01, len(index))
    pred = pd.DataFrame(
        {"score": score, "label": score * 0.3 + rng.normal(0, 0.01, len(index))}, index=index
    )
    report = pd.DataFrame(
        {
            "return": rng.normal(0.001, 0.01, len(dates)),
            "cost": 0.0003,
            "bench": rng.normal(0, 0.01, len(dates)),
            "turnover": 0.1,
        },
        index=dates,
    )
    return pred, report


def test_native_arithmetic_and_ic_do_not_change_inputs():
    pred, report = frames()
    original_pred, original_report = pred.copy(deep=True), report.copy(deep=True)
    figures, summary, omissions = build_qlib_figures(pred, report, include_model_diagnostics=False)
    native = {
        trace.name.replace(" ", "_"): np.asarray(trace.y, dtype=float)
        for trace in figures["portfolio"].data
    }
    expected_net = (report["return"] - report["cost"]).cumsum()
    np.testing.assert_allclose(native["cum_return_w_cost"], np.r_[0, expected_net])
    # A multiplicative equity curve is different and must not be substituted.
    assert not np.allclose(expected_net, (1 + report["return"] - report["cost"]).cumprod() - 1)
    expected_ic = pred.groupby(level="datetime").apply(lambda f: f.score.corr(f.label)).to_numpy()
    np.testing.assert_allclose(figures["score-ic"].data[0].y, expected_ic)
    pd.testing.assert_frame_equal(pred, original_pred)
    pd.testing.assert_frame_equal(report, original_report)
    assert summary["native_risk_sessions_per_year"] == 238
    assert not omissions


@pytest.mark.parametrize("compressed", [False, True])
def test_roundtrip_csv_export_retains_identity_and_native_sources(tmp_path, compressed):
    pred, report = frames()
    run = tmp_path / "run"
    run.mkdir()
    suffix = ".csv.gz" if compressed else ".csv"
    pred.to_csv(run / f"predictions{suffix}")
    report.to_csv(run / f"portfolio{suffix}")
    output = tmp_path / "charts"
    result = export_qlib_run(
        run,
        output,
        model_name="fixture",
        dataset_label="synthetic fixture",
        include_model_diagnostics=False,
        export_png=False,
    )
    saved = json.loads((output / "manifest.json").read_text())
    assert saved == result
    assert saved["inputs"]["instruments"] == 10
    assert len(saved["charts"]) == 7
    assert all(
        chart["native_function"].startswith("qlib.contrib.report.") for chart in saved["charts"]
    )
    html = (output / "report.html").read_text()
    assert "plotly.js v" in html and "not compounded NAV" in html
    assert "https://cdn.plot.ly/plotly-" not in html
    assert json.loads((output / "portfolio.json").read_text())["data"][0]["name"] == "Benchmark"


def test_invalid_or_duplicate_evidence_is_rejected_before_export(tmp_path):
    pred, report = frames()
    pred = pd.concat([pred, pred.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        export_qlib_figures(pred, report, tmp_path / "bad", model_name="fixture", export_png=False)
    assert not (tmp_path / "bad").exists()


def test_constant_scores_explicitly_omit_information_coefficient():
    pred, report = frames()
    pred["score"] = 0.0
    figures, summary, omissions = build_qlib_figures(pred, report, include_model_diagnostics=False)
    assert "score-ic" not in figures
    assert summary["informative_ic_dates"] == 0
    assert omissions[0]["chart"] == "score-ic"
