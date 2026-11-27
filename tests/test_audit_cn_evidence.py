"""Independent evidence checks reject invalid arithmetic and incomplete horizons."""

import copy
import gzip
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def audit():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        yield importlib.import_module("audit_cn_evidence")


def _write_positions(path, rows):
    with gzip.open(path, "wt") as stream:
        # Malformed NaN/Infinity inputs are deliberate in rejection tests.
        json.dump({"positions": rows}, stream)


@pytest.fixture
def positions(tmp_path):
    row = {
        "datetime": "2025-01-03",
        "account": 1000.0,
        "cash": 600.0,
        "holdings": [{"instrument": "AAA", "amount": 20.0, "price": 20.0, "weight": 0.4}],
    }
    report = pd.DataFrame({"account": [1000.0]}, index=pd.to_datetime([row["datetime"]]))
    source = SimpleNamespace(close=lambda symbol: None, last_dates={"AAA": None})
    return tmp_path / "positions.json.gz", row, report, source


def test_valid_numeric_position_reconciles_account_value_and_weight(audit, positions):
    path, row, report, source = positions
    _write_positions(path, [row])
    result = audit.check_numeric_positions(path, report, source)
    assert result["status"] == "checked"
    assert result["days"] == 1
    assert result["maximum_account_relative_error"] == 0
    assert result["maximum_valuation_relative_error"] == 0
    assert result["maximum_weight_absolute_error"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("account", np.nan),
        ("account", np.inf),
        ("account", -np.inf),
        ("account", 0),
        ("account", -1),
        ("cash", np.nan),
        ("cash", np.inf),
        ("cash", -np.inf),
        ("cash", 601),
    ],
)
def test_invalid_position_account_or_cash_is_rejected(audit, positions, field, value):
    path, row, report, source = positions
    row[field] = value
    _write_positions(path, [row])
    with pytest.raises(AssertionError):
        audit.check_numeric_positions(path, report, source)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0, -1])
def test_invalid_report_account_cannot_pass_position_comparison(audit, positions, value):
    path, row, report, source = positions
    report.iloc[0, 0] = value
    _write_positions(path, [row])
    with pytest.raises(AssertionError, match="Portfolio report accounts"):
        audit.check_numeric_positions(path, report, source)


@pytest.mark.parametrize(
    "field,value",
    [("amount", 21), ("amount", -1), ("amount", np.nan), ("price", 0), ("weight", 0.41)],
)
def test_corrupt_holding_quantity_price_or_weight_is_rejected(audit, positions, field, value):
    path, row, report, source = positions
    row["holdings"][0][field] = value
    _write_positions(path, [row])
    with pytest.raises(AssertionError):
        audit.check_numeric_positions(path, report, source)


@pytest.mark.parametrize("date", ["not-a-date", None, "2025-01-03T12:00:00"])
def test_malformed_position_date_is_rejected(audit, positions, date):
    path, row, report, source = positions
    row["datetime"] = date
    _write_positions(path, [row])
    with pytest.raises(AssertionError, match="dates"):
        audit.check_numeric_positions(path, report, source)


def test_duplicate_position_dates_are_rejected_even_if_report_matches(audit, positions):
    path, row, report, source = positions
    _write_positions(path, [row, copy.deepcopy(row)])
    with pytest.raises(AssertionError, match="unique"):
        audit.check_numeric_positions(path, pd.concat([report, report]), source)


@pytest.fixture
def evidence(tmp_path, audit):
    provider, root = tmp_path / "provider", tmp_path / "evidence"
    dates = pd.bdate_range("2025-01-02", periods=13)
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars/day.txt").write_text(
        "\n".join(date.strftime("%Y-%m-%d") for date in dates) + "\n"
    )
    (provider / "instruments/csi300.txt").write_text(
        "".join(f"{symbol}\t{dates[0].date()}\t{dates[-1].date()}\n" for symbol in ["AAA", "BBB"])
    )
    closes = {"AAA": np.full(13, 20, dtype="<f4"), "BBB": np.arange(30, 43, dtype="<f4")}
    for symbol, values in closes.items():
        directory = provider / "features" / symbol.lower()
        directory.mkdir(parents=True)
        np.concatenate([np.array([0], dtype="<f4"), values]).tofile(directory / "close.day.bin")
    model = root / "example"
    model.mkdir(parents=True)
    segments = {
        name: [str(dates[first].date()), str(dates[last].date())]
        for name, first, last in [("train", 0, 1), ("valid", 4, 5), ("test", 8, 9)]
    }
    fingerprint = audit._provider_fingerprint(provider)
    protocol = {
        "provider_fingerprint": fingerprint,
        "segments": segments,
        "models": ["example"],
        "configs": {"example": {"backtest": {"account": 1000.0}}},
    }
    (root / "protocol.json").write_text(json.dumps(protocol))
    (model / "run.json").write_text(json.dumps({"status": "completed", "provider": fingerprint}))
    metrics = {}
    for period, rows, filename in [
        ("valid", [4, 5], "validation_predictions.csv.gz"),
        ("test", [8, 9], "predictions.csv.gz"),
    ]:
        frame = pd.DataFrame(
            [
                {
                    "datetime": dates[row],
                    "instrument": symbol,
                    "label": float(np.float32(values[row + 2] / values[row + 1] - 1)),
                }
                for row in rows
                for symbol, values in closes.items()
            ]
        )
        frame["score"] = frame["label"] + 0.001
        frame.to_csv(model / filename, index=False)
        error = frame["score"] - frame["label"]
        calculated = {
            "prediction_count": 4,
            "evaluated_count": 4,
            "unavailable_label_count": 0,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.abs(error).mean()),
            "zero_return_rmse": float(np.sqrt(np.mean(frame["label"] ** 2))),
            "directional_accuracy": 0.5,
            "ic": 1.0,
            "rank_ic": 1.0,
            "ic_days": 2,
            "rank_ic_days": 2,
        }
        if period == "valid":
            metrics["validation"] = calculated
        else:
            metrics.update(calculated)
    metrics["portfolio"] = {
        "net_cumulative_return": 0.0,
        "gross_cumulative_return": 0.0,
        "benchmark_cumulative_return": 0.0,
        "sum_daily_cost_ratios": 0.0,
        "trade_signal": {"order": ["raw score descending", "instrument ascending"]},
    }
    (model / "metrics.json").write_text(json.dumps(metrics))
    portfolio_dates = dates[9:12]
    report = pd.DataFrame(
        {"account": 1000.0, "return": 0.0, "cost": 0.0, "bench": 0.0}, index=portfolio_dates
    )
    report.to_csv(model / "portfolio.csv")
    _write_positions(
        model / "positions.json.gz",
        [
            {
                "datetime": str(date.date()),
                "account": 1000.0,
                "cash": 600.0,
                "holdings": [{"instrument": "AAA", "amount": 20.0, "price": 20.0, "weight": 0.4}],
            }
            for date in portfolio_dates
        ],
    )
    return root, provider, model


def test_complete_finite_evidence_passes_independent_binary_and_account_checks(audit, evidence):
    root, provider, _ = evidence
    result = audit.audit_evidence(root, provider)
    assert result["status"] == "passed"
    assert result["audited_models"] == ["example"]
    assert result["failures"] == []
    assert result["models"]["example"]["portfolio"]["positions"]["days"] == 3


def test_canonical_labels_preserve_key_alignment_and_missingness(audit):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2025-01-02", periods=2), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    labels = np.array([0.1, np.nan, 0.3, 0.4], dtype=np.float32)
    order = np.array([0, 2, 1, 3])
    canonical = audit._canonical_panel_labels(index, labels)
    assert canonical.equals(audit._canonical_panel_labels(index[order], labels[order]))
    assert canonical.isna().sum() == 1
    assert not canonical.equals(audit._canonical_panel_labels(index[order], labels))
    changed = labels.copy()
    changed[1] = 0
    assert not canonical.equals(audit._canonical_panel_labels(index, changed))
    different_index = pd.MultiIndex.from_tuples(
        [(date, "CCC" if symbol == "BBB" else symbol) for date, symbol in index],
        names=index.names,
    )
    assert not canonical.equals(audit._canonical_panel_labels(different_index, labels))


@pytest.mark.parametrize("audit_kind", ["published", "local"])
@pytest.mark.parametrize("corruption", [None, "label", "index"])
def test_both_auditors_compare_complete_panels_by_keys(audit, evidence, audit_kind, corruption):
    root, provider, original = evidence
    reordered = root / "reordered"
    shutil.copytree(original, reordered)
    protocol = json.loads((root / "protocol.json").read_text())
    protocol["models"].append("reordered")
    protocol["configs"]["reordered"] = copy.deepcopy(protocol["configs"]["example"])
    (root / "protocol.json").write_text(json.dumps(protocol))
    filenames = ["validation_predictions.csv", "predictions.csv"]
    for model in [original, reordered]:
        metadata = json.loads((model / "run.json").read_text())
        metadata.update(experiment_id="test", recorder_id="test")
        (model / "run.json").write_text(json.dumps(metadata))
        if audit_kind == "local":
            for filename in filenames:
                with gzip.open(model / (filename + ".gz"), "rb") as stream:
                    (model / filename).write_bytes(stream.read())
    suffix = ".gz" if audit_kind == "published" else ""
    for filename in filenames:
        path = reordered / (filename + suffix)
        frame = pd.read_csv(path, float_precision="round_trip")
        instrument_major = frame.sort_values(["instrument", "datetime"])
        assert not instrument_major.reset_index(drop=True).equals(frame)
        instrument_major.to_csv(path, index=False)
    if corruption is not None:
        path = reordered / ("predictions.csv" + suffix)
        frame = pd.read_csv(path, float_precision="round_trip")
        if corruption == "label":
            frame.loc[0, "label"] += 0.5
        else:
            frame.loc[0, "instrument"] = "OUTSIDE"
        frame.to_csv(path, index=False)
    # Comparisons may sort an in-memory series, but exports must retain their
    # original bytes and ordering on both successful and rejected audits.
    exported = {
        path: path.read_bytes()
        for model in [original, reordered]
        for filename in filenames
        for path in [model / (filename + suffix)]
    }
    if audit_kind == "published":
        result = audit.audit_evidence(root, provider)
    else:
        result = importlib.import_module("audit_cn_benchmark").audit(root, provider)
    assert {path: path.read_bytes() for path in exported} == exported
    if corruption is None:
        assert result["status"] == "passed", result["failures"]
        assert result["audited_models"] == ["example", "reordered"]
    else:
        assert result["status"] == "failed"
        assert result["audited_models"] == ["example"]
        assert result["failures"][0]["model"] == "reordered"


@pytest.mark.parametrize("column", ["account", "return", "cost", "bench"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_portfolio_report_values_fail_audit(audit, evidence, column, value):
    root, provider, model = evidence
    report = pd.read_csv(model / "portfolio.csv", index_col=0)
    report.loc[report.index[0], column] = value
    report.to_csv(model / "portfolio.csv")
    result = audit.audit_evidence(root, provider)
    assert result["status"] == "failed"
    assert "Portfolio report values must be finite" in result["failures"][0]["error"]


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_portfolio_accounts_fail_audit(audit, evidence, value):
    root, provider, model = evidence
    report = pd.read_csv(model / "portfolio.csv", index_col=0)
    report.iloc[0, report.columns.get_loc("account")] = value
    report.to_csv(model / "portfolio.csv")
    result = audit.audit_evidence(root, provider)
    assert result["status"] == "failed"
    assert "strictly positive" in result["failures"][0]["error"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.01])
def test_invalid_or_inconsistent_published_portfolio_metric_fails_audit(audit, evidence, value):
    root, provider, model = evidence
    metrics = json.loads((model / "metrics.json").read_text())
    metrics["portfolio"]["net_cumulative_return"] = value
    (model / "metrics.json").write_text(json.dumps(metrics))
    result = audit.audit_evidence(root, provider)
    assert result["status"] == "failed"
    assert "Published portfolio" in result["failures"][0]["error"]


@pytest.mark.parametrize("kind", ["duplicate", "malformed", "intraday"])
def test_duplicate_or_malformed_portfolio_dates_fail_audit(audit, evidence, kind):
    root, provider, model = evidence
    report = pd.read_csv(model / "portfolio.csv", index_col=0)
    dates = list(report.index)
    dates[1] = {
        "duplicate": dates[0],
        "malformed": "not-a-date",
        "intraday": dates[1] + " 12:00:00",
    }[kind]
    report.index = dates
    report.to_csv(model / "portfolio.csv")
    result = audit.audit_evidence(root, provider)
    assert result["status"] == "failed"
    assert "dates" in result["failures"][0]["error"]


@pytest.mark.parametrize("remaining_sessions", [0, 1, 2])
def test_insufficient_final_observed_source_reserve_is_rejected(
    audit, evidence, remaining_sessions
):
    root, provider, _ = evidence
    calendar = provider / "calendars/day.txt"
    dates = calendar.read_text().splitlines()
    calendar.write_text("\n".join(dates[: 10 + remaining_sessions]) + "\n")
    protocol = json.loads((root / "protocol.json").read_text())
    protocol["provider_fingerprint"] = audit._provider_fingerprint(provider)
    (root / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(AssertionError, match="three observed sessions"):
        audit.audit_evidence(root, provider)
