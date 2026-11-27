"""Exercise actual-time orchestration and independently calculable label custody."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_prospective import forecasts, setup

from financial_ts import prospective, prospective_service


@pytest.fixture
def clock(monkeypatch):
    value = [pd.Timestamp("2026-11-27T09:00:00Z")]
    monkeypatch.setattr(prospective, "_now", lambda: value[0])
    monkeypatch.setattr(prospective_service, "_now", lambda: value[0])
    return value


def test_before_first_session_tick_preserves_zero_observations(tmp_path, clock):
    store, protocol = setup(tmp_path, clock)
    result = prospective_service.service_tick(store, tmp_path / "feed")
    assert result["status"] == "idle"
    assert result["phase"] == "outside_registered_calendar"
    assert result["protocol_id"] == protocol["protocol_id"]
    assert result["assessment"]["operational_acceptance"]["captured_sessions"] == 0
    assert len(list((store / "service_ticks").glob("*.json"))) == 1


def test_pre_cutoff_tick_only_collects(tmp_path, clock, monkeypatch):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T09:15:00Z")
    calls = []

    def collect(feed, **arguments):
        calls.append(arguments)
        return {"arrival_receipt": "fixture", "receipt_id": "fixture", "observed_at": clock[0].isoformat()}

    monkeypatch.setattr(prospective_service, "collect_arrival", collect)
    result = prospective_service.service_tick(store, tmp_path / "feed")
    assert result["phase"] == "arrival_collected"
    assert calls[0]["expected_session"] == "2026-11-30"
    assert pd.Timestamp(calls[0]["decision_cutoff"]) == pd.Timestamp("2026-11-30T10:00:00Z")
    assert not list((store / "observations").iterdir())


def test_missing_pre_cutoff_arrival_cannot_be_backfilled(tmp_path, clock, monkeypatch):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    monkeypatch.setattr(prospective_service, "collect_arrival", lambda *args, **kwargs: pytest.fail("Late collection"))
    result = prospective_service.service_tick(store, tmp_path / "feed")
    assert result["status"] == "failed"
    assert "cannot be backfilled" in result["error"]
    assert not list((store / "observations").iterdir())
    assert len(list((store / "incidents").glob("*.json"))) == 1


def test_missed_deadline_is_retained_failure(tmp_path, clock):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:31:00Z")
    result = prospective_service.service_tick(store, tmp_path / "feed")
    assert result["status"] == "failed" and "deadline missed" in result["error"]
    assert json.loads(next((store / "service_ticks").glob("*.json")).read_text()) == result


def _provider(path, closes):
    (path / "calendars").mkdir(parents=True)
    (path / "calendars/day.txt").write_text("2026-11-30\n2026-12-01\n2026-12-02\n2026-12-03\n")
    for instrument, values in closes.items():
        destination = path / "features" / instrument.lower()
        destination.mkdir(parents=True)
        np.asarray([0, *values], dtype="<f4").tofile(destination / "close.day.bin")


def test_close_label_reading_retains_missing_and_uses_correct_endpoints(tmp_path):
    _provider(tmp_path, {"AAA": [80, 100, 110, 120], "BBB": [1, 0, np.nan, 1]})
    result = prospective_service._read_closes(tmp_path, ["AAA", "BBB", "CCC"], ["2026-12-01", "2026-12-02"])
    assert result == {"AAA": [100.0, 110.0], "BBB": [None, None], "CCC": [None, None]}
    with pytest.raises(ValueError, match="not observed"):
        prospective_service._read_closes(tmp_path, ["AAA"], ["2026-12-04"])


def test_labels_use_first_admitted_vintage_and_retain_every_instrument(tmp_path, clock, monkeypatch):
    store, _ = setup(tmp_path, clock)
    clock[0] = pd.Timestamp("2026-11-30T10:05:00Z")
    prospective.record_observation(store, **forecasts(store, "2026-11-30"))
    feed = tmp_path / "feed"
    arrivals = {}
    for date, close, stamp in (
        ("2026-12-02", 110, "2026-12-02T10:10:00Z"),
        ("2026-12-03", 150, "2026-12-03T10:10:00Z"),
    ):
        receipt_path = feed / "sessions" / date / "arrival.json"
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text("{}")
        provider = feed / "providers" / date
        _provider(provider, {"AAA": [80, 100, close, 120], "BBB": [1, 2, 3, 4]})
        arrivals[str(receipt_path)] = {
            "expected_session": date, "observed_at": stamp, "receipt_id": date,
            "provider_dir": str(provider), "arrival_receipt": str(receipt_path),
            "provider_fingerprint": {"sha256": date},
        }
    def verify(path):
        if not Path(path).is_file():
            raise ValueError("Referenced arrival disappeared")
        return arrivals[str(path)]

    monkeypatch.setattr(prospective_service, "verify_arrival", verify)
    monkeypatch.setattr(prospective_service, "verify_arrival_metadata", verify)
    clock[0] = pd.Timestamp("2026-12-03T10:15:00Z")
    result = prospective_service.capture_available_labels(store, feed)
    assert [row["session"] for row in result["captured"]] == ["2026-11-30"]
    labels = pd.read_csv(store / "labels/2026-11-30/labels.csv").set_index("instrument")
    assert set(labels.index) == {"AAA", "BBB", "CCC"}
    assert labels.loc["AAA", "label"] == pytest.approx(0.1)
    assert labels.loc["BBB", "label"] == pytest.approx(0.5)
    assert pd.isna(labels.loc["CCC", "label"])
    assert set(labels.available_at) == {"2026-12-02T10:10:00Z"}
    evidence = store / "label_sources/2026-11-30/source.json"
    assert json.loads(evidence.read_text())["endpoint_sessions"] == ["2026-12-01", "2026-12-02"]
    assert prospective_service.capture_available_labels(store, feed)["captured"] == []
    original = evidence.read_bytes()
    evidence.unlink()
    with pytest.raises(ValueError, match="source evidence is missing"):
        prospective_service.capture_available_labels(store, feed)
    evidence.write_bytes(original)
    evidence.write_text("{}")
    with pytest.raises(ValueError, match="source evidence changed"):
        prospective_service.capture_available_labels(store, feed)
    evidence.write_bytes(original)
    (feed / "sessions/2026-12-02/arrival.json").unlink()
    with pytest.raises(ValueError, match="arrival disappeared"):
        prospective_service.capture_available_labels(store, feed)
    clock[0] = pd.Timestamp("2026-12-04T10:15:00Z")
    post_window = prospective_service.service_tick(store, feed)
    assert post_window["phase"] == "outside_registered_calendar"
    assert post_window["status"] == "failed"
    assert "arrival disappeared" in post_window["error"]
