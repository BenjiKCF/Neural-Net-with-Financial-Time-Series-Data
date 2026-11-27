"""One clock-bound collection/publication tick for the registered shadow pilot.

This adapter does not change the registered inference or evaluation source. It
connects genuine arriving snapshots to that frozen runner and retains the raw
close observations used for each first-vintage mature label.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from financial_ts.arrival_feed import collect_arrival, verify_arrival, verify_arrival_metadata
from financial_ts.bundles import atomic_json, canonical_json, file_hash
from financial_ts.operations import read_prediction_artifact
from financial_ts.prospective import (
    _incident,
    _load,
    _verify_receipt,
    evaluate_prospective,
    record_mature_labels,
    run_prospective_day,
)


def _now():
    return pd.Timestamp.now(tz="UTC")


def _read_closes(provider, instruments, dates):
    """Read only observed calendar positions; absent outcomes remain null."""
    provider = Path(provider)
    days = (provider / "calendars/day.txt").read_text().splitlines()
    if len(days) != len(set(days)) or days != sorted(days):
        raise ValueError("Label provider calendar is not sorted and unique.")
    if any(date not in days for date in dates):
        raise ValueError("Label endpoints are not observed in this snapshot.")
    indices = [days.index(date) for date in dates]
    result = {}
    for instrument in instruments:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", instrument):
            raise ValueError("Unsafe label instrument identifier.")
        path = provider / "features" / instrument.lower() / "close.day.bin"
        if path.is_symlink():
            raise ValueError("Label source cannot be a symlink.")
        values = []
        if path.is_file():
            if path.stat().st_size < 4 or path.stat().st_size % 4:
                raise ValueError("Malformed label close array.")
            values = np.fromfile(path, dtype="<f4")
            if not np.isfinite(values[0]) or values[0] < 0 or values[0] != int(values[0]):
                raise ValueError("Malformed label close offset.")
            offset = int(values[0])
            if offset + len(values) - 1 > len(days):
                raise ValueError("Label close array extends beyond observed sessions.")
        result[instrument] = [
            float(values[index - offset + 1])
            if len(values) and 0 <= index - offset < len(values) - 1
            and np.isfinite(values[index - offset + 1])
            and values[index - offset + 1] > 0 else None
            for index in indices
        ]
    return result


def capture_available_labels(store, feed_root):
    """Freeze each due label from its earliest retained, admitted source vintage."""
    store, feed = Path(store), Path(feed_root)
    protocol = _load(store)
    current = _now()
    arrivals = []
    for path in sorted((feed / "sessions").glob("*/arrival.json")):
        # Check all metadata identities without materializing every archived
        # provider. The chosen source is verified in full before reading labels.
        arrivals.append(verify_arrival_metadata(path))
    arrivals.sort(key=lambda item: (pd.Timestamp(item["observed_at"]), item["receipt_id"]))
    calendar = protocol["specification"]["calendar"]["sessions"]
    captured, pending = [], []
    for session in protocol["sessions"]:
        date = session["date"]
        if not (store / "observations" / date).exists():
            continue
        label_dir = store / "labels" / date
        if label_dir.exists():
            receipt = _verify_receipt(label_dir, protocol)
            evidence = store / "label_sources" / date / "source.json"
            if evidence.is_symlink() or not evidence.is_file():
                raise ValueError("Retained mature-label source evidence is missing.")
            expected = str(evidence.absolute()) + "#sha256=" + file_hash(evidence)
            if receipt["source"] != expected:
                raise ValueError("Retained mature-label source evidence changed.")
            retained = json.loads(evidence.read_text())
            if retained["labels_sha256"] != file_hash(label_dir / "labels.csv"):
                raise ValueError("Retained label bytes differ from the source evidence.")
            arrival = verify_arrival_metadata(retained["arrival_receipt"])
            if (
                arrival["receipt_id"] != retained["arrival_receipt_id"]
                or arrival["provider_fingerprint"] != retained["provider_fingerprint"]
            ):
                raise ValueError("Retained label arrival identity changed.")
            continue
        maturity = pd.Timestamp(session["label_mature_at"])
        if current < maturity:
            continue
        index = calendar.index(date)
        endpoints = calendar[index + 1:index + 3]
        source = next((item for item in arrivals if (
            item["expected_session"] >= endpoints[-1]
            and maturity <= pd.Timestamp(item["observed_at"]) <= current
        )), None)
        if source is None:
            pending.append(date)
            continue
        source = verify_arrival(source["arrival_receipt"])
        predictions, _ = read_prediction_artifact(store / "observations" / date / "candidate")
        instruments = sorted(predictions["instrument"].astype(str))
        closes = _read_closes(source["provider_dir"], instruments, endpoints)
        # Recheck the underlying snapshot after reading the target values.
        verify_arrival(source["arrival_receipt"])
        rows = []
        for instrument in instruments:
            left, right = closes[instrument]
            rows.append({
                "datetime": date, "instrument": instrument,
                "label": right / left - 1 if left is not None and right is not None else np.nan,
                "available_at": source["observed_at"],
            })
        frame = pd.DataFrame(rows, columns=["datetime", "instrument", "label", "available_at"])
        evidence = {
            "schema_version": 1, "kind": "first_admitted_mature_label_source",
            "protocol_id": protocol["protocol_id"], "session": date,
            "target": "close[t+2]/close[t+1]-1", "endpoint_sessions": endpoints,
            "arrival_receipt_id": source["receipt_id"],
            "arrival_receipt": source["arrival_receipt"],
            "provider_fingerprint": source["provider_fingerprint"],
            "available_at": source["observed_at"], "observed_closes": closes,
            "labels_sha256": hashlib.sha256(
                frame.to_csv(index=False, float_format="%.17g").encode()
            ).hexdigest(),
            "missing_outcomes": "null, retained in the predicted cross-section",
            "production_ready": False, "orders_submitted": False,
        }
        destination = store / "label_sources" / date / "source.json"
        if destination.exists():
            if destination.read_bytes() != canonical_json(evidence):
                raise ValueError("An interrupted label capture has different source evidence.")
        else:
            atomic_json(destination, evidence)
        receipt = record_mature_labels(
            store, session=date, labels=frame,
            source=str(destination.absolute()) + "#sha256=" + file_hash(destination),
        )
        captured.append({"session": date, "receipt_id": receipt["receipt_id"]})
    return {"captured": captured, "pending_source_sessions": pending}


def service_tick(store, feed_root, *, cache_dir=None):
    """Perform only work permitted at the actual current exchange-local time.

Call before the cutoff to collect, and after it to publish. No public clock
override, retrospective arrival receipt, scheduler, or trading permission is
introduced by this function.
"""
    store, feed = Path(store).absolute(), Path(feed_root).absolute()
    protocol = _load(store)
    now = _now()
    specification = protocol["specification"]
    local = now.tz_convert(specification["timezone"])
    date = local.strftime("%Y-%m-%d")
    result = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "started_at": now.isoformat(), "local_session": date,
        "service_source_sha256": file_hash(__file__),
        "orders_submitted": False, "production_ready": False,
    }
    with FileLock(str(store / ".service.lock"), timeout=0):
        publication_attempt = runner_handles_incident = False
        try:
            if date not in specification["calendar"]["sessions"]:
                result.update(status="idle", phase="outside_registered_calendar")
            else:
                cutoff = pd.Timestamp(f"{date}T{specification['decision_time']}").tz_localize(
                    specification["timezone"]
                )
                session = next((item for item in protocol["sessions"] if item["date"] == date), None)
                path = feed / "sessions" / date / "arrival.json"
                if now < cutoff:
                    arrival = collect_arrival(
                        feed, expected_session=date, decision_cutoff=cutoff.isoformat(),
                        cache_dir=cache_dir,
                    )
                    result.update(status="ready", phase="arrival_collected", arrival={
                        key: arrival[key] for key in ("arrival_receipt", "receipt_id", "observed_at")
                    })
                elif session and (store / "observations" / date).exists():
                    receipt = _verify_receipt(store / "observations" / date, protocol)
                    result.update(status="ready", phase="forecast_already_captured",
                                  observation_receipt_id=receipt["receipt_id"])
                elif session and now <= pd.Timestamp(session["deadline"]):
                    publication_attempt = True
                    if not path.exists():
                        raise ValueError("No genuine pre-cutoff arrival; forecast cannot be backfilled.")
                    arrival = verify_arrival(path)
                    runner_handles_incident = True
                    receipt = run_prospective_day(
                        store, provider=arrival["provider_dir"], arrival_receipt=path,
                    )
                    result.update(status="ready", phase="forecast_captured",
                                  observation_receipt_id=receipt["receipt_id"])
                elif session:
                    publication_attempt = True
                    raise ValueError("Prospective publication deadline missed; no retrospective forecast admitted.")
                else:
                    result.update(status="ready", phase="label_only_session")
            result["labels"] = capture_available_labels(store, feed)
            result["assessment"] = evaluate_prospective(store)
        except Exception as error:
            if publication_attempt and not runner_handles_incident:
                _incident(store, protocol, "service_publication", error, date)
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        result["finished_at"] = _now().isoformat()
        digest = hashlib.sha256(canonical_json(result)).hexdigest()
        atomic_json(store / "service_ticks" / f"{digest}.json", result)
    return result
