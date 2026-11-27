"""Cutoff, schema, revision and quarantine controls use explicit counterexamples."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from financial_ts.data_contracts import (
    FeatureAvailabilityContract,
    audit_snapshot,
    save_snapshot_audit,
    validate_feature_frame,
)


@pytest.fixture
def inputs():
    calendar = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    index = pd.MultiIndex.from_product([calendar[:2], ["A", "B"]], names=["datetime", "instrument"])
    frame = pd.DataFrame({"close": [10., 20., 11., 21.], "factor": 1.}, index=index)
    publication = pd.Series([str(d.date()) + "T18:00:00+08:00" for d, _ in index], index=index)
    contract = FeatureAvailabilityContract(("close", "factor"))
    return frame, publication, contract, calendar


def validate(inputs, **kwargs):
    frame, publication, contract, calendar = inputs
    return validate_feature_frame(frame, contract, cutoff="2025-01-03T19:00:00+08:00",
                                  observed_calendar=calendar, expected_instruments=["A", "B"],
                                  available_at=publication, **kwargs)


def test_explicit_cutoff_uses_publication_time_and_validates_every_history_row(inputs):
    assert validate(inputs)["status"] == "passed"
    frame, publication, contract, calendar = inputs
    publication.iloc[0] = "2025-01-06T18:00:00+08:00"
    with pytest.raises(ValueError, match="after the decision cutoff"):
        validate((frame, publication, contract, calendar))


def test_lag_counts_observed_sessions_not_calendar_days(inputs):
    frame, _, contract, calendar = inputs
    contract = replace(contract, availability_mode="conservative_lag", conservative_lag_sessions=1,
                       conservative_release_time="08:00:00", maximum_staleness_sessions=1)
    result = validate_feature_frame(frame, contract, cutoff="2025-01-06T09:00:00+08:00",
                                    observed_calendar=calendar, expected_instruments=["A", "B"])
    assert result["last_available_at"].startswith("2025-01-06T08:")
    with pytest.raises(ValueError, match="after the decision cutoff"):
        validate_feature_frame(frame, contract, cutoff="2025-01-06T07:00:00+08:00",
                               observed_calendar=calendar, expected_instruments=["A", "B"])


@pytest.mark.parametrize("fault", ["duplicate", "schema", "infinite", "missing", "membership", "naive_publication"])
def test_invalid_features_fail_before_signal_publication(inputs, fault):
    frame, publication, contract, calendar = inputs
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif fault == "schema":
        frame = frame[["factor", "close"]]
    elif fault == "infinite":
        frame.iloc[0, 0] = np.inf
    elif fault == "missing":
        frame.iloc[0, 0] = np.nan
    elif fault == "membership":
        frame = frame[frame.index.get_level_values("instrument") == "A"]
    else:
        publication.iloc[0] = "2025-01-02T18:00:00"
    with pytest.raises(ValueError):
        validate((frame, publication, contract, calendar))


def test_stale_member_does_not_hide_behind_fresh_members(inputs):
    frame, publication, contract, calendar = inputs
    frame = frame.drop((pd.Timestamp("2025-01-03"), "B"))
    publication = publication.reindex(frame.index)
    with pytest.raises(ValueError, match="stale"):
        validate((frame, publication, contract, calendar))


def test_truncated_calendar_cannot_make_months_old_features_fresh(inputs):
    frame, publication, contract, calendar = inputs
    with pytest.raises(ValueError, match="calendar does not cover"):
        validate_feature_frame(
            frame, contract, cutoff="2025-07-03T19:00:00+08:00",
            observed_calendar=calendar[:2], expected_instruments=["A", "B"],
            available_at=publication,
        )


def test_nontrading_cutoff_uses_prior_session_when_calendar_covers_it(inputs):
    frame, publication, contract, calendar = inputs
    result = validate_feature_frame(
        frame, contract, cutoff="2025-01-05T19:00:00+08:00",
        observed_calendar=calendar, expected_instruments=["A", "B"],
        available_at=publication,
    )
    assert result["status"] == "passed"
    assert result["staleness_sessions"] == {"A": 0, "B": 0}


def identity(release):
    return {"release": release, "archive_sha256": "a" * 64, "provider_sha256": "b" * 64}


def test_refresh_records_adjustment_revision_and_requires_cell_specific_source(inputs, tmp_path):
    frame = inputs[0]
    revised = frame.copy()
    revised.iloc[0, 1] = 2
    result = audit_snapshot(revised, identity("new"), previous=frame, previous_identity=identity("old"))
    assert result["status"] == "quarantined" and result["adjustment_changes"] == 1
    path = save_snapshot_audit(result, tmp_path)
    assert save_snapshot_audit(result, tmp_path) == path
    assert "quarantined" in str(path)
    explanation = {("2025-01-02", "A", "factor"): {"reason": "published correction", "source": "https://example.org/correction"}}
    accepted = audit_snapshot(revised, identity("new"), previous=frame, previous_identity=identity("old"),
                              explained_revisions=explanation)
    assert accepted["status"] == "admitted" and len(accepted["revisions"]) == 1
    assert frame.iloc[0, 1] == 1


def test_removed_history_and_duplicate_rows_are_quarantined(inputs):
    frame = inputs[0]
    removed = audit_snapshot(frame.iloc[1:], identity("new"), previous=frame, previous_identity=identity("old"))
    assert "historical_rows_removed" in removed["errors"]
    duplicate = audit_snapshot(pd.concat([frame, frame.iloc[:1]]), identity("new"))
    assert duplicate["status"] == "quarantined"


def test_infinite_revision_receipt_is_serializable_and_same_identity_cannot_change(inputs, tmp_path):
    frame = inputs[0]
    revised = frame.copy()
    revised.iloc[0, 0] = np.inf
    receipt = audit_snapshot(revised, identity("same"), previous=frame, previous_identity=identity("same"))
    assert "same_identity_different_data" in receipt["errors"]
    assert save_snapshot_audit(receipt, tmp_path).is_file()
