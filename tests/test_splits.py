"""Temporal boundary tests use actual sessions, not calendar-day arithmetic."""

from itertools import pairwise

import pandas as pd
import pytest

from financial_ts.workflow import resolve_segments


@pytest.fixture
def holiday_calendar():
    # Jan 15 is absent; the two-session horizon crosses a weekend and a holiday.
    return pd.DatetimeIndex(
        [
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
            "2024-01-16",
            "2024-01-17",
            "2024-01-18",
            "2024-01-19",
            "2024-01-22",
            "2024-01-23",
            "2024-01-24",
            "2024-01-25",
            "2024-01-26",
            "2024-01-29",
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
            "2024-02-05",
        ]
    )


def valid_explicit_segments():
    return {
        "train": ("2024-01-08", "2024-01-11"),
        "valid": ("2024-01-17", "2024-01-19"),
        "test": ("2024-01-24", "2024-02-01"),
    }


def included_sessions(calendar, segment):
    start, end = map(pd.Timestamp, segment)
    return calendar[(calendar >= start) & (calendar <= end)]


def assert_disjoint_label_horizons(calendar, segments, horizon=2):
    parts = [included_sessions(calendar, segments[name]) for name in ("train", "valid", "test")]
    assert all(len(part) for part in parts)
    for previous, following in pairwise(parts):
        last_label_position = calendar.get_loc(previous[-1]) + horizon
        assert last_label_position < calendar.get_loc(following[0])
    assert calendar.get_loc(parts[-1][-1]) + horizon < len(calendar)


def test_automatic_split_preserves_calendar_order_and_purges_horizons():
    calendar = pd.bdate_range("2023-01-03", periods=240).difference(
        pd.DatetimeIndex(["2023-01-16", "2023-02-20", "2023-04-07"])
    )
    segments = resolve_segments(calendar, horizon=2, warmup=60)

    assert set(segments) == {"train", "valid", "test"}
    assert_disjoint_label_horizons(calendar, segments)
    train = included_sessions(calendar, segments["train"])
    valid = included_sessions(calendar, segments["valid"])
    test = included_sessions(calendar, segments["test"])
    assert train[0] == calendar[60]
    assert test[-1] == calendar[-3]
    assert len(train) > len(valid)
    assert abs(len(valid) - len(test)) <= 3
    # Exactly two reserved feature dates before each next partition.
    assert calendar.get_loc(valid[0]) - calendar.get_loc(train[-1]) == 3
    assert calendar.get_loc(test[0]) - calendar.get_loc(valid[-1]) == 3
    assert segments == resolve_segments(calendar, horizon=2, warmup=60)


def test_explicit_two_session_gap_crosses_weekend_and_holiday(holiday_calendar):
    segments = resolve_segments(holiday_calendar, valid_explicit_segments(), horizon=2, warmup=0)
    assert_disjoint_label_horizons(holiday_calendar, segments)
    assert included_sessions(holiday_calendar, segments["train"])[-1] == pd.Timestamp("2024-01-11")


def test_elapsed_days_do_not_substitute_for_two_sessions(holiday_calendar):
    segments = valid_explicit_segments()
    segments["train"] = ("2024-01-08", "2024-01-12")
    # Jan 12 + two sessions is Jan 17: equal to valid start is forbidden,
    # even though five elapsed days separate the feature dates.
    with pytest.raises(ValueError):
        resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)


@pytest.mark.parametrize("boundary", ["train", "valid"])
def test_labels_cannot_end_on_next_partition_start(holiday_calendar, boundary):
    segments = valid_explicit_segments()
    if boundary == "train":
        segments["train"] = ("2024-01-08", "2024-01-12")
    else:
        segments["valid"] = ("2024-01-17", "2024-01-22")
    with pytest.raises(ValueError):
        resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)


def test_explicit_overlapping_partitions_are_rejected(holiday_calendar):
    segments = valid_explicit_segments()
    segments["valid"] = ("2024-01-11", "2024-01-19")
    with pytest.raises(ValueError):
        resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)


def test_nontrading_endpoints_resolve_to_included_sessions(holiday_calendar):
    segments = {
        "train": ("2024-01-08", "2024-01-11"),
        "valid": ("2024-01-17", "2024-01-21"),  # Sunday -> Friday Jan 19
        "test": ("2024-01-24", "2024-02-01"),
    }
    result = resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)
    assert_disjoint_label_horizons(holiday_calendar, result)
    assert included_sessions(holiday_calendar, result["valid"])[-1] == pd.Timestamp("2024-01-19")


def test_last_test_label_may_end_at_last_provider_session(holiday_calendar):
    segments = valid_explicit_segments()
    result = resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)
    last_test = included_sessions(holiday_calendar, result["test"])[-1]
    assert holiday_calendar.get_loc(last_test) + 2 == len(holiday_calendar) - 1


def test_missing_future_label_coverage_is_rejected(holiday_calendar):
    segments = valid_explicit_segments()
    segments["test"] = ("2024-01-24", "2024-02-02")
    with pytest.raises(ValueError):
        resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)


def test_explicit_out_of_range_partition_is_rejected(holiday_calendar):
    segments = valid_explicit_segments()
    segments["train"] = ("2024-01-01", "2024-01-11")
    with pytest.raises(ValueError):
        resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)


def test_partition_without_any_trading_session_is_rejected(holiday_calendar):
    segments = valid_explicit_segments()
    segments["valid"] = ("2024-01-20", "2024-01-21")
    with pytest.raises(ValueError):
        resolve_segments(holiday_calendar, segments, horizon=2, warmup=0)


def test_insufficient_history_for_automatic_split_is_rejected(holiday_calendar):
    with pytest.raises(ValueError):
        resolve_segments(holiday_calendar, horizon=2, warmup=60)
