"""Point-in-time feature contracts and immutable snapshot admission receipts.

An observation date does not establish when data could be used. These checks
require either recorded publication timestamps or an explicit conservative lag.
The latter is an assumption, never a claim of reconstructed historical vintages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def _aware(value, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tz is None:
        raise ValueError(f"{name} must be a finite timezone-aware timestamp")
    return timestamp


def _calendar(values) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(values)
    if (
        calendar.empty or calendar.hasnans or calendar.tz is not None
        or not calendar.is_unique or not calendar.is_monotonic_increasing
        or not (calendar == calendar.normalize()).all()
    ):
        raise ValueError("Observed calendar must contain unique sorted naive session dates")
    return calendar


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Feature panel must be a nonempty DataFrame")
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != [
        "datetime", "instrument"
    ]:
        raise ValueError("Panel index must be [datetime, instrument]")
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError("Duplicate panel rows or columns are forbidden")
    dates = pd.DatetimeIndex(frame.index.get_level_values("datetime"))
    if dates.hasnans or dates.tz is not None or not (dates == dates.normalize()).all():
        raise ValueError("Observation dates must be finite naive session dates")
    if any(not isinstance(x, str) or not x for x in frame.index.get_level_values("instrument")):
        raise ValueError("Instrument identifiers must be nonempty strings")
    return frame.sort_index()


@dataclass(frozen=True)
class FeatureAvailabilityContract:
    """Contract for one feature family; the observed calendar is supplied separately."""

    feature_order: tuple[str, ...]
    timezone: str = "Asia/Shanghai"
    calendar_id: str = "XSHG-observed"
    availability_mode: str = "explicit"
    conservative_lag_sessions: int = 0
    conservative_release_time: str = "18:00:00"
    adjustment_convention: str = "raw"
    revision_policy: str = "quarantine"
    maximum_staleness_sessions: int = 0
    maximum_missing_fraction: float = 0.0
    historical_eligibility: str = "caller-supplied historical membership"
    version: int = 1

    def __post_init__(self):
        object.__setattr__(self, "feature_order", tuple(self.feature_order))
        if not self.feature_order or len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order must be nonempty and unique")
        if any(not isinstance(x, str) or not x for x in self.feature_order):
            raise ValueError("Feature names must be nonempty strings")
        ZoneInfo(self.timezone)
        if not self.calendar_id or not self.historical_eligibility:
            raise ValueError("Calendar and historical eligibility policies are required")
        if self.availability_mode not in {"explicit", "conservative_lag"}:
            raise ValueError("availability_mode must be explicit or conservative_lag")
        for name in ("conservative_lag_sessions", "maximum_staleness_sessions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.adjustment_convention not in {"raw", "split_adjusted", "total_return_adjusted"}:
            raise ValueError("Declare a supported adjustment convention")
        if self.revision_policy != "quarantine":
            raise ValueError("Unexplained historical revisions must be quarantined")
        if not np.isfinite(self.maximum_missing_fraction) or not 0 <= self.maximum_missing_fraction < 1:
            raise ValueError("maximum_missing_fraction must be in [0, 1)")
        pd.Timestamp("2000-01-01 " + self.conservative_release_time)


def validate_feature_frame(
    frame: pd.DataFrame,
    contract: FeatureAvailabilityContract,
    *,
    cutoff,
    observed_calendar,
    expected_instruments,
    available_at: pd.Series | None = None,
) -> dict:
    """Fail closed on schema, availability, coverage, stale or invalid input data.

    Historical context is allowed. Freshness is checked against the latest row
    for each expected instrument, while every row must be available by cutoff.
    The supplied session calendar must extend through the cutoff date (or to
    the following session on a nontrading date). A truncated calendar cannot
    establish that old observations are still the latest eligible session.
    ``expected_instruments`` must be the historically eligible set at cutoff.
    Explicit publication times have exactly the same index as the feature panel.
    """
    panel = _panel(frame)
    calendar = _calendar(observed_calendar)
    cutoff = _aware(cutoff, "cutoff").tz_convert(contract.timezone)
    if list(panel.columns) != list(contract.feature_order):
        raise ValueError("Feature order/schema differs from the registered contract")
    expected = list(expected_instruments)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected historical membership must be nonempty and unique")
    instruments = panel.index.get_level_values("instrument")
    if set(instruments) != set(expected):
        raise ValueError("Observed instruments differ from expected historical membership")
    dates = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    positions = calendar.get_indexer(dates)
    if (positions < 0).any():
        raise ValueError("Observation date is outside the observed calendar")
    decision_day = cutoff.tz_localize(None).normalize()
    if calendar[-1] < decision_day:
        raise ValueError("Observed calendar does not cover the decision cutoff date")
    current = calendar.searchsorted(decision_day, side="right") - 1
    if current < 0 or (positions > current).any():
        raise ValueError("Features include observations after the decision cutoff")
    values = panel.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Feature data contains infinite values")
    missing = float(np.isnan(values).mean())
    if missing > contract.maximum_missing_fraction:
        raise ValueError("Feature missingness exceeds the declared limit")
    if contract.availability_mode == "explicit":
        if available_at is None or not available_at.index.is_unique:
            raise ValueError("Explicit unique publication timestamps are required")
        if set(available_at.index) != set(panel.index):
            raise ValueError("Publication timestamps must cover the exact feature panel")
        published = pd.DatetimeIndex([
            _aware(x, "source publication time") for x in available_at.reindex(panel.index)
        ]).tz_convert(contract.timezone)
    else:
        if available_at is not None:
            raise ValueError("Do not mix explicit timestamps with a conservative-lag contract")
        released_positions = positions + contract.conservative_lag_sessions
        if (released_positions >= len(calendar)).any():
            raise ValueError("Feature conservative availability falls beyond observed sessions")
        published = pd.DatetimeIndex([
            pd.Timestamp(f"{calendar[p].date()} {contract.conservative_release_time}", tz=contract.timezone)
            for p in released_positions
        ])
    if (published > cutoff).any():
        raise ValueError("Feature publication/availability is after the decision cutoff")
    if (published.tz_localize(None).normalize() < dates).any():
        raise ValueError("Publication precedes its claimed observation date")
    latest = pd.Series(positions, index=instruments).groupby(level=0).max()
    stale = {str(k): int(current - v) for k, v in latest.items()}
    if max(stale.values()) > contract.maximum_staleness_sessions:
        raise ValueError("Feature panel is stale for one or more expected instruments")
    return {
        "status": "passed", "cutoff": cutoff.isoformat(), "rows": len(panel),
        "instruments": len(expected), "missing_fraction": missing,
        "staleness_sessions": stale, "contract": asdict(contract),
        "last_available_at": published.max().isoformat(),
        "publication_evidence": "recorded timestamps" if available_at is not None else "conservative assumption",
    }


def _identity(identity: dict) -> dict:
    required = {"release", "archive_sha256", "provider_sha256"}
    if not required.issubset(identity) or not isinstance(identity["release"], str) or not identity["release"]:
        raise ValueError("Snapshot identity requires release, archive_sha256 and provider_sha256")
    for key in required - {"release"}:
        digest = identity[key].removeprefix("sha256:")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Snapshot identities require lowercase SHA256 digests")
    return dict(identity)


def audit_snapshot(
    current: pd.DataFrame,
    identity: dict,
    *,
    previous: pd.DataFrame | None = None,
    previous_identity: dict | None = None,
    explained_revisions: dict[tuple[str, str, str], dict] | None = None,
    maximum_missing_fraction: float = 0.05,
) -> dict:
    """Compare immutable snapshots, retaining every changed historical cell.

    An explanation must contain a nonempty ``reason`` and a ``source`` URL.
    Revision permission is scoped to (date, instrument, column), never global.
    An admitted revision is still retained in the receipt. New symbols and
    coverage dates are reported; removed historical rows and schema changes
    always quarantine. Adjustment-factor revisions are explicitly counted.
    """
    if not np.isfinite(maximum_missing_fraction) or not 0 <= maximum_missing_fraction < 1:
        raise ValueError("Invalid missingness threshold")
    identity = _identity(identity)
    if (previous is None) != (previous_identity is None):
        raise ValueError("Previous data and identity must be supplied together")
    errors, changes = [], []
    try:
        current = _panel(current)
        values = current.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        return {"status": "quarantined", "identity": identity, "errors": [str(exc)]}
    if np.isinf(values).any():
        errors.append("infinite_values")
    missing = float(np.isnan(values).mean())
    if missing > maximum_missing_fraction:
        errors.append("missingness_limit")
    added_symbols, removed_symbols, removed_rows = [], [], []
    if previous is not None:
        previous = _panel(previous)
        _identity(previous_identity)
        if identity == previous_identity and not current.equals(previous):
            errors.append("same_identity_different_data")
        if list(current.columns) != list(previous.columns):
            errors.append("schema_changed")
        current_symbols = set(current.index.get_level_values("instrument"))
        old_symbols = set(previous.index.get_level_values("instrument"))
        added_symbols = sorted(current_symbols - old_symbols)
        removed_symbols = sorted(old_symbols - current_symbols)
        removed_rows = [(str(pd.Timestamp(d).date()), str(s)) for d, s in previous.index.difference(current.index)]
        if removed_rows:
            errors.append("historical_rows_removed")
        common = previous.index.intersection(current.index)
        explanations = explained_revisions or {}
        for column in previous.columns.intersection(current.columns):
            old, new = previous.loc[common, column], current.loc[common, column]
            different = ~(old.eq(new) | (old.isna() & new.isna()))
            for key in common[different]:
                revision_key = (str(pd.Timestamp(key[0]).date()), str(key[1]), str(column))
                explanation = explanations.get(revision_key, {})
                explained = bool(explanation.get("reason")) and str(explanation.get("source", "")).startswith("https://")
                def cell(value):
                    if pd.isna(value):
                        return None
                    return float(value) if np.isfinite(value) else str(value)
                changes.append({
                    "datetime": revision_key[0], "instrument": revision_key[1], "column": str(column),
                    "old": cell(old.loc[key]), "new": cell(new.loc[key]),
                    "explained": explained, "explanation": explanation,
                })
        if any(not item["explained"] for item in changes):
            errors.append("unexplained_historical_revisions")
    dates = current.index.get_level_values("datetime")
    return {
        "status": "quarantined" if errors else "admitted", "errors": errors,
        "identity": identity, "previous_identity": previous_identity,
        "schema": [{"name": str(c), "dtype": str(current[c].dtype)} for c in current],
        "coverage": {"start": str(min(dates).date()), "end": str(max(dates).date()), "rows": len(current),
                     "instruments": current.index.get_level_values("instrument").nunique()},
        "missing_fraction": missing, "missing_by_column": current.isna().sum().astype(int).to_dict(),
        "added_symbols": added_symbols, "removed_symbols": removed_symbols,
        "removed_rows": removed_rows, "revisions": changes,
        "adjustment_changes": sum(x["column"].lstrip("$") == "factor" for x in changes),
    }


def save_snapshot_audit(receipt: dict, output: str | Path) -> Path:
    """Persist a content-addressed receipt; never replace a published snapshot."""
    raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    directory = Path(output) / receipt["status"] / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "audit.json"
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError("Existing audit receipt has been modified") from None
    return path
