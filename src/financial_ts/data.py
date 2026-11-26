"""Validated daily CSV ingestion into a local Qlib binary provider.

CSV prices and volume must be consistently adjusted. Yahoo-style ``Adj Close``
is the one supported raw-data conversion: all OHLC prices are multiplied by
``adj_close / close`` and volume is divided by the same factor. Without an
``adj_close`` column, values are copied as already adjusted; an optional
``factor`` is stored without applying a second adjustment.

The calendar is the union of dates present in the input. Supply all relevant
market sessions: this importer cannot reconstruct sessions absent from every
instrument. Missing individual instrument bars remain NaN, never forward filled.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume", "factor")
_PRICE_FIELDS = ("open", "high", "low", "close")
_SYMBOL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def _column_name(value: str) -> str:
    name = re.sub(r"[\s-]+", "_", value.strip().lower())
    return {"datetime": "date", "ticker": "symbol", "adjclose": "adj_close"}.get(name, name)


def _normalise_symbol(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every row must have a nonempty instrument symbol.")
    symbol = value.strip().upper()
    if not _SYMBOL.fullmatch(symbol) or symbol.endswith(".") or symbol.split(".")[0] in _RESERVED:
        raise ValueError(
            f"Unsafe instrument symbol {value!r}; use letters, digits, '.', '-' or '_'."
        )
    return symbol


def _read_csv(path: Path, symbol: str | None) -> tuple[pd.DataFrame, dict]:
    raw = path.read_bytes()
    header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), [])
    normalised = [_column_name(value) for value in header]
    if len(normalised) != len(set(normalised)):
        raise ValueError(f"Duplicate CSV columns after normalisation in {path.name}.")
    # Preserve numeric-looking ticker codes, including their leading zeroes.
    frame = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype=str)
    frame.columns = normalised
    required = {"date", *_PRICE_FIELDS, "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}.")
    if frame.empty:
        raise ValueError(f"{path.name} has no market observations.")
    if "symbol" not in frame:
        frame["symbol"] = _normalise_symbol(symbol if symbol is not None else path.stem)
    else:
        frame["symbol"] = frame["symbol"].map(_normalise_symbol)
        if symbol is not None and not frame["symbol"].eq(_normalise_symbol(symbol)).all():
            raise ValueError("The explicit symbol conflicts with the CSV symbol column.")

    try:
        dates = pd.to_datetime(frame["date"], format="mixed", errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid dates in {path.name}.") from exc
    if (
        not pd.api.types.is_datetime64_dtype(dates.dtype)
        or dates.isna().any()
        or not dates.eq(dates.dt.normalize()).all()
    ):
        raise ValueError(
            "Dates must be timezone-naive daily dates at midnight, with no missing values."
        )
    frame["date"] = dates

    numeric_fields = [*_PRICE_FIELDS, "volume"]
    numeric_fields += [name for name in ("adj_close", "factor") if name in frame]
    for name in numeric_fields:
        try:
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column {name!r} must contain numeric values.") from exc
        if not np.isfinite(frame[name]).all():
            raise ValueError(f"Column {name!r} contains a missing or non-finite value.")
        invalid_sign = (frame[name] < 0).any() if name == "volume" else (frame[name] <= 0).any()
        if invalid_sign:
            raise ValueError(
                f"Column {name!r} must be {'nonnegative' if name == 'volume' else 'positive'}."
            )

    tolerance = frame[list(_PRICE_FIELDS)].abs().max(axis=1) * 1e-7
    if (frame["high"] + tolerance < frame[["open", "low", "close"]].max(axis=1)).any() or (
        frame["low"] - tolerance > frame[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC bars: low <= open/close <= high is required.")

    if "adj_close" in frame:
        factor = frame["adj_close"] / frame["close"]
        if "factor" in frame and not np.allclose(frame["factor"], factor, rtol=1e-7, atol=0):
            raise ValueError("Provided factor disagrees with adj_close / close.")
        frame["factor"] = factor
        frame[list(_PRICE_FIELDS)] = frame[list(_PRICE_FIELDS)].mul(factor, axis=0)
        frame["volume"] = frame["volume"] / factor
        adjustment = "raw_yahoo_ohlc_times_adj_close_over_close_and_volume_divided_by_factor"
    elif "factor" in frame:
        adjustment = "already_adjusted_ohlcv_with_provided_factor"
    else:
        frame["factor"] = 1.0
        adjustment = "already_adjusted_or_unsplit_ohlcv_with_unit_factor"

    # Qlib's disk format is float32. Reject overflow or positive-value underflow
    # instead of creating a provider which silently changes the input semantics.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        binary_values = frame[list(FIELDS)].to_numpy(dtype="<f4")
    positive_columns = [0, 1, 2, 3, 5]
    if not np.isfinite(binary_values).all() or (binary_values[:, positive_columns] <= 0).any():
        raise ValueError("Adjusted values cannot be represented safely in Qlib float32 storage.")
    metadata = {
        "file": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(frame),
        "adjustment": adjustment,
    }
    return frame[["date", "symbol", *FIELDS]], metadata


def import_csv(source: str | Path, provider: str | Path, symbol: str | None = None) -> dict:
    """Create a new daily Qlib provider from a CSV or directory of CSV files.

    A long CSV can contain many instruments in ``symbol``. With no symbol column,
    use ``symbol=...`` or a valid CSV filename stem. Existing nonempty providers
    are rejected. Import is staged alongside the destination and renamed only
    after every file has passed validation and every binary has been written.

    Returns the same JSON-serialisable provenance metadata written to
    ``provider/metadata.json``. Input files are never modified.
    """
    source = Path(source).expanduser().resolve()
    provider = Path(provider).expanduser().absolute()
    if provider.is_symlink():
        raise ValueError("The provider destination must not be a symbolic link.")
    if provider.exists() and (not provider.is_dir() or any(provider.iterdir())):
        raise FileExistsError(
            f"Provider {provider} already exists and is not empty; choose a new path."
        )
    if source.is_dir():
        paths = sorted(
            path for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".csv"
        )
        if symbol is not None and len(paths) != 1:
            raise ValueError("An explicit symbol is only supported with a single input CSV.")
    elif source.is_file():
        paths = [source]
    else:
        raise FileNotFoundError(f"CSV source does not exist: {source}")
    if not paths:
        raise ValueError(f"No CSV files found in {source}.")

    frames, sources = zip(*(_read_csv(path, symbol) for path in paths))
    observations = pd.concat(frames, ignore_index=True)
    if observations.duplicated(["symbol", "date"]).any():
        raise ValueError(
            "Duplicate (symbol, date) observations are not allowed, including across input files."
        )
    observations = observations.sort_values(["symbol", "date"])
    calendar = pd.DatetimeIndex(observations["date"].unique()).sort_values()
    metadata = {
        "schema_version": 1,
        "frequency": "day",
        "calendar": "union_of_observed_input_dates",
        "start_date": calendar[0].strftime("%Y-%m-%d"),
        "end_date": calendar[-1].strftime("%Y-%m-%d"),
        "sessions": len(calendar),
        "observations": len(observations),
        "instruments": sorted(observations["symbol"].unique().tolist()),
        "fields": list(FIELDS),
        "sources": list(sources),
        "missing_bars": "NaN; no forward fill",
    }
    provider.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{provider.name}-", dir=provider.parent))
    try:
        (stage / "calendars").mkdir()
        (stage / "instruments").mkdir()
        (stage / "features").mkdir()
        (stage / "calendars/day.txt").write_text(
            "\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8"
        )
        instrument_lines = []
        for instrument, frame in observations.groupby("symbol", sort=True):
            start, end = frame["date"].iloc[0], frame["date"].iloc[-1]
            start_index, end_index = calendar.get_indexer([start, end])
            aligned = frame.set_index("date").reindex(calendar[start_index : end_index + 1])
            feature_dir = stage / "features" / instrument.lower()
            feature_dir.mkdir()
            for field in FIELDS:
                values = np.empty(len(aligned) + 1, dtype="<f4")
                values[0] = start_index
                values[1:] = aligned[field].to_numpy(dtype="<f4")
                values.tofile(feature_dir / f"{field}.day.bin")
            instrument_lines.append(f"{instrument}\t{start:%Y-%m-%d}\t{end:%Y-%m-%d}")
        (stage / "instruments/all.txt").write_text(
            "\n".join(instrument_lines) + "\n", encoding="utf-8"
        )
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if provider.exists():
            # rmdir refuses a directory populated since the initial check.
            provider.rmdir()
        stage.rename(provider)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return metadata


def create_demo_csv(
    path: str | Path, seed: int = 42, n_instruments: int = 8, n_days: int = 420
) -> Path:
    """Write reproducible synthetic daily OHLCV for pipeline smoke tests only.

    These independent simulated assets do not establish a profitable strategy.
    Dates are generic business days, not an actual exchange trading calendar.
    """
    if n_instruments < 1 or n_days < 3:
        raise ValueError("The demo needs at least one instrument and three days.")
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(f"Demo CSV already exists: {path}")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    market = rng.normal(0.0002, 0.008, n_days)
    frames = []
    for number in range(n_instruments):
        returns = market * rng.uniform(0.6, 1.2) + rng.normal(0, 0.01, n_days)
        close = (30 + number * 5) * np.exp(np.cumsum(returns))
        previous_close = np.r_[close[0], close[:-1]]
        opening = previous_close * np.exp(rng.normal(0, 0.004, n_days))
        spread = rng.uniform(0.001, 0.012, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "symbol": f"DEMO{number:03d}",
                    "open": opening,
                    "high": np.maximum(opening, close) * (1 + spread),
                    "low": np.minimum(opening, close) * (1 - spread),
                    "close": close,
                    "volume": rng.integers(100_000, 2_000_000, n_days),
                }
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path
