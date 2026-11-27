#!/usr/bin/env python3
"""Explicitly download a local SPY daily-price snapshot and its provenance.

Run from the repository root:
    python scripts/download_spy.py --output .data/backtest

Network access occurs only when this script is executed without --raw-json.
Historical adjusted closes include dividend/split adjustments and are a
total-return proxy. They are not fill quotes. Never add distributions again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_START = "2006-01-01"
DEFAULT_END = "2018-10-12"
FIELDS = ("date", "symbol", "open", "high", "low", "close", "adj_close", "volume")


def source_url(start: date, end: date) -> str:
    if start > end:
        raise ValueError("start must not be after end.")
    first = int(datetime.combine(start, time(), timezone.utc).timestamp())
    last = int(datetime.combine(end + timedelta(days=1), time(), timezone.utc).timestamp())
    query = urllib.parse.urlencode(
        {"period1": first, "period2": last, "interval": "1d", "events": "div,splits"}
    )
    return "https://query1.finance.yahoo.com/v8/finance/chart/SPY?" + query


def _rows(raw: bytes, start: date, end: date):
    try:
        payload = json.loads(raw)
        chart = payload["chart"]
        if chart.get("error") or len(chart["result"]) != 1:
            raise ValueError("The source did not return one successful chart.")
        result = chart["result"][0]
        meta = result["meta"]
        if (meta["symbol"], meta["currency"], meta["instrumentType"]) != ("SPY", "USD", "ETF"):
            raise ValueError("The response must identify the SPY ETF in USD.")
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        arrays = [quote[key] for key in ("open", "high", "low", "close", "volume")]
        if not timestamps or any(len(values) != len(timestamps) for values in [*arrays, adjusted]):
            raise ValueError("Source price arrays must have equal nonzero lengths.")
        rows = []
        for i, timestamp in enumerate(timestamps):
            session = datetime.fromtimestamp(timestamp, ZoneInfo("America/New_York")).date()
            if not start <= session <= end:
                raise ValueError("A returned session is outside the requested date range.")
            row = {
                "date": session.isoformat(),
                "symbol": "SPY",
                **{key: quote[key][i] for key in ("open", "high", "low", "close")},
                "adj_close": adjusted[i],
                "volume": quote["volume"][i],
            }
            if any(
                isinstance(row[key], bool)
                or not isinstance(row[key], (int, float))
                or not math.isfinite(row[key])
                or row[key] <= 0
                for key in ("open", "high", "low", "close", "adj_close")
            ):
                raise ValueError(
                    "Source prices must be finite and positive; missing bars are rejected."
                )
            volume = row["volume"]
            if (
                isinstance(volume, bool)
                or not isinstance(volume, (int, float))
                or not math.isfinite(volume)
                or volume < 0
                or int(volume) != volume
            ):
                raise ValueError("Source volume must be a finite nonnegative integer.")
            tolerance = max(row[key] for key in ("open", "high", "low", "close")) * 1e-7
            if row["high"] + tolerance < max(row["open"], row["low"], row["close"]) or (
                row["low"] - tolerance > min(row["open"], row["high"], row["close"])
            ):
                raise ValueError("Source OHLC bars are inconsistent.")
            rows.append(row)
        dates = [row["date"] for row in rows]
        if dates != sorted(set(dates)):
            raise ValueError("Source sessions must be unique and increasing.")
        events = result.get("events", {})
        return rows, {
            "dividend_events": len(events.get("dividends", {})),
            "split_events": len(events.get("splits", {})),
        }
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
        raise ValueError("Malformed Yahoo daily chart response.") from error


def write_snapshot(
    raw: bytes,
    output: Path,
    *,
    start: date,
    end: date,
    received_at_utc: str,
    acquisition: str = "explicit public HTTPS request",
) -> dict:
    """Validate and materialize supplied bytes; this function never uses network."""
    url = source_url(start, end)
    rows, event_counts = _rows(raw, start, end)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Snapshot output must be a new or empty directory.")
    output.mkdir(parents=True, exist_ok=True)
    raw_path, csv_path = output / "spy_raw.json", output / "spy_daily.csv"
    raw_path.write_bytes(raw)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "source": "Yahoo Finance public chart response",
        "source_url": url,
        "historical_prices_page": "https://finance.yahoo.com/quote/SPY/history/",
        "adjustment_definition": "https://help.yahoo.com/kb/SLN28256.html",
        "acquisition": acquisition,
        "received_at_utc": received_at_utc,
        "requested_start": start.isoformat(),
        "requested_end_inclusive": end.isoformat(),
        "symbol": "SPY",
        "currency": "USD",
        "session_timezone": "America/New_York",
        "rows": len(rows),
        "first_session": rows[0]["date"],
        "last_session": rows[-1]["date"],
        **event_counts,
        "price_basis": "adj_close includes dividend/split adjustments; do not add dividends twice",
        "normalization": "Source values unchanged; timestamps mapped to exchange-local session dates",
        "missing_bars": "rejected; no filling",
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in (raw_path, csv_path)
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", type=date.fromisoformat, default=date.fromisoformat(DEFAULT_START)
    )
    parser.add_argument("--end", type=date.fromisoformat, default=date.fromisoformat(DEFAULT_END))
    parser.add_argument("--output", type=Path, default=Path(".data/backtest"))
    parser.add_argument("--raw-json", type=Path, help="Normalize an existing raw response, offline")
    args = parser.parse_args(argv)
    if args.raw_json:
        raw = args.raw_json.read_bytes()
        received = datetime.fromtimestamp(args.raw_json.stat().st_mtime, timezone.utc).isoformat()
        acquisition = "offline normalization of a previously downloaded raw response"
    else:
        request = urllib.request.Request(
            source_url(args.start, args.end),
            headers={"User-Agent": "financial-ts-qlib/2.2 historical research downloader"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Unexpected source response larger than 16 MiB.")
        received = datetime.now(timezone.utc).isoformat()
        acquisition = "explicit public HTTPS request"
    result = write_snapshot(
        raw,
        args.output,
        start=args.start,
        end=args.end,
        received_at_utc=received,
        acquisition=acquisition,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
