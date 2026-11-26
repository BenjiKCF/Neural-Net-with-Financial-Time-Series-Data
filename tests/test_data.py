"""Protect market-data integrity at the CSV/Qlib boundary."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_ts.data import create_demo_csv, import_csv


def _bars(symbol="AAA"):
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": symbol,
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [100.0, 200.0, 300.0],
        }
    )


def _save(tmp_path, frame, name="source.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _binary(provider, symbol, field):
    return np.fromfile(provider / "features" / symbol.lower() / f"{field}.day.bin", dtype="<f4")


def test_binary_offsets_gaps_and_qlib_roundtrip(tmp_path):
    # BBB defines the complete supplied calendar; AAA is missing a middle bar;
    # CCC starts one session late. Neither absence may be filled or shifted.
    frame = pd.concat([_bars().iloc[[0, 2]], _bars("BBB"), _bars("CCC").iloc[1:]])
    source = _save(tmp_path, frame.sample(frac=1, random_state=4))
    provider = tmp_path / "provider"
    metadata = import_csv(source, provider)
    assert metadata["sessions"] == 3
    assert metadata["observations"] == 7
    assert metadata["sources"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads((provider / "metadata.json").read_text()) == metadata
    np.testing.assert_allclose(_binary(provider, "AAA", "close"), [0, 10.5, np.nan, 12.5])
    np.testing.assert_allclose(_binary(provider, "CCC", "close"), [1, 11.5, 12.5])

    import qlib
    from qlib.data import D

    qlib.init(
        provider_uri=str(provider),
        region="us",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )
    data = D.features(D.instruments("all"), ["$close", "$volume", "$factor"], freq="day")
    assert np.isnan(data.loc[("AAA", pd.Timestamp("2024-01-03")), "$close"])
    assert data.loc[("CCC", pd.Timestamp("2024-01-03")), "$close"] == 11.5
    assert data.loc[("BBB", pd.Timestamp("2024-01-04")), "$volume"] == 300
    assert data.loc[("BBB", pd.Timestamp("2024-01-02")), "$factor"] == 1


def test_yahoo_adjustment_applies_to_all_prices_and_inverse_volume(tmp_path):
    frame = _bars().drop(columns="symbol")
    frame["Adj Close"] = frame["close"] * [0.5, 0.5, 1]
    frame.columns = [name.title() if name != "Adj Close" else name for name in frame]
    source = _save(tmp_path, frame, "^GSPC.csv")
    provider = tmp_path / "provider"
    metadata = import_csv(source, provider, symbol="SPX")
    for field in ("open", "high", "low", "close"):
        np.testing.assert_allclose(
            _binary(provider, "SPX", field)[1:], _bars()[field] * [0.5, 0.5, 1]
        )
    np.testing.assert_allclose(_binary(provider, "SPX", "volume")[1:], [200, 400, 300])
    np.testing.assert_allclose(_binary(provider, "SPX", "factor")[1:], [0.5, 0.5, 1])
    assert "raw_yahoo" in metadata["sources"][0]["adjustment"]


def test_existing_adjusted_factor_is_not_applied_twice(tmp_path):
    frame = _bars()
    frame["factor"] = [0.5, 0.5, 1]
    provider = tmp_path / "provider"
    import_csv(_save(tmp_path, frame), provider)
    np.testing.assert_allclose(_binary(provider, "AAA", "close")[1:], frame["close"])
    np.testing.assert_allclose(_binary(provider, "AAA", "volume")[1:], frame["volume"])
    np.testing.assert_allclose(_binary(provider, "AAA", "factor")[1:], frame["factor"])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("close", np.inf),
        ("close", np.nan),
        ("open", 0),
        ("high", 8),
        ("low", 12),
        ("volume", -1),
        ("volume", "not-a-number"),
        ("factor", 0),
        ("date", "2024-01-02T12:00:00"),
        ("date", "2024-01-02T00:00:00+00:00"),
        ("date", "not-a-date"),
        ("date", ""),
        ("close", 1e100),
    ],
)
def test_invalid_data_never_publishes_partial_provider(tmp_path, column, value):
    frame = _bars()
    if column not in frame:
        frame[column] = 1
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    source = _save(tmp_path, frame)
    provider = tmp_path / "provider"
    with pytest.raises(ValueError):
        import_csv(source, provider)
    assert not provider.exists()
    assert not list(tmp_path.glob(".provider-*"))


@pytest.mark.parametrize(
    "symbol", ["../escape", "AAA/BBB", "AAA\\BBB", ".", "..", "CON", "COM1", "A.", ""]
)
def test_unsafe_symbols_are_rejected(tmp_path, symbol):
    with pytest.raises(ValueError, match="symbol"):
        import_csv(_save(tmp_path, _bars(symbol)), tmp_path / "provider")


def test_numeric_ticker_leading_zeroes_are_preserved(tmp_path):
    frame = _bars("000001").rename(columns={"symbol": "Ticker"})
    provider = tmp_path / "provider"
    metadata = import_csv(_save(tmp_path, frame), provider)
    assert metadata["instruments"] == ["000001"]
    assert (provider / "features/000001/close.day.bin").is_file()


def test_duplicate_rows_across_files_are_rejected(tmp_path):
    directory = tmp_path / "csv"
    directory.mkdir()
    _save(directory, _bars("aaa"), "a.csv")
    _save(directory, _bars("AAA"), "b.csv")
    with pytest.raises(ValueError, match="Duplicate"):
        import_csv(directory, tmp_path / "provider")


def test_csv_directory_and_filename_symbols(tmp_path):
    directory = tmp_path / "csv"
    directory.mkdir()
    _save(directory, _bars().drop(columns="symbol"), "brk-b.csv")
    _save(directory, _bars().drop(columns="symbol"), "SPX.csv")
    metadata = import_csv(directory, tmp_path / "provider")
    assert metadata["instruments"] == ["BRK-B", "SPX"]


def test_existing_provider_is_never_overwritten(tmp_path):
    provider = tmp_path / "provider"
    provider.mkdir()
    sentinel = provider / "my-data.bin"
    sentinel.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        import_csv(_save(tmp_path, _bars()), provider)
    assert sentinel.read_bytes() == b"keep me"
    assert list(provider.iterdir()) == [sentinel]


def test_empty_destination_is_supported(tmp_path):
    provider = tmp_path / "provider"
    provider.mkdir()
    import_csv(_save(tmp_path, _bars()), provider)
    assert (provider / "calendars/day.txt").is_file()


def test_destination_symlink_is_rejected(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    link = tmp_path / "provider"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        import_csv(_save(tmp_path, _bars()), link)
    assert list(target.iterdir()) == []


def test_ambiguous_columns_and_adjustment_are_rejected(tmp_path):
    frame = _bars()
    frame[" Close "] = frame["close"]
    with pytest.raises(ValueError, match="Duplicate CSV columns"):
        import_csv(_save(tmp_path, frame), tmp_path / "provider")
    frame = _bars()
    frame["adj_close"] = frame["close"] / 2
    frame["factor"] = 1
    with pytest.raises(ValueError, match="disagrees"):
        import_csv(_save(tmp_path, frame), tmp_path / "provider")


def test_explicit_symbol_cannot_silently_relabel_data(tmp_path):
    with pytest.raises(ValueError, match="conflicts"):
        import_csv(_save(tmp_path, _bars("AAA")), tmp_path / "provider", symbol="BBB")


def test_demo_is_reproducible_and_importable(tmp_path):
    first = create_demo_csv(tmp_path / "a.csv", n_instruments=3, n_days=80)
    second = create_demo_csv(tmp_path / "b.csv", n_instruments=3, n_days=80)
    assert first.read_bytes() == second.read_bytes()
    metadata = import_csv(first, tmp_path / "provider")
    assert metadata["observations"] == 240
    assert metadata["sessions"] == 80
    assert len(metadata["instruments"]) == 3
    with pytest.raises(FileExistsError):
        create_demo_csv(first)


def test_original_bundled_index_csv_is_importable(tmp_path):
    source = Path(__file__).resolve().parents[1] / "^GSPC.csv"
    metadata = import_csv(source, tmp_path / "provider", symbol="SPX")
    assert metadata["instruments"] == ["SPX"]
    assert metadata["observations"] > 5_000
