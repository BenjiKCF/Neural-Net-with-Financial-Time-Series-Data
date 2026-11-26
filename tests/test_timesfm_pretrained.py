"""These contract tests use a fake forecaster, not pretrained quality evidence."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from financial_ts import timesfm_pretrained
from financial_ts.timesfm_pretrained import OfficialTimesFM3Model


class IndexOnlyDataset:
    def __init__(self, index):
        self.index = index
        self.requests = []

    def prepare(self, segment, **kwargs):
        self.requests.append((segment, kwargs))
        assert kwargs["col_set"] == "feature"
        assert kwargs["data_key"] == "infer"
        return pd.DataFrame(index=self.index)


class FakeForecaster:
    def __init__(self):
        self.contexts = []
        self.batch_sizes = []

    def predict_batch(self, contexts, **kwargs):
        assert kwargs == {"horizon": 2, "return_quantiles": False, "use_symmetric_averaging": False}
        self.contexts.extend([values.copy() for values in contexts])
        self.batch_sizes.append(len(contexts))
        for values in contexts:
            yield SimpleNamespace(forecast=np.array([values[-1] * 1.01, values[-1] * 1.0302]))


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fake; never loaded")
    return tmp_path


@pytest.fixture
def market(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=10)
    index = pd.MultiIndex.from_product([["AAA", "BBB"], dates], names=["instrument", "datetime"])
    prices = pd.DataFrame({"$close": np.r_[np.arange(10) + 100, np.arange(10) + 200]}, index=index)
    calls = []

    class Provider:
        @staticmethod
        def calendar(end_time, freq):
            return dates[dates <= end_time]

        @staticmethod
        def features(symbols, fields, start_time, end_time, freq):
            assert fields == ["$close"]
            calls.append((start_time, end_time))
            keep = prices.index.get_level_values("datetime")
            return prices[(keep >= start_time) & (keep <= end_time)]

    monkeypatch.setattr(timesfm_pretrained, "D", Provider)
    return dates, prices, calls


def test_contexts_stop_at_each_prediction_date_and_keep_original_index(checkpoint, market):
    dates, _, calls = market
    index = pd.MultiIndex.from_tuples(
        [(dates[5], "BBB"), (dates[3], "AAA"), (dates[8], "AAA")],
        names=["datetime", "instrument"],
    )
    dataset = IndexOnlyDataset(index)
    model = OfficialTimesFM3Model(checkpoint, context_length=3, batch_size=2)
    fake = FakeForecaster()
    model._forecaster = fake
    model.fit(dataset)
    assert dataset.requests == []
    result = model.predict(dataset)
    pd.testing.assert_index_equal(result.index, index)
    np.testing.assert_allclose(result, 0.02)
    np.testing.assert_array_equal(fake.contexts[0], [203, 204, 205])
    np.testing.assert_array_equal(fake.contexts[1], [101, 102, 103])
    np.testing.assert_array_equal(fake.contexts[2], [106, 107, 108])
    assert fake.batch_sizes == [2, 1]
    assert calls[-1][1] == dates[8]


def test_internal_missing_session_is_not_dropped_or_interpolated(checkpoint, market):
    dates, prices, _ = market
    prices.loc[("AAA", dates[4]), "$close"] = np.nan
    index = pd.MultiIndex.from_tuples([(dates[5], "AAA")], names=["datetime", "instrument"])
    model = OfficialTimesFM3Model(checkpoint, context_length=3)
    model._forecaster = FakeForecaster()
    model.fit(IndexOnlyDataset(index))
    with pytest.raises(ValueError, match="internal missing"):
        model.predict(IndexOnlyDataset(index))


@pytest.mark.parametrize("forecast", [[0, 1], [1, np.nan], [1], [-1, -2]])
def test_bad_forecast_closes_fail_instead_of_faking_scores(checkpoint, market, forecast):
    dates, _, _ = market
    index = pd.MultiIndex.from_tuples([(dates[5], "AAA")], names=["datetime", "instrument"])
    model = OfficialTimesFM3Model(checkpoint, context_length=3)
    model._forecaster = SimpleNamespace(
        predict_batch=lambda **kwargs: [SimpleNamespace(forecast=forecast)]
    )
    model.fit(IndexOnlyDataset(index))
    with pytest.raises(ValueError, match="two finite positive"):
        model.predict(IndexOnlyDataset(index))


def test_qlib_serialization_omits_loaded_weights(checkpoint):
    model = OfficialTimesFM3Model(checkpoint)
    model._forecaster = FakeForecaster()
    model.fitted = True
    restored = pickle.loads(pickle.dumps(model))
    assert restored.checkpoint_path == str(checkpoint)
    assert restored.fitted
    assert not hasattr(restored, "_forecaster")


def test_remote_id_never_triggers_implicit_download():
    with pytest.raises(ValueError, match="local TimesFM3"):
        OfficialTimesFM3Model("google/timesfm-3.0-pytorch")
