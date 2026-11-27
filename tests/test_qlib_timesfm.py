"""Panel contracts, causal prices, adapter selection, and resumable inference."""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from financial_ts import qlib_timesfm
from financial_ts.qlib_timesfm import QlibTimesFM3Model, _CloseContextBank


@pytest.fixture
def checkpoint(tmp_path):
    target = tmp_path / "checkpoint"
    target.mkdir()
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"fake, never loaded")
    return target


@pytest.fixture
def market(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=12)
    index = pd.MultiIndex.from_product([["AAA", "BBB"], dates], names=["instrument", "datetime"])
    prices = pd.DataFrame({"$close": np.r_[np.arange(12) + 100, np.arange(12) + 200]}, index=index)
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
            selected = prices.index.get_level_values("instrument").isin(symbols)
            return prices[selected & (keep >= start_time) & (keep <= end_time)]

    monkeypatch.setattr(qlib_timesfm, "D", Provider)
    return dates, prices, calls


class PanelDataset:
    def __init__(self, dates):
        self.indices = {
            name: pd.MultiIndex.from_product(
                [days, ["AAA", "BBB"]], names=["datetime", "instrument"]
            )
            for name, days in {"train": dates[2:5], "valid": dates[7:9], "test": dates[10:]}.items()
        }
        self.requests = []
        self.forbid_labels = False

    def prepare(self, segment, *, col_set, data_key):
        self.requests.append((segment, col_set, data_key))
        if col_set == "feature":
            assert data_key == "infer"
            return pd.DataFrame(index=self.indices[segment])
        assert col_set == "label" and data_key == "raw"
        assert segment != "test" and not self.forbid_labels
        # Training pushes the adapter positive, while validation favors epoch zero.
        return pd.DataFrame(
            {"LABEL0": 0.1 if segment == "train" else 0.0}, index=self.indices[segment]
        )


class FakeForecaster:
    def __init__(self):
        self.contexts = []
        self.calls = 0

    def predict_batch(self, contexts, **kwargs):
        assert kwargs == {"horizon": 2, "return_quantiles": False, "use_symmetric_averaging": False}
        self.calls += 1
        self.contexts.extend([context.copy() for context in contexts])
        for context in contexts:
            yield SimpleNamespace(forecast=np.array([context[-1], context[-1] + 1]))


class FakeOfficial:
    parameter_count_ = 42

    def __init__(self):
        self.forecaster = FakeForecaster()

    def _get_forecaster(self):
        return self.forecaster


class FakeLoop(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.tensor(0.0))

    def return_scores(self, contexts):
        return self.adapter.expand(len(contexts)).double()

    def base_parameters(self):
        return iter([self.base])

    def adapter_state_dict(self):
        return {"adapter": self.adapter.detach().cpu().clone()}

    def load_adapter_state_dict(self, state):
        with torch.no_grad():
            self.adapter.copy_(state["adapter"])


def test_causal_fill_preserves_calendar_and_never_uses_later_prices(market):
    dates, prices, calls = market
    prices.loc[("AAA", dates[3]), "$close"] = np.nan
    index = pd.MultiIndex.from_tuples(
        [(dates[4], "AAA"), (dates[7], "BBB")], names=["datetime", "instrument"]
    )
    bank = _CloseContextBank(index, 4)
    contexts, counts = bank.contexts(index, pad=False)
    np.testing.assert_array_equal(contexts[0], [101, 102, 102, 104])
    np.testing.assert_array_equal(contexts[1], [204, 205, 206, 207])
    assert counts == {"forward_filled_values": 1, "leading_padded_values": 0}
    assert calls[-1][1] == dates[7]
    prices.loc[("AAA", dates[5]), "$close"] = 99999
    again, _ = _CloseContextBank(index, 4).contexts(index, pad=False)
    np.testing.assert_array_equal(again[0], contexts[0])


def test_leading_missing_is_trimmed_and_padding_is_explicit(market):
    dates, prices, _ = market
    prices.loc[("AAA", dates[:2]), "$close"] = np.nan
    index = pd.MultiIndex.from_tuples([(dates[3], "AAA")], names=["datetime", "instrument"])
    bank = _CloseContextBank(index, 4)
    frozen, _ = bank.contexts(index, pad=False)
    adapted, fills = bank.contexts(index, pad=True)
    np.testing.assert_array_equal(frozen[0], [102, 103])
    np.testing.assert_array_equal(adapted[0], [102, 102, 102, 103])
    assert fills["leading_padded_values"] == 2


def test_all_missing_context_fails_without_future_backfill(market):
    dates, prices, _ = market
    prices.loc[("AAA", dates[2:6]), "$close"] = np.nan
    index = pd.MultiIndex.from_tuples([(dates[5], "AAA")], names=["datetime", "instrument"])
    bank = _CloseContextBank(index, 4)
    assert not bank.eligible(index)[0]
    with pytest.raises(ValueError, match="No future-price substitution"):
        bank.contexts(index, pad=True)


def test_frozen_fit_reads_no_labels_and_predicts_every_membership_row(
    checkpoint, market, monkeypatch
):
    dates, _, _ = market
    dataset = PanelDataset(dates)
    fake = FakeOfficial()
    monkeypatch.setattr(QlibTimesFM3Model, "_new_backend", lambda self: fake)
    model = QlibTimesFM3Model(checkpoint, context_length=4, batch_size=2)
    model.fit(dataset)
    assert dataset.requests == []
    dataset.forbid_labels = True
    result = model.predict(dataset)
    pd.testing.assert_index_equal(result.index, dataset.indices["test"])
    assert len(result) == 4
    assert fake.forecaster.calls == 2
    assert model.manifest_["inference_sample_cap"] is None


@pytest.mark.parametrize("mode", ["loop", "adapter"])
def test_adapter_selection_includes_epoch_zero_and_omits_test(
    checkpoint, market, monkeypatch, mode
):
    dates, _, calls = market
    dataset = PanelDataset(dates)
    monkeypatch.setattr(QlibTimesFM3Model, "_new_backend", lambda self: FakeLoop())
    model = QlibTimesFM3Model(
        checkpoint,
        mode=mode,
        context_length=4,
        batch_size=2,
        train_samples=4,
        valid_samples=4,
        epochs=2,
    )
    model.fit(dataset)
    assert model.best_epoch_ == 0
    assert model.adapter_state_["adapter"].item() == 0
    assert model.parameter_count_ == 2
    assert model.trainable_parameter_count_ == 1
    assert model.manifest_["optimizer_updates"] == 4
    assert len(model.training_curves_) == 3
    assert model.training_curves_[1]["validation_rmse"] > 0
    assert [request[0] for request in dataset.requests] == ["train", "valid"]
    assert max(call[1] for call in calls) <= dates[8]
    assert all(row["instrument"] in {"AAA", "BBB"} for row in model.samples_["train"])
    assert model._backend.base.grad is None
    dataset.forbid_labels = True
    np.testing.assert_array_equal(model.predict(dataset), 0)


def test_saved_model_excludes_loaded_backbone_and_reloads_small_adapter(
    checkpoint, market, monkeypatch
):
    dates, _, _ = market
    monkeypatch.setattr(QlibTimesFM3Model, "_new_backend", lambda self: FakeLoop())
    model = QlibTimesFM3Model(
        checkpoint, mode="loop", context_length=4, train_samples=2, valid_samples=2, epochs=1
    ).fit(PanelDataset(dates))
    restored = pickle.loads(pickle.dumps(model))
    assert not hasattr(restored, "_backend")
    assert restored.adapter_state_["adapter"].numel() == 1
    assert restored._get_backend().adapter.item() == model.adapter_state_["adapter"].item()


def test_prediction_cache_resumes_without_new_forecasts_and_binds_prices(
    checkpoint, market, monkeypatch, tmp_path
):
    dates, prices, _ = market
    fake = FakeOfficial()
    monkeypatch.setattr(QlibTimesFM3Model, "_new_backend", lambda self: fake)
    model = QlibTimesFM3Model(
        checkpoint, context_length=4, batch_size=2, prediction_cache_dir=tmp_path / "predictions"
    ).fit(None)
    dataset = PanelDataset(dates)
    first = model.predict(dataset)
    calls = fake.forecaster.calls
    second = model.predict(dataset)
    pd.testing.assert_series_equal(first, second)
    assert fake.forecaster.calls == calls
    assert model.last_prediction_["resumed_rows"] == len(first)
    prices.loc[("AAA", dates[11]), "$close"] = 500
    third = model.predict(dataset)
    assert fake.forecaster.calls > calls
    assert not third.equals(first)


def test_invalid_cache_chunk_is_rejected(checkpoint, market, monkeypatch, tmp_path):
    dates, _, _ = market
    monkeypatch.setattr(QlibTimesFM3Model, "_new_backend", lambda self: FakeOfficial())
    cache = tmp_path / "predictions"
    model = QlibTimesFM3Model(checkpoint, context_length=4, prediction_cache_dir=cache).fit(None)
    dataset = PanelDataset(dates)
    model.predict(dataset)
    path = next(cache.glob("test-*/part-*.npz"))
    np.savez(path, start=0, scores=np.array([np.nan]))
    with pytest.raises(ValueError, match="invalid score chunk"):
        model.predict(dataset)


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors"])
def test_replaced_checkpoint_is_rejected_before_cached_replay_or_lazy_load(
    checkpoint, market, monkeypatch, tmp_path, filename
):
    dates, _, _ = market
    fake = FakeOfficial()
    monkeypatch.setattr(QlibTimesFM3Model, "_new_backend", lambda self: fake)
    model = QlibTimesFM3Model(
        checkpoint, context_length=4, prediction_cache_dir=tmp_path / "predictions"
    ).fit(None)
    dataset = PanelDataset(dates)
    model.predict(dataset)
    calls = fake.forecaster.calls
    restored = pickle.loads(pickle.dumps(model))
    (checkpoint / filename).write_bytes(b"replaced checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        model.predict(dataset)
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        restored._get_backend()
    assert fake.forecaster.calls == calls


def test_close_only_handler_keeps_inference_rows_with_missing_labels(monkeypatch):
    captured = {}

    def initialize(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(qlib_timesfm.DataHandlerLP, "__init__", initialize)
    qlib_timesfm.CloseOnlyHandler(start_time="2020-01-01", end_time="2026-11-26")
    assert captured["infer_processors"] == []
    assert captured["learn_processors"] == [{"class": "DropnaLabel"}]
    config = captured["data_loader"]["kwargs"]["config"]
    assert config["feature"] == (["$close"], ["CLOSE"])
    assert config["label"] == (["Ref($close, -2)/Ref($close, -1)-1"], ["LABEL0"])
