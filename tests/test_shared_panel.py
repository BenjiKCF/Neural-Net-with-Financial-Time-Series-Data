"""Exact Qlib window/target equivalence for shared panel storage."""

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP

from financial_ts.model import SequenceForecastModel
from financial_ts.shared_panel import SharedPanelSampler, SharedTSDatasetH


def _frame():
    dates = pd.bdate_range("2024-01-02", periods=13)
    index = pd.MultiIndex.from_product(
        [dates, ["AAA", "BBB", "CCC"]], names=["datetime", "instrument"]
    )
    values = np.arange(len(index) * 3, dtype=np.float32).reshape(-1, 3) / 100
    frame = pd.DataFrame(
        values,
        index=index,
        columns=pd.MultiIndex.from_tuples(
            [("feature", "x"), ("feature", "y"), ("label", "return")]
        ),
    )
    # Late listing, membership exit/reentry, and a date absent from the entire panel.
    holes = [(dates[0], "AAA"), (dates[1], "AAA"), (dates[5], "BBB"), (dates[6], "BBB")]
    holes += [(dates[4], name) for name in ["AAA", "BBB", "CCC"]]
    frame = frame.drop(index=holes)
    frame.loc[(dates[3], "AAA"), ("label", "return")] = np.nan
    frame.loc[(dates[7], "BBB"), ("feature", "x")] = np.nan
    frame.loc[(dates[8], "CCC"), [("feature", "x"), ("feature", "y")]] = np.nan
    return frame, dates


def _handler(frame):
    return DataHandlerLP(
        data_loader={"class": "StaticDataLoader", "kwargs": {"config": frame.copy(deep=True)}},
        infer_processors=[],
        learn_processors=[{"class": "DropnaLabel"}],
    )


def _datasets(step_len=4):
    frame, dates = _frame()
    segments = {
        "train": [dates[0], dates[5]],
        "valid": [dates[6], dates[8]],
        "test": [dates[9], dates[-1]],
    }
    native = TSDatasetH(handler=_handler(frame), segments=segments, step_len=step_len)
    shared = SharedTSDatasetH(handler=_handler(frame), segments=segments, step_len=step_len)
    return native, shared


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("step_len", [1, 4, 32])
def test_every_window_and_index_exactly_matches_native_qlib(dtype, step_len):
    native, shared = _datasets(step_len)
    for segment in ["train", "valid", "test"]:
        expected = native.prepare(segment, col_set="feature", data_key="infer", dtype=dtype)
        actual = shared.prepare(segment, col_set="feature", data_key="infer", dtype=dtype)
        assert isinstance(actual, SharedPanelSampler)
        actual.config(fillna_type="none")
        pd.testing.assert_index_equal(actual.get_index(), expected.get_index())
        assert len(actual) == len(expected)
        for index in range(len(actual)):
            np.testing.assert_array_equal(actual[index], expected[index])
            assert actual[index].dtype == expected[index].dtype
        np.testing.assert_array_equal(
            actual[[0, len(actual) - 1]], expected[[0, len(expected) - 1]]
        )


def test_raw_supervised_targets_and_missing_label_history_match():
    native, shared = _datasets()
    model = SequenceForecastModel(d_feat=2, d_model=8, nhead=2, num_layers=1, num_threads=1)
    for segment in ["train", "valid", "test"]:
        expected, targets = model._supervised(native, segment)
        actual, actual_targets = model._supervised(shared, segment)
        pd.testing.assert_index_equal(actual.get_index(), expected.get_index())
        np.testing.assert_array_equal(actual_targets, targets)
        for index in range(len(actual)):
            np.testing.assert_array_equal(actual[index], expected[index])


def test_segments_share_one_factor_array_and_handler_cache_is_unchanged(tmp_path):
    frame, dates = _frame()
    handler = _handler(frame)
    path = tmp_path / "handler.pkl"
    handler.to_pickle(path, dump_all=True)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = SharedTSDatasetH(
        handler=handler,
        segments={"train": [dates[0], dates[5]], "valid": [dates[6], dates[8]]},
        step_len=4,
    )
    first = dataset.prepare("train", col_set="feature", data_key="infer", dtype=np.float32)
    second = dataset.prepare("valid", col_set="feature", data_key="infer", dtype=np.float32)
    assert first.store is second.store
    assert np.shares_memory(first.store.values, handler._infer.to_numpy(copy=False))
    assert handler._learn.columns.get_level_values(0).unique().tolist() == ["label"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_same_seed_training_parameters_and_predictions_are_bit_exact():
    native, shared = _datasets()
    settings = dict(
        d_feat=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dropout=0.0,
        n_epochs=2,
        early_stop=2,
        batch_size=5,
        seed=27,
        num_threads=1,
    )
    first = SequenceForecastModel(**settings).fit(native)
    second = SequenceForecastModel(**settings).fit(shared)
    assert first.evals_result_ == second.evals_result_
    assert first.target_mean_ == second.target_mean_
    assert first.target_scale_ == second.target_scale_
    for key, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[key])
    pd.testing.assert_series_equal(first.predict(native), second.predict(shared), check_exact=True)


def test_setup_data_invalidates_previously_cached_factor_array():
    _, dataset = _datasets()
    before = dataset.prepare("test", col_set="feature", data_key="infer", dtype=np.float32)
    old_value = before[0][-1, 0]
    dataset.handler._infer = dataset.handler._infer.copy(deep=True)
    dataset.handler._infer.loc[:, ("feature", "x")] += 10
    dataset.setup_data()
    after = dataset.prepare("test", col_set="feature", data_key="infer", dtype=np.float32)
    assert before.store is not after.store
    assert after[0][-1, 0] == pytest.approx(old_value + 10)
