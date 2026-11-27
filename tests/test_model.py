"""Model regressions using real Qlib samplers, without market downloads."""

import numpy as np
import pandas as pd
import pytest
import torch
from qlib.data.dataset import TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP

from financial_ts.model import TemporalTransformerModel


class SmallDataset:
    def __init__(self):
        dates = pd.date_range("2024-01-01", periods=6)
        index = pd.MultiIndex.from_product(
            [dates, ["AAA", "BBB"]], names=["datetime", "instrument"]
        )
        feature = np.arange(len(index) * 2, dtype=np.float64).reshape(-1, 2) / 10
        self.frame = pd.DataFrame(
            np.column_stack([feature, 0.002 * feature[:, 0] - 0.001 * feature[:, 1]]),
            index=index,
            columns=pd.MultiIndex.from_tuples(
                [("feature", "x"), ("feature", "y"), ("label", "return")]
            ),
        )
        self.segments = {
            "train": (dates[0], dates[2]),
            "valid": (dates[3], dates[3]),
            "test": (dates[4], dates[5]),
        }
        self.calls = []

    def prepare(self, segment, col_set, data_key, dtype=None):
        self.calls.append((segment, col_set, data_key))
        if data_key == DataHandlerLP.DK_I and col_set != "feature":
            raise AssertionError("Future labels are unavailable at inference")
        start, end = self.segments[segment]
        return TSDataSampler(
            self.frame[col_set].copy(),
            start=start,
            end=end,
            step_len=4,
            fillna_type="none",
            dtype=dtype,
        )


class LabelFilteredDataset(SmallDataset):
    """Reproduce Qlib DropnaLabel's different learning and inference views."""

    def prepare(self, segment, col_set, data_key, dtype=None):
        self.calls.append((segment, col_set, data_key))
        start, end = self.segments[segment]
        frame = self.frame
        if data_key == DataHandlerLP.DK_L:
            frame = frame.loc[frame[("label", "return")].notna()]
        elif col_set != "feature":
            raise AssertionError("Inference feature windows must not inspect labels")
        return TSDataSampler(
            frame[col_set].copy(),
            start=start,
            end=end,
            step_len=4,
            fillna_type="none",
            dtype=dtype,
        )


def small_model(**overrides):
    config = dict(
        d_feat=2,
        d_model=8,
        nhead=2,
        num_layers=1,
        dropout=0.0,
        n_epochs=2,
        early_stop=2,
        batch_size=3,
        lr=0.01,
        device="cpu",
        num_threads=1,
        seed=23,
    )
    return TemporalTransformerModel(**(config | overrides))


def test_small_validation_singleton_prediction_and_original_index():
    data = SmallDataset()
    results = {}
    model = small_model().fit(data, evals_result=results)
    predictions = model.predict(data)
    expected_index = data.prepare("test", "feature", DataHandlerLP.DK_I).get_index()
    # Two validation rows are smaller than a batch; four test rows leave a singleton.
    assert len(predictions) == 4
    assert np.isfinite(predictions).all()
    pd.testing.assert_index_equal(predictions.index, expected_index)
    assert model.fitted and model.best_epoch_ in {1, 2}
    assert results == model.evals_result_
    assert all(np.isfinite(results[name]).all() for name in ["train", "valid"])


def test_inference_never_fetches_labels_or_copies_padding_from_later_rows():
    data = SmallDataset()
    model = small_model().fit(data)
    first = model.predict(data, segment="train")
    # Delete every label, including history: feature-only inference must still work.
    data.frame = data.frame.drop(columns="label", level=0)
    data.calls.clear()
    captured = []
    hook = model.model.register_forward_pre_hook(lambda module, args: captured.append(args))
    try:
        second = model.predict(data, segment="train")
    finally:
        hook.remove()
    pd.testing.assert_series_equal(first, second)
    assert data.calls == [("train", "feature", DataHandlerLP.DK_I)]
    # The first sample has no three earlier dates. Padding remains zero and masked.
    features, padding = captured[0]
    assert torch.equal(features[0, :3], torch.zeros(3, 2))
    assert padding[0, :3].all()
    assert not padding[0, -1]


def test_target_scaling_uses_training_labels_only():
    data = SmallDataset()
    start, end = data.segments["train"]
    expected = data.frame.loc[(slice(start, end), slice(None)), ("label", "return")].to_numpy()
    valid_start, valid_end = data.segments["valid"]
    data.frame.loc[(slice(valid_start, valid_end), slice(None)), ("label", "return")] += 1.0
    model = small_model().fit(data)
    assert model.target_mean_ == pytest.approx(expected.mean())
    assert model.target_scale_ == pytest.approx(expected.std(ddof=0))


def test_supervised_windows_keep_observed_history_with_unavailable_historical_label():
    data = LabelFilteredDataset()
    missing = (pd.Timestamp("2024-01-02"), "AAA")
    anchor = (pd.Timestamp("2024-01-03"), "AAA")
    data.frame.loc[missing, ("label", "return")] = np.nan
    model = small_model()
    windows, targets = model._supervised(data, "train")
    assert data.calls == [
        ("train", "feature", DataHandlerLP.DK_I),
        ("train", "label", DataHandlerLP.DK_L),
    ]
    # The missing target removes only its own training endpoint.
    assert missing not in windows.get_index()
    assert len(targets) == len(windows) == 5
    assert np.isfinite(targets).all()
    current = windows[windows.get_index().get_loc(anchor)]
    observed_previous_row = data.frame.loc[missing, "feature"].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(current[-2], observed_previous_row)
    inference = data.prepare("train", "feature", DataHandlerLP.DK_I, dtype=np.float32)
    np.testing.assert_array_equal(current, inference[inference.get_index().get_loc(anchor)])
    # Demonstrate the old DK_L implementation would have lost this valid history.
    old = data.prepare("train", "feature", DataHandlerLP.DK_L, dtype=np.float32)
    assert np.isnan(old[old.get_index().get_loc(anchor)][-2]).all()


def test_label_filtered_fit_validation_matches_inference_on_available_targets():
    data = LabelFilteredDataset()
    data.frame.loc[(pd.Timestamp("2024-01-02"), "AAA"), ("label", "return")] = np.nan
    model = small_model().fit(data)
    prediction = model.predict(data, segment="valid")
    raw_label = data.frame[("label", "return")].reindex(prediction.index)
    inference_mse = ((prediction - raw_label) ** 2).mean()
    assert min(model.evals_result_["valid"]) == pytest.approx(inference_mse, rel=1e-6, abs=1e-10)


def test_repeated_seeded_cpu_training_is_reproducible():
    data = SmallDataset()
    first = small_model().fit(data).predict(data)
    second = small_model().fit(data).predict(data)
    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize("segment", ["train", "valid"])
def test_nonfinite_learning_labels_fail_with_actionable_error(segment):
    data = SmallDataset()
    start, _ = data.segments[segment]
    data.frame.loc[(start, "AAA"), ("label", "return")] = np.inf
    with pytest.raises(ValueError, match=f"{segment} segment contains non-finite labels"):
        small_model().fit(data)


def test_explicit_unavailable_cuda_fails_early(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        small_model(device="cuda")


def test_feature_count_mismatch_has_clear_error():
    with pytest.raises(ValueError, match="match d_feat"):
        small_model(d_feat=3).fit(SmallDataset())
