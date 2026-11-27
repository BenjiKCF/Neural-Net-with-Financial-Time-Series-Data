import numpy as np
import pandas as pd
import pytest
import torch
from qlib.data.dataset import TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP

from financial_ts.cross_sectional import CrossSectionalForecastModel, DailyPanel


class PanelDataset:
    def __init__(self):
        dates = pd.bdate_range("2024-01-02", periods=9)
        index = pd.MultiIndex.from_product(
            [dates, ["ZZZ", "AAA", "MID"]], names=["datetime", "instrument"]
        )
        rng = np.random.default_rng(2)
        x = rng.normal(size=(len(index), 3))
        self.frame = pd.DataFrame(
            np.column_stack([x, 0.01 * x[:, 0] - 0.005 * x[:, 1]]),
            index=index,
            columns=pd.MultiIndex.from_tuples(
                [("feature", "x"), ("feature", "y"), ("feature", "z"), ("label", "return")]
            ),
        )
        self.segments = {
            "train": (dates[0], dates[3]),
            "valid": (dates[4], dates[5]),
            "test": (dates[6], dates[8]),
        }
        self.calls = []

    def prepare(self, segment, col_set, data_key, dtype=None):
        self.calls.append((segment, col_set, data_key))
        if col_set == "feature" and data_key != DataHandlerLP.DK_I:
            raise AssertionError("Daily context must use inference features")
        frame = self.frame
        if col_set == "label":
            frame = frame.loc[frame[("label", "return")].notna()]
        start, end = self.segments[segment]
        return TSDataSampler(
            frame[col_set].copy(), start=start, end=end, step_len=4, dtype=dtype, fillna_type="none"
        )


def small_model(**kwargs):
    return CrossSectionalForecastModel(
        **(
            dict(
                d_feat=3,
                d_model=8,
                nhead=2,
                num_layers=1,
                n_epochs=2,
                early_stop=2,
                seed=3,
                device="cpu",
                num_threads=1,
                dropout=0.0,
            )
            | kwargs
        )
    )


def test_daily_groups_have_stable_ids_and_keep_membership_changes():
    data = PanelDataset()
    data.frame = data.frame.drop(index=(data.segments["train"][0], "MID"))
    sampler = data.prepare("train", "feature", DataHandlerLP.DK_I)
    panel = DailyPanel(sampler, 3)
    assert panel[0]["instruments"] == ("AAA", "ZZZ")
    assert panel[1]["instruments"] == ("AAA", "MID", "ZZZ")
    for i in range(len(panel)):
        batch = panel[i]
        assert (
            len(set(sampler.get_index().take(batch["positions"]).get_level_values("datetime"))) == 1
        )


@pytest.mark.parametrize("architecture", ["daily_transformer", "master_style", "quantformer_moe"])
@pytest.mark.parametrize("objective", ["huber", "ic", "ranking"])
def test_models_fit_raw_targets_reproduce_and_predict_without_labels(architecture, objective):
    data = PanelDataset()
    model = small_model(architecture=architecture, objective=objective).fit(data)
    predictions = model.predict(data)
    assert len(predictions) == 9 and np.isfinite(predictions).all()
    assert model.best_epoch_ in {1, 2}
    expected_mse = (
        ((model.predict(data, "valid") - data.frame[("label", "return")]) ** 2).dropna().mean()
    )
    assert model.evals_result_["valid"][model.best_epoch_ - 1] == pytest.approx(
        expected_mse, abs=1e-8
    )
    assert model.training_receipt_["stock_sampling"] == "none"
    data.frame = data.frame.drop(columns="label", level=0)
    data.calls.clear()
    pd.testing.assert_series_equal(model.predict(data), predictions)
    assert data.calls == [("test", "feature", DataHandlerLP.DK_I)]
    restored = CrossSectionalForecastModel(**model.constructor_config())
    restored.model = restored._build_network()
    restored.model.load_state_dict(model.model.state_dict())
    restored.target_mean_, restored.target_scale_ = model.target_mean_, model.target_scale_
    restored.fitted = True
    pd.testing.assert_series_equal(restored.predict(data), predictions)


def test_missing_label_never_deletes_stock_from_attention_context():
    data = PanelDataset()
    date = data.segments["valid"][0]
    data.frame.loc[(date, "AAA"), ("label", "return")] = np.nan
    panel = small_model()._panel(data, "valid", supervised=True)
    batch = panel[0]
    assert batch["instruments"] == ("AAA", "MID", "ZZZ")
    assert torch.isnan(batch["targets"][0])
    assert batch["features"].shape[0] == 3
    model = small_model().fit(data)
    assert len(model.predict(data, "valid")) == 6


def test_future_data_and_labels_cannot_change_past_predictions_or_training_scale():
    data = PanelDataset()
    model = small_model().fit(data)
    expected = model.predict(data, "train")
    test_start = data.segments["test"][0]
    data.frame.loc[(slice(test_start, None), slice(None)), :] += 1000
    pd.testing.assert_series_equal(model.predict(data, "train"), expected)
    second = small_model().fit(data)
    assert model.target_mean_ == second.target_mean_
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[key], atol=0, rtol=0)


def test_pretraining_uses_only_train_dates_and_cost_surrogate_is_explicit():
    model = small_model(pretrain_epochs=1, turnover_weight=0.001).fit(PanelDataset())
    assert len(model.pretraining_receipt_["losses"]) == 1
    assert np.isfinite(model.pretraining_receipt_["losses"]).all()
    assert model.pretraining_receipt_["uses_validation_or_test_features"] is False
    assert model.pretraining_receipt_["last_anchor"] == "2024-01-05"
    assert "not executed cost" in model.training_receipt_["turnover_assumption"]
    history = model.objective_history_
    for i, total in enumerate(history["total_optimized_loss"]):
        assert total == pytest.approx(
            history["base_forecast_loss"][i]
            + history["routing_penalty"][i]
            + history["turnover_penalty"][i],
            abs=1e-6,
        )
        assert history["train"][i] == total
    assert any(value > 0 for value in history["turnover_penalty"])
    assert model.training_receipt_["validation_selection_includes_regularizers"] is False


def test_constructor_rejects_attempt_to_mislabel_temporal_control():
    with pytest.raises(ValueError, match="temporal-only"):
        small_model(
            architecture="daily_transformer", architecture_kwargs={"cross_stock_attention": True}
        )
