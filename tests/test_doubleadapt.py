"""Tests of actual data/model meta adaptation and chronological label use."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from financial_ts.doubleadapt import (
    AdaptationBatch,
    AdaptationTask,
    DoubleAdaptModel,
    PrototypeDataAdapter,
    fit_doubleadapt,
    run_doubleadapt_research,
)


def batch(date, endpoint, seed):
    rng = torch.Generator().manual_seed(seed)
    features = torch.randn(5, 3, generator=rng)
    targets = 0.01 * features[:, 0] - 0.02 * features[:, 1] + 0.003 * seed
    index = pd.MultiIndex.from_product([[pd.Timestamp(date)], ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"])
    return AdaptationBatch(index, features, targets, np.full(5, np.datetime64(endpoint, "ns")))


def tasks():
    return [AdaptationTask("train", batch("2024-01-02", "2024-01-04", 1), batch("2024-01-08", "2024-01-10", 2)),
            AdaptationTask("valid", batch("2024-01-15", "2024-01-17", 3), batch("2024-01-22", "2024-01-24", 4)),
            AdaptationTask("online", batch("2024-02-01", "2024-02-05", 5), batch("2024-02-08", "2024-02-12", 6))]


def fitted():
    train, valid, _ = tasks()
    return fit_doubleadapt([train], [valid], as_of="2024-01-25", epochs=2,
                           hidden=4, heads=2, inner_lr=0.03, model_meta_lr=0.01,
                           data_meta_lr=0.01, seed=42)[0]


def test_adapter_identity_and_paper_headwise_inverse_formula():
    adapter = PrototypeDataAdapter(3, heads=2)
    x = torch.tensor([[1., 2., 3.], [-1., -2., -3.]])
    y = torch.tensor([0.1, -0.2])
    torch.testing.assert_close(adapter(x), x)
    torch.testing.assert_close(adapter.transform_labels(x, y), y)
    torch.testing.assert_close(adapter.restore_labels(x, y), y)
    with torch.no_grad():
        adapter.label_log_scale.copy_(torch.log(torch.tensor([2., 4.])))
        adapter.label_bias.copy_(torch.tensor([1., 2.]))
    expected = (adapter.weights(x) * (y[:, None] - torch.tensor([1., 2.])) / torch.tensor([2., 4.])).sum(-1)
    torch.testing.assert_close(adapter.restore_labels(x, y), expected)
    # A mixture of inverse heads is generally not the inverse of their mixture.
    assert not torch.allclose(adapter.restore_labels(x, adapter.transform_labels(x, y)), y)


def test_inner_update_meta_learns_forecaster_features_and_label_adapter():
    torch.set_num_threads(1)
    train = tasks()[0]
    model = DoubleAdaptModel(3, hidden=4, heads=2, seed=7)
    model.fit_normalization([train], as_of="2024-01-11")
    before = copy.deepcopy(model.state_dict())
    receipt = model._meta_step(train, as_of="2024-01-11")
    assert receipt["model_gradient_norm"] > 0
    assert receipt["feature_adapter_gradient_norm"] > 0
    assert receipt["label_adapter_gradient_norm"] > 0
    assert receipt["gradient_mode"] == "exact_second_order_through_inner_update"
    assert not torch.equal(before["forecaster.0.weight"], model.state_dict()["forecaster.0.weight"])
    assert not torch.equal(before["adapter.feature_weight"], model.state_dict()["adapter.feature_weight"])
    assert not torch.equal(before["adapter.label_bias"], model.state_dict()["adapter.label_bias"])
    torch.testing.assert_close(before["label_mean"], model.label_mean, atol=0, rtol=0)


def test_future_query_labels_cannot_change_already_emitted_predictions():
    model = fitted()
    alternative = copy.deepcopy(model)
    online = tasks()[2]
    scores = []
    for candidate in (model, alternative):
        candidate.begin_task(online.support, inference_start="2024-02-08", as_of="2024-02-05")
        scores.append(candidate.predict(online.query.features, online.query.index))
    torch.testing.assert_close(scores[0], scores[1], atol=0, rtol=0)
    changed = copy.deepcopy(online.query)
    changed.targets += 0.2
    model.finish_task(online.query, as_of="2024-02-12")
    alternative.finish_task(changed, as_of="2024-02-12")
    assert model._state_hash() != alternative._state_hash()
    torch.testing.assert_close(scores[0], scores[1], atol=0, rtol=0)


def test_maturity_and_exact_emitted_identity_are_required_before_meta_updates():
    model = fitted()
    online = tasks()[2]
    with pytest.raises(ValueError, match="matured"):
        model.begin_task(online.support, inference_start="2024-02-08", as_of="2024-02-02")
    model.begin_task(online.support, inference_start="2024-02-08", as_of="2024-02-05")
    with pytest.raises(ValueError, match="already emitted"):
        model.finish_task(online.query, as_of="2024-02-12")
    model.predict(online.query.features, online.query.index)
    before = model._state_hash()
    with pytest.raises(ValueError, match="matured"):
        model.finish_task(online.query, as_of="2024-02-09")
    assert before == model._state_hash() and model.pending is not None
    model.finish_task(online.query, as_of="2024-02-12")
    assert model.pending is None


def test_training_scaler_is_immutable_and_not_fitted_on_validation_labels():
    train, valid, _ = tasks()
    model, receipt = fit_doubleadapt([train], [valid], as_of="2024-01-25", epochs=1, hidden=4, heads=2)
    expected = torch.cat([train.support.targets, train.query.targets]).mean()
    torch.testing.assert_close(model.label_mean, expected)
    with pytest.raises(ValueError, match="immutable"):
        model.fit_normalization([valid], as_of="2024-01-25")
    assert receipt["upstream_reproduction"] is False
    with pytest.raises(ValueError, match="precede validation"):
        fit_doubleadapt([valid], [train], as_of="2024-01-25", epochs=1, hidden=4, heads=2)


def test_support_endpoint_crossing_query_is_rejected():
    task = tasks()[0]
    task.support.label_endpoints[:] = np.datetime64("2024-01-08", "ns")
    with pytest.raises(ValueError, match="strictly before"):
        task.validate(as_of="2024-01-11")


def test_finalization_rejects_changed_features_and_future_trained_historical_replay():
    model = fitted()
    online = tasks()[2]
    model.begin_task(online.support, inference_start="2024-02-08", as_of="2024-02-05")
    model.predict(online.query.features, online.query.index)
    altered = copy.deepcopy(online.query)
    altered.features[0, 0] += 1
    with pytest.raises(ValueError, match="features changed"):
        model.finish_task(altered, as_of="2024-02-12")
    model.finish_task(online.query, as_of="2024-02-12")
    with pytest.raises(ValueError, match="historical inference"):
        model.begin_task(online.support, inference_start="2024-02-08", as_of="2024-02-05")


def test_repeated_prediction_cannot_replace_features_even_if_score_is_unchanged():
    model = fitted()
    online = tasks()[2]
    # A constant predictor makes different features produce exactly the same
    # score. Prediction locking must still retain the original inputs.
    with torch.no_grad():
        for parameter in model.forecaster.parameters():
            parameter.zero_()
        model.adapter.label_bias.zero_()
        model.adapter.label_log_scale.zero_()
    model.begin_task(online.support, inference_start="2024-02-08", as_of="2024-02-05")
    scores = model.predict(online.query.features, online.query.index)
    changed = online.query.features.clone()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="features cannot change"):
        model.predict(changed, online.query.index)
    torch.testing.assert_close(model.predict(online.query.features, online.query.index), scores, atol=0, rtol=0)
    assert torch.equal(model.pending["features"][online.query.index[0]], online.query.features[0])


@pytest.mark.integration
def test_registered_doubleadapt_campaign_runs_all_matched_controls(tmp_path):
    from financial_ts.data import create_demo_csv, import_csv

    provider = tmp_path / "provider"
    import_csv(create_demo_csv(tmp_path / "prices.csv", n_instruments=3, n_days=150), provider)
    calendar = pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
    output = tmp_path / "doubleadapt"
    result = run_doubleadapt_research(provider, output, {
        "region": "us", "universe": "all", "start": str(calendar[60].date()),
        "support_sessions": 2, "query_sessions": 2, "train_tasks": 2,
        "valid_tasks": 1, "online_tasks": 2, "max_instruments": 3,
        "epochs": 1, "num_threads": 1, "model": {"hidden": 4, "heads": 2},
        "bootstrap_repetitions": 100, "bootstrap_block_sessions": 2,
    })
    assert result["status"] == "completed" and not result["deployment_approved"]
    json.dumps(result, allow_nan=False)
    assert set(result["models"]) == {"doubleadapt_paper", "frozen", "naive_incremental", "full_retrain"}
    assert result["online_rows_per_model"] == 12
    predictions = pd.read_csv(output / "predictions.csv")
    assert len(predictions) == 48
    assert predictions.groupby("model").size().nunique() == 1
    locks = json.loads((output / "prediction_locks.json").read_text())
    assert len(locks) == 8 and all(not item["query_labels_in_file"] for item in locks)
    for relative, digest in json.loads((output / "artifacts.json").read_text()).items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="new output"):
        run_doubleadapt_research(provider, output, {})
