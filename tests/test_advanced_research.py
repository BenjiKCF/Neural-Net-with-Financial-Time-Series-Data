import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from financial_ts.advanced_research import (
    HorizonBatch,
    HorizonResearchModel,
    calibrate_intervals,
    interval_metrics,
    multi_horizon_targets,
    run_advanced_research,
)


def test_multi_horizon_targets_use_exchange_sessions_and_do_not_fill_suspensions():
    calendar = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08", "2024-01-09"]
    )
    close = pd.DataFrame(
        {"AAA": [10.0, 20.0, 30.0, 40.0, 50.0], "BBB": [10.0, 10.0, np.nan, 15.0, 20.0]},
        index=calendar,
    )
    labels, endpoints = multi_horizon_targets(close, calendar, horizons=(1, 2))
    assert labels.loc[(calendar[0], "AAA"), "return_1"] == 0.5
    assert labels.loc[(calendar[0], "AAA"), "return_2"] == 1.0
    assert endpoints.loc[(calendar[0], "AAA"), "return_1"] == calendar[2]
    assert np.isnan(labels.loc[(calendar[0], "BBB"), "return_1"])
    assert np.isnan(labels.loc[(calendar[1], "BBB"), "return_1"])
    assert labels.loc[(calendar[0], "BBB"), "return_2"] == 0.5
    assert labels.loc[(calendar[-1], "AAA")].isna().all()


def test_historical_membership_is_signal_date_mask_not_future_survivor_filter():
    dates = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame({"OLD": [10.0, 10.0, 10.0, 10.0, 10.0]}, index=dates)
    eligible = pd.DataFrame({"OLD": [True, True, False, False, False]}, index=dates)
    label, _ = multi_horizon_targets(close, dates, horizons=(1,), eligible=eligible)
    assert label.iloc[:2, 0].eq(0).all()
    assert label.iloc[2:, 0].isna().all()


def _batch(day, endpoint, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(3, 4, 2, generator=generator)
    targets = torch.stack([0.01 * features[:, -1, 0], 0.02 * features[:, -1, 1]], dim=-1)
    return HorizonBatch(
        pd.Timestamp(day),
        features,
        torch.zeros(3, 4, dtype=torch.bool),
        targets,
        np.full((3, 2), np.datetime64(endpoint, "ns")),
    )


def test_joint_heads_train_predict_calibrate_and_retain_training_scaler():
    torch.set_num_threads(1)
    train = [_batch("2024-01-02", "2024-01-04", 1), _batch("2024-01-03", "2024-01-05", 2)]
    valid = [_batch("2024-01-08", "2024-01-10", 3)]
    model = HorizonResearchModel(
        horizons=(1, 2),
        epochs=2,
        seed=7,
        network_kwargs={"d_model": 8, "nhead": 2, "num_layers": 1},
    )
    model.fit(train, valid, as_of="2024-01-12")
    output = model.predict(valid[0].features, valid[0].padding)
    assert output["returns"].shape == (3, 2)
    assert output["quantiles"].shape == (3, 2, 3)
    assert (output["quantiles"].diff(dim=-1) > 0).all()
    means = model.target_mean_.clone()
    update_train = [_batch("2024-01-15", "2024-01-17", 4)]
    update_valid = [_batch("2024-01-19", "2024-01-23", 5)]
    result = model.incremental_update(update_train, update_valid, as_of="2024-01-24")
    assert result["validation_loss"] <= result["initial_validation_loss"]
    torch.testing.assert_close(model.target_mean_, means, atol=0, rtol=0)
    assert model.receipt_["updates"][0]["normalizer_refitted"] is False


def test_unmature_or_cross_boundary_labels_are_rejected_before_optimization():
    train = [_batch("2024-01-02", "2024-01-10", 1)]
    valid = [_batch("2024-01-08", "2024-01-10", 2)]
    model = HorizonResearchModel(horizons=(1, 2), epochs=1)
    with pytest.raises(ValueError, match="matured"):
        model.fit(train, valid, as_of="2024-01-09")
    with pytest.raises(ValueError, match="cross the validation"):
        model.fit(train, valid, as_of="2024-01-11")


def test_failed_incremental_update_restores_exact_prior_weights(monkeypatch):
    torch.set_num_threads(1)
    model = HorizonResearchModel(
        horizons=(1, 2), epochs=1, network_kwargs={"d_model": 8, "nhead": 2, "num_layers": 1}
    )
    model.fit(
        [_batch("2024-01-02", "2024-01-04", 1)],
        [_batch("2024-01-08", "2024-01-10", 2)],
        as_of="2024-01-11",
    )
    before = {key: value.clone() for key, value in model.network.state_dict().items()}

    def bad_update(*args, **kwargs):
        with torch.no_grad():
            next(model.network.parameters()).add_(100)
        raise RuntimeError("interrupted update")

    monkeypatch.setattr(model, "_optimize", bad_update)
    with pytest.raises(RuntimeError, match="interrupted"):
        model.incremental_update(
            [_batch("2024-01-15", "2024-01-17", 3)],
            [_batch("2024-01-19", "2024-01-23", 4)],
            as_of="2024-01-24",
        )
    for key, value in before.items():
        torch.testing.assert_close(model.network.state_dict()[key], value, atol=0, rtol=0)


def test_failed_fit_cannot_publish_partial_weights(monkeypatch):
    model = HorizonResearchModel(
        horizons=(1, 2), epochs=1, network_kwargs={"d_model": 8, "nhead": 2, "num_layers": 1}
    )
    train, valid = [_batch("2024-01-02", "2024-01-04", 1)], [_batch("2024-01-08", "2024-01-10", 2)]

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted fit")

    monkeypatch.setattr(model, "_optimize", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        model.fit(train, valid, as_of="2024-01-11")
    with pytest.raises(ValueError, match="Fit"):
        model.predict(valid[0].features, valid[0].padding)


def test_calibration_groups_whole_dates_and_requires_mature_pre_inference_period():
    dates = pd.date_range("2024-01-01", periods=6).repeat(2)
    lower, upper, target = np.zeros((12, 1)), np.ones((12, 1)), np.arange(12).reshape(-1, 1) / 10
    endpoints = (dates + pd.Timedelta(days=1)).to_numpy()[:, None]
    result = calibrate_intervals(
        lower,
        upper,
        target,
        dates=dates,
        label_endpoints=endpoints,
        cutoff="2024-01-08",
        inference_start="2024-01-09",
        coverage=0.8,
    )
    assert result["calibration_dates_per_horizon"] == [6]
    assert result["expansion"] == pytest.approx([0.1])
    assert interval_metrics(lower - 0.1, upper + 0.1, target)[0]["coverage"] == 1.0
    with pytest.raises(ValueError, match="precede"):
        calibrate_intervals(
            lower,
            upper,
            target,
            dates=dates,
            label_endpoints=endpoints,
            cutoff="2024-01-09",
            inference_start="2024-01-09",
        )
    with pytest.raises(ValueError, match="matured"):
        calibrate_intervals(
            lower,
            upper,
            target,
            dates=dates,
            label_endpoints=endpoints,
            cutoff="2024-01-06",
            inference_start="2024-01-09",
        )


def test_insufficient_calibration_dates_do_not_claim_nominal_coverage():
    with pytest.raises(ValueError, match="Insufficient"):
        calibrate_intervals(
            np.zeros((1, 1)),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            dates=["2024-01-01"],
            label_endpoints=[["2024-01-02"]],
            cutoff="2024-01-03",
            inference_start="2024-01-04",
            coverage=0.95,
        )


@pytest.mark.integration
def test_registered_provider_experiment_retains_all_controls_predictions_and_hashes(tmp_path):
    from financial_ts.data import create_demo_csv, import_csv

    source = create_demo_csv(tmp_path / "market.csv", n_instruments=3, n_days=180)
    provider = tmp_path / "provider"
    import_csv(source, provider)
    calendar = pd.DatetimeIndex(
        pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines())
    )
    output = tmp_path / "experiment"
    result = run_advanced_research(
        provider,
        output,
        {
            "region": "us",
            "universe": "all",
            "start": str(calendar[60].date()),
            "horizons": [1, 2],
            "max_instruments": 3,
            "epochs": 1,
            "num_threads": 1,
            "network": {"d_model": 8, "nhead": 2, "num_layers": 1},
            "segment_sessions": {
                "train": 6,
                "valid": 4,
                "calibration": 5,
                "test": 4,
                "update_train": 5,
                "update_valid": 4,
                "post_update_test": 4,
            },
        },
    )
    json.dumps(result, allow_nan=False)
    assert result["status"] == "completed"
    assert set(result["trials"]) == {
        "joint_point",
        "joint_quantile",
        "single_1",
        "single_2",
        "frozen",
        "incremental",
        "full_retrain",
    }
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions.groupby(["datetime", "instrument", "model", "horizon"]).size().max() == 1
    assert {"uncalibrated_lower", "uncalibrated_upper", "lower", "upper"}.issubset(predictions)
    manifest = json.loads((output / "artifacts.json").read_text())
    for filename, digest in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="new or empty"):
        run_advanced_research(provider, output, {})
