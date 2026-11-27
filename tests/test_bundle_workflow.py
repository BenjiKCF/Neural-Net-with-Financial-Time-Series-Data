"""Native Qlib training -> numerical bundle -> source-factor inference parity."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from financial_ts.bundles import NumericalPredictor, verify_bundle
from financial_ts.cli import main
from financial_ts.data import create_demo_csv, import_csv
from financial_ts.operations import (
    build_provider_packet,
    register_bundle,
    run_daily_batch,
    set_alias,
)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["ridge", "lightgbm", "transformer", "master_style"])
def test_live_qlib_bundle_replays_original_provider_forecasts(tmp_path, name):
    source = create_demo_csv(tmp_path / "market.csv", n_instruments=4, n_days=130)
    provider = tmp_path / "provider"
    import_csv(source, provider)
    params = (
        {
            "num_boost_round": 3,
            "early_stopping_rounds": 2,
            "min_data_in_leaf": 10,
            "num_threads": 1,
            "verbosity": -1,
        }
        if name == "lightgbm"
        else {}
        if name == "ridge"
        else {
            "n_epochs": 1,
            "early_stop": 1,
            "d_model": 8,
            "nhead": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "num_threads": 1,
            "device": "cpu",
        }
    )
    config = {
        "provider_uri": str(provider),
        "output_dir": str(tmp_path / "run"),
        "region": "us",
        "instruments": "all",
        "seed": 7,
        "warmup_sessions": 60,
        "sequence_length": 8,
        "model": {"name": name, "params": params},
        "backtest": {"enabled": False},
        "export_inference_bundle": True,
    }
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["train", "--config", str(config_path)]) == 0
    bundle = tmp_path / "run/inference_bundle"
    assert verify_bundle(bundle)["schema_version"] == 1
    assert json.loads((bundle / "parity.json").read_text())["status"] == "passed"
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert metadata["market_timezone"] == "America/New_York"
    fit_end = pd.Timestamp(metadata["fit_end"]).tz_convert("America/New_York")
    assert (fit_end.hour, fit_end.minute, fit_end.second) == (23, 59, 59)
    expected = pd.read_csv(tmp_path / "run/predictions.csv", dtype={"instrument": str})
    day = expected.datetime.min()
    packet, features, padding = build_provider_packet(
        provider,
        bundle,
        decision_date=day,
        decision_cutoff=pd.Timestamp(f"{day} 18:00", tz="America/New_York").isoformat(),
        data_available_at=pd.Timestamp(f"{day} 17:00", tz="America/New_York").isoformat(),
    )
    actual = NumericalPredictor(bundle).predict(
        features, padding=padding, feature_names=packet["feature_names"]
    )
    original = (
        expected.loc[expected.datetime == day]
        .set_index("instrument")
        .score.reindex(packet["instruments"])
    )
    np.testing.assert_allclose(actual, original, atol=2e-6, rtol=0)
    assert packet["timezone"] == "America/New_York"
    version = register_bundle(tmp_path / "registry", bundle)["version"]
    set_alias(tmp_path / "registry", "shadow", version)
    receipt = run_daily_batch(
        tmp_path / "registry",
        "shadow",
        tmp_path / "daily",
        packet=packet,
        features=features,
        padding=padding,
    )
    assert receipt["status"] == "ready"
