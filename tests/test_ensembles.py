import numpy as np
import pandas as pd
import pytest

from financial_ts.ensembles import blend_predictions, select_blend


def frame(scores):
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["A", "B", "C"]], names=["datetime", "instrument"]
    )
    return pd.DataFrame({"score": scores, "label": [-0.1, 0.01, 0.2, -0.2, 0.0, 0.1]}, index=index)


def test_blend_selects_validation_and_retains_all_weights():
    left = frame([-0.1, 0.01, 0.2, -0.2, 0.0, 0.1])
    right = frame([0.2, 0.01, -0.1, 0.1, 0.0, -0.2])
    choice = select_blend(left, right)
    assert choice["left_weight"] == 1.0 and not choice["evaluation_used"]
    assert len(choice["history"]) == 5
    # An arbitrarily bad later test never enters selection; chosen weight stays fixed.
    test = blend_predictions(right, left, choice["left_weight"])
    pd.testing.assert_series_equal(test.score, right.score)


def test_no_silent_intersection_or_target_mismatch():
    left = frame(np.zeros(6))
    with pytest.raises(ValueError, match="panels differ"):
        blend_predictions(left, left.iloc[:-1], 0.5)
    right = left.copy()
    right.iloc[0, 1] += 0.1
    with pytest.raises(ValueError, match="labels differ"):
        blend_predictions(left, right, 0.5)


def test_permutations_and_undefined_ic_tie_break():
    left = frame(np.zeros(6))
    right = frame(np.ones(6))
    choice = select_blend(left.iloc[::-1], right, weights=[0, 1])
    assert choice["left_weight"] == 1
    assert choice["history"][0]["metrics"]["rank_ic"] is None
    with pytest.raises(ValueError):
        select_blend(left, right, weights=[0.5, 0.5])


def test_blend_rejects_existing_or_protected_output_before_parent_analysis(tmp_path, monkeypatch):
    from financial_ts import rolling
    from financial_ts.ensembles import run_rolling_blend

    source, provider = tmp_path / "source", tmp_path / "provider"
    source.mkdir()
    provider.mkdir()
    protocol = {"output_dir": str(source), "provider_uri": str(provider)}
    monkeypatch.setattr(rolling, "_load_protocol", lambda value: protocol)
    monkeypatch.setattr(
        rolling,
        "summarize_rolling_campaign",
        lambda *args, **kwargs: pytest.fail("must validate output before parent analysis"),
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="new immutable"):
        run_rolling_blend(source, existing)
    for protected in (source / "new-output", provider / "new-output"):
        with pytest.raises(ValueError, match="separate"):
            run_rolling_blend(source, protected)
        assert not protected.exists()


@pytest.mark.integration
def test_registered_blend_reads_rolling_workflow_paths_and_replays_fixed_control(tmp_path):
    import json
    from pathlib import Path

    from financial_ts.bundles import file_hash
    from financial_ts.data import create_demo_csv, import_csv
    from financial_ts.ensembles import run_rolling_blend
    from financial_ts.rolling import prepare_rolling_campaign, run_rolling_campaign

    source = create_demo_csv(tmp_path / "prices.csv", n_instruments=3, n_days=140)
    provider = tmp_path / "provider"
    metadata = import_csv(source, provider)
    metadata["synthetic"] = True
    (provider / "metadata.json").write_text(json.dumps(metadata))
    spec = {
        "region": "us",
        "instruments": "all",
        "models": ["zero", "ridge"],
        "seeds": [7, 42],
        "warmup_sessions": 60,
        "epochs": 1,
        "rolling": {"train_sessions": 30, "valid_sessions": 10, "test_sessions": 8, "max_folds": 1},
        "backtest": {"enabled": True, "topk": 2, "n_drop": 1, "account": 100000},
        "export_inference_bundle": False,
        "bootstrap": {"block_lengths": [2], "draws": 100, "seed": 11},
        "cost_scenarios": [{"id": "base", "fee_multiplier": 1}],
    }
    protocol = prepare_rolling_campaign(provider, tmp_path / "campaign", spec)
    assert run_rolling_campaign(protocol)["status"] == "completed"
    campaign = Path(protocol["output_dir"])
    # Preserve a historical analysis annotation that regeneration would erase.
    comparison = campaign / "comparison.json"
    original_summary = json.loads(comparison.read_text())
    original_summary["frozen_analysis_marker"] = "preserve original derived report"
    comparison.write_text(json.dumps(original_summary))
    parent_hashes = {
        str(path.relative_to(campaign)): file_hash(path)
        for path in campaign.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "blended"
    result = run_rolling_blend(protocol["output_dir"], output, left="zero", right="ridge")
    assert result["status"] == "completed", result
    assert result["registration"]["seeds"] == [7, 42]
    assert parent_hashes == {
        str(path.relative_to(campaign)): file_hash(path)
        for path in campaign.rglob("*")
        if path.is_file()
    }
    input_audit = output / "input_audit"
    receipt = json.loads((input_audit / "analysis_receipt.json").read_text())
    assert receipt["original_comparison"]["sha256"] == parent_hashes["comparison.json"]
    assert result["registration"]["input_audit"]["analysis_receipt_sha256"] == file_hash(
        input_audit / "analysis_receipt.json"
    )
    fold = result["folds"][0]
    assert set(fold["controls"]) == {"zero", "ridge", "fixed_50_50"}
    assert fold["portfolio"]["status"] == "completed"
    assert len(fold["prediction_lineage"]) == 8
    assert all("/workflow/" in filename for filename in fold["prediction_lineage"])
    assert all(row["portfolio"]["status"] == "completed" for row in fold["controls"].values())
    selected = json.loads((output / "fold-000" / "selection.json").read_text())
    assert selected["selection_split"] == "validation" and not selected["evaluation_used"]
    # Model/config recipes may differ now; raw market identity cannot silently change.
    calendar = provider / "calendars/day.txt"
    calendar.write_text(calendar.read_text() + "\n2020-12-31")
    bad_output = tmp_path / "changed_source"
    with pytest.raises(ValueError, match="provider bytes changed"):
        run_rolling_blend(protocol["output_dir"], bad_output, left="zero", right="ridge")
    assert not bad_output.exists()
    with pytest.raises(ValueError, match="separate"):
        run_rolling_blend(
            protocol["output_dir"],
            Path(protocol["output_dir"]) / "bad",
            left="zero",
            right="ridge",
            replay=False,
        )
