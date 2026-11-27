"""Public command overrides for the production roadmap models."""

import pytest
import yaml

from financial_ts.cli import main
from financial_ts.workflow import load_config, run_workflow


@pytest.mark.parametrize("name", ["daily_transformer", "master_style", "quantformer_moe"])
def test_train_cli_forwards_device_to_cross_sectional_models(tmp_path, monkeypatch, name):
    config = tmp_path / "config.yaml"
    config.write_text(f"model:\n  name: {name}\n  params:\n    device: auto\n")
    seen = []
    monkeypatch.setattr("financial_ts.workflow.run_workflow", lambda value: seen.append(value))
    monkeypatch.setattr("financial_ts.cli._display", lambda value: None)
    assert main(["train", "--config", str(config), "--device", "cpu"]) == 0
    assert len(seen) == 1
    assert seen[0]["model"]["params"]["device"] == "cpu"


@pytest.mark.parametrize("enabled", [True, False])
def test_train_cli_accepts_boolean_bundle_export(tmp_path, monkeypatch, enabled):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"export_inference_bundle": enabled}), encoding="utf-8")
    seen = []
    monkeypatch.setattr("financial_ts.workflow.run_workflow", lambda value: seen.append(value))
    monkeypatch.setattr("financial_ts.cli._display", lambda value: None)
    assert main(["train", "--config", str(config)]) == 0
    assert seen == [{"export_inference_bundle": enabled}]


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", "true", []])
def test_bundle_export_rejects_non_boolean_values_before_workflow_io(tmp_path, invalid):
    output = tmp_path / "run"
    config = {"output_dir": str(output), "export_inference_bundle": invalid}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="export_inference_bundle must be boolean"):
        load_config(path)
    with pytest.raises(ValueError, match="export_inference_bundle must be boolean"):
        run_workflow(config)
    assert not output.exists()
