"""Public command overrides for the production roadmap models."""

import pytest

from financial_ts.cli import main


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
