"""The large execution grid cannot be reduced, duplicated or accepted by counters alone."""

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from financial_ts import campaign_control as control
from financial_ts import rolling
from financial_ts.data import create_demo_csv, import_csv


def _spec():
    return {
        "region": "us",
        "instruments": "all",
        "models": ["zero", "ridge"],
        "seeds": [7, 42],
        "epochs": 1,
        "warmup_sessions": 60,
        "rolling": {"train_sessions": 30, "valid_sessions": 10, "test_sessions": 5, "max_folds": 1},
        "backtest": {"enabled": False},
        "export_inference_bundle": False,
        "bootstrap": {"block_lengths": [2], "draws": 100, "seed": 11},
    }


@pytest.fixture
def campaign(tmp_path):
    provider = tmp_path / "provider"
    (provider / "calendars").mkdir(parents=True)
    dates = pd.bdate_range("2020-01-01", periods=150)
    (provider / "calendars/day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")))
    return rolling.prepare_rolling_campaign(provider, tmp_path / "campaign", _spec())


def test_all_shards_cover_exact_declared_trials_once(campaign):
    shards = [control.trial_partition(campaign, i, 3) for i in range(3)]
    ids = [trial["id"] for shard in shards for trial in shard]
    assert len(ids) == len(set(ids)) == 4
    assert set(ids) == {trial["id"] for trial in campaign["trials"]}
    for index, count in ((True, 3), (0, 0), (-1, 2), (2, 2), (0, 5)):
        with pytest.raises(ValueError):
            control.trial_partition(campaign, index, count)


def test_full_recipe_hash_binds_the_original_budget():
    recipe = json.loads((Path(__file__).parents[1] / "configs/rolling_full.json").read_text())
    assert rolling._digest(recipe) == control.FULL_SPECIFICATION_SHA256
    recipe["rolling"]["train_sessions"] = 30
    assert rolling._digest(recipe) != control.FULL_SPECIFICATION_SHA256


def test_pending_and_unverified_counter_cannot_pass(campaign):
    status = control.campaign_status(campaign)
    assert status["expected_trials"] == 4
    assert status["counts"] == {"pending": 4, "running": 0, "failed": 0, "completed": 0}
    assert status["verified_completed_trials"] == 0
    assert status["full_campaign_execution_gate"] == "pending"
    assert not status["promotion_ready"] and not status["prospective_confirmation"]


def _receipt(campaign, trial, *, status="completed", artifacts=None):
    path = Path(campaign["output_dir"]) / "trials" / trial["id"] / "attempt-0001"
    rolling._save(
        path / "receipt.json",
        {
            "protocol_sha256": campaign["protocol_sha256"],
            "trial_sha256": rolling._digest(trial),
            "trial_id": trial["id"],
            "status": status,
            "artifacts": artifacts or {},
            "attempt": 1,
            "elapsed_seconds": 0.0,
        },
    )
    return path


def test_completed_receipt_without_evidence_is_not_acceptance(campaign):
    trial = campaign["trials"][0]
    _receipt(campaign, trial)
    assert control.campaign_status(campaign)["counts"]["completed"] == 1
    with pytest.raises(ValueError, match="missing required evidence"):
        control.campaign_status(campaign, verify_artifacts=True)


def test_foreign_receipt_cannot_count_toward_declared_grid(campaign):
    trial = campaign["trials"][0]
    path = _receipt(campaign, trial)
    receipt = json.loads((path / "receipt.json").read_text())
    receipt["trial_id"] = "another/candidate/seed-1"
    rolling._save(path / "receipt.json", receipt)
    with pytest.raises(ValueError, match="identity"):
        control.campaign_status(campaign)


def test_freeze_reuse_and_tamper_detection(tmp_path):
    destination = tmp_path / "runtime"
    first = control.freeze_runtime(destination, _spec())
    assert control.freeze_runtime(destination, _spec()) == first
    changed = copy.deepcopy(_spec())
    changed["seeds"] = [42]
    with pytest.raises(ValueError, match="different identity"):
        control.freeze_runtime(destination, changed)
    path = destination / "src/financial_ts/rolling.py"
    path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="source integrity"):
        control.verify_runtime(destination)


def test_injected_runtime_source_is_detected(tmp_path):
    destination = tmp_path / "runtime"
    control.freeze_runtime(destination, _spec())
    (destination / "src/financial_ts/new_file.py").write_text("pass\n")
    with pytest.raises(ValueError, match="source integrity"):
        control.verify_runtime(destination)


def test_interrupted_directory_kept_and_failed_receipt_recovered(campaign, monkeypatch):
    trial = campaign["trials"][0]
    attempt = Path(campaign["output_dir"]) / "trials" / trial["id"] / "attempt-0001"
    attempt.mkdir(parents=True)
    (attempt / "partial_evidence.txt").write_text("preserve me")
    monkeypatch.setattr(rolling, "run_workflow", lambda _: pytest.fail("retry not authorized"))
    rolling.run_rolling_campaign(campaign, trial_ids=[trial["id"]], summarize=False)
    receipt = json.loads((attempt / "receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["started_at_utc"] is None
    assert (attempt / "partial_evidence.txt").read_text() == "preserve me"
    assert control.campaign_status(campaign)["preserved_failed_attempts"] == 1


def test_missing_attempt_history_cannot_pass_audit(campaign):
    path = _receipt(campaign, campaign["trials"][0])
    path.rename(path.with_name("attempt-0002"))
    with pytest.raises(ValueError, match="contiguous numbered chain"):
        control.campaign_status(campaign, verify_artifacts=True)


def test_previous_attempt_pointer_is_audited(campaign):
    path = _receipt(campaign, campaign["trials"][0], status="failed")
    receipt = json.loads((path / "receipt.json").read_text())
    receipt["previous_attempt"] = "invented/previous/attempt-0000"
    rolling._save(path / "receipt.json", receipt)
    with pytest.raises(ValueError, match="attempt chain"):
        control.campaign_status(campaign, verify_artifacts=True)


def test_nonterminal_earlier_attempt_cannot_disappear_from_failure_count(campaign):
    path = _receipt(campaign, campaign["trials"][0], status="running")
    second = path.with_name("attempt-0002")
    receipt = json.loads((path / "receipt.json").read_text())
    receipt.update(
        status="failed", attempt=2, previous_attempt=str(path.relative_to(campaign["output_dir"]))
    )
    rolling._save(second / "receipt.json", receipt)
    with pytest.raises(ValueError, match="terminal interruption receipt"):
        control.campaign_status(campaign, verify_artifacts=True)


def test_abandoned_running_attempt_preserves_receipt_and_counts_failure(campaign, monkeypatch):
    trial = campaign["trials"][0]
    path = _receipt(campaign, trial, status="running")
    original = (path / "receipt.json").read_bytes()
    monkeypatch.setattr(
        rolling, "run_workflow", lambda _: pytest.fail("must not run without retry")
    )
    rolling.run_rolling_campaign(campaign, trial_ids=[trial["id"]], summarize=False)
    receipt = json.loads((path / "receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert (path / "receipt.before-recovery.json").read_bytes() == original
    assert receipt["interruption_recovery"]["previous_receipt_sha256"] == rolling._sha256(
        path / "receipt.before-recovery.json"
    )
    assert (
        control.campaign_status(campaign, verify_artifacts=True)["preserved_failed_attempts"] == 1
    )
    (path / "receipt.before-recovery.json").write_text("{}")
    with pytest.raises(ValueError, match="recovery evidence"):
        control.campaign_status(campaign, verify_artifacts=True)


def _merge_module():
    path = Path(__file__).parents[1] / "scripts/merge_campaign_shards.py"
    spec = importlib.util.spec_from_file_location("merge_campaign_shards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _restore_module():
    path = Path(__file__).parents[1] / "scripts/restore_campaign_provider.py"
    spec = importlib.util.spec_from_file_location("restore_campaign_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata_campaign(campaign, tmp_path):
    provider = Path(campaign["provider_uri"])
    metadata = {
        "source": "synthetic/source",
        "region": "us",
        "synthetic": True,
        "retrieved_at_utc": "2026-11-26T18:00:00+00:00",
        "release_tag": "fixture-v1",
        "validation": {"archive_sha256": "source-bytes-v1"},
        "assets": {
            "archive": {
                "id": 1,
                "digest": "source-bytes-v1",
                "download_count": 10,
                "uploader": {"login": "fixture-author"},
            }
        },
    }
    (provider / "metadata.json").write_text(json.dumps(metadata, indent=3) + "\n")
    original = tmp_path / "registered-provider-metadata.json"
    original.write_bytes((provider / "metadata.json").read_bytes())
    protocol = rolling.prepare_rolling_campaign(provider, tmp_path / "metadata-campaign", _spec())
    return provider, metadata, original, protocol


def test_cache_miss_restores_exact_manifest_and_preserves_fresh_retrieval(campaign, tmp_path):
    provider, metadata, original, protocol = _metadata_campaign(campaign, tmp_path)
    fresh = copy.deepcopy(metadata)
    fresh["retrieved_at_utc"] = "2026-11-27T18:00:00+00:00"
    fresh["assets"]["archive"]["download_count"] += 80
    fresh["assets"]["archive"]["uploader"]["display_name"] = "Current profile field"
    (provider / "metadata.json").write_text(json.dumps(fresh, indent=2))
    observed = (provider / "metadata.json").read_bytes()
    result = _restore_module().restore_provider(
        protocol, provider, original, tmp_path / "retrievals"
    )
    assert result["manifest_restored"] and result["provider_verified"]
    assert (provider / "metadata.json").read_bytes() == original.read_bytes()
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["observed_metadata_text"].encode() == observed
    assert receipt["observed_retrieved_at_utc"] != receipt["registered_retrieved_at_utc"]
    assert receipt["registered_metadata_sha256"] == protocol["data_manifest_sha256"]
    again = _restore_module().restore_provider(
        protocol, provider, original, tmp_path / "retrievals"
    )
    assert not again["manifest_restored"]


def test_cache_miss_rejects_other_archive_or_changed_provider_bytes(campaign, tmp_path):
    provider, metadata, original, protocol = _metadata_campaign(campaign, tmp_path)
    metadata["validation"]["archive_sha256"] = "wrong-source"
    (provider / "metadata.json").write_text(json.dumps(metadata))
    before = (provider / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="source, archive or semantic"):
        _restore_module().restore_provider(protocol, provider, original, tmp_path / "retrievals")
    assert (provider / "metadata.json").read_bytes() == before
    (provider / "metadata.json").write_bytes(original.read_bytes())
    with (provider / "calendars/day.txt").open("a") as stream:
        stream.write("\n2099-01-01")
    with pytest.raises(ValueError, match="provider bytes"):
        _restore_module().restore_provider(protocol, provider, original, tmp_path / "retrievals")
    assert not list((tmp_path / "retrievals").glob("retrieval-*.json"))


def test_incomplete_shard_returns_failure_for_remote_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(
        control,
        "run_campaign_queue",
        lambda *args, **kwargs: {
            "invocation_status": "finished",
            "invocation_failed": True,
            "invocation_incomplete_trials": ["fold-000/ridge/seed-7"],
        },
    )
    assert control.main(["_run", "--output", str(tmp_path)]) == 1
    monkeypatch.setattr(
        control,
        "run_campaign_queue",
        lambda *args, **kwargs: {
            "invocation_status": "stopped_after_active",
            "invocation_failed": False,
            "invocation_incomplete_trials": ["fold-000/ridge/seed-7"],
        },
    )
    assert control.main(["_run", "--output", str(tmp_path)]) == 0


def test_shard_merge_preserves_verified_interruption_transition(campaign, monkeypatch, tmp_path):
    root = Path(campaign["output_dir"])
    trial = campaign["trials"][0]
    path = _receipt(campaign, trial, status="running")
    original = (path / "receipt.json").read_bytes()
    inputs = tmp_path / "inputs"
    shutil.copytree(root / "trials", inputs / "old/trials")
    monkeypatch.setattr(rolling, "run_workflow", lambda _: pytest.fail("no retry"))
    rolling.run_rolling_campaign(campaign, trial_ids=[trial["id"]], summarize=False)
    shutil.copytree(root / "trials", inputs / "new/trials")
    (path / "receipt.json").write_bytes(original)
    (path / "receipt.before-recovery.json").unlink()
    result = _merge_module().merge_shards(inputs, root)
    assert result["artifacts"] == 2
    assert json.loads((path / "receipt.json").read_text())["status"] == "failed"
    assert (path / "receipt.before-recovery.json").read_bytes() == original
    assert (
        control.campaign_status(campaign, verify_artifacts=True)["preserved_failed_attempts"] == 1
    )
    assert _merge_module().merge_shards(inputs, root)["files_added"] == 0


def test_shard_merge_rejects_ordinary_conflict_before_writing(campaign, tmp_path):
    root = Path(campaign["output_dir"])
    _receipt(campaign, campaign["trials"][0], status="failed")
    inputs = tmp_path / "inputs"
    shutil.copytree(root / "trials", inputs / "one/trials")
    shutil.copytree(root / "trials", inputs / "two/trials")
    changed = next((inputs / "two").rglob("receipt.json"))
    receipt = json.loads(changed.read_text())
    receipt["error"] = "different unrelated evidence"
    rolling._save(changed, receipt)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Conflicting shard receipt"):
        _merge_module().merge_shards(inputs, root)
    assert all(Path(p).read_bytes() == b for p, b in before.items())


@pytest.mark.integration
def test_real_process_queue_preserves_frozen_identity_and_resumes(tmp_path):
    provider = tmp_path / "provider"
    source = create_demo_csv(tmp_path / "prices.csv", n_instruments=3, n_days=150)
    metadata = import_csv(source, provider)
    metadata["synthetic"] = True
    (provider / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "queue"
    registered = control.register_campaign(provider, output, _spec())
    assert registered["counts"]["pending"] == 4
    descriptor = json.loads((output / "execution.json").read_text())
    environment = control._environment(descriptor["runtime"])
    control.request_stop(output)
    paused_command = [
        sys.executable,
        "-m",
        "financial_ts.campaign_control",
        "_run",
        "--output",
        str(output),
    ]
    subprocess.run(
        paused_command, env=environment, check=True, capture_output=True, text=True, timeout=30
    )
    assert control.campaign_status(output)["counts"]["pending"] == 4
    command = [
        sys.executable,
        "-m",
        "financial_ts.campaign_control",
        "_run",
        "--output",
        str(output),
        "--workers",
        "2",
        "--clear-stop",
        "--max-trials",
        "2",
    ]
    subprocess.run(
        command, env=environment, check=True, capture_output=True, text=True, timeout=120
    )
    partial = control.campaign_status(output, verify_artifacts=True)
    assert partial["counts"]["completed"] == 2 and partial["counts"]["pending"] == 2
    receipts = {str(p): p.read_bytes() for p in (output / "trials").rglob("receipt.json")}
    subprocess.run(
        command, env=environment, check=True, capture_output=True, text=True, timeout=120
    )
    complete = control.campaign_status(output, verify_artifacts=True)
    assert complete["counts"]["completed"] == complete["verified_completed_trials"] == 4
    assert (
        complete["full_campaign_execution_gate"] == "pending"
    )  # Tiny fixture is never full-budget evidence.
    assert all(Path(path).read_bytes() == data for path, data in receipts.items())
    assert not (output / "comparison.json").exists()  # Workers do not race report writers.
    assert len(list((output / "control").glob("invocation-*.json"))) == 3
    assert (
        json.loads((output / "protocol.json").read_text())["source_code_sha256"]
        == control.verify_runtime(descriptor["runtime"])["source_code_sha256"]
    )
    summary_command = [
        sys.executable,
        "-m",
        "financial_ts.campaign_control",
        "summarize",
        "--output",
        str(output),
    ]
    subprocess.run(
        summary_command, env=environment, check=True, capture_output=True, text=True, timeout=120
    )
    assert json.loads((output / "comparison.json").read_text())["completed_trials"] == 4
