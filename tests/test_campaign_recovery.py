"""Regression coverage for cached-data failure and preserving an original remote run."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _module(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata():
    return {
        "source": "chenditc/investment_data",
        "release_tag": "2026-11-26",
        "region": "cn",
        "frequency": "day",
        "synthetic": False,
        "retrieved_at_utc": "2026-11-27T10:00:00Z",
    }


@pytest.mark.parametrize("empty_exists", [False, True])
def test_missing_download_then_cache_hit_never_redownloads(tmp_path, empty_exists):
    module = _module("restore_campaign_provider")
    provider = tmp_path / "provider"
    if empty_exists:
        provider.mkdir()
    downloads = []

    def downloader(path, *, release):
        downloads.append((path, release))
        path.mkdir(exist_ok=True)
        (path / "metadata.json").write_text(json.dumps(_metadata()))
        (path / "market.bin").write_bytes(b"immutable market bytes")

    first = module.ensure_provider_available(provider, "2026-11-26", downloader=downloader)
    original = {p.name: p.read_bytes() for p in provider.iterdir()}
    again = module.ensure_provider_available(provider, "2026-11-26", downloader=downloader)
    assert first["downloaded"] and not again["downloaded"]
    assert len(downloads) == 1
    assert original == {p.name: p.read_bytes() for p in provider.iterdir()}
    assert "Availability only" in again["qualification"]


@pytest.mark.parametrize(
    "kind", ["partial", "other_release", "other_source", "synthetic", "symlink"]
)
def test_existing_invalid_cache_is_preserved_and_never_overwritten(tmp_path, kind):
    provider = tmp_path / "provider"
    provider.mkdir()
    metadata = _metadata()
    if kind == "partial":
        (provider / "partial.bin").write_bytes(b"retain")
    elif kind == "symlink":
        original = tmp_path / "metadata.json"
        original.write_text(json.dumps(metadata))
        (provider / "metadata.json").symlink_to(original)
    else:
        key = {"other_release": "release_tag", "other_source": "source", "synthetic": "synthetic"}[
            kind
        ]
        metadata[key] = True if kind == "synthetic" else "changed"
        (provider / "metadata.json").write_text(json.dumps(metadata))
    before = {p.name: p.read_bytes() for p in provider.iterdir()}
    with pytest.raises(ValueError):
        _module("restore_campaign_provider").ensure_provider_available(
            provider, "2026-11-26", downloader=lambda *_a, **_k: pytest.fail("do not redownload")
        )
    assert before == {p.name: p.read_bytes() for p in provider.iterdir()}


def test_actual_workflow_cache_step_accepts_nonempty_provider(tmp_path):
    root = Path(__file__).parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/full-five-seed.yml").read_text())
    commands = []
    for job in ("register", "shard"):
        step = next(
            s
            for s in workflow["jobs"][job]["steps"]
            if s.get("name") == "Reuse cached provider or download missing pinned data"
        )
        commands.append(step["run"])
    assert commands[0] == commands[1]
    provider = tmp_path / ".data/chenditc/2026-11-26"
    provider.mkdir(parents=True)
    (provider / "metadata.json").write_text(json.dumps(_metadata()))
    (provider / "market.bin").write_bytes(b"cached")
    args = commands[0].split()
    args[0] = sys.executable
    args[1] = str(root / args[1])
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        args, cwd=tmp_path, env=environment, capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["downloaded"] is False


def _api_fixture():
    artifact = {
        "id": 17,
        "name": "full-five-seed-registration",
        "expired": False,
        "digest": "sha256:original",
    }
    repo = "owner/project"
    routes = {
        "actions/runs/200/artifacts?per_page=100&page=1": {"artifacts": []},
        "actions/runs/100": {
            "id": 100,
            "repository": {"full_name": repo},
            "head_repository": {"full_name": repo},
            "path": ".github/workflows/full-five-seed.yml",
            "status": "completed",
            "conclusion": "failure",
            "head_sha": "original-source-sha",
            "html_url": "https://github.com/owner/project/actions/runs/100",
        },
        "actions/runs/100/artifacts?per_page=100&page=1": {"artifacts": [artifact]},
        "actions/runs/100/jobs?per_page=100&page=1": {
            "jobs": [
                {
                    "id": 1,
                    "name": "shard (0)",
                    "status": "completed",
                    "conclusion": "failure",
                    "steps": [
                        {
                            "name": "Verify the pinned provider",
                            "status": "completed",
                            "conclusion": "failure",
                        },
                        {
                            "name": "Run the unchanged registered shard",
                            "status": "completed",
                            "conclusion": "skipped",
                        },
                    ],
                }
            ]
        },
    }
    return routes, repo


def test_terminal_recovery_uses_original_registration_and_retains_workflow_failure():
    routes, repo = _api_fixture()
    result = _module("select_campaign_recovery").select_registration(
        routes.__getitem__, repo, "200", "100"
    )
    assert result["outputs"] == {
        "found": "true",
        "current_registration": "false",
        "registration_run_id": "100",
        "source_run_id": "100",
    }
    receipt = result["receipt"]
    assert receipt["registration_artifact"]["id"] == 17
    assert receipt["source_run"]["head_sha"] == "original-source-sha"
    assert receipt["source_job_counts"] == {"failure": 1}
    assert receipt["source_jobs"][0]["training_step"]["conclusion"] == "skipped"


def test_rerun_prefers_current_registration_and_keeps_original_source_provenance():
    routes, repo = _api_fixture()
    current = copy.deepcopy(
        routes["actions/runs/100/artifacts?per_page=100&page=1"]["artifacts"][0]
    )
    current["id"] = 18
    routes["actions/runs/200/artifacts?per_page=100&page=1"]["artifacts"] = [current]
    result = _module("select_campaign_recovery").select_registration(
        routes.__getitem__, repo, "200", "100"
    )
    assert result["outputs"]["registration_run_id"] == "200"
    assert result["outputs"]["current_registration"] == "true"
    assert result["receipt"]["source_registration_artifact"]["id"] == 17


@pytest.mark.parametrize(
    "change", ["active", "foreign_repo", "foreign_head", "wrong_workflow", "missing", "expired"]
)
def test_invalid_recovery_source_cannot_start_a_second_campaign(change):
    routes, repo = _api_fixture()
    run = routes["actions/runs/100"]
    if change == "active":
        run["status"] = "in_progress"
    elif change == "foreign_repo":
        run["repository"]["full_name"] = "elsewhere/repo"
    elif change == "foreign_head":
        run["head_repository"]["full_name"] = "elsewhere/repo"
    elif change == "wrong_workflow":
        run["path"] = ".github/workflows/ci.yml"
    elif change == "missing":
        routes["actions/runs/100/artifacts?per_page=100&page=1"]["artifacts"] = []
    else:
        routes["actions/runs/100/artifacts?per_page=100&page=1"]["artifacts"][0]["expired"] = True
    with pytest.raises(ValueError):
        _module("select_campaign_recovery").select_registration(
            routes.__getitem__, repo, "200", "100"
        )


@pytest.mark.parametrize(
    "value,message",
    [
        ("abc", ""),
        ("0", ""),
        ("", "[resume-five-seed:x]"),
        ("100", "[resume-five-seed:101]"),
        ("", "[resume-five-seed:100] [resume-five-seed:100]"),
    ],
)
def test_invalid_or_ambiguous_recovery_requests_fail(value, message):
    with pytest.raises(ValueError):
        _module("select_campaign_recovery").recovery_run_id(value, message)


def test_exact_recovery_marker_and_dispatch_input():
    parse = _module("select_campaign_recovery").recovery_run_id
    assert parse("", "Fix provider cache [resume-five-seed:100]") == "100"
    assert parse("100", "") == "100"
    assert parse("", "ordinary fix") == ""


def test_shard_recovery_finds_prior_artifacts_without_neighbor_or_expired_matches():
    routes, _ = _api_fixture()
    routes["actions/runs/100/artifacts?per_page=100&page=1"]["artifacts"].extend(
        [
            {"name": "full-five-seed-shard-2-attempt-1", "expired": False},
            {"name": "full-five-seed-shard-20-attempt-1", "expired": False},
        ]
    )
    routes["actions/runs/200/artifacts?per_page=100&page=1"]["artifacts"] = [
        {"name": "full-five-seed-shard-2-attempt-1", "expired": True},
    ]
    module = _module("select_campaign_recovery")
    assert module.shard_artifacts(routes.__getitem__, "200", "100", "2") == {
        "current_found": "false",
        "source_found": "true",
    }
    assert module.shard_artifacts(routes.__getitem__, "200", "100", "3") == {
        "current_found": "false",
        "source_found": "false",
    }
