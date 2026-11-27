"""Select an immutable Actions registration and retain operational recovery provenance.

Only a terminal run of this repository's full-campaign workflow can be a recovery
source. This wrapper never registers a replacement experiment or changes its code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

WORKFLOW = ".github/workflows/full-five-seed.yml"
REGISTRATION = "full-five-seed-registration"


def recovery_run_id(explicit, message):
    explicit = (explicit or "").strip()
    markers = re.findall(r"\[resume-five-seed:([1-9][0-9]*)\]", message or "")
    if (message or "").count("[resume-five-seed:") != len(markers) or len(markers) > 1:
        raise ValueError("Use one exact [resume-five-seed:RUN_ID] marker.")
    if explicit and not re.fullmatch(r"[1-9][0-9]*", explicit):
        raise ValueError("resume_run_id must be a positive numeric run ID.")
    if explicit and markers and explicit != markers[0]:
        raise ValueError("Dispatch input and recovery marker name different source runs.")
    return explicit or (markers[0] if markers else "")


def list_items(api_get, path, key):
    items, page = [], 1
    while True:
        batch = api_get(f"{path}?per_page=100&page={page}")[key]
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def _artifacts(api_get, run_id):
    return list_items(api_get, f"actions/runs/{run_id}/artifacts", "artifacts")


def _registration(artifacts):
    matches = [a for a in artifacts if a["name"] == REGISTRATION and not a["expired"]]
    if len(matches) > 1:
        raise ValueError("Ambiguous immutable registration artifacts.")
    return matches[0] if matches else None


def select_registration(api_get, repository, current_run_id, source_run_id=""):
    """Resolve current-rerun or terminal-source custody without replacing evidence."""
    current_run_id = str(current_run_id)
    source_run_id = str(source_run_id or "")
    if not re.fullmatch(r"[1-9][0-9]*", current_run_id):
        raise ValueError("Invalid current run ID.")
    if source_run_id == current_run_id:
        raise ValueError("A run cannot recover itself through the source-run input.")
    current = _registration(_artifacts(api_get, current_run_id))
    source, registration, jobs = None, None, []
    if source_run_id:
        if not re.fullmatch(r"[1-9][0-9]*", source_run_id):
            raise ValueError("Invalid recovery source run ID.")
        source = api_get(f"actions/runs/{source_run_id}")
        if (
            source.get("id") != int(source_run_id)
            or source.get("repository", {}).get("full_name") != repository
            or source.get("head_repository", {}).get("full_name") != repository
            or source.get("path", "").split("@", 1)[0] != WORKFLOW
        ):
            raise ValueError("Recovery source is not this repository's full-campaign workflow.")
        if source.get("status") != "completed":
            raise ValueError(
                "Recovery source is still active; wait for a terminal run to avoid duplicate work."
            )
        registration = _registration(_artifacts(api_get, source_run_id))
        if registration is None:
            raise ValueError("Recovery source has no available immutable registration artifact.")
        jobs = list_items(api_get, f"actions/runs/{source_run_id}/jobs", "jobs")
    selected = current or registration
    counts = {}
    for job in jobs:
        outcome = job.get("conclusion") or job.get("status")
        counts[outcome] = counts.get(outcome, 0) + 1
    return {
        "outputs": {
            "found": str(selected is not None).lower(),
            "current_registration": str(current is not None).lower(),
            "registration_run_id": current_run_id if current else source_run_id,
            "source_run_id": source_run_id,
        },
        "receipt": {
            "schema_version": 1,
            "kind": "github_workflow_operational_recovery",
            "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repository": repository,
            "current_run_id": current_run_id,
            "source_run_id": source_run_id or None,
            "registration_artifact": selected,
            "source_registration_artifact": registration,
            "source_run": source,
            "source_job_counts": counts,
            "source_jobs": [
                {
                    "id": job["id"],
                    "name": job["name"],
                    "status": job["status"],
                    "conclusion": job.get("conclusion"),
                    "html_url": job.get("html_url"),
                    "failed_steps": [
                        s["name"] for s in job.get("steps", []) if s.get("conclusion") == "failure"
                    ],
                    "training_step": next(
                        (
                            s
                            for s in job.get("steps", [])
                            if s.get("name") == "Run the unchanged registered shard"
                        ),
                        None,
                    ),
                }
                for job in jobs
            ],
            "qualification": "Workflow recovery only. Original registration, scientific runtime and existing trial attempts remain authoritative; source workflow failures are not invented trial attempts.",
        },
    }


def shard_artifacts(api_get, current_run_id, source_run_id, shard):
    if not re.fullmatch(r"[0-9]+", str(shard)) or not 0 <= int(shard) < 80:
        raise ValueError("Shard must be in the unchanged 80-shard partition.")
    prefix = f"full-five-seed-shard-{shard}-attempt-"
    result = {}
    for label, run_id in (("current", str(current_run_id)), ("source", str(source_run_id or ""))):
        artifacts = _artifacts(api_get, run_id) if run_id else []
        result[label + "_found"] = str(
            any(a["name"].startswith(prefix) and not a["expired"] for a in artifacts)
        ).lower()
    return result


def _api_get(repository, token):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid repository identity.")

    def get(path):
        request = Request(
            f"https://api.github.com/repos/{repository}/{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        with urlopen(request, timeout=60) as response:
            return json.load(response)

    return get


def _write_outputs(values):
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in values.items():
            if "\n" in str(value) or "\r" in str(value):
                raise ValueError("Unsafe workflow output.")
            stream.write(f"{key}={value}\n")


def _receipt_path(directory):
    run_id, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[1-9][0-9]*", attempt):
        raise ValueError("Invalid workflow run or attempt.")
    return Path(directory) / f"run-{run_id}-attempt-{attempt}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "shards", "bind"))
    parser.add_argument("--receipt-dir", type=Path, default=Path("runs/full-five-seed-recovery"))
    parser.add_argument("--shard")
    args = parser.parse_args()
    if args.command == "bind":
        path = _receipt_path(args.receipt_dir)
        receipt = json.loads(path.read_text())
        protocol = Path("runs/full-five-seed/protocol.json")
        runtime = Path("runs/full-five-seed-runtime/manifest.json")
        receipt["protocol_sha256"] = json.loads(protocol.read_text())["protocol_sha256"]
        receipt["frozen_runtime_sha256"] = json.loads(runtime.read_text())["runtime_sha256"]
        receipt["retained_registration_files"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                protocol,
                runtime,
                Path("runs/full-five-seed/execution.json"),
                Path("runs/full-five-seed-environment.txt"),
                Path("runs/full-five-seed-provider-metadata.json"),
            )
        }
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        return
    get = _api_get(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    if args.command == "select":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        message = (event.get("head_commit") or {}).get("message", "")
        source = recovery_run_id(os.environ.get("RESUME_RUN_ID"), message)
        selected = select_registration(
            get, os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"], source
        )
        path = _receipt_path(args.receipt_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Recovery selection receipt already exists; preserve it.")
        path.write_text(json.dumps(selected["receipt"], indent=2) + "\n")
        _write_outputs(selected["outputs"])
    else:
        _write_outputs(
            shard_artifacts(
                get, os.environ["GITHUB_RUN_ID"], os.environ.get("SOURCE_RUN_ID"), args.shard
            )
        )


if __name__ == "__main__":
    main()
