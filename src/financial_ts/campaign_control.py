"""Frozen, resumable process supervision for an unchanged rolling protocol.

The controller changes execution order/concurrency, never the scientific grid.
Each trial remains the rolling runner's independently locked, audited attempt.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from filelock import FileLock

from financial_ts import rolling

FULL_SPECIFICATION_SHA256 = "83b04ee3ee08b3d6e7180a48c529d95c7de4b550c044b8e0982a2b7af9081aa8"


def freeze_runtime(destination, specification):
    """Retain all Python sources counted by rolling's code-identity contract."""
    destination = Path(destination).expanduser().resolve()
    package = Path(rolling.__file__).parent
    sources = {
        str(path.relative_to(package)): rolling._sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    identity = {
        "schema_version": 1,
        "execution_api_version": 2,
        "source_code_sha256": rolling._digest(sources),
        "sources": sources,
        "specification": specification,
        "specification_sha256": rolling._digest(specification),
        "packages": rolling._package_versions(),
    }
    manifest = {**identity, "runtime_sha256": rolling._digest(identity)}
    if destination.exists():
        if verify_runtime(destination) != manifest:
            raise ValueError("Frozen runtime already exists with a different identity.")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent))
    try:
        for relative, digest in sources.items():
            source = package / relative
            target = temporary / "src" / "financial_ts" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if rolling._sha256(target) != digest:
                raise ValueError(
                    "Source changed while freezing runtime; retry before registration."
                )
        if rolling._code_identity() != identity["source_code_sha256"]:
            raise ValueError("Source changed while freezing runtime; retry before registration.")
        rolling._save(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def verify_runtime(destination):
    destination = Path(destination).expanduser().resolve()
    manifest = json.loads((destination / "manifest.json").read_text())
    identity = {k: v for k, v in manifest.items() if k != "runtime_sha256"}
    if rolling._digest(identity) != manifest.get("runtime_sha256"):
        raise ValueError("Frozen runtime manifest integrity failed.")
    root = destination / "src" / "financial_ts"
    observed = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink():
            raise ValueError("Frozen runtime sources must not be symlinks.")
        observed[str(path.relative_to(root))] = rolling._sha256(path)
    if (
        observed != manifest["sources"]
        or rolling._digest(observed) != manifest["source_code_sha256"]
    ):
        raise ValueError("Frozen runtime source integrity failed.")
    if rolling._digest(manifest["specification"]) != manifest["specification_sha256"]:
        raise ValueError("Frozen recipe integrity failed.")
    return manifest


def _descriptor(root):
    root = Path(root).expanduser().resolve()
    descriptor = json.loads((root / "execution.json").read_text())
    manifest = verify_runtime(descriptor["runtime"])
    protocol = rolling._load_protocol(root)
    if (
        descriptor["protocol_sha256"] != protocol["protocol_sha256"]
        or descriptor["runtime_sha256"] != manifest["runtime_sha256"]
        or manifest["source_code_sha256"] != protocol["source_code_sha256"]
        or manifest["specification"] != protocol["specification"]
        or manifest["packages"] != protocol["packages"]
    ):
        raise ValueError("Execution descriptor does not match the frozen protocol/runtime.")
    return root, descriptor, protocol


def _environment(runtime):
    environment = os.environ.copy()
    # Replace inherited source paths; editing the development tree cannot change workers.
    environment["PYTHONPATH"] = str(Path(runtime) / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "2"
    return environment


def register_campaign(provider, output, specification, *, runtime=None):
    output = Path(output).expanduser().resolve()
    runtime = (
        output.with_name(output.name + "-runtime")
        if runtime is None
        else Path(runtime).expanduser().resolve()
    )
    if runtime == output or runtime.is_relative_to(output) or output.is_relative_to(runtime):
        raise ValueError("Frozen runtime and campaign need separate directories.")
    manifest = freeze_runtime(runtime, specification)
    command = [
        sys.executable,
        "-m",
        "financial_ts.campaign_control",
        "_register",
        "--provider",
        str(Path(provider).expanduser().resolve()),
        "--output",
        str(output),
        "--runtime",
        str(runtime),
    ]
    subprocess.run(command, env=_environment(runtime), check=True)
    protocol = rolling._load_protocol(output)
    if protocol["source_code_sha256"] != manifest["source_code_sha256"]:
        raise ValueError("Registration used a different source runtime.")
    return campaign_status(output, verify_artifacts=True)


def trial_partition(protocol, shard_index=0, shard_count=1):
    for name, value in (("shard_count", shard_count), ("shard_index", shard_index)):
        if type(value) is not int or value < (1 if name == "shard_count" else 0):
            raise ValueError(f"Invalid {name}.")
    if shard_index >= shard_count or shard_count > len(protocol["trials"]):
        raise ValueError("Shard index/count exceeds the registered trial grid.")
    return [trial for i, trial in enumerate(protocol["trials"]) if i % shard_count == shard_index]


def campaign_status(value, *, verify_artifacts=False):
    """Count exact registered trials; only an artifact audit can pass execution."""
    protocol = rolling._load_protocol(value)
    root = Path(protocol["output_dir"])
    rows, counts = [], {key: 0 for key in ("pending", "running", "failed", "completed")}
    failure_count, audited, scenario_count = 0, 0, 0
    for trial in protocol["trials"]:
        attempts = rolling._attempts(root, trial)
        history = []
        for attempt in attempts:
            path = attempt / "receipt.json"
            if not path.exists():
                # A process may be between mkdir and its first atomic receipt.
                history.append(
                    {"status": "running", "attempt_path": str(attempt.relative_to(root))}
                )
                continue
            receipt = json.loads(path.read_text())
            if (
                receipt.get("protocol_sha256") != protocol["protocol_sha256"]
                or receipt.get("trial_sha256") != rolling._digest(trial)
                or receipt.get("trial_id") != trial["id"]
            ):
                raise ValueError("Trial receipt identity differs from registered grid.")
            if receipt.get("status") not in {"running", "failed", "completed"}:
                raise ValueError("Unknown trial receipt status.")
            ordinal = int(attempt.name.removeprefix("attempt-"))
            previous = attempt.parent / f"attempt-{ordinal - 1:04d}"
            expected_previous = str(previous.relative_to(root)) if ordinal > 1 else None
            if (
                type(receipt.get("attempt")) is not int
                or receipt.get("attempt") != ordinal
                or receipt.get("previous_attempt") != expected_previous
            ):
                raise ValueError("Trial receipt attempt chain is invalid.")
            if ordinal > 1 and not (previous / "receipt.json").is_file():
                raise ValueError("Trial attempt chain is missing its previous receipt.")
            if ordinal < len(attempts) and receipt["status"] == "running":
                raise ValueError(
                    "An earlier attempt lacks an audited terminal interruption receipt."
                )
            if verify_artifacts:
                rolling._verify_attempt(attempt, trial, protocol)
                if receipt["status"] == "completed":
                    required = {
                        "run.json",
                        "metrics.json",
                        "predictions.csv",
                        "validation_predictions.csv",
                    }
                    if not required.issubset(receipt.get("artifacts", {})):
                        raise ValueError("Completed trial is missing required evidence artifacts.")
                    if trial["config"]["backtest"]["enabled"]:
                        if "cost_scenarios.json" not in receipt["artifacts"]:
                            raise ValueError(
                                "Completed trial is missing registered cost scenarios."
                            )
                        scenarios = json.loads(
                            (attempt / "workflow" / "cost_scenarios.json").read_text()
                        )
                        ids = [row["id"] for row in scenarios.get("scenarios", [])]
                        if scenarios.get("status") != "completed" or ids != [
                            row["id"] for row in protocol["cost_scenarios"]
                        ]:
                            raise ValueError(
                                "Completed cost scenarios differ from the registered grid."
                            )
            failure_count += int(receipt["status"] == "failed")
            history.append(
                {
                    "status": receipt["status"],
                    "attempt_path": str(attempt.relative_to(root)),
                    "elapsed_seconds": receipt.get("elapsed_seconds"),
                    "error": receipt.get("error"),
                }
            )
        state = history[-1]["status"] if history else "pending"
        counts[state] += 1
        audited += int(verify_artifacts and state == "completed")
        if verify_artifacts and state == "completed" and trial["config"]["backtest"]["enabled"]:
            scenario_count += len(protocol["cost_scenarios"])
        rows.append({"trial_id": trial["id"], "status": state, "attempts": history})
    expected = len(protocol["folds"]) * len(protocol["candidates"]) * len(protocol["seeds"])
    actual_grid = {(x["fold"], x["candidate"], x["seed"]) for x in protocol["trials"]}
    expected_grid = {
        (fold["id"], candidate["id"], seed)
        for fold in protocol["folds"]
        for candidate in protocol["candidates"]
        for seed in protocol["seeds"]
    }
    if actual_grid != expected_grid or len(protocol["trials"]) != expected:
        raise ValueError(
            "Registered trials do not cover the exact declared fold/candidate/seed grid."
        )
    full_grid = (
        protocol["seeds"] == [7, 42, 2026, 2027, 2028]
        and len(protocol["folds"]) == 16
        and [candidate["id"] for candidate in protocol["candidates"]]
        == [
            "zero",
            "ridge",
            "lightgbm",
            "transformer",
            "daily_transformer",
            "master_style",
            "quantformer_moe",
        ]
        and protocol["specification"].get("epochs") == 12
        and len(protocol["cost_scenarios"]) == 5
    )
    full_recipe = (
        full_grid and rolling._digest(protocol["specification"]) == FULL_SPECIFICATION_SHA256
    )
    return {
        "schema_version": 1,
        "observed_at_utc": rolling._now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "registered_at_utc": protocol["registered_at_utc"],
        "source_code_sha256": protocol["source_code_sha256"],
        "specification_sha256": rolling._digest(protocol["specification"]),
        "expected_trials": expected,
        "folds": len(protocol["folds"]),
        "candidates": len(protocol["candidates"]),
        "seeds": protocol["seeds"],
        "epoch_budget": protocol["specification"].get("epochs"),
        "expected_scenario_replays": expected * len(protocol["cost_scenarios"]),
        "counts": counts,
        "preserved_failed_attempts": failure_count,
        "artifacts_verified": bool(verify_artifacts),
        "verified_completed_trials": audited,
        "verified_completed_scenario_replays": scenario_count,
        "matches_full_campaign_grid": full_grid,
        "matches_full_campaign_recipe": full_recipe,
        "full_campaign_execution_gate": "passed"
        if full_recipe and audited == expected
        else "pending",
        "candidate_quality_gate": "requires_registered_quality_review",
        "prospective_confirmation": False,
        "promotion_ready": False,
        "qualification": "Historical research. Execution completion is separate from economic, lifecycle and prospective acceptance.",
        "trials": rows,
    }


def run_campaign_queue(
    output,
    *,
    workers=1,
    retry_failed=False,
    max_trials=None,
    shard_index=0,
    shard_count=1,
    clear_stop=False,
):
    """Run a bounded-concurrency queue; termination leaves original attempt receipts."""
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer in [1, 16].")
    if max_trials is not None and (type(max_trials) is not int or max_trials < 1):
        raise ValueError("max_trials must be a positive invocation limit, not a scientific budget.")
    root, descriptor, protocol = _descriptor(output)
    manifest = verify_runtime(descriptor["runtime"])
    if rolling._code_identity() != manifest["source_code_sha256"]:
        raise ValueError("Run the queue through the frozen runtime launcher.")
    if rolling._package_versions() != protocol["packages"]:
        raise ValueError("Runtime package versions changed after registration.")
    shard = trial_partition(protocol, shard_index, shard_count)
    control = root / "control"
    control.mkdir(exist_ok=True)
    # One active controller per campaign prevents two different shard plans competing.
    with FileLock(str(control / ".supervisor.lock"), timeout=0):
        stop_path = control / "stop_after_active.json"
        if clear_stop:
            stop_path.unlink(missing_ok=True)
        status = campaign_status(root, verify_artifacts=True)
        states = {row["trial_id"]: row["status"] for row in status["trials"]}
        queue = [
            trial
            for trial in shard
            if states[trial["id"]] == "pending"
            or (retry_failed and states[trial["id"]] in {"failed", "running"})
        ]
        if max_trials is not None:
            queue = queue[:max_trials]
        invocation = {
            "protocol_sha256": protocol["protocol_sha256"],
            "started_at_utc": rolling._now(),
            "workers": workers,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "retry_failed": retry_failed,
            "invocation_limit": max_trials,
            "selected_trial_ids": [trial["id"] for trial in queue],
            "status": "running",
            "worker_exits": [],
        }
        receipt_path = control / ("invocation-" + str(time.time_ns()) + ".json")
        rolling._save(receipt_path, invocation)
        active = []
        stop_requested = False
        try:
            while queue or active:
                if stop_path.exists() and not stop_requested:
                    stop = json.loads(stop_path.read_text())
                    if stop.get("protocol_sha256") != protocol["protocol_sha256"]:
                        raise ValueError("Stop request belongs to another campaign.")
                    stop_requested = True
                    invocation["stop_request"] = stop
                    invocation["not_started_due_to_stop"] = [trial["id"] for trial in queue]
                    queue.clear()
                while queue and len(active) < workers:
                    trial = queue.pop(0)
                    log_dir = control / "logs" / trial["id"]
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / (str(time.time_ns()) + ".log")
                    stream = log_path.open("w")
                    command = [
                        sys.executable,
                        "-m",
                        "financial_ts.campaign_control",
                        "_trial",
                        "--output",
                        str(root),
                        "--trial-id",
                        trial["id"],
                    ]
                    if retry_failed:
                        command.append("--retry-failed")
                    process = subprocess.Popen(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env=_environment(descriptor["runtime"]),
                    )
                    active.append((process, stream, trial["id"], log_path))
                finished = []
                for entry in active:
                    process, stream, trial_id, log_path = entry
                    code = process.poll()
                    if code is not None:
                        stream.close()
                        invocation["worker_exits"].append(
                            {
                                "trial_id": trial_id,
                                "exit_code": code,
                                "log": str(log_path.relative_to(root)),
                            }
                        )
                        finished.append(entry)
                for entry in finished:
                    active.remove(entry)
                if finished:
                    rolling._save(receipt_path, invocation)
                    rolling._save(control / "status.json", campaign_status(root))
                if active:
                    time.sleep(1)
            invocation["status"] = "stopped_after_active" if stop_requested else "finished"
        except BaseException as exc:
            invocation["status"] = "interrupted"
            invocation["error"] = f"{type(exc).__name__}: {exc}"
            for process, stream, _, _ in active:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                stream.close()
            raise
        finally:
            invocation["finished_at_utc"] = rolling._now()
            rolling._save(receipt_path, invocation)
        result = campaign_status(root, verify_artifacts=True)
        final_states = {row["trial_id"]: row["status"] for row in result["trials"]}
        required_ids = (
            invocation["selected_trial_ids"]
            if max_trials is not None
            else [trial["id"] for trial in shard]
        )
        result["invocation_status"] = invocation["status"]
        result["invocation_incomplete_trials"] = [
            trial_id for trial_id in required_ids if final_states[trial_id] != "completed"
        ]
        result["invocation_failed"] = (
            bool(result["invocation_incomplete_trials"]) and not stop_requested
        )
        rolling._save(control / "status.json", result)
        return result


def _compact(result):
    return {key: value for key, value in result.items() if key != "trials"}


def request_stop(output):
    """Ask a compatible controller to finish active trials without launching more."""
    _, descriptor, protocol = _descriptor(output)
    if verify_runtime(descriptor["runtime"]).get("execution_api_version", 1) < 2:
        raise ValueError("This older frozen controller has no graceful-stop interface.")
    receipt = {"protocol_sha256": protocol["protocol_sha256"], "requested_at_utc": rolling._now()}
    rolling._save(Path(protocol["output_dir"]) / "control/stop_after_active.json", receipt)
    return {**receipt, "status": "stop_requested", "active_trials_terminated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "register",
            "run",
            "stop",
            "status",
            "summarize",
            "_register",
            "_trial",
            "_run",
            "_summarize",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--trial-id")
    args = parser.parse_args(argv)
    if args.command == "register":
        if args.provider is None or args.config is None:
            parser.error("register requires --provider and --config")
        result = register_campaign(
            args.provider, args.output, json.loads(args.config.read_text()), runtime=args.runtime
        )
    elif args.command == "_register":
        manifest = verify_runtime(args.runtime)
        protocol = rolling.prepare_rolling_campaign(
            args.provider, args.output, manifest["specification"]
        )
        rolling._save(
            args.output / "execution.json",
            {
                "schema_version": 1,
                "runtime": str(args.runtime.resolve()),
                "runtime_sha256": manifest["runtime_sha256"],
                "protocol_sha256": protocol["protocol_sha256"],
            },
        )
        result = {
            "protocol_sha256": protocol["protocol_sha256"],
            "registered_trials": len(protocol["trials"]),
        }
    elif args.command == "run":
        root, descriptor, _ = _descriptor(args.output)
        command = [
            sys.executable,
            "-m",
            "financial_ts.campaign_control",
            "_run",
            "--output",
            str(root),
            "--workers",
            str(args.workers),
            "--shard-index",
            str(args.shard_index),
            "--shard-count",
            str(args.shard_count),
        ]
        if args.max_trials is not None:
            command += ["--max-trials", str(args.max_trials)]
        if args.retry_failed:
            command.append("--retry-failed")
        if args.clear_stop:
            command.append("--clear-stop")
        return subprocess.call(command, env=_environment(descriptor["runtime"]))
    elif args.command == "_run":
        result = run_campaign_queue(
            args.output,
            workers=args.workers,
            max_trials=args.max_trials,
            retry_failed=args.retry_failed,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            clear_stop=args.clear_stop,
        )
    elif args.command == "_trial":
        if not args.trial_id:
            parser.error("_trial requires --trial-id")
        result = rolling.run_rolling_campaign(
            args.output, trial_ids=[args.trial_id], retry_failed=args.retry_failed, summarize=False
        )
        protocol = rolling._load_protocol(args.output)
        trial = next(trial for trial in protocol["trials"] if trial["id"] == args.trial_id)
        attempts = rolling._attempts(Path(protocol["output_dir"]), trial)
        result["trial_status"] = rolling._verify_attempt(attempts[-1], trial, protocol)["status"]
    elif args.command == "status":
        result = campaign_status(args.output, verify_artifacts=args.verify_artifacts)
    elif args.command == "stop":
        result = request_stop(args.output)
    elif args.command == "summarize":
        root, descriptor, _ = _descriptor(args.output)
        command = [
            sys.executable,
            "-m",
            "financial_ts.campaign_control",
            "_summarize",
            "--output",
            str(root),
        ]
        return subprocess.call(command, env=_environment(descriptor["runtime"]))
    else:
        # Summary has one writer and reads audited immutable trial artifacts.
        _, descriptor, protocol = _descriptor(args.output)
        if (
            rolling._code_identity() != protocol["source_code_sha256"]
            or rolling._package_versions() != protocol["packages"]
        ):
            raise ValueError(
                "Canonical summary must use the registered source and package versions."
            )
        control = args.output / "control"
        control.mkdir(exist_ok=True)
        with FileLock(str(control / ".supervisor.lock"), timeout=0):
            result = rolling.summarize_rolling_campaign(args.output)
    print(json.dumps(_compact(result), indent=2, allow_nan=False))
    if (args.command == "_run" and result.get("invocation_failed")) or (
        args.command == "_trial" and result.get("trial_status") != "completed"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
