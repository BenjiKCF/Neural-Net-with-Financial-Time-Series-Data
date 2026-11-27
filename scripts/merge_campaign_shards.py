"""Merge retained shard attempts without overwriting conflicting evidence.

Download the one registration artifact first. Each input child directory is an
unpacked worker artifact containing trials/ and, optionally, control/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from filelock import FileLock

from financial_ts import rolling


def _recovery_transition(first, second):
    """Allow only the audited running→interrupted transition, preserving original bytes."""
    for older, newer in ((first, second), (second, first)):
        old = json.loads(older.read_text())
        new = json.loads(newer.read_text())
        recovery = new.get("interruption_recovery", {})
        original = newer.parent / "receipt.before-recovery.json"
        changed_fields = {
            "status",
            "finished_at_utc",
            "elapsed_seconds",
            "error",
            "interruption_recovery",
        }
        if (
            old.get("status") == "running"
            and new.get("status") == "failed"
            and recovery.get("previous_receipt_sha256") == rolling._sha256(older)
            and original.is_file()
            and original.read_bytes() == older.read_bytes()
            and {k: v for k, v in old.items() if k not in changed_fields}
            == {k: v for k, v in new.items() if k not in changed_fields}
        ):
            return newer
    raise ValueError("Conflicting shard receipt is not a verified interruption recovery.")


def merge_shards(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    protocol = rolling._load_protocol(output)
    trials = {trial["id"]: trial for trial in protocol["trials"]}
    planned = {}
    artifacts = sorted(path for path in source.iterdir() if path.is_dir())
    if not artifacts:
        raise ValueError("No unpacked shard artifacts were supplied.")
    for artifact in artifacts:
        if artifact.is_symlink():
            raise ValueError("Shard artifacts must not be symlinks.")
        for path in artifact.rglob("*"):
            if path.is_symlink():
                raise ValueError("Shard evidence must not contain symlinks.")
            if not path.is_file():
                continue
            relative = path.relative_to(artifact)
            if relative.parts[0] not in {"trials", "control"}:
                raise ValueError(f"Unexpected shard artifact path: {relative}")
            if path.name.endswith(".lock"):
                continue
            if relative.parts[0] == "trials":
                if len(relative.parts) < 6:
                    raise ValueError("Malformed trial artifact path.")
                trial_id = "/".join(relative.parts[1:4])
                if trial_id not in trials or not relative.parts[4].startswith("attempt-"):
                    raise ValueError("Shard artifact is outside the registered trial grid.")
                if relative.parts[-1] == "receipt.json" and len(relative.parts) == 6:
                    receipt = json.loads(path.read_text())
                    if (
                        receipt.get("protocol_sha256") != protocol["protocol_sha256"]
                        or receipt.get("trial_sha256") != rolling._digest(trials[trial_id])
                        or receipt.get("trial_id") != trial_id
                    ):
                        raise ValueError("Shard receipt differs from the immutable registration.")
            digest = rolling._sha256(path)
            if relative in planned and planned[relative][1] != digest:
                if relative.parts[0] == "trials" and relative.name == "receipt.json":
                    path = _recovery_transition(planned[relative][0], path)
                    digest = rolling._sha256(path)
                else:
                    raise ValueError(f"Conflicting shard evidence: {relative}")
            planned[relative] = (path, digest)
    copied = 0
    (output / "control").mkdir(exist_ok=True)
    with FileLock(str(output / "control/.supervisor.lock"), timeout=0):
        for relative, (_, digest) in planned.items():
            target = output / relative
            if target.is_symlink() or any(
                parent.is_symlink() for parent in target.parents if parent.is_relative_to(output)
            ):
                raise ValueError("Existing campaign evidence must not contain symlinks.")
            if target.exists() and (not target.is_file() or rolling._sha256(target) != digest):
                if (
                    target.is_file()
                    and relative.parts[0] == "trials"
                    and relative.name == "receipt.json"
                ):
                    selected = _recovery_transition(target, planned[relative][0])
                    planned[relative] = (selected, rolling._sha256(selected))
                else:
                    raise ValueError(f"Existing campaign evidence conflicts with shard: {relative}")
        for relative, (path, _) in planned.items():
            target = output / relative
            if not target.exists() or rolling._sha256(target) != rolling._sha256(path):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".merge-tmp")
                shutil.copyfile(path, temporary)
                temporary.replace(target)
                copied += 1
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "artifacts": len(artifacts),
        "files_seen": len(planned),
        "files_added": copied,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_shards(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
