"""Verify every archived byte before restoring the frozen prospective store."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def restore(manifest_path, destination):
    manifest_path, destination = Path(manifest_path), Path(destination)
    manifest = json.loads(manifest_path.read_text())
    archive = manifest_path.parent / manifest["archive"]
    raw = archive.read_bytes()
    if len(raw) != manifest["bytes"] or hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Prospective archive size or SHA256 differs from the retained manifest.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Use a new, empty restoration destination.")
    with tarfile.open(archive, "r:xz") as stream:
        members = stream.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members.")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError("Unsafe archive member.")
        files = [member for member in members if member.isfile()]
        if {member.name for member in files} != set(manifest["files"]):
            raise ValueError("Prospective archive inventory differs.")
        for member in files:
            data = stream.extractfile(member).read()
            expected = manifest["files"][member.name]
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError(f"Archived file identity differs: {member.name}")
        destination.mkdir(parents=True)
        for member in members:
            if member.isdir():
                (destination / member.name).mkdir(parents=True, exist_ok=True)
        # Extract the already validated regular files manually for Python 3.10+.
        for member in files:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(stream.extractfile(member).read())
    protocol = json.loads((destination / "store/protocol.json").read_text())
    if protocol["protocol_id"] != manifest["protocol_id"]:
        raise ValueError("Restored protocol differs from the retained manifest.")
    return {
        "protocol_id": protocol["protocol_id"], "verified_files": len(files),
        "destination": str(destination.absolute()), "environment_recreated": False,
        "orders_submitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.manifest, args.destination), indent=2))


if __name__ == "__main__":
    main()
