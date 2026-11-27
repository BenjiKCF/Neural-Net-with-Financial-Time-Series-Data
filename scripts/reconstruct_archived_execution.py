#!/usr/bin/env python3
"""Reproduce and retain the execution chain behind an archived Qlib account."""

import argparse
import json

from financial_ts.archived_reconstruction import (
    reconstruct_archived_execution,
    verify_archived_reconstruction,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--provider")
    parser.add_argument("--lifecycle-assessment")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_archived_reconstruction(args.output)
    elif args.run_dir and args.provider and args.lifecycle_assessment:
        result = reconstruct_archived_execution(args.run_dir, args.provider, args.lifecycle_assessment, args.output)
    else:
        parser.error("--run-dir, --provider and --lifecycle-assessment are required unless --verify is supplied")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
