#!/usr/bin/env python3
"""Reconcile retained account evidence, or verify an existing settlement report."""

import argparse
import json

from financial_ts.settlement import run_account_settlement, verify_account_settlement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_account_settlement(args.output)
    elif args.spec:
        result = run_account_settlement(args.spec, args.output)
    else:
        parser.error("--spec is required unless --verify is supplied")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
