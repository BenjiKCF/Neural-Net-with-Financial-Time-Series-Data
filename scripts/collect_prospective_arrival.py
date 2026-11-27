"""Acquire and audit real chenditc arrivals for the prospective shadow runner."""

import argparse
import json
from pathlib import Path

from financial_ts.arrival_feed import collect_arrival, verify_arrival


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--feed-root", type=Path, required=True)
    collect.add_argument("--session", required=True)
    collect.add_argument("--decision-cutoff", required=True)
    collect.add_argument("--cache-dir", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect_arrival(
            args.feed_root, expected_session=args.session,
            decision_cutoff=args.decision_cutoff, cache_dir=args.cache_dir,
        )
    else:
        result = verify_arrival(args.receipt)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
