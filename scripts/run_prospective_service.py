"""Run one genuine-time collection or publication tick; never backdate evidence."""

import argparse
import json
from pathlib import Path

from financial_ts.prospective_service import service_tick


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--feed-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = service_tick(args.store, args.feed_root, cache_dir=args.cache_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
