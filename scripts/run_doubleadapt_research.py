"""Run the independently implemented DoubleAdapt paper-method comparator."""

import argparse
import json
from pathlib import Path

from financial_ts.doubleadapt import run_doubleadapt_research


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/doubleadapt_screening.json"))
    args = parser.parse_args()
    result = run_doubleadapt_research(args.provider, args.output, json.loads(args.config.read_text()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
