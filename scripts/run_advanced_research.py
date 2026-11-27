"""Run the explicitly registered multi-horizon/quantile/adaptation experiment."""

import argparse
import json
from pathlib import Path

from financial_ts.advanced_research import run_advanced_research


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/advanced_screening.json"))
    args = parser.parse_args()
    result = run_advanced_research(args.provider, args.output, json.loads(args.config.read_text()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
