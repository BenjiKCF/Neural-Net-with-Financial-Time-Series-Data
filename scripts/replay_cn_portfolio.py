"""Replay a CN native Qlib portfolio from raw or gzip CSV predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from financial_ts.cn_portfolio import replay_cn_portfolio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--provider", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay_cn_portfolio(args.run_dir, args.provider, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "ties": result["exact_score_ties"],
            }
        )
    )


if __name__ == "__main__":
    main()
