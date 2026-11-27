#!/usr/bin/env python3
"""Export actual Qlib diagnostics from one saved run; never train or select models.

Example, from the project root after installing ``.[visualization]``::

    python scripts/render_qlib_benchmark.py \
        --run-dir runs/chenditc/lightgbm/seed-42 \
        --output docs/assets/qlib --model LightGBM \
        --dataset 'chenditc 2026-11-26 · CSI300'
"""

from __future__ import annotations

import argparse
import json

from financial_ts.qlib_visuals import export_qlib_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--no-model-diagnostics", action="store_true")
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    manifest = export_qlib_run(
        args.run_dir,
        args.output,
        model_name=args.model,
        dataset_label=args.dataset,
        include_model_diagnostics=not args.no_model_diagnostics,
        export_png=not args.no_png,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "charts": len(manifest["charts"]),
                "omissions": manifest["omissions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
