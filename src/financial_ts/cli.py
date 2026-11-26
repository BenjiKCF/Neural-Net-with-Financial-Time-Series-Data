"""Small command-line entry point; importing it never starts training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fts", description="Financial time-series research with Qlib and PyTorch."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser(
        "import-csv", help="Validate adjusted OHLCV and create a Qlib provider"
    )
    ingest.add_argument(
        "--csv", required=True, type=Path, help="CSV file or directory of CSV files"
    )
    ingest.add_argument("--provider", required=True, type=Path, help="New Qlib provider directory")
    ingest.add_argument("--symbol", help="Symbol for a single CSV without a symbol column")

    train = commands.add_parser(
        "train", help="Train, evaluate, and optionally backtest a YAML workflow"
    )
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--provider", type=Path, help="Override provider_uri")
    train.add_argument("--output", type=Path, help="Override output_dir; must be new/empty")
    train.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], help="Transformer device override"
    )
    group = train.add_mutually_exclusive_group()
    group.add_argument("--backtest", action="store_true")
    group.add_argument("--no-backtest", action="store_true")

    demo = commands.add_parser("demo", help="Run a fully offline synthetic software check")
    demo.add_argument(
        "--output",
        type=Path,
        default=Path("runs/demo"),
        help="New directory for all demo artifacts",
    )
    demo.add_argument("--model", choices=["lightgbm", "transformer", "both"], default="lightgbm")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    demo.add_argument(
        "--epochs", type=int, default=3, help="Short Transformer smoke-training limit"
    )
    return parser


def _display(result: dict) -> None:
    metrics = result["metrics"]
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "synthetic": result["run"]["synthetic"],
                "prediction_count": metrics["prediction_count"],
                "rmse": metrics["rmse"],
                "zero_return_rmse": metrics["zero_return_rmse"],
                "recorder_id": result["run"]["recorder_id"],
            },
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        from financial_ts.data import create_demo_csv, import_csv
        from financial_ts.workflow import load_config, run_workflow

        if args.command == "import-csv":
            print(json.dumps(import_csv(args.csv, args.provider, args.symbol), indent=2))
        elif args.command == "train":
            config = load_config(args.config)
            if args.provider:
                config["provider_uri"] = str(args.provider)
            if args.output:
                config["output_dir"] = str(args.output)
            if args.device:
                if config.get("model", {}).get("name") != "transformer":
                    raise ValueError("--device applies to the transformer configuration only.")
                config["model"].setdefault("params", {})["device"] = args.device
            if args.backtest or args.no_backtest:
                config.setdefault("backtest", {})["enabled"] = args.backtest
            _display(run_workflow(config))
        else:
            root = args.output.expanduser().resolve()
            if root.exists() and (not root.is_dir() or any(root.iterdir())):
                raise ValueError(f"Demo output already exists: {root}. Choose a new --output.")
            if args.epochs < 1:
                raise ValueError("--epochs must be positive.")
            if args.model in {"transformer", "both"}:
                try:
                    import torch  # noqa: F401
                except ImportError as exc:
                    raise ValueError(
                        "Install neural support first: python -m pip install -e '.[neural]'"
                    ) from exc
            root.mkdir(parents=True, exist_ok=True)
            print(
                "SYNTHETIC SOFTWARE CHECK: these results do not measure a real trading edge.",
                flush=True,
            )
            csv = create_demo_csv(root / "synthetic.csv", seed=args.seed)
            provider = root / "provider"
            manifest = import_csv(csv, provider)
            manifest["synthetic"] = True
            (provider / "metadata.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            models = ["lightgbm", "transformer"] if args.model == "both" else [args.model]
            comparison = {}
            for name in models:
                params = (
                    {"num_boost_round": 30, "early_stopping_rounds": 5, "min_data_in_leaf": 20}
                    if name == "lightgbm"
                    else {
                        "d_model": 32,
                        "nhead": 4,
                        "num_layers": 1,
                        "n_epochs": args.epochs,
                        "early_stop": 2,
                        "batch_size": 128,
                        "device": args.device,
                    }
                )
                result = run_workflow(
                    {
                        "provider_uri": str(provider),
                        "output_dir": str(root / name),
                        "region": "us",
                        "instruments": "all",
                        "seed": args.seed,
                        "experiment_name": "synthetic_demo",
                        "sequence_length": 20,
                        "model": {"name": name, "params": params},
                        "backtest": {
                            "enabled": True,
                            "topk": 3,
                            "n_drop": 1,
                            "open_cost": 0.001,
                            "close_cost": 0.001,
                            "benchmark": "DEMO000",
                        },
                    }
                )
                comparison[name] = result["metrics"]
                _display(result)
            (root / "comparison.json").write_text(
                json.dumps(
                    {
                        "synthetic": True,
                        "purpose": "software verification only",
                        "models": comparison,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return 0
    except (ValueError, FileExistsError, FileNotFoundError, ImportError) as exc:
        print(f"fts: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
