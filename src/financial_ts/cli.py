"""Small command-line entry point; importing it never starts training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from financial_ts.catalog import MODEL_NAMES, NEURAL_MODELS, TRAINABLE_MODELS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fts", description="Financial time-series research with Qlib and PyTorch."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download-chenditc", help="Download and verify the latest or a pinned chenditc Qlib release"
    )
    download.add_argument("--provider", required=True, type=Path)
    download.add_argument("--release", default="latest")
    download.add_argument("--cache-dir", type=Path)
    cn = commands.add_parser(
        "benchmark-cn", help="Rerun the CSI300 comparison on verified chenditc data"
    )
    cn.add_argument("--provider", required=True, type=Path)
    cn.add_argument("--output", type=Path, default=Path("runs/chenditc-csi300"))
    cn.add_argument(
        "--models", nargs="+", help="Declared model names; default runs all available variants"
    )
    cn.add_argument("--epochs", type=int, default=6)
    cn.add_argument("--seed", type=int, default=42)
    cn.add_argument("--threads", type=int, default=2)
    cn.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    cn.add_argument("--timesfm-checkpoint", type=Path)
    cn.add_argument("--summarize-only", action="store_true")
    cn.add_argument(
        "--retry-failed",
        action="store_true",
        help="Archive incomplete attempts and retry identical configurations",
    )
    plots = commands.add_parser(
        "qlib-report", help="Export Qlib native figures, PNGs and interactive HTML"
    )
    plots.add_argument("--run-dir", required=True, type=Path)
    plots.add_argument("--output", required=True, type=Path)
    plots.add_argument("--model")
    plots.add_argument("--dataset-label", default="")
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
        "--device", choices=["auto", "cpu", "cuda"], help="Neural model device override"
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
    demo.add_argument("--model", choices=[*TRAINABLE_MODELS, "both", "all"], default="lightgbm")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    demo.add_argument("--epochs", type=int, default=3, help="Short neural smoke-training limit")
    commands.add_parser("models", help="List trainable models and official pretrained TimesFM3")
    benchmark = commands.add_parser("benchmark", help="Run a declared model/seed comparison")
    benchmark.add_argument("--provider", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, default=Path("runs/architecture-benchmark"))
    benchmark.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(TRAINABLE_MODELS)
    )
    benchmark.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 2026])
    benchmark.add_argument("--epochs", type=int, default=12)
    benchmark.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    benchmark.add_argument(
        "--timesfm-checkpoint", type=Path, help="Local official TimesFM3 checkpoint"
    )
    loop = commands.add_parser(
        "finetune-timesfm", help="Fine-tune local gates/adapters on a looped TimesFM3 checkpoint"
    )
    loop.add_argument("--provider", type=Path, required=True)
    loop.add_argument("--checkpoint", type=Path, required=True)
    loop.add_argument("--output", type=Path, default=Path("runs/timesfm3-loop"))
    loop.add_argument("--epochs", type=int, default=3)
    loop.add_argument("--train-samples", type=int, default=256)
    loop.add_argument("--valid-samples", type=int, default=128)
    loop.add_argument("--context-length", type=int, default=128)
    loop.add_argument("--batch-size", type=int, default=4)
    loop.add_argument("--lr", type=float, default=0.001)
    loop.add_argument("--seed", type=int, default=42)
    loop.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    research = commands.add_parser("research", help="Fit the fixed small-model candidate campaign")
    research.add_argument("--provider", type=Path, required=True)
    research.add_argument("--output", type=Path, default=Path("runs/model-research"))
    research.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 2026])
    research.add_argument("--epochs", type=int, default=12)
    research.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    papers = commands.add_parser(
        "research-papers",
        help="Test ES and directional weight editing on the selected mini ensemble",
    )
    papers.add_argument("--provider", type=Path, required=True)
    papers.add_argument("--base-runs", type=Path, default=Path("runs/model-research"))
    papers.add_argument("--spy-csv", type=Path, required=True)
    papers.add_argument("--output", type=Path, default=Path("runs/paper-posttraining"))
    papers.add_argument("--fit-samples", type=int, default=512)
    papers.add_argument("--population", type=int, default=12)
    papers.add_argument("--generations", type=int, default=8)
    comparison = commands.add_parser(
        "compare-timesfm", help="Compare frozen, loop and adapter-only TimesFM on common dates"
    )
    comparison.add_argument("--provider", type=Path, required=True)
    comparison.add_argument("--checkpoint", type=Path, required=True)
    comparison.add_argument("--output", type=Path, default=Path("runs/timesfm-comparison"))
    comparison.add_argument(
        "--original-evidence", type=Path, default=Path("reports/timesfm_loop_pilot")
    )
    comparison.add_argument("--resume", action="store_true")
    comparison.add_argument(
        "--allow-numerical-replication",
        action="store_true",
        help="Explicitly retain a same-recipe retrain when exact archived predictions cannot be reproduced",
    )
    backtest = commands.add_parser(
        "backtest", help="Replay saved forecasts as delayed-close long/cash SPY strategies"
    )
    backtest.add_argument("--predictions", type=Path, required=True)
    backtest.add_argument("--spy-csv", type=Path, required=True)
    backtest.add_argument("--output", type=Path, default=Path("runs/backtest"))
    backtest.add_argument("--costs-bps", nargs="+", type=float, default=[0, 5, 10, 20])
    report = commands.add_parser(
        "compare-research",
        help="Join completed experiments and export common scores and SPY backtests",
    )
    report.add_argument(
        "--historical-predictions", type=Path, default=Path("reports/architecture_predictions.csv")
    )
    report.add_argument(
        "--historical-report", type=Path, default=Path("reports/architecture_benchmark.json")
    )
    report.add_argument("--scratch", type=Path, default=Path("runs/model-research"))
    report.add_argument("--timesfm", type=Path, default=Path("runs/timesfm-comparison"))
    report.add_argument("--spy-csv", type=Path, required=True)
    report.add_argument("--output", type=Path, default=Path("runs/financial-comparison"))
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

        if args.command == "download-chenditc":
            from financial_ts.chenditc import download_chenditc

            result = download_chenditc(
                args.provider, release=args.release, cache_dir=args.cache_dir
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "benchmark-cn":
            from financial_ts.cn_benchmark import (
                prepare_cn_benchmark,
                run_cn_models,
                summarize_cn_benchmark,
            )

            protocol = prepare_cn_benchmark(
                args.provider,
                args.output,
                epochs=args.epochs,
                seed=args.seed,
                device=args.device,
                num_threads=args.threads,
                checkpoint=args.timesfm_checkpoint,
            )
            if not args.summarize_only:
                run_cn_models(protocol, models=args.models, retry_failed=args.retry_failed)
            print(json.dumps(summarize_cn_benchmark(args.output), indent=2, allow_nan=False))
        elif args.command == "qlib-report":
            from financial_ts.qlib_visuals import export_qlib_run

            result = export_qlib_run(
                args.run_dir, args.output, model_name=args.model, dataset_label=args.dataset_label
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "models":
            print("\n".join(MODEL_NAMES))
        elif args.command == "research-papers":
            from financial_ts.paper_research import run_paper_research

            result = run_paper_research(
                args.provider,
                args.base_runs,
                args.spy_csv,
                args.output,
                fit_samples=args.fit_samples,
                population=args.population,
                generations=args.generations,
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "research":
            from financial_ts.research import run_scratch_research

            result = run_scratch_research(
                args.provider, args.output, seeds=args.seeds, epochs=args.epochs, device=args.device
            )
            print(json.dumps(result["metrics"], indent=2, allow_nan=False))
        elif args.command == "compare-timesfm":
            from financial_ts.timesfm_comparison import run_timesfm_comparison

            result = run_timesfm_comparison(
                args.provider,
                args.checkpoint,
                args.output,
                args.original_evidence,
                resume=args.resume,
                allow_numerical_replication=args.allow_numerical_replication,
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "backtest":
            from financial_ts.comparison import run_saved_backtest

            result = run_saved_backtest(
                args.predictions, args.spy_csv, args.output, costs_bps=args.costs_bps
            )
            print(
                json.dumps(
                    {
                        "output_dir": result["output_dir"],
                        "strategies": len({row["model"] for row in result["metrics"]}),
                        "cost_scenarios_bps": args.costs_bps,
                        "result_rows": len(result["metrics"]),
                        "files": ["metrics.csv", "ledger.csv", "settings.json"],
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
        elif args.command == "compare-research":
            from financial_ts.comparison import build_financial_comparison

            result = build_financial_comparison(
                args.historical_predictions,
                args.historical_report,
                args.scratch,
                args.timesfm,
                args.spy_csv,
                args.output,
            )
            print(
                json.dumps(
                    {
                        "output_dir": result["output_dir"],
                        "models": int(result["metrics"]["model"].nunique()),
                        "test_count": result["settings"]["test_count"],
                    },
                    indent=2,
                )
            )
        elif args.command == "finetune-timesfm":
            from financial_ts.loop_experiment import run_loop_experiment

            result = run_loop_experiment(
                args.provider,
                args.checkpoint,
                args.output,
                epochs=args.epochs,
                train_samples=args.train_samples,
                valid_samples=args.valid_samples,
                context_length=args.context_length,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                device=args.device,
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "benchmark":
            from financial_ts.benchmark import run_benchmark

            result = run_benchmark(
                args.provider,
                args.output,
                args.models,
                args.seeds,
                args.epochs,
                args.device,
                args.timesfm_checkpoint,
            )
            print(json.dumps(result["aggregate"], indent=2))
        elif args.command == "import-csv":
            print(json.dumps(import_csv(args.csv, args.provider, args.symbol), indent=2))
        elif args.command == "train":
            config = load_config(args.config)
            if args.provider:
                config["provider_uri"] = str(args.provider)
            if args.output:
                config["output_dir"] = str(args.output)
            if args.device:
                name = config.get("model", {}).get("name")
                if name not in (*NEURAL_MODELS, "timesfm3_pretrained"):
                    raise ValueError("--device applies to neural model configurations only.")
                if name == "timesfm3_pretrained" and args.device == "auto":
                    import torch

                    args.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            if args.model != "lightgbm":
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
            models = (
                list(TRAINABLE_MODELS)
                if args.model == "all"
                else ["lightgbm", "transformer"]
                if args.model == "both"
                else [args.model]
            )
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
