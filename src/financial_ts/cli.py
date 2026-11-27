"""Small command-line entry point; importing it never starts training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from financial_ts.catalog import (
    CROSS_SECTIONAL_MODELS,
    EXPERIMENTAL_MODEL_NAMES,
    MODEL_NAMES,
    NEURAL_MODELS,
    TRAINABLE_MODELS,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fts", description="Financial time-series research with Qlib and PyTorch."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    rolling = commands.add_parser(
        "rolling-evaluate", help="Register and run causal folds, seeds and objective/model controls"
    )
    rolling.add_argument("--provider", type=Path)
    rolling.add_argument("--config", type=Path)
    rolling.add_argument("--output", type=Path, required=True)
    rolling.add_argument("--register-only", action="store_true")
    rolling.add_argument("--summarize-only", action="store_true")
    rolling.add_argument(
        "--summary-output",
        type=Path,
        help="Write an audited derived summary separately, preserving original campaign reports",
    )
    rolling.add_argument("--retry-failed", action="store_true")
    rolling.add_argument("--trial-ids", nargs="+")
    bundle = commands.add_parser(
        "verify-bundle", help="Check every immutable numerical model artifact"
    )
    bundle.add_argument("--bundle", type=Path, required=True)
    register = commands.add_parser(
        "register-bundle", help="Register an immutable numerical model version"
    )
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--bundle", type=Path, required=True)
    alias = commands.add_parser(
        "set-model-alias",
        help="Assign a reviewed version or roll back a compatible retained version",
    )
    alias.add_argument("--registry", type=Path, required=True)
    alias.add_argument("--alias", required=True)
    alias.add_argument("--version", required=True)
    alias.add_argument("--evidence", type=Path)
    alias.add_argument("--rollback", action="store_true")
    daily = commands.add_parser(
        "daily-infer", help="Publish a validated, immutable daily prediction batch"
    )
    daily.add_argument("--registry", type=Path, required=True)
    daily.add_argument("--alias", default="shadow")
    daily.add_argument("--output", type=Path, required=True)
    daily.add_argument("--policy", type=Path)
    source = daily.add_mutually_exclusive_group(required=True)
    source.add_argument("--provider", type=Path)
    source.add_argument("--input-npz", type=Path)
    daily.add_argument("--input-manifest", type=Path)
    daily.add_argument("--decision-date")
    daily.add_argument("--decision-cutoff")
    daily.add_argument("--data-available-at")
    monitor = commands.add_parser(
        "monitor-labels", help="Evaluate forecasts only against matured labels"
    )
    monitor.add_argument("--predictions", type=Path, required=True)
    monitor.add_argument("--labels", type=Path, required=True)
    monitor.add_argument("--cutoff", required=True)
    monitor.add_argument("--minimum-rank-ic", type=float)
    monitor.add_argument("--output", type=Path)
    blend = commands.add_parser(
        "rolling-blend", help="Evaluate a validation-selected tree/neural ensemble"
    )
    blend.add_argument("--campaign", type=Path, required=True)
    blend.add_argument("--output", type=Path, required=True)
    blend.add_argument("--left", default="lightgbm")
    blend.add_argument("--right", default="master_style")
    blend.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    blend.add_argument("--metric", choices=["rank_ic", "ic", "rmse"], default="rank_ic")
    lifecycle = commands.add_parser(
        "lifecycle-assess", help="Assess archived holdings against sourced corporate actions"
    )
    lifecycle.add_argument("--run-dir", type=Path, required=True)
    lifecycle.add_argument("--provider", type=Path, required=True)
    lifecycle.add_argument("--events", type=Path, default=Path("configs/lifecycle_events.json"))
    lifecycle.add_argument("--output", type=Path, required=True)
    settlement = commands.add_parser(
        "account-settlement", help="Reconcile source-bound raw-share account settlement evidence"
    )
    settlement.add_argument("--spec", type=Path, required=True)
    settlement.add_argument("--output", type=Path, required=True)
    settlement_audit = commands.add_parser(
        "audit-account-settlement", help="Recheck retained account evidence and reconciliation"
    )
    settlement_audit.add_argument("--output", type=Path, required=True)
    acceptance = commands.add_parser(
        "roadmap-acceptance", help="Revalidate evidence for the four remaining roadmap gates"
    )
    acceptance.add_argument("--campaign", type=Path)
    acceptance.add_argument("--account-settlement", type=Path)
    acceptance.add_argument("--prospective", type=Path)
    acceptance.add_argument("--output", type=Path)
    scheduled = commands.add_parser(
        "scheduled-shadow", help="Run one scheduled shadow batch using an admitted arrival receipt"
    )
    scheduled.add_argument("--config", type=Path, required=True)
    advanced = commands.add_parser(
        "research-advanced",
        help="Run registered multi-horizon, uncertainty and adaptation controls",
    )
    advanced.add_argument("--provider", type=Path, required=True)
    advanced.add_argument("--config", type=Path, required=True)
    advanced.add_argument("--output", type=Path, required=True)
    doubleadapt = commands.add_parser(
        "research-doubleadapt",
        help="Run a source-documented DoubleAdapt-style comparator and matched controls",
    )
    doubleadapt.add_argument("--provider", type=Path, required=True)
    doubleadapt.add_argument("--config", type=Path, required=True)
    doubleadapt.add_argument("--output", type=Path, required=True)
    paper = commands.add_parser(
        "paper-replay", help="Replay a registered paper execution session with durable receipts"
    )
    paper.add_argument("--spec", type=Path, required=True)
    paper.add_argument("--output", type=Path, required=True)
    snapshot = commands.add_parser(
        "snapshot-audit", help="Admit or quarantine an immutable tabular snapshot"
    )
    snapshot.add_argument("--current", type=Path, required=True)
    snapshot.add_argument("--identity", type=Path, required=True)
    snapshot.add_argument("--previous", type=Path)
    snapshot.add_argument("--previous-identity", type=Path)
    snapshot.add_argument("--output", type=Path, required=True)
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

        if args.command == "rolling-evaluate":
            from financial_ts.rolling import (
                prepare_rolling_campaign,
                run_rolling_campaign,
                summarize_rolling_campaign,
            )

            if args.config is not None:
                if args.provider is None:
                    raise ValueError("A new rolling campaign requires --provider with --config.")
                import yaml

                prepare_rolling_campaign(
                    args.provider, args.output, yaml.safe_load(args.config.read_text())
                )
            if args.register_only:
                result = {"status": "registered", "protocol": str(args.output / "protocol.json")}
            else:
                if not args.summarize_only:
                    run_rolling_campaign(
                        args.output, retry_failed=args.retry_failed, trial_ids=args.trial_ids
                    )
                result = summarize_rolling_campaign(
                    args.output, summary_output_dir=args.summary_output
                )
            print(json.dumps(result, indent=2, allow_nan=False))
            if result.get("status") == "failed":
                return 1
        elif args.command == "verify-bundle":
            from financial_ts.bundles import verify_bundle

            print(json.dumps(verify_bundle(args.bundle), indent=2, allow_nan=False))
        elif args.command == "register-bundle":
            from financial_ts.operations import register_bundle

            print(
                json.dumps(register_bundle(args.registry, args.bundle), indent=2, allow_nan=False)
            )
        elif args.command == "set-model-alias":
            from financial_ts.operations import set_alias

            evidence = json.loads(args.evidence.read_text()) if args.evidence else None
            print(
                json.dumps(
                    set_alias(
                        args.registry,
                        args.alias,
                        args.version,
                        evidence=evidence,
                        rollback=args.rollback,
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
        elif args.command == "daily-infer":
            import numpy as np

            from financial_ts.operations import (
                BatchPolicy,
                build_provider_packet,
                resolve_alias,
                run_daily_batch,
            )

            policy = (
                BatchPolicy(**json.loads(args.policy.read_text())) if args.policy else BatchPolicy()
            )
            if args.provider:
                if not all((args.decision_date, args.decision_cutoff, args.data_available_at)):
                    raise ValueError(
                        "Provider inference requires --decision-date, --decision-cutoff and --data-available-at."
                    )
                path, pinned_version = resolve_alias(args.registry, args.alias)
                packet, features, padding = build_provider_packet(
                    args.provider,
                    path,
                    decision_date=args.decision_date,
                    decision_cutoff=args.decision_cutoff,
                    data_available_at=args.data_available_at,
                )
            else:
                if args.input_manifest is None:
                    raise ValueError("Numerical input requires --input-manifest.")
                packet = json.loads(args.input_manifest.read_text())
                pinned_version = None
                with np.load(args.input_npz, allow_pickle=False) as archive:
                    features = archive["features"].copy()
                    padding = archive["padding"].copy() if "padding" in archive else None
            print(
                json.dumps(
                    run_daily_batch(
                        args.registry,
                        args.alias,
                        args.output,
                        packet=packet,
                        features=features,
                        padding=padding,
                        policy=policy,
                        pinned_bundle_id=pinned_version,
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
        elif args.command == "monitor-labels":
            import pandas as pd

            from financial_ts.bundles import atomic_json
            from financial_ts.operations import monitor_mature_labels

            result = monitor_mature_labels(
                pd.read_csv(args.predictions),
                pd.read_csv(args.labels),
                cutoff=args.cutoff,
                minimum_rank_ic=args.minimum_rank_ic,
            )
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "scheduled-shadow":
            from financial_ts.operations import run_scheduled_shadow

            result = run_scheduled_shadow(json.loads(args.config.read_text()))
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "research-advanced":
            from financial_ts.advanced_research import run_advanced_research

            result = run_advanced_research(
                args.provider, args.output, json.loads(args.config.read_text())
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "research-doubleadapt":
            from financial_ts.doubleadapt import run_doubleadapt_research

            result = run_doubleadapt_research(
                args.provider, args.output, json.loads(args.config.read_text())
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "paper-replay":
            from financial_ts.execution import run_paper_session

            print(json.dumps(run_paper_session(args.spec, args.output), indent=2, allow_nan=False))
        elif args.command == "snapshot-audit":
            import pandas as pd

            from financial_ts.data_contracts import audit_snapshot, save_snapshot_audit

            current = pd.read_csv(
                args.current, parse_dates=["datetime"], dtype={"instrument": str}
            ).set_index(["datetime", "instrument"])
            previous = (
                pd.read_csv(
                    args.previous, parse_dates=["datetime"], dtype={"instrument": str}
                ).set_index(["datetime", "instrument"])
                if args.previous
                else None
            )
            result = audit_snapshot(
                current,
                json.loads(args.identity.read_text()),
                previous=previous,
                previous_identity=json.loads(args.previous_identity.read_text())
                if args.previous_identity
                else None,
            )
            path = save_snapshot_audit(result, args.output)
            print(json.dumps({**result, "receipt": str(path)}, indent=2, allow_nan=False))
            if result["status"] == "quarantined":
                return 1
        elif args.command == "rolling-blend":
            from financial_ts.ensembles import run_rolling_blend

            result = run_rolling_blend(
                args.campaign,
                args.output,
                left=args.left,
                right=args.right,
                weights=args.weights,
                metric=args.metric,
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command == "lifecycle-assess":
            from financial_ts.lifecycle import assess_archived_lifecycles

            result = assess_archived_lifecycles(
                args.run_dir, args.provider, args.events, args.output
            )
            print(json.dumps(result, indent=2, allow_nan=False))
        elif args.command in {"account-settlement", "audit-account-settlement"}:
            from financial_ts.settlement import (
                run_account_settlement,
                verify_account_settlement,
            )

            result = (
                run_account_settlement(args.spec, args.output)
                if args.command == "account-settlement"
                else verify_account_settlement(args.output)
            )
            print(json.dumps(result, indent=2, allow_nan=False))
            if result["status"] != "passed":
                return 1
        elif args.command == "roadmap-acceptance":
            from financial_ts.acceptance import assess_roadmap_acceptance
            from financial_ts.bundles import atomic_json

            result = assess_roadmap_acceptance(
                campaign=args.campaign,
                account_settlement=args.account_settlement,
                prospective=args.prospective,
            )
            if args.output is not None:
                atomic_json(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
            if result["status"] != "passed_for_review":
                return 1
        elif args.command == "download-chenditc":
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
            print("\n".join(EXPERIMENTAL_MODEL_NAMES))
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
                if name not in (*NEURAL_MODELS, *CROSS_SECTIONAL_MODELS, "timesfm3_pretrained"):
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
