"""Finish the pinned CSI300 campaign from existing successful worker exports.

Run only against the explicitly identified GitHub Actions artifacts. Do not
retrain, change the registered split, or select a model using test performance.
The result is a staging bundle; nothing is published until the full audit passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from collect_cn_evidence import SOURCE_FILES, collect

from financial_ts.cn_benchmark import summarize_cn_benchmark
from financial_ts.cn_inference_shards import assemble_shards
from financial_ts.qlib_visuals import export_qlib_run

WORKERS = ("deepseek_v41_mini", "qwen4_mini", "timesfm3_frozen_128")
SHARDED = ("timesfm3_frozen_256", "timesfm3_loop_128", "timesfm3_adapter_128")
WORKER_COMMIT = "149605860f0c1644912bc5a17c4ad60ab5f0df58"
SHARD_COMMIT = "9a3e3dd412fa34fc4d49b800c6cfd8426a7ee861"
WORKER_RUN = "45928631920"
SHARD_RUN = "45931274928"
EVIDENCE = Path("reports/chenditc_csi300")
GALLERY = Path("docs/assets/qlib")
REQUIRED_CHARTS = ("portfolio", "score-ic", "monthly-ic", "group-return", "risk-summary")
DISPLAY = {
    "lightgbm": "LightGBM",
    "transformer": "Temporal Transformer",
    "deepseek_v41_mini": "DeepSeek V4.1 Mini",
    "glm54_mini": "GLM-5.4 Loop Mini",
    "qwen4_mini": "Qwen4 Mini",
    "timesfm3_mini": "TimesFM 3 Mini",
    "multiscale_residual_mixer": "Multiscale Residual Mixer",
    "regime_linear_moe": "Regime Linear MoE",
    "timesfm3_frozen_128": "Official TimesFM 3 (128)",
    "timesfm3_frozen_256": "Official TimesFM 3 (256)",
    "timesfm3_loop_128": "Official TimesFM 3 Loop + adapter",
    "timesfm3_adapter_128": "Official TimesFM 3 Adapter-only",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_worker(path, model, fingerprint):
    manifest = read(path / "checksums.json")
    if (
        manifest["status"] != "completed"
        or manifest["model"] != model
        or manifest["provider_fingerprint"] != fingerprint
        or manifest["source"]["git_commit"] != WORKER_COMMIT
        or str(manifest["source"]["workflow"]["GITHUB_RUN_ID"]) != WORKER_RUN
        or manifest["release"] != "2026-11-26"
    ):
        raise ValueError(f"Worker identity mismatch: {model}")
    expected = set()
    for item in manifest["files"]:
        name = item["path"]
        file = path / name
        if (
            name not in SOURCE_FILES
            or name in expected
            or file.is_symlink()
            or not file.is_file()
            or file.stat().st_size != item["bytes"]
            or sha(file) != item["sha256"]
        ):
            raise ValueError(f"Worker artifact checksum or allowlist mismatch: {name}")
        expected.add(name)
    if {p.name for p in path.iterdir()} != expected | {"checksums.json"}:
        raise ValueError(f"Unexpected worker artifacts: {model}")


def verify_shard(path, model, shard, fingerprint):
    receipt = read(path / "shard.json")
    identity = receipt["identity"]
    if (
        receipt["status"] != "completed"
        or receipt["model"] != model
        or receipt["shard"] != shard
        or receipt["count"] != 4
        or identity["model"] != model
        or identity["git_commit"] != SHARD_COMMIT
        or identity["provider_fingerprint"] != fingerprint
        or str(read(path / "source.json")["workflow"]["GITHUB_RUN_ID"]) != SHARD_RUN
    ):
        raise ValueError(f"Shard identity mismatch: {model}/{shard}")
    names = {item["path"] for item in receipt["files"]}
    allowed = {
        "fit_manifest.json", "fit_samples.json", "training_curves.json",
        "resolved_config.json", "data_source.json", "protocol.json", "source.json",
        "validation_predictions.csv.gz", "predictions.csv.gz",
    }
    if names != allowed or {p.name for p in path.iterdir()} != names | {"shard.json"}:
        raise ValueError(f"Unexpected shard files: {model}/{shard}")
    if any(p.is_symlink() or not p.is_file() for p in path.iterdir()):
        raise ValueError("Shard files must be regular files.")


def run_script(name, *args):
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name(name)), *map(str, args)],
        check=True,
    )


def replace_block(source, start, end, content):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Missing or duplicate documentation markers: {start}")
    left, rest = source.split(start, 1)
    _, right = rest.split(end, 1)
    return left + start + "\n" + content.strip() + "\n" + end + right


def fmt(value, kind="number"):
    if value is None:
        return "—"
    return f"{value:+.2%}" if kind == "pct" else f"{value:.6f}"


def result_table(summary):
    rows = [
        "| Model | Validation Rank IC ↑ | Test RMSE ↓ | Test Rank IC ↑ | "
        "Qlib simulated net return | Max drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = summary["models"][0]
    rows += [
        f"| Zero-return forecast | — | {fmt(baseline['zero_return_rmse'])} | — | — | — |",
        f"| CSI300 benchmark | — | — | — | {fmt(baseline['benchmark_return'], 'pct')} | — |",
    ]
    for row in summary["models"]:
        model = row["model"]
        name = DISPLAY[model]
        if model == summary["primary_model"]:
            name = f"**{name} (validation-selected)**"
        fields = [
            name, fmt(row["validation_rank_ic"]), fmt(row["test_rmse"]),
            fmt(row["test_rank_ic"]), fmt(row["net_return"], "pct"),
            fmt(row["max_drawdown"], "pct"),
        ]
        rows.append("| " + " | ".join(fields) + " |")
    return "\n".join(rows)


def validate_gallery(path, manifest):
    from PIL import Image

    if manifest["status"] != "completed":
        raise ValueError("Chart generation was incomplete")
    charts = {item["name"]: item for item in manifest["charts"]}
    for name in REQUIRED_CHARTS:
        if name not in charts or not charts[name]["native_function"].startswith("qlib."):
            raise ValueError(f"Missing native diagnostic: {name}")
        with Image.open(path / charts[name]["png"]) as im:
            if min(im.size) < 100:
                raise ValueError(f"Unexpected image dimensions: {name}")
            im.verify()
    for name, info in manifest["files"].items():
        if Path(name).name != name or sha(path / name) != info["sha256"]:
            raise ValueError(f"Chart export hash mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.output.resolve()
    if work.exists():
        raise ValueError("Use a fresh staging directory.")
    bundle = work / "bundle"
    target = bundle / EVIDENCE
    shutil.copytree(EVIDENCE, target)
    protocol = read(target / "protocol.json")
    fingerprint = protocol["provider_fingerprint"]
    before = read(target / "comparison.json")
    if set(before["pending"]) != set(WORKERS + SHARDED):
        raise ValueError("Unexpected pending roster; do not overwrite existing results.")

    sources = {}
    for model in WORKERS:
        source = args.workers / f"chenditc-20261126-{model}-1"
        verify_worker(source, model, fingerprint)
        sources[model] = source
    for model in SHARDED:
        paths = [args.shards / f"official-shard-{model}-{i}-1" for i in range(4)]
        for i, path in enumerate(paths):
            verify_shard(path, model, i, fingerprint)
        source = work / "assembled" / model
        assemble_shards(paths, source)
        sources[model] = source

    for model, source in sources.items():
        print(f"Replaying {model}", flush=True)
        replay = work / "replays" / model
        run_script("replay_cn_portfolio.py", "--run-dir", source,
                   "--provider", args.provider, "--output", replay)
        collect(source, replay, target / model)

    # Do not accept a partial audit. The registered roster must be complete.
    run_script("audit_cn_evidence.py", "--evidence", target, "--provider", args.provider)
    audit = read(target / "audit.json")
    if (
        audit["status"] != "passed"
        or audit["pending_models"]
        or audit["failures"]
        or set(audit["audited_models"]) != set(protocol["models"])
    ):
        raise ValueError("Full numerical audit did not pass.")
    summary = summarize_cn_benchmark(target)
    if summary["status"] != "completed" or summary["pending"] or len(summary["models"]) != 12:
        raise ValueError("Full 12-model comparison is not complete.")
    primary = summary["primary_model"]
    if primary not in DISPLAY:
        raise ValueError("No validation-selected chart model.")
    gallery = bundle / GALLERY
    manifest = export_qlib_run(
        target / primary, gallery, model_name=DISPLAY[primary],
        dataset_label="chenditc 2026-11-26 · historical CSI300",
    )
    validate_gallery(gallery, manifest)

    qualification = (
        "All 12 declared model setups are complete and independently audited. "
        "The primary scratch model was selected by full-validation daily Rank IC, "
        "not test returns. Results use one declared seed (42); the separate "
        "official TimesFM setups have different inputs and fitting budgets. "
        "These are native Qlib simulations, not an executable profit claim. "
        "The source's terminated-series/corporate-action valuation issue remains unresolved."
    )
    table = result_table(summary)
    link = "[Full comparison CSV](reports/chenditc_csi300/comparison.csv) · [Numerical audit](reports/chenditc_csi300/audit.json)"
    readme = Path("README.md").read_text(encoding="utf-8")
    readme = replace_block(
        readme, "<!-- CHENDITC_RESULTS_START -->", "<!-- CHENDITC_RESULTS_END -->",
        qualification + "\n\n" + table + "\n\n" + link,
    )
    chart_labels = {
        "portfolio": "Native Qlib portfolio report",
        "score-ic": "Native Qlib daily IC and rank IC",
        "monthly-ic": "Native Qlib monthly IC heatmap",
        "group-return": "Native Qlib score-quintile return diagnostics",
        "risk-summary": "Native Qlib risk analysis",
    }
    gallery_text = (
        f"**Validation-selected model: {DISPLAY[primary]}.** "
        "These figures are generated from the audited full test forecasts and "
        "deterministic Qlib portfolio replay. "
        "[Offline interactive report](docs/assets/qlib/report.html) · "
        "[Chart provenance and hashes](docs/assets/qlib/manifest.json)\n\n"
    )
    gallery_text += "\n\n".join(
        f"![{chart_labels[name]}](docs/assets/qlib/{name}.png)" for name in REQUIRED_CHARTS
    )
    gallery_text += "\n\n[All additional native Qlib diagnostics](docs/assets/qlib/)"
    readme = replace_block(
        readme, "<!-- CHENDITC_GALLERY_START -->", "<!-- CHENDITC_GALLERY_END -->",
        gallery_text,
    )
    (bundle / "README.md").write_text(readme, encoding="utf-8")
    docs = Path("docs/chenditc-benchmark.md")
    detail = replace_block(
        docs.read_text(encoding="utf-8"),
        "<!-- CHENDITC_DOC_RESULTS_START -->", "<!-- CHENDITC_DOC_RESULTS_END -->",
        qualification + "\n\n" + table + "\n\n"
        "[Full comparison](../reports/chenditc_csi300/comparison.csv) · "
        "[Audit](../reports/chenditc_csi300/audit.json) · "
        "[Interactive chart report](assets/qlib/report.html)",
    )
    (bundle / docs).write_text(detail, encoding="utf-8")
    receipt = {
        "status": "completed",
        "worker_run": WORKER_RUN,
        "shard_run": SHARD_RUN,
        "recovered_models": list(sources),
        "audited_models": audit["audited_models"],
        "primary_model": primary,
        "charts": list(manifest["files"]),
        "corporate_action_qualification": audit["qualification"],
    }
    (target / "verification/finalization_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
