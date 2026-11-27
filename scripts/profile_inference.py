#!/usr/bin/env python3
"""Profile immutable same-date stock-panel inference at FP32 and CPU BF16.

No model fitting, label requests, weight mutation or stock-axis microbatching.
Each precision runs in a fresh subprocess; the retained numerical bundle is
verified before and after the experiment. Numerical failure is a measured
outcome, never a reason to silently omit a precision or change its thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_ts.bundles import (  # noqa: E402
    NumericalPredictor,
    atomic_json,
    file_hash,
    verify_bundle,
)


def _now():
    return pd.Timestamp.now(tz="UTC").isoformat()


def _path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _array_hash(array):
    values = np.asarray(array, dtype="<f8").copy()
    values[np.isnan(values)] = np.nan
    digest = hashlib.sha256(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _environment():
    versions = {}
    for name in ("torch", "numpy", "pandas", "pyqlib", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu = "unknown"
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": psutil.cpu_count(),
        "process_available_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "packages": versions,
        "load_average_1_5_15m": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }


def _process_peak_rss():
    # Linux ru_maxrss is KiB; this is process-lifetime high-water memory,
    # including imports/model/data loading and warmup, not an allocator peak.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1 if sys.platform == "darwin" else 1024))


class _RSSSampler:
    def __init__(self, interval_ms):
        self.interval = interval_ms / 1000
        self.process = psutil.Process()
        self.before = self.process.memory_info().rss
        self.peak = self.before
        self.samples = 1
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)
            self.samples += 1

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.after = self.process.memory_info().rss
        self.peak = max(self.peak, self.after)
        self.samples += 1

    def receipt(self):
        return {
            "rss_before_bytes": self.before,
            "rss_after_bytes": self.after,
            "sampled_peak_rss_bytes": self.peak,
            "sampled_peak_increase_bytes": max(self.peak - self.before, 0),
            "rss_samples": self.samples,
            "process_lifetime_peak_rss_bytes": _process_peak_rss(),
        }


def _worker(protocol_path, input_path, mode, output):
    import torch

    protocol = json.loads(Path(protocol_path).read_text())
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    predictor = NumericalPredictor(protocol["bundle_path_absolute"])
    if predictor.info["kind"] != "cross_sectional":
        raise ValueError("This profile requires the complete-panel cross-sectional adapter.")
    if predictor.manifest["bundle_id"] != protocol["bundle_id"]:
        raise ValueError("Worker resolved a different immutable bundle.")
    # This in-memory override changes autocast policy only. On-disk constructor,
    # weights, preprocessing, raw input and full stock panel remain identical.
    predictor.info["constructor"]["precision"] = mode
    torch.set_num_threads(protocol["num_threads"])
    predictor.network.eval()
    with np.load(input_path, allow_pickle=False) as data:
        features, padding = data["features"], data["padding"]
    load_seconds = time.perf_counter() - started
    records, outputs = [], []
    for phase, count in (("warmup", protocol["warmup"]), ("measured", protocol["repetitions"])):
        for repeat in range(count):
            row = {"phase": phase, "repeat": repeat, "mode": mode}
            with _RSSSampler(protocol["rss_sample_interval_ms"]) as memory:
                begin = time.perf_counter_ns()
                try:
                    scores = predictor.predict(features, padding=padding)
                    if scores.shape != (len(features),) or not np.isfinite(scores).all():
                        raise ValueError("Nonfinite or incorrectly shaped profile scores.")
                    row.update(status="completed", scores_sha256=_array_hash(scores))
                    if phase == "measured":
                        outputs.append(scores.copy())
                        row["score_row"] = len(outputs) - 1
                except Exception as error:
                    row.update(status="failed", error=f"{type(error).__name__}: {error}")
                row["latency_ms"] = (time.perf_counter_ns() - begin) / 1e6
            row.update(memory.receipt())
            records.append(row)
    measured = [row for row in records if row["phase"] == "measured"]
    successful = [row for row in measured if row["status"] == "completed"]
    latency = np.array([row["latency_ms"] for row in successful])
    result = {
        "mode": mode,
        "status": "completed"
        if all(row["status"] == "completed" for row in records)
        else "numerical_failure",
        "environment": _environment(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "bundle_load_and_input_seconds": load_seconds,
        "rows": records,
        "measured_successes": len(successful),
        "measured_failures": len(measured) - len(successful),
        "latency_ms": {
            "p50": float(np.quantile(latency, 0.5)) if len(latency) else None,
            "p95": float(np.quantile(latency, 0.95)) if len(latency) else None,
            "mean": float(latency.mean()) if len(latency) else None,
            "std_population": float(latency.std()) if len(latency) else None,
            "min": float(latency.min()) if len(latency) else None,
            "max": float(latency.max()) if len(latency) else None,
        },
        "maximum_sampled_rss_bytes": max(row["sampled_peak_rss_bytes"] for row in measured),
        "maximum_sampled_rss_increase_bytes": max(
            row["sampled_peak_increase_bytes"] for row in measured
        ),
        "process_lifetime_peak_rss_bytes": _process_peak_rss(),
        "within_mode_max_abs_repeat_difference": float(
            np.max(np.abs(np.stack(outputs) - outputs[0]))
        )
        if outputs
        else None,
        "weights_sha256_after": file_hash(Path(protocol["bundle_path_absolute"]) / "weights.npz"),
    }
    output = Path(output)
    atomic_json(output.with_suffix(".json"), result)
    np.savez_compressed(
        output.with_suffix(".npz"),
        scores=np.stack(outputs) if outputs else np.empty((0, len(features))),
    )


def _rank_summary(reference, candidate, instruments, topk):
    frame = pd.DataFrame({"instrument": instruments, "fp32": reference, "bfloat16": candidate})
    for mode in ("fp32", "bfloat16"):
        frame[mode + "_average_rank"] = frame[mode].rank(method="average", ascending=False)
        ordered = frame.sort_values([mode, "instrument"], ascending=[False, True]).index
        rank = pd.Series(np.arange(1, len(frame) + 1), index=ordered)
        frame[mode + "_ordinal_rank"] = rank.reindex(frame.index).to_numpy()
    difference = np.abs(frame.bfloat16 - frame.fp32)
    rank_shift = np.abs(frame.bfloat16_ordinal_rank - frame.fp32_ordinal_rank)
    correlation = frame.fp32_average_rank.corr(frame.bfloat16_average_rank)
    first = set(frame.loc[frame.fp32_ordinal_rank <= topk, "instrument"])
    second = set(frame.loc[frame.bfloat16_ordinal_rank <= topk, "instrument"])
    i, j = np.triu_indices(len(frame), k=1)
    original_sign, candidate_sign = (
        np.sign(reference[i] - reference[j]),
        np.sign(candidate[i] - candidate[j]),
    )
    strict = (original_sign != 0) & (candidate_sign != 0)
    summary = {
        "max_abs_score_difference": float(difference.max()),
        "mean_abs_score_difference": float(difference.mean()),
        "score_difference_rmse": float(np.sqrt(np.mean((candidate - reference) ** 2))),
        "spearman_rank_correlation": float(correlation) if np.isfinite(correlation) else None,
        "max_ordinal_rank_shift": int(rank_shift.max()),
        "mean_ordinal_rank_shift": float(rank_shift.mean()),
        "fraction_with_changed_ordinal_rank": float((rank_shift != 0).mean()),
        "topk": topk,
        "topk_overlap_fraction": len(first & second) / topk,
        "baseline_tied_pairs": int((original_sign == 0).sum()),
        "bfloat16_tied_pairs": int((candidate_sign == 0).sum()),
        "discordant_strict_pairs": int(((original_sign != candidate_sign) & strict).sum()),
        "strict_pairs": int(strict.sum()),
        "tie_break": "score descending then instrument ascending",
    }
    return summary, frame


def profile(args):
    from financial_ts.operations import build_provider_packet

    bundle = Path(args.bundle).expanduser().resolve()
    provider = Path(args.provider).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ValueError("Profile output must be new; prior measurements are immutable.")
    for protected in (bundle, provider):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError("Profile output must be separate from model and provider.")
    if args.warmup < 1 or args.repetitions < 3 or args.num_threads < 1 or args.topk < 1:
        raise ValueError(
            "Use positive warmup/threads/topk and at least three measured repetitions."
        )
    if (
        args.rss_sample_interval_ms <= 0
        or args.score_atol <= 0
        or not 0 <= args.min_rank_correlation <= 1
    ):
        raise ValueError("Invalid numerical or memory sampling tolerances.")
    manifest = verify_bundle(bundle)
    metadata = json.loads((bundle / "metadata.json").read_text())
    model = json.loads((bundle / "model.json").read_text())
    if model["kind"] != "cross_sectional":
        raise ValueError("Select an actual cross-sectional numerical model bundle.")
    date = pd.Timestamp(args.decision_date)
    cutoff = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=20)
    fit_end = pd.Timestamp(metadata["fit_end"])
    if pd.isna(fit_end) or fit_end.tzinfo is None or fit_end >= cutoff:
        raise ValueError("Profile decision date must follow the recorded model selection cutoff.")
    protocol = {
        "schema_version": 1,
        "registered_at_utc": _now(),
        "bundle_path": _path(bundle),
        "bundle_path_absolute": str(bundle),
        "bundle_id": manifest["bundle_id"],
        "weights_sha256": file_hash(bundle / "weights.npz"),
        "architecture": model["constructor"]["architecture"],
        "provider_path": _path(provider),
        "expected_provider_fingerprint": metadata["data_fingerprint"],
        "decision_date": str(date.date()),
        "mode_order": ["float32", "bfloat16"],
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "num_threads": args.num_threads,
        "rss_sample_interval_ms": args.rss_sample_interval_ms,
        "tolerances": {
            "max_abs_score_difference": args.score_atol,
            "minimum_spearman_rank_correlation": args.min_rank_correlation,
            "maximum_ordinal_rank_shift": args.max_rank_shift,
            "minimum_topk_overlap_fraction": args.min_topk_overlap,
        },
        "topk": args.topk,
        "batching": "one complete eligible decision-date stock cross section; never split stock attention",
        "timed_scope": "NumPy preprocessing, CPU tensor preparation, full-panel model inference and raw-return rescaling; excludes bundle load, factor extraction, file publication and warmup",
        "memory_method": "psutil process RSS sampled on a separate thread at the registered interval around each call, with before/after samples; Linux ru_maxrss*1024 records process-lifetime high-water RSS including imports/load/warmup. Sampled peaks may miss short transients and are not allocator peaks.",
        "precision_method": "Identical FP32 weights; change only in-memory constructor.precision so the existing NumericalPredictor uses torch.autocast(cpu, dtype=bfloat16) for eligible operations. Normalization and returned raw-return rescaling retain their existing precision.",
        "isolation": "fresh subprocess per precision, sequential fixed mode order; host may concurrently run the research campaign",
        "script_sha256": file_hash(__file__),
        "source_files": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in sorted((ROOT / "src" / "financial_ts").rglob("*.py"))
        },
        "environment": _environment(),
        "decision_rule": "Retain FP32 unless every measured BF16 run is finite, all registered numerical/rank tolerances pass, and measured p50 improves; this one-day diagnostic alone never changes deployment defaults.",
    }
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", protocol)
    started = time.perf_counter()
    packet, features, padding = build_provider_packet(
        provider,
        bundle,
        decision_date=args.decision_date,
        decision_cutoff=cutoff.isoformat(),
        data_available_at=(cutoff - pd.Timedelta(hours=1)).isoformat(),
    )
    preparation_seconds = time.perf_counter() - started
    if packet["data_fingerprint"] != metadata["data_fingerprint"]:
        raise ValueError("Provider changed from the pinned bundle training snapshot.")
    if packet["instruments"] != packet["expected_instruments"]:
        raise ValueError("The profile requires the complete eligible current stock panel.")
    if len(features) < args.topk:
        raise ValueError("Top-k comparison exceeds the available complete stock panel.")
    if padding is None or padding.shape != features.shape[:-1] or padding[:, -1].any():
        raise ValueError("Invalid complete-panel padding.")
    feature_receipt = {
        "packet": packet,
        "shape": list(features.shape),
        "input_bytes": int(features.nbytes + padding.nbytes),
        "features_sha256": _array_hash(features),
        "padding_sha256": hashlib.sha256(padding.tobytes()).hexdigest(),
        "provider_feature_preparation_seconds": preparation_seconds,
        "labels_requested": False,
        "arrival_timestamp_qualification": "Historical profiling clock only; not evidence of actual source publication time.",
    }
    atomic_json(output / "input_receipt.json", feature_receipt)
    mode_results, scores = {}, {}
    with tempfile.TemporaryDirectory(prefix="fts-precision-") as temporary:
        temporary = Path(temporary)
        input_path = temporary / "features.npz"
        np.savez(input_path, features=features, padding=padding)
        for mode in protocol["mode_order"]:
            target = temporary / mode
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-mode",
                mode,
                "--worker-protocol",
                str(output / "protocol.json"),
                "--worker-input",
                str(input_path),
                "--worker-output",
                str(target),
            ]
            try:
                run = subprocess.run(command, capture_output=True, text=True, timeout=60)
                if run.returncode:
                    raise RuntimeError(f"Worker exited {run.returncode}: {run.stderr[-3000:]}")
                result = json.loads(target.with_suffix(".json").read_text())
                with np.load(target.with_suffix(".npz"), allow_pickle=False) as data:
                    scores[mode] = data["scores"].copy()
                result["worker_stderr_tail"] = run.stderr[-3000:]
            except Exception as error:
                result = {"status": "worker_failed", "error": f"{type(error).__name__}: {error}"}
            mode_results[mode] = result
            atomic_json(output / f"{mode}.json", result)
    comparisons, gates = None, {}
    if all(mode in scores and len(scores[mode]) for mode in protocol["mode_order"]):
        comparisons, rows = _rank_summary(
            scores["float32"][0], scores["bfloat16"][0], packet["instruments"], args.topk
        )
        rows.to_csv(output / "score_comparison.csv", index=False, float_format="%.17g")
        gates = {
            "all_calls_succeeded": all(
                mode_results[mode]["status"] == "completed" for mode in protocol["mode_order"]
            ),
            "max_abs_score_difference": comparisons["max_abs_score_difference"] <= args.score_atol,
            "rank_correlation": comparisons["spearman_rank_correlation"] is not None
            and comparisons["spearman_rank_correlation"] >= args.min_rank_correlation,
            "maximum_rank_shift": comparisons["max_ordinal_rank_shift"] <= args.max_rank_shift,
            "topk_overlap": comparisons["topk_overlap_fraction"] >= args.min_topk_overlap,
        }
    fp32_median = mode_results["float32"].get("latency_ms", {}).get("p50")
    bf16_median = mode_results["bfloat16"].get("latency_ms", {}).get("p50")
    ratio = fp32_median / bf16_median if fp32_median and bf16_median else None
    integrity = (
        verify_bundle(bundle)["bundle_id"] == protocol["bundle_id"]
        and file_hash(bundle / "weights.npz") == protocol["weights_sha256"]
    )
    result = {
        "schema_version": 1,
        "status": "completed",
        "architecture": protocol["architecture"],
        "bundle_id": protocol["bundle_id"],
        "decision_date": protocol["decision_date"],
        "shape": list(features.shape),
        "protocol_sha256": file_hash(output / "protocol.json"),
        "input_receipt_sha256": file_hash(output / "input_receipt.json"),
        "bundle_unchanged": integrity,
        "modes": {
            mode: {
                key: value
                for key, value in data.items()
                if key not in {"rows", "worker_stderr_tail"}
            }
            for mode, data in mode_results.items()
        },
        "score_and_rank_comparison": comparisons,
        "numerical_gates": gates,
        "numerical_acceptance": "passed" if gates and all(gates.values()) else "failed",
        "fp32_p50_over_bfloat16_p50": ratio,
        "measured_bfloat16_p50_improved": bool(ratio and ratio > 1),
        "deployment_default": "float32",
        "promotion_changed": False,
        "limitations": [
            "Single historical full cross section and one CPU environment; no training-speed, GPU, market-quality or production-latency claim.",
            "Other work may share this host; fixed mode order and shared CPU load confound small latency differences.",
            "RSS includes native runtime and model/input memory; sampled interval peaks are lower bounds on brief transients, while process-lifetime high-water includes startup.",
            "Passing numerical tolerances on one date does not establish robust rank stability across dates or authorize changing production defaults.",
        ],
        "finished_at_utc": _now(),
    }
    atomic_json(output / "summary.json", result)
    print(
        json.dumps(
            {
                "output": _path(output),
                "architecture": result["architecture"],
                "numerical_acceptance": result["numerical_acceptance"],
                "latency_ratio_fp32_over_bf16": ratio,
                "comparison": comparisons,
            },
            indent=2,
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle")
    parser.add_argument("--provider")
    parser.add_argument("--decision-date")
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--rss-sample-interval-ms", type=float, default=2.0)
    parser.add_argument("--score-atol", type=float, default=0.0001)
    parser.add_argument("--min-rank-correlation", type=float, default=0.999)
    parser.add_argument("--max-rank-shift", type=int, default=3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--min-topk-overlap", type=float, default=0.95)
    parser.add_argument("--worker-mode", choices=("float32", "bfloat16"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-protocol", help=argparse.SUPPRESS)
    parser.add_argument("--worker-input", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_mode:
        return _worker(
            args.worker_protocol, args.worker_input, args.worker_mode, args.worker_output
        )
    if not all((args.bundle, args.provider, args.decision_date, args.output)):
        parser.error("--bundle, --provider, --decision-date and --output are required")
    return profile(args)


if __name__ == "__main__":
    main()
