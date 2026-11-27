# Measured inference precision profile

**Result, 27 November 2026: retain FP32.** CPU bfloat16 completed every call but
failed the predeclared maximum-rank-shift gate. Its median measured latency was
11.53% higher on this shared CPU host. This is a one-date inference
measurement, with no change to training, model selection, or deployment defaults.

[Registered protocol](../reports/production_roadmap/precision_profile/protocol.json) ·
[Summary](../reports/production_roadmap/precision_profile/summary.json) ·
[Input receipt](../reports/production_roadmap/precision_profile/input_receipt.json) ·
[Per-stock comparison](../reports/production_roadmap/precision_profile/score_comparison.csv)

## What was measured

The actual **quantformer_moe**, fold 000, seed 7 model from the registered
CSI300 screening campaign was already trained and exported when this profile
started. The profile used immutable bundle
`cb751278b5fa266773bcf4909c693f47439a8fa50ca39b6e92d2d4f3b444d7e4`. It did not substitute a toy model or refit weights.

The input was the complete historical CSI300 cross section on **2024-08-21**:
**300 stocks × 32 sessions × 157 features**.
The feature builder requested historical factors only, applied historical universe
membership, and ended every sequence on that decision date. No return labels
were requested. Both precisions used identical weights, raw features, fitted
normalization and padding. The bundle identity and weight hash were unchanged
at completion.

Each full stock cross section stayed together in one model call. Splitting
stocks into independent microbatches would change cross-stock attention and
would therefore change the model being measured.

The protocol fixed **2 warmup calls and 10 measured calls per precision**,
one intra-operation CPU thread, one inter-operation thread, a fresh subprocess
for each precision, and FP32 followed by bfloat16. Bundle loading and its retained
fixture replay occurred before those calls. All measured repetitions within
each precision produced identical scores in this run.

Environment: AMD EPYC 9V74 80-Core Processor; 9
logical CPUs available to the process; Python 3.12.14; PyTorch
2.15.0+cpu; Qlib 0.9.7.
The research campaign was running on the same host, so small timing differences
are subject to scheduling and mode-order effects.

## Latency and memory

| Measurement | FP32 | CPU bfloat16 |
| --- | ---: | ---: |
| Successful measured calls | 10/10 | 10/10 |
| Median call latency | 104.361 ms | 116.395 ms |
| 95th-percentile call latency | 131.059 ms | 131.229 ms |
| Mean call latency | 108.603 ms | 118.576 ms |
| Highest sampled resident set during measured calls | 483.73 MiB | 471.24 MiB |
| Largest sampled increase above a call's starting RSS | 43.83 MiB | 0.01 MiB |
| Process-lifetime high-water RSS | 603.00 MiB | 603.00 MiB |

The timed call includes NumPy preprocessing, CPU tensor preparation, full-panel
inference and rescaling to raw-return units. It excludes factor extraction,
bundle loading, warmup and output publication. Factor preparation, including
provider identity checks, took **69.086 seconds**;
the numerical input occupied **12,067,200 bytes**. These costs are recorded
separately because reducing model arithmetic does not remove input preparation.

RSS was sampled every 2 ms on a separate thread, with samples
before and after each call. Brief allocation peaks can fall between samples.
Linux `resource.getrusage(...).ru_maxrss × 1024` reports a separate process-lifetime
high-water value, which includes imports, bundle/data loading and warmup. These
are process memory measurements, not PyTorch allocator peaks. The measurements
do not establish a general memory saving from bfloat16.

The full individual observations are retained in
[float32.json](../reports/production_roadmap/precision_profile/float32.json) and
[bfloat16.json](../reports/production_roadmap/precision_profile/bfloat16.json).

## Numerical and ranking acceptance

Tolerances were written to `protocol.json` before feature preparation and
precision trials. They were not changed after inspecting the result.

| Registered check | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Maximum absolute return-score difference | ≤ 0.0001 | 0.000029917 | Pass |
| Spearman correlation of stock ranks | ≥ 0.999 | 0.999809435 | Pass |
| Maximum ordinal rank movement | ≤ 3 places | 9 places | **Fail** |
| Top-20 membership overlap | ≥ 95% | 100.0% | Pass |
| Finite, successful warmup and measured calls | Every call | Every call | Pass |

The maximum score difference was about
**0.299 basis points of return**. Nevertheless,
**64.33%** of stocks changed ordinal rank;
the mean absolute movement was **1.267 places**.
Bfloat16 introduced **237 tied stock pairs**, against
**0** in FP32. The deterministic ordering is score descending,
then instrument identifier ascending. Small score errors can change rankings
when scores are close together, which is why the profile checks ranks as well
as absolute error.

The top-20 set was unchanged on this date. That observation does not establish
unchanged portfolio decisions across dates, holdings, dropout policies or costs.
The predeclared rank gate failed and the measured median did not improve, so
this experiment supplies no basis for enabling bfloat16 by default.

## Reproduce

Run after the numerical bundle exists:

```bash
python scripts/profile_inference.py \
  --bundle runs/roadmap-screening-20261127/trials/fold-000/quantformer_moe/seed-7/attempt-0001/workflow/inference_bundle \
  --provider .data/chenditc/2026-11-26 \
  --decision-date 2024-08-21 \
  --warmup 2 --repetitions 10 --num-threads 1 \
  --score-atol 0.0001 --min-rank-correlation 0.999 \
  --max-rank-shift 3 --topk 20 --min-topk-overlap 0.95 \
  --output runs/precision-profile-replay
```

Use a new output directory for every experiment; the script rejects overwriting
existing evidence. The source bundle and provider must remain separate from the
output. The stored protocol records exact package versions, source-file hashes,
weight identity, input identity, mode order, tolerance values and memory method.
Returned scores and errors remain visible even when a precision fails.

This uses PyTorch's documented
[`torch.autocast("cpu", dtype=torch.bfloat16)` inference pattern](https://docs.pytorch.org/docs/2.15/amp.html).
Autocast selects operation precision; it does not replace the stored weights
with a separately trained model. No gradients or optimizer steps occur here.
Broader adoption would require multiple dates, representative machines,
portfolio-sensitive ranking checks and a declared operational latency budget.
