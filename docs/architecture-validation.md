# Architecture validation — 27 November 2026

All six trainable models completed the declared three-seed historical comparison
before the PR update. The official TimesFM 3 checkpoint also completed a separate
fixed-weight replay. All **114 automated tests passed** locally on Python 3.12 CPU.

Evidence: [full JSON](../reports/architecture_benchmark.json),
[18 supervised trials](../reports/architecture_benchmark.csv), and
[aligned labels and all 19 model forecasts](../reports/architecture_predictions.csv).
The JSON contains full configurations, learning curves, package versions, data and
source hashes, all seeds, predeclared protocols, and independent checkpoint audit.

## Fixed comparison protocol

The retained `^GSPC.csv` contains 7,252 observations from 1990-01-02 to 2018-10-10.
All scratch models use the same 157 OHLCV factors and
`Ref($close, -2) / Ref($close, -1) - 1` target. Yahoo price adjustments apply before
feature construction. Feature and neural target normalization fit on training only.

| Partition | Feature dates |
| --- | --- |
| Training | 1990-03-28 through 2007-05-04 |
| Validation | 2007-05-09 through 2013-01-18 |
| Test | 2013-01-24 through 2018-10-08 |

Actual-session purging prevents labels crossing split boundaries. There are
**1,438 aligned, finite test predictions per run**, all with observed labels.
The target permits a decision after close t and entry at close t+1.

The declared seeds are 7, 42, 2026. Each scratch neural model uses a 32-session
window, width 32, four heads, a depth setting of 2, batch size 128, AdamW
(`lr=0.001`, `weight_decay=0.0001`), dropout 0.1, auxiliary coefficient 0.01,
at most 12 epochs, and patience 4. The best epoch is selected solely by validation
forecast MSE. No seed is discarded or selected using test error. LightGBM uses
single-date factors and up to 300 boosting rounds with patience 30.

Depth is architecture-specific: DeepSeek has two encoder plus two decoder layers;
GLM has two shared blocks run twice; Qwen has two groups of three delta mixers plus
one sparse-attention mixer; TimesFM has two temporal/variate mixing layers.
Parameters and computation are consequently unequal. These are small research
configurations, not a controlled equal-compute comparison or an optimized recipe.

## Historical supervised results

RMSE is in decimal return units; lower is better. The final column is the
population standard deviation across all three seeds, **not a confidence interval**.
Validation RMSE differs substantially from test RMSE because these periods include
different return distributions; it should only be compared within the same column.

| Model | Parameters | Mean validation RMSE | Mean test RMSE | Test RMSE seed std |
| --- | ---: | ---: | ---: | ---: |
| Zero-return forecast | 0 | — | 0.00773293 | — |
| LightGBM | trees | 0.01598985 | 0.00773053 | 0.00000000 |
| Temporal Transformer | 30,561 | 0.01599798 | 0.00777545 | 0.00002057 |
| DeepSeek V4.1 Flash 1017 mini | 103,969 | 0.01601134 | 0.00788319 | 0.00003098 |
| GLM-5.4 mini | 45,189 | 0.01602342 | 0.00787864 | 0.00003437 |
| Qwen4 mini | 122,258 | 0.01601151 | 0.00783621 | 0.00002633 |
| TimesFM 3 mini | 30,157 | 0.01599370 | 0.00772459 | 0.00000243 |

TimesFM 3 mini has the lowest mean test RMSE in this fixed experiment. Its
improvement over zero return is only about 0.108% in relative RMSE. LightGBM is
also essentially tied with zero return; the other neural models are worse.
The more elaborate architectures therefore do not show a general advantage on
this small single-index dataset under these settings.

This evidence reports the run outcomes without selecting a trading strategy.
The interval was already inspected during the initial migration, so it is an
exploratory historical comparison, not a fresh final holdout. One index, one
chronological split and three seeds cannot establish statistical significance,
current-market generalization, or trading profitability. No portfolio backtest
is attached to this index because the index itself is not directly tradable.

## Actual official TimesFM 3 replay

The separately downloaded official checkpoint at
`43046b85ec22d584a13f8098c2ed39c889e129c2` contains **330,710,976 parameters**.
Using `timesfm[torch]==3.0.2`, the adapter consumes 256 observed provider-adjusted
closes per prediction and forecasts the next two closes in univariate mode.
It converts them to `forecast[t+2] / forecast[t+1] - 1`, matching the same target.
It loads fixed weights without fitting this repository's labels.

| Forecast | Test predictions | Test RMSE |
| --- | ---: | ---: |
| Zero return | 1,438 | 0.00773293 |
| Official pretrained TimesFM 3 | 1,438 | 0.00780899 |

The official replay is worse than the zero-return forecast on this sample.
Its historical pretraining overlap is unknown and its inputs/training history
differ from the supervised models; it must not be interpreted as a clean
unseen-data comparison against them.

An independent check recomputed every label and both RMSE values, then reloaded
the checkpoint for the first, middle, and last prediction dates. The captured
inputs were exactly 256 closes ending at each date, with no future observations.
All three scores reproduced exactly (maximum difference 0.0). The Qlib adapter
pickle is 450 bytes and contains no pretrained weights. The report records the
actual checkpoint hashes and 1,322,898,824-byte weight-file size.

Google's separate [TimesFM Non-Commercial License](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE)
applies to the official weights. This repository does not redistribute them.
See [TimesFM documentation](architectures/timesfm.md) for download and usage.

## Software verification and scope

- **114 tests passed**, with 11 third-party warnings; no failures or skips.
- All six scratch models completed the offline demo with 576 predictions each,
  native Qlib records, and costed synthetic portfolio reports.
- All four mini models trained through real Qlib in integration tests. Reloaded
  model and dataset artifacts reproduced every score/index exactly, with feature
  normalizer refitting forbidden during replay.
- Mechanism tests cover causal behavior, missing-observation isolation, expert
  and indexer gradients, shared loop parameters and caches, actual delta
  recurrence, residual gates, prefix normalization, and forecast refinement.
- The official adapter's mocked tests check its contract; the actual checkpoint
  replay above is separate evidence of working official inference.
- Ruff checks and formatting checks pass. CI is configured for Python 3.10 and 3.12,
  the complete suite, and a six-model offline demo. Fresh CI status belongs to the
  updated PR, not to the older two-model run.

Actual official pretrained inference was exercised on Python 3.12.14, CPU
PyTorch 2.15.0; compatibility metadata is not an executed Python 3.10 checkpoint
test. GPU execution has not been verified. Several training subsets ran in
parallel, so recorded elapsed times are not a controlled speed benchmark.

A final-output JSON conversion bug in neural-only benchmark subsets was found
after their trial artifacts were saved. It was fixed without changing numerical
training or forecasts. A strict serialization regression test and an actual
single-model CLI benchmark completed successfully afterward. Every historical
trial and exported metric was checked independently before inclusion here.

## Reproduce

From an installed checkout, use new provider/output directories:

```bash
python -m pytest -q
fts demo --model all --epochs 1 --device cpu --output runs/check-all
fts import-csv --csv "^GSPC.csv" --symbol SPX --provider .data/check-spx
fts benchmark --provider .data/check-spx --output runs/check-architectures --models lightgbm transformer deepseek_v41_mini glm54_mini qwen4_mini timesfm3_mini --seeds 7 42 2026 --epochs 12 --device cpu
```

For the official model, install the optional extra and download the pinned
checkpoint as documented, then run:

```bash
fts train --config configs/timesfm3_pretrained.yaml --provider .data/check-spx --output runs/check-official-timesfm3 --device cpu --no-backtest
```

The older initial migration results remain in [validation.md](validation.md).
