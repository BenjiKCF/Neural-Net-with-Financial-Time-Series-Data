# Neural Networks with Financial Time-Series Data

**A reproducible Qlib workflow for compact DeepSeek V4.1, GLM-5.4, Qwen4, and TimesFM 3 architecture experiments, with Transformer and LightGBM comparison baselines.**

Version 2.1 adds four independently implemented PyTorch models trained from random initialization, plus a separate adapter for Google's official pretrained TimesFM 3. The shared workflow includes validated daily OHLCV ingestion, causal feature processing, chronological training/validation/test splits, prediction exports, experiment tracking, and optional portfolio backtesting.

The obsolete notebooks, scripts, cached models/data, and output images have been removed from the working tree. The original material remains in [Git history at commit `15e0640`](https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data/tree/15e0640c799e106c3a3fb8901e25663a13197574); `^GSPC.csv` is retained for the offline historical example. See [the legacy audit](docs/legacy-audit.md) for the cleanup inventory and evaluation problems addressed.

## What is included

| Component | Current implementation |
| --- | --- |
| Data | Local adjusted CSV → validated Qlib binary provider; supports the bundled Yahoo-style index CSV |
| Features | 157 OHLCV-computable factors from Qlib Alpha158; excludes the true-VWAP field unavailable in ordinary OHLCV |
| Trainable architectures | Four compact numeric adaptations of DeepSeek-V4.1-Flash-1017, GLM-5.4, Qwen4, and TimesFM 3 |
| Comparison baselines | PyTorch temporal Transformer and Qlib LightGBM |
| Optional fixed-weight model | Official pretrained TimesFM 3 through an explicit local-checkpoint adapter |
| Neural training | Historical windows, AdamW, gradient clipping, auxiliary routing/index losses where applicable, validation early stopping |
| Splits | Chronological 60/20/20 by default, with label-horizon purging using actual provider sessions |
| Preprocessing | Robust feature normalization fitted on training dates only; raw return labels |
| Evaluation | RMSE, MAE, zero-return forecast baseline, directional accuracy; daily IC/Rank IC for multiple instruments |
| Portfolio | Optional Qlib top-k strategy, next-close execution, explicit trading costs, gross/net reports |
| Reproducibility | Seed, resolved task, source/provider hashes, package versions, git revision, MLflow records |
| Verification | Offline all-model demo, mechanism and integration tests, CPU CI, reproducible multi-seed historical comparison |

A successful demo verifies the software. It does not establish profitable forecasting or trading.

## Model choices and provenance

The four `*_mini` models implement selected mechanisms disclosed for the named versions. They use continuous financial factors and a scalar return head, with dimensions reduced for CPU experiments. **They contain no pretrained or distilled flagship weights and inherit none of the source models' pretrained capabilities or published performance.** The architecture notes distinguish implemented, adapted, and omitted mechanisms and link the primary sources.

| Model name / configuration | Architecture retained in the numeric adaptation | Details |
| --- | --- | --- |
| [`deepseek_v41_mini`](configs/deepseek_v41_mini.yaml) | **DeepSeek-V4.1-Flash-1017**: causal encoder/decoder, CSA2-inspired compressed retrieval, shared decoder KV and index reuse, four-stream mHC, routed and shared experts | [DeepSeek notes](docs/architectures/deepseek.md) |
| [`glm54_mini`](configs/glm54_mini.yaml) | **GLM-5.4**: repeated shared blocks, separate per-pass gates, reused global KV/indices plus current local KV, sparse experts | [GLM notes](docs/architectures/glm_qwen.md#glm-54-adaptation) |
| [`qwen4_mini`](configs/qwen4_mini.yaml) | **Qwen4**: three gated-delta mixers per sparse-attention mixer, four gated residual streams, sparse experts, causal numeric substitute for lexical N-grams | [Qwen notes](docs/architectures/glm_qwen.md#qwen4-adaptation) |
| [`timesfm3_mini`](configs/timesfm3_mini.yaml) | **TimesFM 3**: temporal/variate attention, patched latent variates, contiguous masked horizon, prefix normalization and forecast-statistic refinement | [TimesFM notes](docs/architectures/timesfm.md#timesfm3_mini) |
| [`transformer`](configs/transformer.yaml) | Plain temporal Transformer for comparison | Shared Qlib training and target |
| [`lightgbm`](configs/lightgbm.yaml) | Gradient-boosted trees for comparison | Single-date factors with validation early stopping |

The supplied mini configurations use width 32, four heads, a 32-session history, and a maximum of 12 epochs. `num_layers` has architecture-specific meaning: DeepSeek counts layers per encoder/decoder stage, GLM counts shared blocks, Qwen counts four-mixer groups, and TimesFM counts temporal/variate mixing layers. Equal settings therefore do not imply equal depth, parameters, or compute. Run artifacts report the actual parameter count.

`timesfm3_pretrained` is an additional, separate choice using [Google's official TimesFM 3 checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch). Its weights have a separate noncommercial license; see [official pretrained inference](#official-pretrained-timesfm-3). List all accepted model names with `fts models` after installation.

## Install

Use **Python 3.10–3.12** in a fresh virtual environment. Python 3.12 is the locally verified environment.

```bash
git clone https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data.git
cd Neural-Net-with-Financial-Time-Series-Data
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the base workflow and run the offline LightGBM check:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
fts demo --output runs/demo-lightgbm
```

For all six trainable models, including the four new architectures:

```bash
python -m pip install -e ".[neural]"
fts demo --model all --epochs 2 --device cpu --output runs/demo-all
```

The demo creates synthetic data locally and exercises feature generation, training, prediction recording, portfolio simulation, and cost reporting. `--model all` runs LightGBM, the Transformer, and all four minis; it writes a `comparison.json` plus a directory per model. After dependencies are installed it requires no network access or API keys. It does not load or download the optional official checkpoint. Choose a new output path for each run.

Use `--model deepseek_v41_mini` (or another model name) for a single architecture. The existing `--model both` option runs only the LightGBM and Transformer baselines.

For a smaller **Linux/Windows CPU-only** PyTorch install, install PyTorch before the neural extra:

```bash
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[neural]"
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch build appropriate to the machine using the [official installation selector](https://pytorch.org/get-started/locally/). Set `device: cuda` or pass `--device cuda`. `auto` selects CUDA if available, otherwise CPU; CPU also works on macOS. GPU execution has not been validated by this repository's CPU CI.

### Dependency policy

The installable baseline is pinned to `pyqlib==0.9.7`, the published PyPI release checked for this modernization. MLflow and filelock are bounded to avoid known fresh-install compatibility changes; NumPy/Pandas major versions are also bounded. Exact versions are recorded in each `run.json`.

Upstream [Qlib v0.9.8](https://github.com/microsoft/qlib/releases/tag/v0.9.8) is a newer **source** release with artifact-loading hardening. It was not published on PyPI when this migration was verified. This package does not claim compatibility with an untested source upgrade. The supported CLI uses data-only YAML and numeric market data and only reloads artifacts just created by its own run. Qlib 0.9.7 pickle artifacts must come from trusted local training; never unpickle downloaded models or datasets from an untrusted source.

## Run on the original S&P 500 history

The repository already contains `^GSPC.csv`, so this path also works offline:

```bash
fts import-csv --csv "^GSPC.csv" --symbol SPX --provider .data/spx-history
fts train --config configs/lightgbm.yaml --provider .data/spx-history --output runs/spx-lightgbm --no-backtest
fts train --config configs/transformer.yaml --provider .data/spx-history --output runs/spx-transformer --device cpu --no-backtest
fts train --config configs/deepseek_v41_mini.yaml --provider .data/spx-history --output runs/spx-deepseek --device cpu --no-backtest
```

This is a historical **single-index forecasting** example. The index itself is not a directly tradable security, so these commands evaluate predictions without a portfolio simulation. Cross-sectional label normalization is deliberately absent: ranking or z-scoring a single instrument would destroy its target.

The bundled data is old. Replace it with a recent, appropriately licensed dataset for current-market research.

### Reproduce the three-seed architecture comparison

After importing the provider above, run the declared six-model comparison:

```bash
fts benchmark --provider .data/spx-history --output runs/spx-architecture-benchmark --models lightgbm transformer deepseek_v41_mini glm54_mini qwen4_mini timesfm3_mini --seeds 7 42 2026 --epochs 12 --device cpu
```

This trains each model under each seed. All six use the same 157 factors, delayed-return target, and chronological partitions; the five neural models use the same 32-session context and optimizer settings. LightGBM receives one factor row and uses boosting. Neural parameter counts, effective depth, and runtime differ. Early stopping uses validation only; the command does not choose a winner from test scores or run a portfolio backtest.

The output contains the declared `protocol.json`, individual run directories, per-run `comparison.csv`, mean/dispersion `aggregate.csv`, and `comparison.json`. The protocol records the provider fingerprint and complete configurations before training. The final comparison includes parameter counts, epochs, validation and test metrics, and timing. Dispersion across three seeds is a measure of initialization sensitivity, not a statistical confidence interval.

See [architecture validation](docs/architecture-validation.md) and its [machine-readable evidence](reports/architecture_benchmark.json) for recorded results and limitations. These results are an exploratory comparison on historical data; a previously inspected test interval is not a fresh final holdout. The earlier two-model modernization evidence is retained separately in [docs/validation.md](docs/validation.md).

### Official pretrained TimesFM 3

The small supervised `timesfm3_mini` and the official `timesfm3_pretrained` adapter have different training histories and input contracts. The adapter loads fixed Google weights, forecasts two future closes from up to 256 historical closes, and converts them to the same delayed-return target. It uses each instrument independently in univariate mode; `fit` loads weights without fitting to this repository's labels. Its default context also differs from the scratch neural models' 32-session factor windows.

Install the optional official implementation:

```bash
python -m pip install -e ".[timesfm]"
```

Obtain the [official checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch/tree/43046b85ec22d584a13f8098c2ed39c889e129c2) separately, retaining its license, and set `model.params.checkpoint_path` in [configs/timesfm3_pretrained.yaml](configs/timesfm3_pretrained.yaml) to its local directory. The directory must contain `config.json` and `model.safetensors`. The adapter does not download weights, and this repository does not redistribute them.

```bash
fts train --config configs/timesfm3_pretrained.yaml --provider .data/spx-history --output runs/spx-timesfm3-pretrained --device cpu --no-backtest
```

For a separately recorded fixed-weight comparison through the benchmark CLI:

```bash
fts benchmark --provider .data/spx-history --output runs/spx-timesfm3-official --models timesfm3_pretrained --timesfm-checkpoint /absolute/path/to/timesfm-3.0-pytorch --device cpu
```

The pretrained model runs once rather than repeating identical fixed weights for each seed. Its pretraining overlap with the bundled historical market data is unknown, so this is a pretrained inference experiment with different provenance from the scratch models. Checkpoint loading and inference evidence must be identified separately from mocked adapter tests.

Google's code is Apache-2.0, while the checkpoint uses the separate [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE), restricted to noncommercial, nonproduction use and prohibiting redistribution of the model or derivatives. The code license does not grant commercial rights to the weights. The [TimesFM architecture notes](docs/architectures/timesfm.md) document the adapter, pinned implementation, checkpoint revision, and limitations.

## Import your own market data

Supported long-form CSV columns:

```csv
date,symbol,open,high,low,close,volume
2024-01-02,AAA,100,102,99,101,1000000
2024-01-03,AAA,101,103,100,102,1100000
2024-01-02,BBB,50,51,49,50.5,500000
2024-01-03,BBB,50.5,52,50,51.5,600000
```

The rows above only illustrate the schema; training needs substantially more history.

```bash
fts import-csv --csv /path/to/market.csv --provider .data/market
fts train --config configs/lightgbm.yaml --provider .data/market --output runs/market-lightgbm
fts train --config configs/qwen4_mini.yaml --provider .data/market --output runs/market-qwen --no-backtest
```

A directory of per-symbol CSV files is also accepted. Without a `symbol` column, the filename stem supplies the symbol, or use `--symbol` for one CSV. Header names are case-insensitive, and Yahoo `Date`/`Adj Close` names are recognized.

**Adjustment contract:**

- With `Adj Close`/`adj_close`, the input is treated as raw Yahoo-style OHLCV. The importer computes `factor = adj_close / close`, multiplies every OHLC price by it, and divides volume by it.
- Without `adj_close`, OHLCV must already use a consistent adjustment convention, or describe a security without relevant corporate actions. An optional `factor` is retained without adjusting a second time; otherwise it is 1.
- True VWAP is not recoverable from a daily OHLCV bar. The feature handler excludes `VWAP0` instead of substituting a made-up value.

Dates must be daily, timezone-naive session dates. Duplicate symbol/date pairs, invalid prices, inconsistent bars, unsafe symbols, and negative volumes are rejected. Missing instrument bars remain missing. The calendar is the union of observed dates; provide every relevant market session because sessions absent from every input instrument cannot be reconstructed.

Each import writes `metadata.json` with source hashes, adjustment semantics, and coverage. Existing populated providers are protected from accidental replacement.

The importer accepts a historical universe you supply; it does not reconstruct historical index membership or delisted securities. A current constituent list can introduce survivorship bias. Vendor revisions and corporate-action adjustments also need point-in-time scrutiny before interpreting performance.

## Target, timing, and leakage controls

At the close of session **t**, the model uses features available through that close and predicts:

```text
close[t+2] / close[t+1] - 1
```

This is Qlib's delayed next-session **simple return** target. It allows a decision after close t and entry at close t+1. The old notebooks targeted next-day log returns, so their screenshots are not comparable to this workflow's metrics.

Default splits:

1. Reserve the first 60 sessions for feature history.
2. Divide the remaining prediction period chronologically at approximately 60% / 20% / 20%.
3. Remove the final two feature dates before each next partition so their labels cannot cross the boundary.
4. Reserve two sessions after the last prediction for observed labels. Portfolio runs reserve one additional real session for Qlib's executor boundary.

You may specify explicit ranges in YAML:

```yaml
segments:
  train: [2000-01-03, 2010-12-28]
  valid: [2011-01-03, 2014-12-29]
  test: [2015-01-05, 2018-06-29]
```

These are illustrative ranges and must lie inside your actual provider. The validator resolves included sessions and rejects overlapping label horizons or insufficient future coverage. It does not approximate a two-session gap with two elapsed days.

Feature normalization is fitted on training dates and reused for validation and test. For the scratch neural models, target standardization is also fitted only on training labels and inverted before prediction. Early stopping sees validation data only. Training batches may shuffle within their already fixed partition. Historical sequence context may extend before a validation/test boundary; no sequence uses observations later than its own prediction timestamp.

## Configure models and backtesting

Choose a YAML from the [model table](#model-choices-and-provenance), then set the provider, output directory, and training parameters. Paths are relative to your working directory; these examples assume the repository root. The six trainable models share feature definitions, the raw target, split rules, and evaluation exports. Neural models additionally receive historical windows. The official TimesFM adapter reads provider-adjusted close histories and is documented separately above.

Set common neural dimensions directly under `model.params`. Architecture-specific options belong under `architecture_kwargs`, for example:

```yaml
model:
  name: glm54_mini
  params:
    d_model: 32
    nhead: 4
    num_layers: 2
    aux_loss_weight: 0.01
    architecture_kwargs:
      loop_steps: 2
      sparse_k: 8
      local_window: 4
```

The shared trainer adds the weighted auxiliary loss for models that expose one, while reporting and early-stopping on forecast MSE. Target scaling is undone before evaluation.

For a US equity universe with at least five instruments, set the desired portfolio parameters and run:

```bash
fts train --config configs/lightgbm.yaml --provider .data/market --output runs/market-portfolio --backtest
```

Relevant configuration:

```yaml
backtest:
  enabled: true
  topk: 5
  n_drop: 1
  account: 100000
  risk_degree: 0.95
  benchmark: SPY  # must exist in the provider; null is an explicit zero-return cash benchmark
  open_cost: 0.001
  close_cost: 0.001
  min_cost: 0.0
```

The default costs are illustrative 10-basis-point fees per buy/sell. Replace them with assumptions suitable for the instruments and broker. Backtesting uses next-session close prices, fractional shares, and a long-only top-k portfolio. It does not model order-book fills, market impact, borrow, taxes, or latency, and final positions are marked to market without forced liquidation. Results include actual simulated cost deductions.

The supplied portfolio assumptions are for US-market research. Chinese-market feature/model training can use a compatible provider with `region: cn`; portfolio simulation is rejected until appropriate exchange rules are implemented.

For serious evaluation, predeclare model selection criteria, use multiple chronological/walk-forward periods, include realistic costs and a defensible benchmark, and reserve a fresh final holdout. This repository provides one transparent holdout workflow, not an automatic strategy search or live-trading system.

## Run artifacts

Each completed output directory includes:

| File/directory | Contents |
| --- | --- |
| `predictions.csv` | Exact datetime/instrument index, model score, and raw delayed-return label |
| `metrics.json` | Forecast metrics, zero-return baseline, parameter/training details, Qlib statistics, portfolio results when enabled |
| `learning_curves.json` | Training and validation forecast MSE by epoch for scratch neural models |
| `portfolio.csv` | Daily portfolio return, benchmark return, trading costs, turnover, and account values; present when backtesting is enabled |
| `resolved_config.json` | Effective paths, dates, model settings, and execution configuration |
| `qlib_task.yaml` | Resolved native Qlib task for inspection/replay in the installed environment |
| `run.json` | Run status, seed-containing configuration reference, data/source hashes, package versions, git revision |
| `mlruns/` | Qlib/MLflow model, fitted dataset metadata, predictions, and analysis artifacts |

A failed run records its error where possible. Run directories are never silently overwritten. Use a new name for the next attempt.

The model adapters implement Qlib's `fit(dataset)` and `predict(dataset, segment="test")` interface. Inference never requests labels, retains a final one-sample batch, and returns scores in the original units and exact sample order. Saved scratch models retain their learned parameters. A saved official TimesFM adapter retains its configuration and local checkpoint path; restoring it requires the checkpoint to remain available.

## Development and verification

```bash
python -m pip install -e ".[dev,neural]"
ruff check src tests
python -m pytest -q
fts demo --model all --epochs 2 --device cpu --output runs/check-all
```

The tests cover CSV adjustment/validation, actual Qlib binary round-trips, calendar-aware purging, train-only preprocessing under future-data perturbation, small-batch neural training, singleton inference, label/index alignment, and portfolio costs. Architecture tests exercise causal masks, padding isolation, sparse expert/router gradients, learned indexers, memory reuse, gated-delta recurrence, residual-stream gates, patch normalization/refinement, and saved-model replay. The pretrained adapter has separate contract tests. CI uses Python 3.10 and 3.12 with CPU PyTorch.

Read [docs/architecture-validation.md](docs/architecture-validation.md) for execution evidence from this revision rather than inferring model quality from a passing software check. The earlier [modernization validation](docs/validation.md) records the initial two-baseline migration. The removed notebook chain and its evaluation defects are documented in [docs/legacy-audit.md](docs/legacy-audit.md).

## References

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [Qlib data and adjustment conventions](https://qlib.readthedocs.io/en/stable/component/data.html)
- [Qlib benchmark implementations](https://github.com/microsoft/qlib/tree/v0.9.7/examples/benchmarks)
- [Qlib PyPI distribution](https://pypi.org/project/pyqlib/)
- [PyTorch](https://pytorch.org/)
