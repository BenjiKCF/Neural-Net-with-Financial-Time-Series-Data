# Workflow and configuration guide

Start with [the beginner guide](getting-started.md) for installation and your first run.
This page covers data contracts, model configuration, evaluation, and run artifacts.

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

Choose a YAML from the [model table](../README.md#model-catalog), then set the provider, output directory, and training parameters. Paths are relative to your working directory; these examples assume the repository root. The six trainable models share feature definitions, the raw target, split rules, and evaluation exports. Neural models additionally receive historical windows. The official TimesFM adapter reads provider-adjusted close histories; see its [separate setup guide](architectures/timesfm.md#timesfm3_pretrained).

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

Read [docs/architecture-validation.md](architecture-validation.md) for execution evidence from this revision rather than inferring model quality from a passing software check. The earlier [modernization validation](validation.md) records the initial two-baseline migration. The removed notebook chain and its evaluation defects are documented in [docs/legacy-audit.md](legacy-audit.md).

## References

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [Qlib data and adjustment conventions](https://qlib.readthedocs.io/en/stable/component/data.html)
- [Qlib benchmark implementations](https://github.com/microsoft/qlib/tree/v0.9.7/examples/benchmarks)
- [Qlib PyPI distribution](https://pypi.org/project/pyqlib/)
- [PyTorch](https://pytorch.org/)

## Dependency policy

The installable baseline is pinned to `pyqlib==0.9.7`, the published PyPI release checked for this modernization. MLflow and filelock are bounded to avoid known fresh-install compatibility changes; NumPy/Pandas major versions are also bounded. Exact versions are recorded in each `run.json`.

Upstream [Qlib v0.9.8](https://github.com/microsoft/qlib/releases/tag/v0.9.8) is a newer **source** release with artifact-loading hardening. It was not published on PyPI when this migration was verified. This package does not claim compatibility with an untested source upgrade. The supported CLI uses data-only YAML and numeric market data and only reloads artifacts just created by its own run. Qlib 0.9.7 pickle artifacts must come from trusted local training; never unpickle downloaded models or datasets from an untrusted source.
