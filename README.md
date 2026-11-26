# Neural Networks with Financial Time-Series Data

**A reproducible Qlib research workflow with a PyTorch temporal Transformer and a LightGBM benchmark.**

Version 2 replaces the supported execution path of the original 2017–2018 TensorFlow/Keras project. It includes validated daily OHLCV ingestion, causal feature processing, chronological training/validation/test splits, prediction exports, experiment tracking, and optional portfolio backtesting.

The original notebooks, data, and images remain available as historical material. Start with the commands below for the maintained workflow. See [the legacy audit](docs/legacy-audit.md) for the concrete compatibility and evaluation problems addressed.

## What is included

| Component | Current implementation |
| --- | --- |
| Data | Local adjusted CSV → validated Qlib binary provider; supports the bundled Yahoo-style index CSV |
| Features | 157 OHLCV-computable factors from Qlib Alpha158; excludes the true-VWAP field unavailable in ordinary OHLCV |
| Neural model | PyTorch temporal Transformer over 20 historical sessions, positional encoding, AdamW, clipping, validation early stopping |
| Benchmark model | Qlib LightGBM with validation early stopping |
| Splits | Chronological 60/20/20 by default, with label-horizon purging using actual provider sessions |
| Preprocessing | Robust feature normalization fitted on training dates only; raw return labels |
| Evaluation | RMSE, MAE, zero-return forecast baseline, directional accuracy; daily IC/Rank IC for multiple instruments |
| Portfolio | Optional Qlib top-k strategy, next-close execution, explicit trading costs, gross/net reports |
| Reproducibility | Seed, resolved task, source/provider hashes, package versions, git revision, MLflow records |
| Verification | Offline synthetic demo, unit/integration tests, CPU CI |

A successful demo verifies the software. It does not establish profitable forecasting or trading.

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

For the neural model:

```bash
python -m pip install -e ".[neural]"
fts demo --model both --epochs 3 --output runs/demo-both
```

The demo creates synthetic data locally and exercises feature generation, both requested models, prediction recording, portfolio simulation, and cost reporting. After dependencies are installed it requires no network access or API keys. Choose a new output path for each run.

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
```

This is a historical **single-index forecasting** example. The index itself is not a directly tradable security, so these commands evaluate predictions without a portfolio simulation. Cross-sectional label normalization is deliberately absent: ranking or z-scoring a single instrument would destroy its target.

The bundled data is old. Replace it with a recent, appropriately licensed dataset for current-market research.

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
fts train --config configs/transformer.yaml --provider .data/market --output runs/market-transformer
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

Feature normalization is fitted on training dates and reused for validation and test. Neural target standardization is also fitted only on training labels and inverted before prediction. Early stopping sees validation data only. Training batches may shuffle within their already fixed partition. Historical sequence context may extend before a validation/test boundary; no sequence uses observations later than its own prediction timestamp.

## Configure models and backtesting

Edit [configs/lightgbm.yaml](configs/lightgbm.yaml) or [configs/transformer.yaml](configs/transformer.yaml). Paths are relative to your working directory; the examples assume the repository root. Both models use the same feature definitions, raw target, split rules, and evaluation exports. The Transformer also receives historical windows.

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
| `metrics.json` | Forecast metrics, zero-return baseline, Qlib statistics, portfolio results when enabled |
| `portfolio.csv` | Daily portfolio return, benchmark return, trading costs, turnover, and account values |
| `resolved_config.json` | Effective paths, dates, model settings, and execution configuration |
| `qlib_task.yaml` | Resolved native Qlib task for inspection/replay in the installed environment |
| `run.json` | Run status, seed-containing configuration reference, data/source hashes, package versions, git revision |
| `mlruns/` | Qlib/MLflow model, fitted dataset metadata, predictions, and analysis artifacts |

A failed run records its error where possible. Run directories are never silently overwritten. Use a new name for the next attempt.

The model implements Qlib's `fit(dataset)` and `predict(dataset, segment="test")` interface. Neural inference requests features only, retains a final one-sample batch, and returns scores in the original units and exact sample order.

## Development and verification

```bash
python -m pip install -e ".[dev,neural]"
ruff check src tests
python -m pytest -q
fts demo --model both --epochs 2 --output runs/check
```

The tests cover CSV adjustment/validation, actual Qlib binary round-trips, calendar-aware purging, train-only preprocessing under future-data perturbation, small-batch neural training, singleton inference, label/index alignment, and portfolio costs. CI runs on Python 3.10 and 3.12 with CPU PyTorch.

The old notebooks and `Old version/` are historical references, excluded from the supported package and CI. Their obsolete downloaders and dependency/API patterns are described in [docs/legacy-audit.md](docs/legacy-audit.md).

## References

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [Qlib data and adjustment conventions](https://qlib.readthedocs.io/en/stable/component/data.html)
- [Qlib benchmark implementations](https://github.com/microsoft/qlib/tree/v0.9.7/examples/benchmarks)
- [Qlib PyPI distribution](https://pypi.org/project/pyqlib/)
- [PyTorch](https://pytorch.org/)
