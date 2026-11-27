# Chenditc data, CSI300 benchmark, and native Qlib charts

This is the primary recent-data workflow. It downloads and verifies a named
chenditc Qlib release, trains the repository's models on historical CSI300
membership, and evaluates them with a shared China-market portfolio simulation.
The earlier SPX/SPY results remain available in the [archived research report](financial-research.md).

## Which data are used?

The campaign resolves **[chenditc/investment_data release 2026-11-26](https://github.com/chenditc/investment_data/releases/tag/2026-11-26)**,
the latest available release checked on **27 November 2026**. Release publication
and market coverage are distinct: the archive's last observed trading date is
**26 November 2026**. The importer excludes the separate future calendar from
market observations.

| Property | Verified snapshot |
| --- | --- |
| Source archive | `qlib_bin.tar.gz`, 573,153,280 bytes |
| Archive SHA-256 | `5c7d2391e569896f22fdccd146f9a0466fc363b7bbabb528c68bd2cc56f11559` |
| Observed calendar | 2000-01-04–2026-11-26; 6,518 trading sessions |
| Full provider | 6,178 instruments; 17,600,664 finite close observations |
| Instruments with a finite close on the cutoff | 5,583 |
| CSI300 membership file | 949 historical instruments across 1,227 intervals; 300 active at the cutoff |
| Other supplied universes | CSI500, CSI800, CSI1000, and broader instrument lists |
| Missing bars | Missing values remain missing; no price forward fill |
| Future-calendar entries | 25 scheduling dates, excluded from observed data |

The complete archive is downloaded, but **this benchmark trains and trades the
historical CSI300 universe**. It does not train on all 6,178 instruments. The
provider includes daily prices, volume, amount, adjustment factors, and index
membership files. The scratch-model handler uses the project's existing **157
OHLCV factors**; official TimesFM uses adjusted-close histories.

`fts download-chenditc` resolves the GitHub release once, checks archive size and
SHA-256 against the release asset and matching provenance manifest, validates
archive paths and dates, and inspects the extracted close files. It writes
`metadata.json` beside the provider's `calendars`, `instruments`, and `features`
directories. The benchmark records that metadata and its hash in `protocol.json`
before training. See the [downloader](../src/financial_ts/chenditc.py) and
[campaign implementation](../src/financial_ts/cn_benchmark.py).

The bundled `^GSPC.csv` is a different dataset: **7,252 daily S&P 500 index rows
from 1990-01-02 to 2018-10-10**. Earlier reported tests used 2013–2018 SPX signals,
with SPY adjusted prices for the trading simulation. An RMSE from that single
index is not directly comparable with an RMSE across hundreds of Chinese
stocks in 2025–2026.

## Chronological protocol

| Segment | Signal dates | Purpose |
| --- | --- | --- |
| Historical warmup | 2019, before the training window | Build causal rolling features and price contexts |
| Train | 2020-01-02–2023-12-27 | Fit feature normalization, target scaling, and model weights |
| Validation | 2024-01-02–2024-12-27 | Select checkpoints and the primary model |
| Test | 2025-01-02–2026-11-23 | Report the declared models' forecast and portfolio results |

A feature window ending after close **t** predicts
`Ref($close, -2) / Ref($close, -1) - 1`: the return from close **t+1** to close
**t+2**. Two observed trading sessions are purged at each split boundary, so
training labels do not cross into validation and validation labels do not
cross into test. The last three observed sessions reserve the two label closes
and Qlib's executor boundary. A newer release may extend the test endpoint;
use the pinned release below to reproduce this exact period.

The primary model is chosen by the **highest mean daily validation Rank IC
among the eight scratch models**, with ties resolved by model name. Checkpoint
selection stays within validation. The test table includes all declared models,
including underperforming ones. Choosing a chart subject by test profit would
be a different protocol.

## Models and training budget

The declared seed is **42**. This campaign is a single-seed comparison; the
three-seed dispersion from the old SPX study does not apply to these results.

| Family | Models | Inputs and fitting budget |
| --- | --- | --- |
| Boosted-tree control | `lightgbm` | All eligible training rows; 157 current-day factors; up to 300 boosting rounds and validation early stopping |
| Scratch neural models | `transformer`, `deepseek_v41_mini`, `glm54_mini`, `qwen4_mini`, `timesfm3_mini`, `multiscale_residual_mixer`, `regime_linear_moe` | All eligible training rows; 32-session factor windows; width 32; batch 512; AdamW, learning rate 0.001; up to six epochs and early stopping patience four |
| Frozen official TimesFM | `timesfm3_frozen_128`, `timesfm3_frozen_256` | Released checkpoint, no weight updates; 128 or 256 adjusted closes per forecast |
| Official adaptation | `timesfm3_loop_128`, `timesfm3_adapter_128` | Frozen base; 128-close contexts; 1,024 training samples, 512 validation-selection samples, three epochs; complete validation and test inference afterward |

The loop reuses pretrained layers and trains gates/adapters. The adapter-only
control has a matched trainable parameter count, with different computation.
They are distinct from `timesfm3_mini`, which is a small, randomly initialized
numeric architecture. Official-model pretraining and inputs differ from the
scratch models; this is an explicit comparison of practical forecasting setups,
not a claim that all models received identical information or compute.

The original ES and directional weight-edit experiments remain archived as
SPX-only results. They are not silently relabeled as CSI300 measurements.

## Portfolio simulation

All declared models use Qlib's `TopkDropoutStrategy` with the same parameters.

| Setting | Value |
| --- | --- |
| Universe | Historical CSI300 membership |
| Portfolio rule | Target the 50 highest-ranked stocks; drop up to five holdings at rebalance |
| Benchmark | `SH000300`, CSI300 index |
| Initial account | CNY 100,000,000 |
| Invested target | 95% |
| Execution | Close on the session after the signal |
| Purchase cost | 0.0005 of traded value, or 5 basis points |
| Sale cost | 0.0015 of traded value, or 15 basis points |
| Minimum cost | CNY 5 per order |
| Trade unit | 100-share lots, using Qlib's adjustment-factor handling |
| Limit threshold | Fixed 9.5% daily price-change filter |
| End of run | Remaining positions marked to market |

These are declared simulation settings. They do not recreate every board's
price-limit rules, every historical fee change, slippage, or market impact.
The account size is an accounting input, not demonstrated strategy capacity.

### Terminated series and portfolio valuation

The completed LightGBM backtest exposes a concrete limitation in the source
and native simulator. `SH600837` has no new source prices after **2025-02-05**, but
the strategy continues to hold it. Qlib carries its last adjusted quote,
**38.66765**, for **440 portfolio dates**, through 2026-11-25. That stale-valued
holding is **1.9434% of the final simulated account value**.

The data audit also identifies **98 missing stock-days within recorded
membership intervals** across three terminated series: `SH600837`, `SH601989`,
and `SH601059`. Historical members remain in the benchmark; deleting them
after observing their termination would change the universe retrospectively.
Correct merger/termination valuation has not yet been established.

For this reason, tables label these figures **Qlib simulated net return**.
LightGBM's measured **+8.58% simulated net return**, compared with **+2.04% for the
benchmark**, is an output of these accounting assumptions. It does not establish
an investable return until the corporate-action treatment is resolved. Native
portfolio charts inherit the same limitation. Forecast RMSE, IC, and Rank IC
are evaluated on the available finite labels, with unavailable labels counted
separately; they do not repair the portfolio valuation issue.

## Results

<!-- CHENDITC_DOC_RESULTS_START -->
**Status: running.** No complete CSI300 result table is published yet. The
campaign preserves per-model `run.json`, `metrics.json`, predictions, and
portfolio reports as models finish. Only completed artifacts will populate
the result table and measured chart gallery.
<!-- CHENDITC_DOC_RESULTS_END -->

Read **RMSE** as return forecast error, **IC** as a daily cross-sectional Pearson
correlation, and **Rank IC** as the corresponding Spearman correlation.
Forecast accuracy and portfolio profit measure different outcomes. A model
can have a lower RMSE and still rank stocks poorly or trade too often.

The comparison summary calculates net portfolio return by compounding
`1 + return - cost`. Its annualized return uses 252 sessions, and its maximum
drawdown is measured against a running compounded peak including the initial
account. The native Qlib chart conventions below are different and explicitly
labeled.

## Read the native Qlib charts

`fts qlib-report` uses the actual functions in
[`qlib.contrib.report`](https://github.com/microsoft/qlib/tree/v0.9.7/qlib/contrib/report).
It adds presentation labels without replacing the native calculations.

| Figure | What it shows | How to read it |
| --- | --- | --- |
| `portfolio.png` | Native seven-panel portfolio report | Portfolio and benchmark returns, costs, turnover, and drawdowns; net series include the declared costs |
| `score-ic.png` | Daily IC and rank IC | Positive values indicate useful same-day cross-sectional ranking; daily variation matters |
| `monthly-ic.png` | Monthly mean IC heatmap | Inspect consistency across months; an empty cell means no observation |
| `group-return.png` | Score-quintile return curves | Group 1 has the highest scores; these are label-return diagnostics before costs |
| `risk-summary.png` | Native excess-return risk analysis | Compare excess returns, volatility, information ratio, and drawdown before/after costs |
| Other exports | Daily IC bars, IC/Q-Q and group distributions, monthly risk, score autocorrelation | Diagnose instability, changing regimes, and persistent signals |

**Qlib's default charts accumulate daily returns by addition.** Their curves
and drawdowns are not compounded NAV or compound drawdown. Native daily risk
analysis in the pinned Qlib version uses **238 sessions per year**. In
particular, the native “annualized return” is a mean daily return scaled by 238;
it is not CAGR. These conventions are recorded in the exported manifest and
interactive report.

The quintile plots do not implement the Top50 execution simulation. They show
equal-weight future label returns grouped by score, before transaction costs.
Their long-short line is a diagnostic spread, not evidence of an executable
short portfolio in this market.

The native heatmap auto-scales its palette to the observed IC range. Read the
numerical color bar rather than treating a warm color alone as positive IC.

Each export includes static PNGs, Plotly JSON files, `report.html` with embedded
JavaScript for offline viewing, and `manifest.json` recording source functions,
package versions, data-frame hashes, date coverage, and file hashes. Undefined
statistics are reported explicitly; the exporter does not invent missing
correlations or distributions.

## Reproduce the campaign

After the [installation guide](getting-started.md), download the current release:

```bash
fts download-chenditc --provider .data/cn --release latest
fts benchmark-cn --provider .data/cn --output runs/csi300
```

For the exact dataset used here, pin the release and use a new provider folder:

```bash
fts download-chenditc --provider .data/cn-20261126 --release 2026-11-26
fts benchmark-cn --provider .data/cn-20261126 --output runs/csi300-20261126
```

The eight scratch architectures run by default. To include all four official
TimesFM variants, install the optional dependencies and follow the
[checkpoint download guide](architectures/timesfm.md#timesfm3_pretrained), then
start the campaign with a local checkpoint:

```bash
python -m pip install -e ".[timesfm,visualization]"
fts benchmark-cn --provider .data/cn-20261126 --output runs/csi300-all-20261126 --timesfm-checkpoint .data/timesfm3-checkpoint
```

The complete market panel is much larger than the bundled SPX example. The
official checkpoint also requires approximately 1.32 GB of local weight files.
CPU is supported; use `--device cuda` when the selected environment has a
supported CUDA PyTorch installation. Runtime depends on the model and machine.

Inspect these files as the campaign runs:

| Output | Contents |
| --- | --- |
| `protocol.json` | Pinned source metadata, split dates, roster, budgets, and selection rule |
| `comparison.csv` / `comparison.json` | Available completed results and the campaign's pending models |
| `<model>/run.json` | Run status, data provenance, package versions, and timing |
| `<model>/metrics.json` | Full validation/test forecast metrics and portfolio summary |
| `<model>/predictions.csv` | Dated per-stock test scores and labels |
| `<model>/portfolio.csv` | Native Qlib daily portfolio returns, costs, benchmark, and turnover |
| `<model>/learning_curves.json` | Available training/validation checkpoint history |

Generate charts for a completed model:

```bash
fts qlib-report --run-dir runs/csi300-20261126/lightgbm --output runs/csi300-20261126/lightgbm-plots --model LightGBM --dataset-label "chenditc 2026-11-26 · CSI300"
```

Open the generated `report.html` locally. For the README gallery, use the model
selected by the registered validation rule, and retain the comparison table
for all models. A chart export requires a new or empty output directory.

## Interpretation limits

Historical membership intervals avoid applying only today's constituent list
to the whole past. They do not certify that every underlying price adjustment
or membership correction was available at its original publication time.
The upstream snapshot may contain revisions. The archived source and hashes
make this run reproducible against this snapshot, not a certified publication-time
database.

One seed measures one initialization. It does not quantify seed robustness,
and a historical replay is not prospective validation. Official TimesFM's
pretraining overlap remains unknown. These limits remain relevant even when
a model shows positive IC or a profitable simulated portfolio.
