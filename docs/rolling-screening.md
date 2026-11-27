# Completed CSI300 rolling screen

**27 November 2026: 40 of 40 registered trials completed, with no failed training attempts.**
The validation rule selected `daily_ranking` in both folds. Its held-out rank IC
was negative in both. This small historical screen supplies no basis for promoting
the new models or claiming a persistent trading advantage.

[Registered recipe](../configs/rolling_screening.json) ·
[Complete evidence export](../reports/production_roadmap/rolling_screening/README.md) ·
[Analysis correction](rolling-analysis-audit.md) ·
[Native Transformer diagnostics](assets/rolling/transformer-fold000/report.html)

## Registered experiment

The source snapshot, provider identity, chronological folds, candidates, seeds,
validation rule, bootstrap grid and cost scenarios were written before fitting.
The pinned chenditc release is 26 November 2026, using historical CSI300 membership.
The 2024 periods have already been inspected and are explicitly exploratory.

| Setting | Registered value |
| --- | --- |
| Trials | 2 folds × 10 candidate/objective configurations × seeds 7 and 42 |
| Neural budget | 2 epochs; width 16; 2 heads; 1 layer; 32-session context |
| Tabular controls | Zero-return forecast, ridge, LightGBM |
| Daily objectives | MSE, Huber, stabilized IC and pairwise ranking |
| Validation selection | Mean validation rank IC across all declared seeds; RMSE fallback only when no finite primary score exists; candidate identifier breaks equal primary scores |
| Portfolio | Native Qlib CSI300 replay, TopkDropout top 20/drop 4, CNY 10 million base account |
| Scenarios | Base fees; doubled fees; one-tenth account size; additional 5 bps slippage |
| Uncertainty | Paired whole-date circular blocks of 5, 10 and 20 sessions; 1,000 draws |

| Fold | Training | Validation | Validation label endpoint | Held-out forecast dates |
| --- | --- | --- | --- | --- |
| 000 | 2024-01-02–2024-07-03 | 2024-07-08–2024-08-16 | 2024-08-20 | 2024-08-21–2024-09-19 |
| 001 | 2024-01-30–2024-07-31 | 2024-08-05–2024-09-13 | 2024-09-19 | 2024-09-20–2024-10-24 |

The two-session target information endpoints determine the gaps. Each native
portfolio covers 21 return dates, including execution/holding endpoints after
its 20 signal dates. Each fold starts a separate funded account; their returns
must not be concatenated into a live track record.

## All candidate results

The table reports **arithmetic means across the two individual seed runs**.
RMSE is in raw fractional-return units. Return is each independently funded
base-scenario portfolio's compounded net return, averaged over seeds; it is not
a seed-ensemble portfolio return. The exported summary retains individual
runs, seed dispersion, matched seed-mean forecast comparisons and all four cost
scenarios. Native Qlib chart panels separately use Qlib's additive convention.

| Candidate | Fold 000 RMSE | Fold 000 rank IC | Fold 000 net return | Fold 001 RMSE | Fold 001 rank IC | Fold 001 net return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 0.019452 | undefined | -4.00% | 0.050017 | undefined | +14.43% |
| ridge | 0.019948 | -0.0360 | -5.48% | 0.050876 | -0.0259 | +30.74% |
| lightgbm | 0.019452 | -0.0300 | -6.84% | 0.049932 | -0.0541 | +24.80% |
| transformer | 0.019885 | -0.0568 | -5.89% | 0.049990 | -0.0111 | +25.87% |
| daily_transformer | 0.019575 | -0.0492 | -4.90% | 0.050448 | -0.0479 | +17.02% |
| master_style | 0.019473 | -0.0305 | -5.49% | 0.050534 | -0.0799 | +13.23% |
| quantformer_moe | 0.019459 | -0.0412 | -5.36% | 0.050356 | -0.0398 | +21.49% |
| daily_huber | 0.019617 | -0.0485 | -5.86% | 0.050634 | -0.0583 | +15.65% |
| daily_ic | 0.020900 | -0.0303 | -6.86% | 0.051565 | -0.0730 | +11.89% |
| daily_ranking | 0.019791 | -0.0291 | -7.32% | 0.050628 | -0.0659 | +16.46% |

The zero-return forecast has undefined IC. Its portfolio is the same deterministic
TopkDropout rule applied to tied scores, making it an arbitrary tied-score stock
control. It is not a cash account or a zero-return portfolio.

The selected ranking objective had validation rank IC of **+0.0665** and
**+0.0776**, followed by held-out rank IC of **−0.0291** and **−0.0659**. Selection
is retained as registered; the held-out results are not used to switch winners.
All nonconstant candidates had negative mean held-out rank IC in both short
folds. Some portfolios gained in the second interval, but absolute portfolio
gains do not establish forecast skill or excess performance.

## Statistical qualification

The original frozen analysis allowed a circular block as long as the full sample.
For 20 signal dates and block length 20, every resample contains every date and
produces a degenerate interval. The corrected analysis keeps the registered grid,
marks those intervals `insufficient_dates`, and preserves the original results
for inspection. The correction changes no prediction, fitted parameter,
validation selection, portfolio, or point estimate.

Every accepted interval also reports effective block count. A 20-session block
on 21 portfolio dates has only 1.05 block lengths and is explicitly marked as
having very limited resampling information. Even the shorter-block intervals
come from short, dependent historical periods with multiple candidate
comparisons. They are exploratory evidence, not a prospective confirmation.

The [larger five-seed recipe](../configs/rolling_full.json) remains an executable
future experiment. It is not included in this completion count. The original
12-setup benchmark remains byte-for-byte unchanged.

## Reproduction and integrity

The evidence export contains all attempts, validation and evaluation forecasts,
learning curves, numerical bundles and reload fixtures, native scenario ledgers,
original reports and the separately audited report. A portable verifier checks
receipt-listed artifacts against their original hashes without loading native
pickle files. Exact frozen training source and its identity are retained in
[the runtime receipt](../reports/production_roadmap/runtime_source.json).

Reproduction on a different host uses a new output path, the same pinned source
snapshot and provider, and the registered recipe. Original absolute artifact
locations are preserved in the historical protocol instead of being silently
rewritten. See the evidence export's README for archive verification and
extraction commands.
