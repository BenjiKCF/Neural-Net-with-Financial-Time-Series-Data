# LightGBM and MASTER rolling blend

The completed historical blend screen selected **0% LightGBM / 100% MASTER** in
the first fold and **50% LightGBM / 50% MASTER** in the second fold. All eight
forecast panels and all 32 native portfolio scenarios completed, with no failed
folds. The results do not establish a consistent blend advantage or justify
production promotion.

The complete output is retained in
[`reports/production_roadmap/rolling_blend`](../reports/production_roadmap/rolling_blend/summary.json).
The [registered protocol](../reports/production_roadmap/rolling_blend/protocol.json),
[independent verification](../reports/production_roadmap/rolling_blend/verification.json),
and [export manifest](../reports/production_roadmap/rolling_blend/export_manifest.json)
preserve the source identities and original execution paths.

## Selection and matched controls

Each constituent is the arithmetic mean of its forecasts from **both registered
seeds, 7 and 42**. The runner waited for all eight constituent attempts across
two folds to complete; it did not substitute an available seed subset.
LightGBM weight `w` produces `w * LightGBM + (1 - w) * MASTER` in raw return
units. Exact date/instrument keys and labels must agree before averaging or
blending.

The per-fold search maximizes mean daily validation Rank IC over the registered
grid `[0, 0.25, 0.5, 0.75, 1]`, breaking ties by validation RMSE and then smaller
LightGBM weight. The selected weight is saved before the evaluation prediction
files are loaded. No test return, cost scenario, or test correlation chooses the
weight. Every attempted validation weight is retained in
[`validation_grid.csv`](../reports/production_roadmap/rolling_blend/validation_grid.csv).

| Fold | Validation signal dates | Evaluation signal dates | Selected LightGBM weight | Selected validation Rank IC |
| --- | --- | --- | ---: | ---: |
| `fold-000` | 2024-07-08–2024-08-16 | 2024-08-21–2024-09-19 | 0.00 | 0.046315 |
| `fold-001` | 2024-08-05–2024-09-13 | 2024-09-20–2024-10-24 | 0.50 | 0.052944 |

The two constituent means and a fixed 50/50 blend are evaluated alongside the
selected blend. In fold 000, the selected forecast is exactly the MASTER
control. In fold 001, it is exactly the fixed 50/50 control. These equalities are
retained rather than being presented as additional independent results.

## Forecast results

Each panel contains 6,000 forecasts across 20 signal dates and the contemporaneous
CSI300 membership. Fold 000 has 5,948 evaluable labels and 52 unavailable labels;
fold 001 has 5,984 evaluable labels and 16 unavailable labels. Missing labels
remain recorded and are not replaced with invented exits or zero returns.

Source:
[`forecast_metrics.csv`](../reports/production_roadmap/rolling_blend/forecast_metrics.csv).
The zero-return RMSE is 0.0194522403 in fold 000 and 0.0500174895 in fold 001.

| Fold | Method | RMSE | Mean daily IC | Mean daily Rank IC |
| --- | --- | ---: | ---: | ---: |
| 000 | Validation selected | 0.01946489 | -0.031098 | -0.031454 |
| 000 | LightGBM seed mean | 0.01945235 | -0.019476 | -0.029979 |
| 000 | MASTER seed mean | 0.01946489 | -0.031098 | -0.031454 |
| 000 | Fixed 50/50 | 0.01943732 | -0.032112 | -0.031449 |
| 001 | Validation selected | 0.05019684 | -0.093262 | -0.111092 |
| 001 | LightGBM seed mean | 0.04993162 | -0.021548 | -0.054080 |
| 001 | MASTER seed mean | 0.05053228 | -0.077953 | -0.078124 |
| 001 | Fixed 50/50 | 0.05019684 | -0.093262 | -0.111092 |

All mean evaluation IC and Rank IC values in this table are negative. The
validation-selected forecast has a higher RMSE than the zero-return forecast
in both folds. These observations must remain visible alongside any positive
portfolio return.

## Native portfolio results

The registered Qlib strategy uses top 20 holdings, four replacements, a base
account of CNY 10 million, and the parent campaign's generic trading rules and
cost assumptions. Each fold is independently funded. Its native report has 21
sessions: 2024-08-22–2024-09-23 for fold 000 and
2024-09-23–2024-10-28 for fold 001. **These portfolios are not concatenated into
a continuous strategy return.**

Source:
[`portfolio_metrics.csv`](../reports/production_roadmap/rolling_blend/portfolio_metrics.csv).
Net return compounds the native daily `return - cost`; drawdown includes the
initial account value. Turnover is the mean native daily turnover ratio.

| Fold | Method | Base net return | Base maximum drawdown | Mean daily turnover |
| --- | --- | ---: | ---: | ---: |
| 000 | Validation selected | -5.2641% | -8.1944% | 31.4656% |
| 000 | LightGBM seed mean | -6.8362% | -7.4187% | 40.1085% |
| 000 | MASTER seed mean | -5.2641% | -8.1944% | 31.4656% |
| 000 | Fixed 50/50 | -5.1358% | -8.3257% | 37.2573% |
| 001 | Validation selected | +21.7428% | -12.6366% | 36.2082% |
| 001 | LightGBM seed mean | +24.7962% | -14.4804% | 34.8564% |
| 001 | MASTER seed mean | +14.9542% | -5.7685% | 31.0268% |
| 001 | Fixed 50/50 | +21.7428% | -12.6366% | 36.2082% |

### Cost and account-size stresses

Each row below is a separate actual native replay. The fee stress doubles the
registered proportional fees and minimum fee. The smaller account is CNY
1 million. The five-basis-point slippage scenario adds five basis points to
each side's proportional execution cost; it is a cost proxy, not an order-book
impact model. Account-size changes can alter lot rounding and the resulting
holdings.

| Fold | Method | Base return | Fees ×2 | Account ×0.1 | Additional 5 bps per side |
| --- | --- | ---: | ---: | ---: | ---: |
| 000 | Validation selected | -5.2641% | -5.8400% | -5.2375% | -5.5762% |
| 000 | LightGBM seed mean | -6.8362% | -7.5745% | -6.7852% | -7.2286% |
| 000 | MASTER seed mean | -5.2641% | -5.8400% | -5.2375% | -5.5762% |
| 000 | Fixed 50/50 | -5.1358% | -5.8283% | -5.0999% | -5.5073% |
| 001 | Validation selected | +21.7428% | +20.9146% | +20.3117% | +21.2966% |
| 001 | LightGBM seed mean | +24.7962% | +23.9802% | +24.4467% | +24.3477% |
| 001 | MASTER seed mean | +14.9542% | +14.2653% | +14.8456% | +14.5805% |
| 001 | Fixed 50/50 | +21.7428% | +20.9146% | +20.3117% | +21.2966% |

All 32 source-gap audits report zero final stale weight in these short windows.
That limited observation does not qualify the provider's complete corporate-action
accounting, historical publication timestamps, or real-world execution. The
archived longer-horizon lifecycle issues remain a separate qualification.

## Frozen implementation and reanalysis qualification

This is an exploratory historical follow-up on already completed constituent
forecasts, not a new untouched holdout. It uses the frozen runtime at
`/workspace/scratch/f328bea0f8d6/runtime-screen/src`, whose registered source-code
SHA-256 is
`05dd6d449d27c2f586fe1685996d9602d4505748d026d60d5899187fbe0186f3`.
The parent protocol SHA-256 is
`52489beb3bc0a813660da1d939962bbd2bc39aee8120150ff4dcfdaf973e87df`.

The frozen runner invoked the parent campaign's summarizer before blend
registration. Subsequent audit/reanalysis of the parent summary is separate and
does not rewrite these blend forecasts or native portfolios. Later source fixes
are not silently attributed to this frozen execution. The retained protocol and
receipts preserve the original `runs/roadmap-screening-20261127` and
`runs/roadmap-blend-20261127` paths; copying the completed evidence into `reports`
does not change its source identity.

The current CLI can register a separate follow-up in a new output directory:

```bash
fts rolling-blend --campaign runs/roadmap-screening-20261127 \
  --output runs/rolling-blend-new \
  --left lightgbm --right master_style
```

The independent verification recomputed the selected validation weights,
matched seed means, all blend scores, forecast metrics, and all 32 portfolio
summaries from the retained daily reports. It verified 24 registered input
files and all 308 artifacts belonging to the eight completed constituent
attempts. Maximum differences in blend arithmetic and portfolio statistics
were both exactly zero in this environment.

The retained export contains 173 copied files, about 4.4 MB, plus its export
manifest. It includes every forecast, validation grid, fold/control receipt,
scenario return file, numeric holdings ledger, and source-gap audit. The
[artifact manifest](../reports/production_roadmap/rolling_blend/artifact_manifest.json)
hashes the 172 result and verification files; the export manifest additionally
hashes that artifact manifest and verifies the copy byte for byte.

`promotion_eligible` remains `false`. The mixed results, short windows, negative
cross-sectional correlations, retrospective provider, and generic native
exchange assumptions do not support a winning-model or deployment claim.
