# Production roadmap: implementation and evidence

**Version 2.3 · software and retained exploratory research, 27 November 2026.**

The roadmap now has executable data contracts, lifecycle accounting, registered
rolling evaluation, daily cross-sectional models and objectives, adaptation
comparators, numerical model bundles and a shadow-operation runner. The
implementation, tests, configuration and reproduction instructions are
accompanied by the retained rolling, adaptation, blend, accounting and operation
evidence. The previously published CSI300 comparison and native Qlib gallery
retain their original meaning and qualifications.

These artifacts preserve the original protocols, forecasts, checkpoints, failed
attempts and source identities. They report bounded historical research and
software acceptance; they do not establish a persistent trading advantage or a
production deployment. Publication adds access to retained evidence, without
retraining models or relabeling old results as tests of later software.

## Implementation map

| Work package | Implementation and configuration | Executable verification |
| --- | --- | --- |
| Feature availability and refresh admission | [Data contracts](../src/financial_ts/data_contracts.py), [raw daily contract](../configs/data_contract_daily.json) | [Publication cutoffs, freshness, membership and revision tests](../tests/test_data_contracts.py) |
| Lifecycle and independent accounting | [Event ledger](../src/financial_ts/lifecycle.py), [sourced event catalog](../configs/lifecycle_events.json) | [Hand-calculated events, unresolved values and ledger reconciliation](../tests/test_lifecycle.py) |
| Paper execution | [Dated rules, fills and durable sessions](../src/financial_ts/execution.py), [synthetic session](../configs/paper_session_demo.json) | [Lot, settlement, participation, cash and retry tests](../tests/test_execution.py) |
| Registered rolling evaluation | [Campaign runner](../src/financial_ts/rolling.py), [screening recipe](../configs/rolling_screening.json), [larger recipe](../configs/rolling_full.json) | [Split maturity, selection, preserved attempts and paired comparison tests](../tests/test_rolling.py) |
| Same-date objectives | [Daily model adapter](../src/financial_ts/cross_sectional.py), [loss functions](../src/financial_ts/objectives.py) | [Grouping and chronology](../tests/test_cross_sectional.py), [ties, constant predictions and gradients](../tests/test_objectives.py) |
| Cross-stock and MoE models | [Architecture implementation](../src/financial_ts/architectures/cross_stock.py), [MASTER-style recipe](../configs/master_style.yaml), [MoE recipe](../configs/quantformer_moe.yaml) | [Masking, permutation and architecture controls](../tests/test_cross_stock.py) |
| Simple controls and ensembles | [Baselines](../src/financial_ts/baselines.py), [validation-selected blends](../src/financial_ts/ensembles.py) | [Baseline tests](../tests/test_baselines.py), [aligned forecasts and blend selection](../tests/test_ensembles.py) |
| Multi-horizon, uncertainty and simple adaptation | [Advanced runner](../src/financial_ts/advanced_research.py), [screening configuration](../configs/advanced_screening.json) | [Target endpoints, calibration and update guards](../tests/test_advanced_research.py) |
| DoubleAdapt-style comparator | [Independent method implementation](../src/financial_ts/doubleadapt.py), [registered recipe](../configs/doubleadapt_screening.json) | [Meta-gradients, prediction locks and mature-label controls](../tests/test_doubleadapt.py) |
| Model bundles and daily operation | [Numerical bundles](../src/financial_ts/bundles.py), [registry and batch runner](../src/financial_ts/operations.py) | [Integrity and reload tests](../tests/test_bundles.py), [Qlib export integration](../tests/test_bundle_workflow.py), [operational failure tests](../tests/test_operations.py) |
| Optional precision measurement | [Same-input inference profiler](../scripts/profile_inference.py) | Produces hardware-specific timing, memory and numerical comparisons after a local model has been trained. |

Read the [data and lifecycle guide](data-lifecycle.md),
[architecture guide](architectures/cross-sectional.md),
[DoubleAdapt method guide](doubleadapt.md), and
[operation guide](production-operation.md) for the corresponding contracts.

## Published research evidence

| Study | Completed scope | Finding and evidence |
| --- | --- | --- |
| Rolling evaluation | 40 trials: two folds, two seeds and ten candidate/objective configurations; 160 native cost-scenario replays. | Validation selected `daily_ranking` in both folds; held-out Rank IC was negative in both. [Full candidate table](rolling-screening.md) · [Complete archive and audit](../reports/production_roadmap/rolling_screening/README.md). |
| Tree/neural blends | Two folds, eight forecast panels and 32 native cost scenarios, with constituent and fixed-blend controls. | Selection uses validation only; results and costs are retained without promoting a test winner. [Blend report](rolling-blend.md). |
| Multi-horizon and uncertainty | Separate/joint 1-, 5- and 20-session forecasts, quantile calibration and frozen/incremental/full-retraining controls. | Mixed improvements across horizons; wider calibrated intervals do not establish useful trading uncertainty. [Bounded-screen results](architectures/cross-sectional.md#completed-bounded-screen). |
| DoubleAdapt-style comparator | Compact meta-learning experiment with three matched learned controls and prediction-before-label locks. | No advantage established; paired RMSE intervals against all learned controls include zero. [Method and results](doubleadapt.md#actual-compact-screen-results). |
| Precision profile | Same weights and full 300-stock inference context, float32 and CPU bfloat16 in separate processes. | Float32 retained; bfloat16 was slower and failed the registered rank-change gate. [Profile](precision-profile.md). |
| Lifecycle and paper execution | Archived holdings assessment plus a separate synthetic durable execution session. | Corporate-action allocation/tradability remains unresolved; the synthetic session reconciles and reuses identical requests. [Assessment](../reports/production_roadmap/lifecycle_transformer/assessment.json) · [Synthetic receipt](../reports/production_roadmap/paper_session_demo/completion.json). |
| Historical operations | Three exported models, each replayed on 300 eligible stocks with 11 checks. | Historical raw-factor parity, publication, recovery and rollback checks passed in their original runtime. See the qualification below. |

## Research provenance and software versions

| Evidence layer | Identity and scope |
| --- | --- |
| Frozen 40-trial training | The [runtime source receipt](../reports/production_roadmap/runtime_source.json) and [campaign protocol](../reports/production_roadmap/rolling_screening/protocol.json) pin the original training source. The [archive README](../reports/production_roadmap/rolling_screening/README.md) explains complete integrity checks and reproduction at fresh paths. |
| Corrected derived rolling analysis | The [analysis receipt](../reports/production_roadmap/rolling_screening/analysis_receipt.json) identifies the frozen training source and the later summary source separately. The [audit guide](rolling-analysis-audit.md) explains the noninformative full-period bootstrap correction; no forecast, selection or portfolio was replaced. |
| Original implementation acceptance | The unchanged [historical receipt](../reports/production_roadmap/acceptance.json) records **563 passing tests**, its source identity and the original bounded research. This is the earlier implementation's acceptance record. |
| Later hardened software | The [software receipt](validation/production-software.json) records **570 passing tests**, dependency and wheel checks, and source identity after the calendar-coverage and prediction-manifest fixes. Its original source-only publication scope describes that earlier delivery stage; the present evidence publication does not rewrite the receipt. |
| Evidence publication and review fixes | The [publication receipt](validation/production-publication.json) records **600 passing tests**, exact archive reassembly and all 1,724 archived files/1,556 receipt-listed artifacts verified. Standalone `fts train` accepts a strictly Boolean bundle-export option, and ledger valuations require calendar coverage through the exchange-local cutoff. |

The retained research was not rerun on the later hardened software. In
particular, historical operation outputs use the original artifact schema;
current schema-2 readers reject schema-1 daily prediction artifacts. Preserve
them as historical evidence and use a fresh output path for any new rehearsal.
The [compatibility instructions](production-operation.md#3-publish-one-daily-prediction-batch)
explain the exact distinction. No historical output is silently migrated.

## Accounting finding

The retained source assessment reproduces the archived Transformer's **6.489845%**
final exposure to terminated series. Sourced final ratios identify the three
successor mappings, including SH601989's final **0.1339** ratio, which differs
from its preliminary proposal. However, converting the archived adjusted
positions back to raw shares produces noninteger quantities for two symbols,
and account-level share allocation/tradability evidence is incomplete.

The ledger therefore retains unresolved exposures and unavailable values. It
neither invents a last-price liquidation nor marks a losing stock as deleted.
This assessment does **not** replace the archived -0.94% simulated return with an
unverified corrected return. Exact details and primary issuer sources are in the
[assessment](../reports/production_roadmap/lifecycle_transformer/assessment.json)
and [event catalog](../configs/lifecycle_events.json).

The [durable paper demonstration](../reports/production_roadmap/paper_session_demo/completion.json)
is explicitly synthetic. Its final CNY 9,975.105 cash reconciles independently,
and rerunning the identical session does not duplicate a fill.

## Historical operational acceptance

In the original recorded runtime, independent source-factor replays tested
ridge, Transformer and MASTER-style
bundles on all 300 eligible CSI300 stocks for 21 August 2024. Each model passed
11 checks, including fresh raw-factor parity, immutable registration, atomic
publication, duplicate reuse, stale/late-data rejection, failure recovery,
conflicting-request rejection, alias pinning, compatible rollback and blocked
production promotion.

| Exported model | Forecasts | Maximum absolute raw-return score difference | Evidence |
| --- | ---: | ---: | --- |
| Ridge | 300 | 7.54 × 10⁻⁹ | [Receipt](../reports/production_roadmap/operations_acceptance/ridge/results.json) |
| Transformer | 300 | 4.14 × 10⁻⁹ | [Receipt](../reports/production_roadmap/operations_acceptance/transformer/results.json) |
| MASTER-style | 300 | 1.90 × 10⁻⁹ | [Receipt](../reports/production_roadmap/operations_acceptance/master_style/results.json) |

These are historical software replays with explicitly hypothetical arrival
timestamps. They do not certify point-in-time publication or replace prospective
shadow observation. No orders were submitted and no production alias was assigned.

## Research budgets and interpretation

The main rolling screen completed **40 of 40 trials**: two folds, two seeds and ten
candidate/objective configurations, with no failed training attempts. It evaluates successive 2024 intervals.
All folds use the same pinned chenditc provider; each candidate retains its
validation forecasts, held-out forecasts, model budget, selection receipt,
numerical bundle and cost-scenario positions. The validation rule selected
`daily_ranking` in both folds; its held-out rank IC was negative in both. See
[all candidate results and qualifications](rolling-screening.md) and the
[complete evidence archive](../reports/production_roadmap/rolling_screening/README.md).
The corrected analysis preserves the original frozen summary and explicitly
marks full-period circular-bootstrap intervals as noninformative.

The advanced screen is a deliberately small historical experiment with a universe
selected at the first training date and separate chronological training,
validation, calibration, evaluation and adaptation periods. Joint learning and
incremental updates do not improve every horizon. Calibration widens the tested
intervals, and the calibration method does not guarantee coverage for dependent
future financial returns. See its complete table in the
[architecture report](architectures/cross-sectional.md#completed-bounded-screen).

The larger five-seed rolling recipe and a genuinely prospective confirmation
period are distinct campaigns. A configured larger budget is not reported as an
executed experiment.

The separate [DoubleAdapt-style screen](doubleadapt.md) also retains a negative
result. Its RMSE is approximately 0.018611, versus 0.018539 for frozen weights
and 0.018404 for zero return. All paired RMSE intervals against the three learned
controls include zero. The implementation documents its compact MLP and exact
differentiation choices rather than claiming an exact upstream reproduction.

## Run the software checks

Use the supported Python environment from the [installation guide](getting-started.md).
From the repository root:

```bash
python -m pip install -e ".[dev,neural]"
python -m ruff check src tests
python -m pytest -q

fts paper-replay --spec configs/paper_session_demo.json --output runs/paper-demo
fts paper-replay --spec configs/paper_session_demo.json --output runs/paper-demo
```

The second paper command checks identical-run reuse. It should not add another
fill. The session is a synthetic fixture and submits no live order. The
[CI workflow](../.github/workflows/ci.yml) also exercises the offline model demo,
wheel installation, CLI and retained historical reproduction paths. Consult
the CI run for the revision being reviewed; a test result from an earlier
checkout does not certify a later change.

The latest [600-test publication validation receipt](validation/production-publication.json)
records the review fixes and evidence integrity checks. CI also reconstructs the
four archive parts and streams the complete receipt-chain audit. The earlier
[570-test software receipt](validation/production-software.json) and
[563-test historical acceptance](../reports/production_roadmap/acceptance.json)
remain unchanged.
Remote CI is linked from the implementation PR; evaluate the run associated with
the revision being reviewed.

## Generate a local research campaign

Download a pinned provider snapshot through the existing workflow, then register
and execute the screening recipe:

```bash
fts download-chenditc --provider .data/cn-20261126 --release 2026-11-26
fts rolling-evaluate --provider .data/cn-20261126 \
  --config configs/rolling_screening.json --output runs/rolling-screen --register-only
fts rolling-evaluate --output runs/rolling-screen

fts rolling-blend --campaign runs/rolling-screen --output runs/rolling-blend \
  --left lightgbm --right master_style

fts research-advanced --provider .data/cn-20261126 \
  --config configs/advanced_screening.json --output runs/advanced-screen
fts research-doubleadapt --provider .data/cn-20261126 \
  --config configs/doubleadapt_screening.json --output runs/doubleadapt-screen
```

The screening budgets are deliberately small and are specified in the linked
recipes. Their retained results are described above. The larger rolling recipe
now has a [separate five-seed execution campaign](five-seed-campaign.md); its
560-trial completion and acceptance remain pending. Passing software tests do
not supply its empirical results. For a new campaign, inspect the
generated protocol, every successful or failed attempt, aligned forecasts,
selection decision, metrics and cost scenarios before drawing conclusions.

Each campaign binds its actual source, dependencies, provider and output
location. Changing any of those identities requires a fresh campaign directory.
Use the [rolling operations instructions](production-operation.md#1-run-a-registered-rolling-comparison)
for exact resume and separate-analysis behavior. Historical periods already
examined during development remain exploratory.

## Remaining acceptance gates

The [remaining-acceptance continuation](remaining-acceptance.md) adds an exact-grid
campaign auditor, account settlement verification and a genuinely future shadow
pilot. Its evidence report retains the open requirements below.

| Gate | What the software supplies | What still requires external or future evidence |
| --- | --- | --- |
| Data availability | Timestamp/lag contracts, historical membership checks and refresh quarantine. | Publication vintages or defensible source arrival records for the intended decision times. |
| Accounting | Raw-share event ledger, unresolved-value handling and dated execution rules. | Actual account allocations, full corporate-action coverage, successor tradability and a newly qualified portfolio replay. |
| Candidate quality | Registered chronological selection, controls, multiple seeds and paired uncertainty calculations. | Adequate experiment budgets, a practical improvement under costs and genuinely uninspected confirmation observations. |
| Operation | Content-addressed bundles, atomic outputs, explicit failures, idempotency and compatible rollback. | A declared host/owner, measured service objectives and a prospective shadow observation period. |
| Deployment | A registry that checks scope-specific promotion evidence. | Reviewed data/model usage rights, execution reconciliation, risk limits and the intended deployment decision. |

The [promotion gates](production-roadmap.md#promotion-gates) remain open until
their required evidence exists. No production alias, installed scheduler or live
broker connection is created by publishing this source.
