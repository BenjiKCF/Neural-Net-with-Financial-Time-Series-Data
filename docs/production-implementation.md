# Production roadmap: software implementation

**Version 2.3 · source-only delivery, 27 November 2026.**

The roadmap now has executable data contracts, lifecycle accounting, registered
rolling evaluation, daily cross-sectional models and objectives, adaptation
comparators, numerical model bundles and a shadow-operation runner. This release
contains the implementation, test fixtures, configuration and reproduction
instructions. The previously published CSI300 comparison and native Qlib gallery
retain their original meaning and qualifications.

The additional research runs are not distributed with this source-only change.
New provider-derived arrays, per-stock predictions or labels, fitted parameters,
native pickle checkpoints, figures and the complete local evidence archive are
outside its publication scope. The commands below generate a new, inspectable
campaign in the user's own environment. There is no new performance leaderboard
or claim that a newly implemented architecture outperforms the existing models.

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

The [source-only software validation receipt](validation/production-software.json)
records the local test, dependency and wheel checks for this publication. It
contains software verification metadata rather than market-data or fitted-model
artifacts. Remote CI is linked from the implementation PR.

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
recipes. The larger rolling recipe is a separate experiment. Neither a recipe
nor a passing software test supplies its missing empirical results. Inspect the
generated protocol, every successful or failed attempt, aligned forecasts,
selection decision, metrics and cost scenarios before drawing conclusions.

Each campaign binds its actual source, dependencies, provider and output
location. Changing any of those identities requires a fresh campaign directory.
Use the [rolling operations instructions](production-operation.md#1-run-a-registered-rolling-comparison)
for exact resume and separate-analysis behavior. Historical periods already
examined during development remain exploratory.

## Remaining acceptance gates

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
