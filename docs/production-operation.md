# Numerical bundles and daily shadow operation

The roadmap supplies executable research and operational components alongside
[retained experimental evidence](production-implementation.md#published-research-evidence).
The commands below create a new local campaign or operational replay. The
[original acceptance receipt](../reports/production_roadmap/acceptance.json)
describes the earlier research runtime; the
[later software receipt](validation/production-software.json) verifies the
subsequent calendar and manifest hardening. Use these separate records with the
[promotion gates](production-roadmap.md#promotion-gates).

## 1. Run a registered rolling comparison

Use the pinned chenditc provider from the existing download workflow:

```bash
fts download-chenditc --provider .data/cn-20261126 --release 2026-11-26
fts rolling-evaluate --provider .data/cn-20261126 \
  --config configs/rolling_screening.json --output runs/rolling-screen
```

The screening recipe declares two chronological folds, two seeds, ten candidates,
a two-epoch neural budget, four portfolio scenarios and date-block comparisons.
It is an implementation/research screen. The separate
[full recipe](../configs/rolling_full.json) uses longer histories and five seeds;
it is a larger experiment, not a result already established by the small screen.

The runner writes `protocol.json` before fitting. It binds every trial, label
endpoint, source file, dependency version, provider fingerprint, fitting window,
validation selection rule and cost assumption. A changed recipe, source or data
requires a new output directory. Completed outputs are hash-checked on reuse.

```bash
# Register without fitting.
fts rolling-evaluate --provider .data/cn-20261126 \
  --config configs/rolling_screening.json --output runs/rolling-screen --register-only

# Resume the exact registered campaign; retain failed attempts.
fts rolling-evaluate --output runs/rolling-screen --retry-failed

# Inspect completed, failed and pending work without fitting.
fts rolling-evaluate --output runs/rolling-screen --summarize-only

# Preserve the original analysis when auditing with a revised reporting runtime.
fts rolling-evaluate --output runs/rolling-screen --summarize-only \
  --summary-output runs/rolling-screen-audited

# Train/validation-only blend selection, followed by common portfolio scenarios.
fts rolling-blend --campaign runs/rolling-screen --output runs/rolling-blend \
  --left lightgbm --right master_style
```

Never concatenate independently funded fold returns into a supposedly continuous
live portfolio. Forecast uncertainty resamples whole dates, retaining each stock
cross section. Seed dispersion and dependence across market dates are different
sources of uncertainty. Undefined correlations remain missing and are counted.
A failed candidate/seed cannot silently disappear from the selection budget.
The runner marks a block length that covers the complete forecast sample as
`insufficient_dates`: circular resampling then has no useful variation. Check
the effective number of blocks and block-length sensitivity before interpreting
any interval. The [retained analysis audit](rolling-analysis-audit.md) preserves
the original and separately corrected summaries.
[Rolling tests](../tests/test_rolling.py) exercise these boundaries.

## 2. Export and inspect a model bundle

Registered rolling runs enable `export_inference_bundle: true`. A standalone
`fts train` workflow can set the same key. The live Qlib training record exports
its fitted model and inference processors without asking a deployment process to
load Qlib's executable pickle artifacts.

The option must be the YAML Boolean `true` or `false`; quoted strings, numbers
and `null` are rejected before training. The integration tests exercise this
through `fts train --config`, including numerical replay for ridge, LightGBM,
Transformer and MASTER-style bundles.

Each `inference_bundle/` contains:

| Artifact | Purpose |
| --- | --- |
| `manifest.json` | Content identity and the SHA256/size of every artifact. |
| `schema.json` | Ordered features and historical context length. |
| `preprocessing.json` | Training-fitted robust centers/scales, clipping and missing-value policy. |
| `model.json`, `weights.npz` or `booster.txt` | Allowlisted model configuration and numerical parameters. |
| `metadata.json` | Source-file hashes, provider identity, raw target, universe, seed and selection maturity cutoff. |
| `environment.json` | Observed runtime versions. |
| `reload_fixture.npz`, `parity.json` | Numerical reload checks and the declared tolerance. |

A locally generated fixture is reconstructed from processed validation inputs
and tests state/processor numerical parity. It can contain source-derived
features; keep it within the permitted scope for that data. The
[bundle integration tests](../tests/test_bundle_workflow.py) also recompute raw
Alpha158-derived factors from a synthetic provider and compare them with native
Qlib forecasts. A model loader rechecks the retained fixture under its current
code and dependencies; an old `passed` receipt is not accepted as proof that a
changed runtime still produces the same scores.

```bash
fts verify-bundle --bundle runs/rolling-screen/trials/FOLD/MODEL/seed-SEED/attempt-0001/workflow/inference_bundle
fts register-bundle --registry runs/registry \
  --bundle runs/rolling-screen/trials/FOLD/MODEL/seed-SEED/attempt-0001/workflow/inference_bundle
```

Use the actual trial and attempt path printed in the campaign's `comparison.json`.
The registry returns a 64-character immutable version. Assign it to a shadow
alias after reviewing the reload receipt:

```bash
fts set-model-alias --registry runs/registry --alias shadow --version BUNDLE_SHA256
```

The registry copies immutable bundles under their content identity and updates
aliases under a lock. `--rollback` requires a previously assigned, retained
version with the same feature schema/context. Every alias change records its
previous version and evidence.

The special `champion` alias requires a bundle-specific production decision,
passed referenced gates, declared service objectives, incident ownership and
approved model/data usage rights. A shadow approval cannot authorize a production
promotion. No current research bundle is automatically promoted. Official
research-only TimesFM checkpoints are excluded from this export path.

## 3. Publish one daily prediction batch

```bash
fts daily-infer --registry runs/registry --alias shadow \
  --provider .data/cn-20261126 --decision-date 2026-11-26 \
  --decision-cutoff '2026-11-26T19:00:00+08:00' \
  --data-available-at '2026-11-26T18:15:00+08:00' \
  --policy configs/production/batch_policy.json --output runs/daily-shadow
```

The timestamps above illustrate the argument format. Supply the **actual arrival
receipt**, rather than copying the example or inventing an earlier arrival. A
current historical snapshot cannot certify what was published in a past market
session. Historical replay retains that qualification.

The provider adapter requests feature expressions only and ends them at the
registered decision session. It uses the historical eligibility intervals so
that a re-entering security has the same history mask as during training. It
loads fitted normalization from the bundle and never refits on inference data.

The batch rejects future or stale arrivals, the wrong decision session, duplicate
or unexpected instruments, insufficient coverage, reordered features, missing
current rows, excessive missingness, nonfinite scores and service-objective
breaches. Padding cannot conceal an absent tabular observation. The model's
selection-label information cutoff must precede the prediction cutoff.
Exported cutoffs and provider packets use the exchange timezone for the declared
region, including daylight-saving offsets for US sessions. Label maturity is not
interpreted in the execution host's timezone.

Successful output is an atomically published dated directory containing
`predictions.csv`, `input_manifest.json` and `manifest.json`. Its receipt binds
model/data identity, request identity, cutoff, coverage, score distribution,
runtime and resident memory. Consumers should call
`read_prediction_artifact()` to verify the output before use. Verification covers
the prediction CSV and its input manifest; a missing or altered input manifest
invalidates reuse. Repeating exactly
the same request returns the existing result; changing input, policy or model
for an already published date raises a conflict. Failures have separate receipts
and never create a usable prediction artifact.

**Prediction-artifact compatibility:** new outputs use manifest schema version
2. A valid directory contains exactly three regular files: `manifest.json`,
`input_manifest.json` and `predictions.csv`. The manifest binds the CSV and input
manifest by SHA256; the reader also checks the packet's identity and instruments.
Older schema-version-1 outputs are rejected because they do not establish this
input-manifest integrity. Preserve them as historical evidence and regenerate
into a fresh output path using the actual arrival receipt. Editing old hashes or
adding a version field does not migrate an artifact.

The published [historical operation receipts](../reports/production_roadmap/operations_acceptance/summary.json)
include version-1 outputs from the earlier runtime. Their original successful
checks are historical evidence, not evidence that those directories satisfy the
new version-2 reader. The later [software validation](validation/production-software.json)
covers the strengthened checks separately. For a historical rehearsal, label
hypothetical arrivals explicitly; only an actual arrival receipt can support
prospective operation.

An upstream application may instead supply raw factors as numerical arrays:
`--input-npz batch.npz --input-manifest batch.json`. The NPZ contains `features`
and optionally `padding`; it is loaded with `allow_pickle=False`. The manifest
must declare the ordered feature names, current/expected instruments, observation
session, publication time, decision cutoff and immutable data identity.

## 4. Schedule shadow observation

The [sample service](../deployment/financial-ts-shadow.service) and
[timer](../deployment/financial-ts-shadow.timer) invoke:

```bash
fts scheduled-shadow --config configs/production/shadow_schedule.json
```

They are examples for a declared Linux host, not installed services. Configure
the service account, repository/venv paths, incident owner, data-arrival schedule
and local objectives before enabling the timer. The sample runs on weekdays at
19:00 Hong Kong time. A sourced holiday receipt is required to skip a nontrading
session; an absent receipt is an outage.

The upstream admitted arrival receipt must identify `expected_session`,
`decision_cutoff`, `data_available_at`, the exact `provider_fingerprint`, and
`snapshot_audit_status: admitted`. `scheduled-shadow` refuses a stale receipt,
a different provider, an unadmitted refresh, or an invocation outside its
publication window. It cannot manufacture a receipt or fall back to an older
snapshot. The runner records an operator-visible error/exit status. Configure the
host's existing alert transport; the package does not send unsolicited messages.

`fts snapshot-audit` checks a raw tabular refresh against a previous snapshot.
Its current/previous CSVs use `datetime,instrument` keys and named numeric fields.
Identity JSON requires the release, archive and provider identities described in
[data lifecycle contracts](data-lifecycle.md). Unexplained historical cell
revisions, missing keys or schema changes are quarantined. Published snapshots
are retained unchanged.

## 5. Paper execution and monitoring

```bash
fts paper-replay --spec configs/paper_session_demo.json --output runs/paper-demo
# The identical repeat verifies artifacts and reuses the completed receipt.
fts paper-replay --spec configs/paper_session_demo.json --output runs/paper-demo

fts monitor-labels --predictions runs/daily-shadow/2026-11-26/predictions.csv \
  --labels matured-labels.csv --cutoff '2026-11-30T19:00:00+08:00' \
  --output runs/monitoring/2026-11-30.json
```

The paper-session specification registers initial raw-share holdings/cash,
calendar, dated trading rules, fee/slippage assumptions, order IDs, observed bars
and corporate actions. Its event ledger records actual partial/rejected fills,
settlement, fees, positions and unresolved exposures. A restart can replay from
the registered initial account and reproduce the same journal. Duplicate order
IDs cannot cause a second fill, and a conflicting reused ID is rejected.

Monitoring labels require `datetime,instrument,label,available_at` columns.
Every availability timestamp must include an explicit timezone. Future labels
are deferred until their information endpoints mature; a mature row with a
missing return reduces evaluated coverage and triggers the coverage check. Score/rank
drift can be compared with `compare_daily_artifacts()`. Alerts open a review;
they do not retrain or promote a model. Paper fills are simulations, not actual
broker acknowledgements. A real execution adapter still needs authenticated
order state, broker fills and reconciliation before deployment.

## 6. Rehearse operations and profile numerical choices

After training two compatible, distinct model versions, the
[acceptance script](../scripts/run_operations_acceptance.py) can verify raw-factor
parity, registry reuse, atomic publication, failure recovery, alias pinning and
rollback against their local forecasts:

```bash
python scripts/run_operations_acceptance.py --provider .data/cn-20261126 \
  --primary-run runs/primary-model --rollback-run runs/previous-model \
  --output runs/operations-check --work-dir runs/operations-check-work
```

Both run directories must contain completed workflows and exported bundles. The
script uses explicit hypothetical historical arrival times; its result is a
software rehearsal, not a prospective shadow record. The retained
[three-model historical rehearsal](production-implementation.md#historical-operational-acceptance)
reports its original source and outputs separately from a new invocation.

The [precision profiler](../scripts/profile_inference.py) compares float32 and
CPU bfloat16 on the same weights and full same-date stock context in separate
processes. Declare tolerances before running it:

```bash
python scripts/profile_inference.py --bundle runs/primary-model/inference_bundle \
  --provider .data/cn-20261126 --decision-date 2024-08-21 \
  --output runs/inference-profile --score-atol 0.0001 \
  --min-rank-correlation 0.999 --max-rank-shift 3
```

Choose an eligible date after the bundle's fitting/selection cutoffs. Inspect
runtime, resident memory, score differences, ranking and top-k changes together.
The existence of a precision option is not evidence of speed, memory or quality
improvement on a particular machine. The
[retained CPU profile](precision-profile.md) favored float32: bfloat16 was slower
and failed the declared rank-change gate on that measured machine.

## Environment and release checks

The exact CPU Python 3.12.14 research environment is recorded in
[locked-cpu-py312.txt](../requirements/locked-cpu-py312.txt). It includes optional
TimesFM and visualization dependencies used by this repository. The lock is a
platform-specific reproduction profile; Python 3.10 and 3.12 remain covered by
CI with the package's supported dependency ranges.

```bash
python -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/locked-cpu-py312.txt
python -m pip install --no-deps .
python -m pytest -q
```

Run dependency checks in the selected environment with `python -m pip check`.
The [CI workflow](../.github/workflows/ci.yml) builds a wheel, installs it into an
isolated package environment, verifies that imports resolve to that wheel and
exercises the CLI. That package check reuses the runner's installed dependencies;
CI itself installs those dependencies on clean runners. Model bundles record
the observed versions and verify numerical fixtures on each load. Cross-hardware
claims remain limited by their explicit numerical tolerance.

For optional MLflow lineage, `export_to_mlflow()` registers the immutable numerical
bundle and copies the selected alias into a configured registry. It does not make
the bundle an MLflow `pyfunc` model. Load it with `NumericalPredictor`. MLflow's
[registry documentation](https://mlflow.org/docs/2.20.3/model-registry.html)
describes aliases/version lineage. Qlib's
[online-serving documentation](https://qlib.readthedocs.io/en/stable/component/online.html)
notes that daily signal updates and next-day executable orders have different
requirements.
