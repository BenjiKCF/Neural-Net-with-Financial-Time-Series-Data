# TimesFM 3 loop fine-tuning: a bounded feasibility pilot

**The real pretrained model can be converted into a shared-block loop and
fine-tuned through small gates and adapters.** A fixed three-epoch CPU pilot
completed 192 optimizer updates with the original 330,710,976 parameters frozen.
Validation selected epoch 2. This establishes a working conversion and training
path; it does not establish a looping-specific forecasting advantage.

**Follow-up:** a [common-period evaluation](#follow-up-evaluation-on-common-periods)
now compares a fresh same-recipe loop retrain, the frozen checkpoint, and a
parameter-matched gated adapter-only control. The loop improves on the frozen
checkpoint but does not beat zero return or the previously reported TimesFM
mini. The original pilot tables and evidence below remain unchanged.

The run completed on **27 November 2026, Hong Kong time**. Machine-readable
timestamps are preserved in UTC. All numerical evidence below comes from the
[saved run manifest](../reports/timesfm_loop_pilot/manifest.json),
[metrics](../reports/timesfm_loop_pilot/metrics.json),
[learning curves](../reports/timesfm_loop_pilot/learning_curves.json),
[sample records](../reports/timesfm_loop_pilot/samples.json), and
[validation predictions](../reports/timesfm_loop_pilot/validation_predictions.csv).

## What was trained

The [loop backend](../src/financial_ts/timesfm_loop.py) keeps the official model's
20 layers and replays layers 8–11 once using the same layer objects. This produces
24 layer evaluations. A bounded token gate and rank-8 adapter control the extra
pass. Their identity initialization preserves the original forecast before
training. See the [architecture and gradient path](architectures/timesfm-loop.md).

| Setting | Fixed value |
| --- | --- |
| Base checkpoint | `google/timesfm-3.0-pytorch` |
| Checkpoint revision | `43046b85ec22d584a13f8098c2ed39c889e129c2` |
| Official package | `timesfm==3.0.2` |
| Frozen base parameters | 330,710,976 |
| Trainable gate/adapter parameters | 21,761; approximately 0.00658% of total parameters |
| Replayed layers / total block traversals | `[8, 12)` / 2 |
| Gate bound / adapter rank | 0.1 / 8 |
| Context / forecast horizon | 128 observed closes / 2 future closes |
| Training / validation sample budget | 256 / 128 |
| Epochs / batch size / seed | 3 / 4 / 42 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0, gradient norm clipped to 1 |
| CPU threads | 2 |
| Selection | Lowest sampled validation RMSE among epochs 0, 1, 2, and 3 |

Only four new tensors enter the optimizer: gate weights, gate bias, adapter
down-projection, and adapter up-projection. The runner checks that all base
parameters are frozen, these trainable names exactly match the saved adapter
keys, and base parameters have no gradients after each epoch.

## Data boundaries

The provider was imported from the repository's `^GSPC.csv`: 7,252 daily
observations of SPX from 2 January 1990 through 10 October 2018. Prices are the
provider's adjusted closes. Each input ends at close `t`; the target is
`close[t+2] / close[t+1] - 1`. There is no future filling of incomplete windows.

| Partition | Existing partition bounds | Actual sampled prediction dates | Count |
| --- | --- | --- | ---: |
| Training | 1990-03-28 → 2007-05-04 | 1990-07-03 → 2007-05-04 | 256 |
| Validation | 2007-05-09 → 2013-01-18 | 2007-05-09 → 2013-01-18 | 128 |
| Test | 2013-01-24 → 2018-10-08 | Not evaluated | 0 |

The longer context moves the first eligible training example to 3 July 1990.
Samples are evenly spaced among eligible dates; multi-instrument inputs use
balanced instrument quotas. The exact dates, context hashes, and target values
are retained in `samples.json`.

Two observed trading sessions are purged at each partition boundary. The latest
training target exits on 8 May 2007, before validation starts. The latest
validation target exits on 23 January 2013, before test starts. The price query
ends on that validation exit date. Provider/calendar metadata and file hashes
describe the complete dataset, but no test-period price or label is used for
training, selection, or scoring in this runner.

Target mean and scale are fitted on the sampled training labels only:
0.0007839964 and 0.0093150372. They scale the squared-error objective; forecast
scores and reported RMSE remain in decimal return units.

## Every epoch, including the unchanged checkpoint

| State | Training RMSE | Validation RMSE ↓ | Cumulative updates | Selection |
| --- | ---: | ---: | ---: | --- |
| Epoch 0: unchanged checkpoint | 0.00954067 | 0.01600509 | 0 | Eligible baseline |
| Epoch 1 | 0.00954619 | 0.01598686 | 64 | — |
| **Epoch 2** | **0.00946775** | **0.01587005** | **128** | **Selected** |
| Epoch 3 | 0.00916589 | 0.01599213 | 192 | — |

The selected validation RMSE is **0.84375% lower** than the unchanged checkpoint.
A zero-return forecast on the same 128 validation rows has RMSE **0.01589699**.
The selected adapter is only slightly below this simple reference, and the
third epoch is worse than the second. All declared epochs ran; the report does
not hide later degradation or choose an epoch from test results.

Training RMSE at epoch zero evaluates a fixed state. For epochs 1–3, it is the
aggregate of predictions made immediately before each batch update, so weights
change within that measurement. Validation always evaluates one fixed end-of-epoch
state. The two training measurements should not be treated as identical metrics.

The run processed 768 training examples in total and took approximately
172 seconds in the recorded CPU environment, including setup and validation.
This timing is one machine's observation, not a runtime guarantee.

## Conversion and replay checks

- **Initial forecast parity:** maximum absolute forecast difference was **0.0**
  for the first four validation contexts, comparing the loop's initialization
  with the unchanged stack.
- **Actual learning:** the selected gate tensors moved by an L2 norm of
  **0.44596215**. The full selected gate/adapter state moved by **2.15630316**.
  These are parameter changes, not model-quality scores.
- **Selected-state replay:** saving and loading the selected adapter reproduced
  the first four selected validation scores with maximum absolute difference
  **0.0**. This is a checked batch, not a claim that every exported row was
  independently replayed.
- **Source provenance:** the manifest records the runner, backend, workflow,
  data-provider, source CSV, checkpoint configuration, and checkpoint-weight
  hashes, together with installed package versions.

The unit tests additionally cover shared-layer identity, masks and positions,
causal prefixes, finite gradients, freezing, strict adapter serialization,
epoch-zero selection when training worsens validation, and unchanged training
results after perturbing every test-period price. These tests use small doubles
and do not require downloading the official weights.

## What the result does and does not establish

The conversion preserves the original starting forecast, trains successfully,
and saves a replayable selected adapter. The small validation improvement is
exploratory: it comes from one index, one seed, a sampled validation interval,
and checkpoint selection on those same validation observations. There is no
independent test score, confidence interval, transaction-cost evaluation, or
profitability claim for this pilot. Pretraining overlap with historical market
data is unknown.

Both the repeated shared block and the adapter are new. Their combined effect
cannot isolate a benefit from looping. The follow-up below adds a gated
adapter-only control with matched trainable parameters, initialization, data,
optimizer, training budget, and selection. Its gate has a different role and
it uses less compute. More seeds, a stricter compute/architecture ablation,
and an independently justified evaluation period are still needed for a
stronger causal or performance conclusion.

This pilot uses 128-close contexts and validation rows. The earlier official
TimesFM benchmark uses 256-close contexts and 1,438 test rows. Their RMSE values
must not be ranked against each other. The existing six-model test benchmark
and its source hashes are preserved unchanged.

## Follow-up: evaluation on common periods

The subsequent comparison completed on **27 November 2026 at 08:01:37 Hong
Kong time**. It uses 128-close contexts for all three official-checkpoint
variants, the same return target, all **1,436 validation dates**, and exactly
the **1,438 test dates** used in the existing architecture benchmark. All test
labels and their order match that benchmark exactly.

| Model | Full validation RMSE ↓ | Test RMSE ↓ |
| --- | ---: | ---: |
| Zero-return reference | **0.01599491** | **0.00773293** |
| Frozen TimesFM 3, 128-close context | 0.01620599 | 0.00790151 |
| Loop fine-tuning, fresh same-recipe retrain | 0.01615178 | 0.00787693 |
| Parameter-matched gated adapter-only control | 0.01615656 | 0.00788071 |

The loop retrain has **0.3111% lower test RMSE** than the frozen model under
this common input protocol. It is **1.8621% worse than zero return** and
**1.9721% worse** than the prior TimesFM mini mean, 0.00772459. That mini uses
32-session factor windows and random initialization; it is a different input
and training regime, while the test dates and labels are shared.

The validation/test scale difference is mostly visible in the target itself:
zero return has RMSE 0.01599491 on the full 2007–2013 validation interval and
0.00773293 on the 2013–2018 test interval, a ratio of approximately **2.0684**.
The original 128-row sampled-validation score around 0.016 therefore cannot
be ranked directly against the old test score around 0.0077. The selected
adapter's small sampled-validation advantage over zero return also did not
extend to the full validation interval.

[Complete comparison and provenance](../reports/timesfm_comparison/comparison.json)
· [All aligned predictions](../reports/timesfm_comparison/all_period_predictions.csv)
· [Shared backtest, plots, and new candidate models](financial-research.md)

### Numerical replication, not restoration of the original adapter

The original local adapter was unavailable in the restored execution
environment. The original recipe was run **once**, with identical checkpoint
and provider bytes, loop training sources, sample records, seed, and recorded
package versions. The strict numerical comparison failed; this failure is
[preserved](../reports/timesfm_comparison/strict_recovery_attempt.json).

| Reproduction check | Observation |
| --- | --- |
| Original selected sampled-validation RMSE | 0.0158700501993 |
| Fresh retrain selected sampled-validation RMSE | 0.0158702974238 |
| Selected epoch | 2 in both runs |
| Maximum learning-curve RMSE discrepancy | 2.4722e-7 |
| Maximum selected-score discrepancy over all 128 rows | 8.4155e-6 |
| RMSE between those 128 selected-score vectors | 1.7757e-6 |
| Checked context/label records | All 384 match exactly |
| Adapter archive hash | Different |

The comparison resumed using that one retrain, with no additional training
attempt to force agreement. Its status is **`numerical_replication_only`**,
and the test/backtest forecast represents this fresh retrain. The original
adapter's exact test performance was not measured. Small prediction changes
can alter a trading signal's sign, so the distinction matters for backtesting.
The current platform, CPU capability, and Torch parallel configuration are
recorded. Original CPU details were not recorded; the cause of the drift is
not established.

The retrain ran from 07:43:33 to 07:45:50 Hong Kong time. The continued control
and evaluation run began at 07:49:50; both adapter selections were frozen at
**07:52:08**, before the first test-price query. Machine-readable timestamps
remain in UTC. The two original loop training modules were preserved
byte-for-byte. The comparison source SHA-256 was
`a57420079918da48e878165a9a4002210b2a1b4a8d0cdc4545599cb8532f42fc`;
the manifests retain the remaining source/checkpoint hashes and versions.

### Completed gated adapter-only control

Both variants start with the same gate and rank-8 adapter tensors and train
**21,761 parameters** under the same 256-training/128-validation-sample budget,
seed, minibatch order, optimizer, three epochs, and epoch-zero eligibility.
Both selected epoch 2. Full validation and test scoring began after selection
and did not change either checkpoint.

The control inserts `h + (1 + gate(h)) * adapter(h)` after layer 11 and executes
the original 20 layers. The loop uses
`h + gate(h) * (F_shared(h) - h) + adapter(h)` and executes 24 layer applications.
The trainable parameter counts match, but the gate's role and computation
budget differ. The small test gap, 0.00787693 versus 0.00788071, does not
establish a benefit specifically caused by looping. This remains one seed on
a previously inspected historical interval, with unknown pretraining overlap.

The [comparison archive](../reports/timesfm_comparison) contains 13
allowlisted JSON/CSV files: the comparison, aligned predictions, strict-check
failure, and both runs' manifests, metrics, learning curves, sample records,
and sampled-validation predictions. They are byte-identical copies of the
run outputs. Checkpoint and adapter tensors are excluded. The
[`recovered_loop` directory](../reports/timesfm_comparison/recovered_loop)
retains the original run layout; its recorded status explicitly identifies
the fresh numerical retrain.

## Reproduce the pilot

Follow the [beginner setup](getting-started.md), then obtain the separately
licensed local checkpoint using the [official inference guide](architectures/timesfm.md#timesfm3_pretrained).
From the repository root, with the virtual environment activated:

```bash
python -m pip install -e ".[timesfm]"
fts import-csv --csv "^GSPC.csv" --symbol SPX --provider .data/my-spx
fts finetune-timesfm --provider .data/my-spx --checkpoint .data/timesfm3-checkpoint --output runs/my-timesfm-loop --epochs 3 --train-samples 256 --valid-samples 128 --context-length 128 --batch-size 4 --lr 0.001 --seed 42 --device cpu
```

Skip the import if that provider already exists. Choose a fresh output directory
for another run. The command refuses to overwrite a completed experiment and
records failures in its manifest. Budget flags are sample limits; a smaller
provider may have fewer eligible examples, which are recorded explicitly.

The output contains `manifest.json`, `samples.json`, `learning_curves.json`,
`metrics.json`, `validation_predictions.csv`, and the local `adapter.pt`.
The saved adapter requires the same base checkpoint and loop settings when
restored. The report directory publishes the five numerical/provenance files
as byte-identical copies of this run. Absolute paths in the manifest describe
the execution environment; they are not required installation locations.

The [checkpoint license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE)
permits noncommercial, nonproduction research and prohibits redistribution of
the model and its derivatives. **The original checkpoint and learned adapter
weights are excluded from this repository.** The manifest retains the local
adapter's hash for provenance, but the binary is intentionally not published.

### Reproduce the common-period comparison

With the same imported provider and local official checkpoint, run:

```bash
python -m financial_ts.timesfm_comparison --provider .data/my-spx --checkpoint .data/timesfm3-checkpoint --output runs/my-timesfm-comparison --original-evidence reports/timesfm_loop_pilot
```

This checks checkpoint/provider identity, performs one original-recipe
retrain, and checks all recorded sampled predictions before training the
control. If strict numerical reproduction fails, inspect the generated
comparison and retrain evidence. A completed retrain that stopped before
control training and test evaluation can be explicitly continued as a fresh
numerical replication:

```bash
python -m financial_ts.timesfm_comparison --provider .data/my-spx --checkpoint .data/timesfm3-checkpoint --output runs/my-timesfm-comparison --original-evidence reports/timesfm_loop_pilot --resume --allow-numerical-replication
```

That continuation reuses the saved retrain and preserves the failed strict
attempt. It does not rerun training to seek closer agreement. Changed source,
checkpoint/provider bytes, sampled contexts/labels, or selected epoch remain
errors. Completed comparisons cannot be overwritten.
