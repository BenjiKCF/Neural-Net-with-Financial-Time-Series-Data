# Daily cross-sectional research models

This source-only roadmap implementation adds three research models alongside
the archived eight-model catalog. Recipes and tests are included; new trained
weights, data-derived fixtures and experiment outputs are generated locally. Their batching and label contracts are shared,
so changes in the objective can be compared without silently changing which
stocks the model observes.

| Model | Temporal attention | Cross-stock attention | Market context | Experts |
| --- | --- | --- | --- | --- |
| `daily_transformer` | Yes | Off | Off | One |
| `master_style` | Yes | On by default | On by default | One |
| `quantformer_moe` | Yes | On by default | On by default | Four by default |

These implementations are trained from scratch on the repository's own data.
`master_style` is inspired by the temporal/inter-stock aggregation and
market-conditioned feature selection of
[MASTER](https://arxiv.org/abs/2312.15235). It is **not an exact reproduction**:
its observable market descriptors are cross-sectional mean and dispersion of
the current eligible Alpha157 factors; MASTER uses its own market-index
descriptors. No upstream code, preprocessed data, or trained weights are
imported. The [official repository](https://github.com/SJTU-DMTai/MASTER)
documents important validation/preprocessing distinctions, which is why this
implementation uses the same feature-only inference view for fitting,
validation, and prediction.

## Date and instrument semantics

Each network call receives the complete eligible cross section from **one
decision date**, shaped `[stock, history, feature]`. Every history ends at the
decision cutoff. Instruments are sorted by their stable identifier within each
date, but stock position is not embedded: permuting the stocks permutes their
scores. Padding is excluded from temporal attention, cross-stock attention,
and market descriptors. Membership changes alter the actual eligible panel.

Missing labels mask only the loss. They do not delete an otherwise eligible
stock from the attention context. This matters at validation, near delistings,
and for different label horizons. Whole dates are used without subsampling;
`batch_size` remains accepted for workflow compatibility but never chunks a
cross section. The current adapter makes one equally weighted optimizer update
per usable date and records skipped/undefined days.

## Objective controls

`objective` accepts `mse`, `huber`, `ic`, `ranking`, or `combined`.

- `ic` optimizes negative daily Pearson correlation, with a declared small
  auxiliary Huber term. It does not differentiate through Spearman ranks.
- `ranking` uses the mean logistic loss on untied target pairs, with the same
  auxiliary Huber term. Tied predictions remain in training.
- `combined` explicitly weights Huber, Pearson, and pairwise ranking terms.
- Constant predictions are retained. Constant/singleton targets have an
  undefined correlation term, which is counted; the auxiliary regression
  signal remains available. Denominators are stabilized in float32.

The auxiliary Huber coefficient for IC/ranking defaults to `0.05` and can be
changed through `objective_kwargs`. Checkpoint selection is declared as
`selection_metric: objective` or `mse` and uses validation only. Standard
`learning_curves.json` retains raw-unit MSE for compatibility; the actual
optimized losses, skipped-date counts, routing, and selected checkpoint are
exported separately in the run's objective/training/pretraining/routing receipts.
The objective history separates the base forecast loss, weighted routing and
turnover penalties, and the total optimized loss. Validation selection uses
the declared forecast criterion and excludes those training regularizers.

The optional `turnover_weight` adds turnover of explicitly mapped long-only
softmax portfolio weights, including cash. Previous weights are aligned by
instrument ID, and absent instruments are frozen. This training surrogate
assumes current names are tradable. It is **not an executed cost, impact model,
or substitute for the independent execution ledger**.

## Architectural and efficiency ablations

`architecture_kwargs` supports `cross_stock_attention`, `market_context`,
`num_experts`, and `uniform_routing`. For the MoE, compare one expert, a uniform
mixture, and a learned gate. The learned gate uses observable market context;
daily weights, mean utilization, entropy, and effective expert count can be
inspected. Expert names do not imply discovered economic regimes.

`pretrain_epochs` enables masked historical-feature reconstruction on the
training panel before supervised fitting. The receipt records the actual
training anchors, mask probability, reconstruction losses, and extra runtime.
Validation and test features are excluded. The appropriate control has the
same architecture and `pretrain_epochs: 0`.

The default precision is float32. `precision: bfloat16` enables autocast while
keeping objectives and normalization arithmetic in float32. Numerical and
ranking parity must be measured for the intended hardware before changing the
default. No speed or memory improvement is asserted by the option's existence.

## Run a native Qlib experiment

```bash
fts train --config configs/master_style.yaml \
  --provider .data/chenditc/2026-11-26 --output runs/master-style --no-backtest

fts train --config configs/quantformer_moe.yaml \
  --provider .data/chenditc/2026-11-26 --output runs/quantformer-moe --no-backtest
```

Use the registered rolling runner for meaningful comparisons. The native Qlib
workflow keeps its existing label `close[t+2] / close[t+1] - 1`. Extra horizon
heads use a separate runner so their semantics cannot be mistaken for that
one-session portfolio replay.

## Multi-horizon, intervals, and adaptation

```bash
python scripts/run_advanced_research.py \
  --provider .data/chenditc/2026-11-26 \
  --config configs/advanced_screening.json \
  --output runs/advanced-screening
```

The runner registers its protocol and provider fingerprints before fitting.
It chooses a bounded, fixed stock set from historical eligibility on the first
training day and respects later membership. The same training-fitted Alpha157
processor feeds these controls:

1. A shared encoder with 1-, 5-, and 20-session return heads.
2. The shared model with monotonic 10%, 50%, and 90% quantile heads.
3. A separate single-head model for each horizon.
4. Frozen, guarded incremental, and full-retraining versions of the point model
   evaluated on a separate post-update period.

All returns start at the same next-session entry. The maximum target endpoint
is therefore 21 sessions, and that many observed sessions are purged between
periods. Features remain available from earlier history, but every fitted or
selected label must have matured by the declared cutoff. Incremental updates
retain the old target normalizer and compare every updated checkpoint with
epoch zero; a harmful update or interrupted optimizer restores the retained
weights. The simple update is **not a DoubleAdapt reproduction**.
The full-retraining control refits all model weights and its target scaler on
the original and new training panels. All three adaptation controls retain the
original feature scaler, which isolates model adaptation from a change in
feature preprocessing; this is not a full pipeline retraining experiment.

Quantile calibration uses a separate chronological segment after validation.
Each date contributes its worst cross-sectional noncoverage residual; the
reported expansion uses a conservative finite-sample order statistic. Insufficient
calibration dates fail explicitly. This is empirical calibration; market
dependence means no exchangeability or future-coverage guarantee is asserted.
A local run records raw and calibrated intervals, coverage and width for
inspection. The separate [DoubleAdapt comparator](../doubleadapt.md) provides
an explicit meta-learning control for incremental adaptation.

## Verify and interpret a local run

The architecture and adapter tests use constructed inputs to check contracts:

```bash
python -m pytest -q tests/test_cross_stock.py tests/test_cross_sectional.py \
  tests/test_objectives.py tests/test_advanced_research.py
```

Inspect the generated protocol before fitting and retain the full output after
fitting. The advanced runner records preprocessing, segment membership, model
configurations, numerical checkpoints, prediction indices, interval calibration
and adaptation decisions. These files belong to that local campaign and are not
published as part of this source-only release.

Keep each architecture/objective ablation separate. Compare identical date and
instrument panels, preserve epoch-zero controls, disclose extra pretraining or
meta-learning compute, and inspect adverse outcomes. A bounded smoke recipe can
exercise the machinery; a scientific claim needs sufficient chronological
folds/seeds, meaningful uncertainty estimates and a separate confirmation period.
Multi-horizon forecast quality also requires an execution policy that matches
the intended holding horizon before a trading conclusion is justified.
