# Legacy notebook audit

The 2017–2018 notebooks remain available as historical material. The supported
workflow is now the Qlib package and CLI described in the main README. The legacy
notebooks were inspected as source, not executed, during migration.

## Why dependency upgrades alone are insufficient

| Finding | Source | Impact |
| --- | --- | --- |
| Training and holdout independently fit `RobustScaler`, including the target column. | Notebook 5, `preprocess_df` and the following split cell; repeated in notebook 6. | Holdout observations determine their own transformation, and the model sees a different scale at evaluation. Inverse transformation also uses holdout target statistics. |
| Sequences are shuffled before Keras `validation_split=0.1`. | Notebooks 5 and 6. | Overlapping 60-day windows are mixed between training and internal validation; this is not chronological model selection. Shuffling batches after a chronological split is acceptable. |
| `Y = lrets.shift(-1)` is constructed before the outer split, without purging. | Notebooks 5 and 6. | The final training label reaches across the nominal holdout boundary. |
| Hyperparameter search optimizes `accuracy` / `val_acc` for continuous returns. | Notebook 6, `create_model`. | The search objective does not measure return-regression quality. |
| Removed interfaces include standalone `CuDNNLSTM`, pandas `.ix` and `DataFrame.append`; older scripts use TensorFlow 1 sessions/placeholders. | Model, news and preprocessing notebooks; `Old version/`. | The old pipeline is incompatible with current supported dependencies. |
| The preparation notebook reads `GSPC.csv`, but the tracked file is `^GSPC.csv`; preprocessing writes `data2`, while training reads `data2.h5`. | Notebooks 2, 4 and 5. | A fresh clone cannot reproduce the notebook chain as written. |
| Inverse scaling hardcodes 184 prediction rows and nine other columns. | Notebook 5, prediction postprocessing; repeated in notebook 6. | Changing data length or feature count breaks inference. |
| News publication timestamps are reduced to dates without an exchange-session cutoff. | News downloader and news/technical-data combination notebooks. | Same-date news is not established as available by the trading decision time. |

Some legacy files refer to API credentials. The new workflow neither runs these
downloaders nor copies their credentials. Its offline example needs no API key.

## Prediction timing changed deliberately

The original model's target was the next-day **log return**:

```text
log(close[t+1] / close[t])
```

The supported Qlib target is the delayed next-day **simple return**:

```text
Ref($close, -2) / Ref($close, -1) - 1
```

Features observed through close `t` predict the return from close `t+1` to close
`t+2`. This allows an entry at the following close rather than assuming a fill at
the same close used to construct the features. The old plots and scores therefore
cannot be treated as a directly comparable baseline, or as verified trading
performance.

## Evaluation requirements

Fit preprocessing on training data only and reuse it unchanged for validation
and test. Partition chronologically before model selection; tune and stop on
validation, then evaluate the held-out test period. For the two-session label
horizon, remove the final two feature dates before each next partition so that
earlier labels end strictly before that partition begins. Count actual provider
calendar sessions, including holidays and gaps, rather than elapsed days.

Keep raw returns for evaluation even if a learning-only label transform is used.
Cross-sectional label normalization is unsuitable for the bundled single-index
example because one instrument produces degenerate labels. Validation/test
sequences may use already observed historical feature context, but never future
observations. Fees, execution timing and data quality still need explicit
assumptions in any portfolio backtest. Synthetic runs verify execution, not
profitability.
