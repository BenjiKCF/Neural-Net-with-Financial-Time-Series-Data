# DoubleAdapt-style adaptation comparator

[`doubleadapt.py`](../src/financial_ts/doubleadapt.py) independently implements
feature adaptation, label adaptation and a model initialization trained through
a differentiable inner update. It supplies a research comparator to frozen
weights, ordinary incremental updates and full retraining. The simpler update
in [`advanced_research.py`](../src/financial_ts/advanced_research.py) remains a
separate method.

The method, tests and registered recipe accompany the retained
[results](../reports/production_roadmap/doubleadapt_screening/results.json) and
[protocol](../reports/production_roadmap/doubleadapt_screening/protocol.json).
The compact screen does not establish an advantage over the controls. Its
original predictions, numerical states and source identity are preserved;
publishing them does not rerun the study on later code or reproduce an upstream
benchmark.

## Method reference and scope

The methodological reference is
[*DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend
Forecasting*](https://arxiv.org/abs/2306.09862v3), with the
[official repository](https://github.com/SJTU-DMTai/DoubleAdapt) as a reference.
The local implementation does not import that project or distribute its code,
processed data or weights. The run protocol identifies the method as
`independent_DoubleAdapt_paper_method` with `upstream_reproduction: false`.

Its compact forecaster uses current-date Alpha158-derived OHLCV factors and an
MLP. It preserves the data/model adaptation structure while making its model,
normalization, task order and differentiation choices explicit.

## Computation

For processed feature vector \(x\), prototype weights are

\[
w_i(x)=\operatorname{softmax}_i\big(\operatorname{cos}(x,p_i)/\tau\big).
\]

Residual feature heads produce

\[
g(x)=x+\sum_i w_i(x)(A_i x+b_i).
\]

The target normalizer uses unique finite meta-training labels and remains fixed.
For raw return \(y\), let \(z=(y-\mu)/\sigma\). Affine label heads transform and
restore the internal target as follows:

\[
h(x,z)=\sum_i w_i(x)\big(\exp(a_i)z+c_i\big),
\qquad
r(x,u)=\sum_i w_i(x)\frac{u-c_i}{\exp(a_i)}.
\]

The restoration is the weighted sum of the inverse of each affine head. It is
generally not the algebraic inverse of the weighted mixture. Positive scales
keep each head invertible; checks reject nonfinite or unsafe updates. Scores
are returned to original return units through \(\mu+\sigma r(x,u)\).

The support set supplies one inner model step:

\[
\theta'=\theta-\eta_{\mathrm{inner}}\nabla_\theta
\operatorname{MSE}\big(F_\theta(g(X_s)),h(X_s,z_s)\big).
\]

After query scores have been recorded and query labels have matured, the
upper-level objective is

\[
\operatorname{MSE}\big(r(X_q,F_{\theta'}(g(X_q))),z_q\big)
+\lambda\operatorname{MSE}\big(h(X_s,z_s),z_s\big).
\]

Gradients update both the initial model and the feature/label adapter.
`torch.autograd.grad(..., create_graph=True)` retains differentiation through
the inner update. The [tests](../tests/test_doubleadapt.py) check that model,
feature-adapter and label-adapter parameters receive gradients and change on a
constructed task, as well as checking the headwise restoration formula.

## Deliberate implementation choices

| Component | Local choice and comparison requirement |
| --- | --- |
| Forecaster | A compact `Tanh` MLP on current-day factors. This differs from a recurrent model over Alpha360 inputs. |
| Inner differentiation | Exact differentiation through the inner step. Its computational cost differs from a first-order approximation. |
| Label scale | `exp(log_scale)` constrains each affine scale to be positive. |
| Initial adapters | Zero feature residuals/biases and unit label scales; registered random initialization for prototypes and forecaster. |
| Offline sequence | Chronological tasks and validation-only checkpoint selection; epoch zero remains eligible. |
| Normalization | Feature processing fits in the allowed training window; the target normalizer uses meta-training labels only. |
| Task construction | Support/query windows and their information-endpoint gaps use observed exchange sessions. |
| Controls | Same eligible panel, raw targets, MLP width, seed and immutable target scale; different update counts are disclosed rather than described as equal compute. |

Exact values for widths, heads, learning rates, epochs, task sizes and uncertainty
calculations are declared in
[`doubleadapt_screening.json`](../configs/doubleadapt_screening.json). This
bounded recipe exercises the method; it does not define a sufficient publication
or production experiment budget.

## Chronology and prediction locking

The signal-date target is `close[t+2] / close[t+1] - 1`, with information endpoint
`t+2` on the observed session calendar. Missing entry/exit prices remain missing.
The target builder does not shift over missing stock dates or invent terminal
prices. Raw return units here mean unstandardized returns from the provider's
price series; they do not turn adjusted prices into executable cash quotes.

The screening cohort is selected from historical membership on the first
training date, with later membership checked at each signal date. Its compact
alphabetical subset is a computational choice, not a representative random
sample of all CSI300 stocks. A retrospective provider does not certify
historical intraday availability.

The online sequence is explicit:

1. `begin_task(support, inference_start=..., as_of=...)` validates support labels
   and requires them to mature before the query period. It retains the adapted
   parameters and the original model-state hash.
2. `predict(features, index)` receives no labels. The model operates row by row,
   retaining emitted scores and original features by exact date/instrument ID.
3. The runner writes a scores-only file and its hash before admitting the
   corresponding query labels.
4. `finish_task(query, as_of=...)` requires mature labels, unchanged initial
   state, the exact emitted identities and original features. It updates future
   model state and records its causal cutoff.

Repeated predictions must preserve both the score and original features.
Invalid finalization leaves the pending task and model state intact. Once a
later label update has occurred, the resulting model cannot be reused for an
earlier historical inference cutoff. The local hash receipts make this sequence
auditable; they are not an independent timestamp service or proof of original
market-data publication times.

## Matched controls

| Method | Updates permitted at an online task |
| --- | --- |
| `doubleadapt_paper` | Support adaptation, transformed-feature query prediction, then a meta update after the query scores are locked and labels mature. |
| `frozen` | Supervised initialization fitted on allowed initial history; no online weight updates. |
| `naive_incremental` | Fine-tunes current weights on mature support labels, predicts, then uses mature query labels after locking scores. |
| `full_retrain` | Initializes a new MLP from the registered seed and fits retained mature history plus current support; current query labels enter history only after prediction. |

Control training epochs are selected within allowed validation history. Controls
share the fixed meta-training target scale; they do not use DoubleAdapt's later
parameters, gradients or online statistics. Retain their individual cutoffs,
fitting budgets and predictions when comparing outcomes.

## Actual compact-screen results

Source: the immutable input release `chenditc/investment_data` dated
`2026-11-26`, with provider SHA-256
`1b6c53d8fa6a7caa1c59719e0621555f8930713c3deaacd9985dedc812ceb7a8`.
Full release metadata and archive identity are in the protocol. There are 20
stocks, 15 online query dates, and 300 evaluated date/instrument rows per method;
no label is missing in this particular evaluation panel. All methods share the
same zero-return RMSE of **0.0184036102**.

| Method | RMSE | MAE | Mean daily IC | Mean daily Rank IC | Directional accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| DoubleAdapt paper method | 0.0186110865 | 0.0134941833 | -0.058878 | -0.079347 | 46.00% |
| Frozen MLP | 0.0185393784 | 0.0134797283 | -0.055063 | -0.086575 | 48.00% |
| Naive incremental MLP | 0.0186246671 | 0.0134887127 | -0.040569 | -0.067208 | 44.67% |
| Full retrain MLP | 0.0185483191 | 0.0134730786 | -0.051134 | -0.080705 | 46.33% |

The declared DoubleAdapt validation candidates were epoch zero, one, and two.
Their validation MSEs were respectively `0.0002057818492`,
`0.0002059955150`, and `0.0002062095155`. **Epoch zero was selected.** This means
the offline meta-training passes did not improve the registered validation
criterion within this budget. The selected initialization still undergoes the
declared mature-validation and online meta updates. The receipts retain the
attempted offline candidate updates as well; they must not be interpreted as
all belonging to the selected checkpoint's ancestry.

### Paired date-block uncertainty

The implementation uses the repository's paired circular moving-block
bootstrap, with five-session blocks, 200 draws, and seed 42. Each draw retains
whole daily cross sections and exactly matched date/instrument labels. The
differences below are **DoubleAdapt minus the control**. Negative RMSE
differences and positive IC differences favor DoubleAdapt.

| Control | RMSE difference | RMSE 95% interval | IC difference | IC 95% interval |
| --- | ---: | --- | ---: | --- |
| Frozen | +0.0000717092 | [-0.0000602650, +0.0002308659] | -0.003815 | [-0.039208, +0.032398] |
| Naive incremental | -0.0000135803 | [-0.0001379518, +0.0001258687] | -0.018309 | [-0.099562, +0.045841] |
| Full retrain | +0.0000627682 | [-0.0000354436, +0.0001814676] | -0.007744 | [-0.032738, +0.014708] |

| Control | Rank IC difference | Rank IC 95% interval |
| --- | ---: | --- |
| Frozen | +0.007228 | [-0.038286, +0.053572] |
| Naive incremental | -0.012139 | [-0.101239, +0.056282] |
| Full retrain | +0.001358 | [-0.028128, +0.031894] |

All intervals contain zero, and the sample contains only three five-session
query blocks. These intervals are exploratory measures of uncertainty, not a
broad significance or generalization claim. No test-driven tuning was
performed after reading these results. There is no evidence here to promote
this model or to replace an existing strategy.

## Run and inspect

After installing the neural dependencies and obtaining a pinned provider:

```bash
fts research-doubleadapt --provider .data/cn-20261126 \
  --config configs/doubleadapt_screening.json --output runs/doubleadapt-screen

python -m pytest -q tests/test_doubleadapt.py
```

The retained experiment includes [prediction locks](../reports/production_roadmap/doubleadapt_screening/prediction_locks.json),
[meta-update receipts](../reports/production_roadmap/doubleadapt_screening/meta_update_receipts.json),
[control receipts](../reports/production_roadmap/doubleadapt_screening/control_receipts.json),
[aligned forecasts](../reports/production_roadmap/doubleadapt_screening/predictions.csv)
and [artifact hashes](../reports/production_roadmap/doubleadapt_screening/artifacts.json).
Earlier attempts remain in the evidence directory with their diagnostics. The
canonical `doubleadapt_screening` record reflects the prediction-lock correction;
those earlier attempts are not additional tuning trials.

The command writes a protocol before fitting and retains prediction locks,
selection decisions, meta-update/control receipts, numerical states, aligned
predictions and paired statistics in the new output directory. Preserve failed
attempts and adverse results. Changing a recipe or runtime requires a new
campaign identity.

Compare raw-return errors and daily correlations on exactly matching dates and
instruments. Date-block uncertainty should preserve each cross section; inspect
sample length and effective block count before interpreting its intervals.
This forecast runner does not provide a costed strategy or calibrated position
size. Those decisions require a separately registered policy, execution
accounting and prospective confirmation under the
[production gates](production-roadmap.md#promotion-gates).
