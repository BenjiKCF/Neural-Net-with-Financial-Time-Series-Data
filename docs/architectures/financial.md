# Small financial forecasting candidates

These two models are **local, randomly initialized research designs** for the
repository's daily-return task. Their definitions were fixed before the new
test comparison. They study whether restrained capacity and historical context
help when a simple zero-return forecast is already competitive. Their inclusion
does not imply that they outperform the existing models.

Both use the existing `SequenceForecastModel` and Qlib preprocessing: 157 numeric
factors, 32-session windows, training-only feature and target transformations,
and one delayed-return prediction. The shipped settings use width 32, two layers,
dropout 0.1, AdamW at learning rate 0.001, and up to 12 epochs with validation
early stopping after four non-improving epochs. Seeds are 7, 42, and 2026 in the
registered comparison.

## Shared historical summaries

At every token, both models compute trailing means over **1, 4, 16, and 32
session positions**, ending at that token. Missing rows contribute neither a
value nor an observation count. A missing session remains a time gap; the window
does not extend into older observations to compensate. Empty histories remain
finite. These operations never inspect a later token.

The output heads start at zero in normalized target units. The shared adapter
then adds the training-label mean back when predicting in return units, so the
initial forecast is the **training mean**, not a raw zero return. The zero-return
and training-mean controls should be reported separately.

## Multiscale residual mixer

**CLI name:** `multiscale_residual_mixer`

**Class:** `MultiScaleResidualMixerEncoder`

**Configuration:** [multiscale_residual_mixer.yaml](../../configs/multiscale_residual_mixer.yaml)

Each historical scale passes through the same feature projection. A learned
softmax mixes the four projected representations. Two causal mixer blocks then
apply depthwise temporal convolution and a feature MLP, each with a modest
learned residual gain. The temporal convolutions have kernel size three and
dilations one and two in the shipped configuration. They pad only on the left.

The forecast combines a linear map of all four original feature summaries with
a gated nonlinear readout:

```text
summary_t = concat(mean_1, mean_4, mean_16, mean_32)
hidden_t = causal_mixer(softmax_scale_mix(projected_means))
prediction_t = linear(summary_t) + sigmoid(gain) * residual_head(hidden_t)
```

Both output heads start at zero. The final residual gain and each block's update
gains start with logit -2, corresponding to approximately 0.119. These are learned
model parameters, not a validation-fitted output calibration.

The hypothesis is that a linear path can handle simple factor relationships,
while a small temporal residual represents departures from them without a large
attention network. `forward_features` exposes causal hidden states.

## Regime-gated linear experts

**CLI name:** `regime_linear_moe`

**Class:** `RegimeLinearMoEEncoder`

**Configuration:** [regime_linear_moe.yaml](../../configs/regime_linear_moe.yaml)

Three independent linear experts receive the same concatenated trailing means.
A small gate, with two width-32 hidden layers, supplies softmax mixture weights.
Its six descriptors are current feature amplitude, three changes between
adjacent-scale means, 16-session feature dispersion, and 32-session coverage.
Nonnegative amplitude/dispersion descriptors use `log1p` to moderate scale.

```text
expert_k = weight_k @ summary_t + bias_k
mixture_weights = softmax(gate(historical_descriptors_t))
prediction_t = sum_k mixture_weights_k * expert_k
```

These descriptors summarize normalized factors. They do **not** constitute a
separately estimated physical return-volatility process or named market regimes.
All three experts run on every example; the model is a cheap dense mixture,
without sparse routing or a load-balancing loss.

Expert heads start at zero, while the gate retains independent random
initialization so the experts receive different weighted updates. Gate gradients
start after expert predictions diverge. `routing_weights` exposes the mixture,
and `forward_features` returns each expert's weighted contribution over time.

## Dimensions and interpretation

`num_layers` counts temporal/feature blocks in the mixer and hidden gating
layers in the linear mixture. The common `nhead` argument is retained for the
shared training interface; neither model contains attention heads. Equal values
of the common width/layer arguments do not imply equal parameter count or compute.
The experiment reports actual parameter counts and runtime.

| Shipped configuration | Trainable parameters |
| --- | ---: |
| Multiscale residual mixer | 14,395 |
| Regime-gated linear experts | 3,266 |

These counts use 157 input features, width 32, two layers, and the four fixed
historical scales. All parameters are initialized locally and trained.

## Sources and scope

The implementations are independently written. Relevant architectural motivation:

- [Google Research: TSMixer](https://research.google/pubs/tsmixer-an-all-mlp-architecture-for-time-series-forecasting/)
  studies MLP mixing along time and feature dimensions.
- [TimeMixer, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/a7ac8a21e5a27e7ab31a5f42a0117bdb-Abstract-Conference.html)
  studies complementary information across temporal scales. This local model
  does not reproduce TimeMixer's complete decomposition or prediction pipeline.
- [LTSF-Linear, AAAI 2023](https://github.com/cure-lab/LTSF-Linear)
  motivates retaining simple linear comparators in time-series experiments.
- [Jacobs et al., Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
  introduces learned allocation among specialist predictors.

Those sources do not demonstrate financial superiority for these local designs.
All scores, forecast calibration, and trading policies must be evaluated under
the repository's explicit chronological protocol. A better historical RMSE is
also distinct from a useful strategy after turnover and transaction costs.

## Verification

The focused tests check trailing-window values against hand-calculated cases,
padding invariance, finite empty histories, causal prefixes, normalized
data-dependent routing, gradients through the intended components, exact
state-dict replay, and learning of a known synthetic historical signal. The
shared Qlib workflow provides separate integration and data-boundary checks.
