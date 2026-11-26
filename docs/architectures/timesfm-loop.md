# Looping and fine-tuning the official TimesFM3 checkpoint

**Status: experimental research architecture.** This converts the existing
official TimesFM3 checkpoint into a shared-block loop model and fine-tunes
small gates and adapters. It is not an official Google model release, and it is
separate from this repository's randomly initialized `timesfm3_mini`.

The implementation is in
[`financial_ts.timesfm_loop`](../../src/financial_ts/timesfm_loop.py). It uses the
[official TimesFM3 model and decode implementation](https://github.com/google-research/timesfm/blob/918df02c9d366b2145bc5876135fa2131d3ed2a6/src/timesfm3/torch/model.py)
and the original
[mixing transformer layers](https://github.com/google-research/timesfm/blob/918df02c9d366b2145bc5876135fa2131d3ed2a6/src/timesfm3/torch/transformer.py)
through the pinned `timesfm==3.0.2` package. No pretrained weights are copied
into the source files.

## What changes

The original 20-layer model executes normally through layer 11, using zero-based
indices. The middle block, layers `[8, 12)`, then executes once more using the
**same four layer objects and parameters**. The result is gated into the hidden
state before the original remaining layers execute. Each extra traversal has
its own small gate and adapter; the pretrained middle block is shared.

For a hidden token `h`, shared block `F`, and refinement step `r`:

```text
n = RMSNorm(h)
gate_r(h) = gate_scale * tanh(W_gate_r n + b_gate_r)
adapter_r(h) = W_up_r GELU(W_down_r n)
h_next = h + gate_r(h) * (F(h) - h) + adapter_r(h)
```

The gate is one bounded scalar per token, including each variate/patch
position. Temporal and variate masks, segment identifiers, and rotary positions
are passed unchanged to each replay. A loop iteration does not advance the
time index. Nonempty autoregressive KV caches are rejected; this implementation
uses the original full-sequence forecasting path.

Gate weights and biases start at zero, and the adapter's up-projection starts
at zero. Therefore the initial refinement is exactly the identity. The gate
and up-projection receive gradients immediately; the down-projection starts
receiving gradients after the up-projection changes.

| Default setting | Value |
| --- | --- |
| Frozen official parameters | 330,710,976 |
| Middle block | Layers 8–11, inclusive |
| Total traversals of that block | 2 |
| Extra layer evaluations | 4; 24 total rather than the original 20 |
| Adapter rank | 8 |
| Gate magnitude bound | 0.1 |
| Trainable gate/adapter parameters | 21,761 |
| Total parameters including frozen backbone | 330,732,737 |

All pretrained parameters remain frozen. Each additional traversal adds another
21,761 trainable parameters at this width/rank. `loop_steps=1` is the original
model without trainable refinements. Gates modulate the loop's contribution;
they do not implement adaptive halting or skip computation. Weight sharing
limits added parameters, while retaining the full pretrained model and adding
compute.

## Forecast and training API

Install the optional package and supply the separately stored local checkpoint:

```bash
python -m pip install -e '.[timesfm]'
```

```python
import torch
from financial_ts.timesfm_loop import LoopTimesFM3

model = LoopTimesFM3(
    checkpoint_path=".data/timesfm3-checkpoint",
    device="cpu",
    num_threads=2,
    loop_start=8,
    loop_end=12,
    loop_steps=2,
    adapter_rank=8,
    gate_scale=0.1,
)

# contexts is a floating [batch, context_length] tensor of provider $close prices.
forecast_closes = model.forecast(contexts)       # [batch, 2], differentiable
return_predictions = model.return_scores(contexts)  # [batch], float64 ratio
unchanged_baseline = model.baseline_forecast(contexts)

optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters() if parameter.requires_grad],
    lr=0.001,
)
```

Contexts must contain finite positive closes and end at the prediction date.
They are prices in the provider's units, including upstream adjustments, not
normalized Alpha158 factors or unadjusted tape prices. The return score is
`forecast_close[t+2] / forecast_close[t+1] - 1`, matching the existing Qlib label.

The official public forecaster disables gradients. This backend explicitly
unwraps the pinned model's `no_grad` decode decorator and calls the same decode
body with gradients enabled. It retains official preprocessing, detrending,
RevIN, masked horizon construction, stitching, and output quantiles. Quantiles
are sorted before selecting the median, matching the public forecaster's
default behavior. The pretrained backbone stays in evaluation mode even during
adapter training.

Use training labels alone to determine any target scaling. Choose adapter
checkpoints using validation data, include the unmodified initialization as an
eligible candidate, and keep the test period out of training and model
selection. A small fixed-budget experiment establishes that this training path
runs; it does not establish that looping improves prediction quality. The
official checkpoint's pretraining overlap with historical market data remains
unknown.

This experiment changes both looping and adapters. Comparing it with the
unchanged checkpoint measures their combined effect and cannot attribute any
improvement specifically to repeated transformer computation. A matched
adapter-only control, using the same data, adapter placement/rank, optimizer,
training budget, and validation selection, is needed before claiming a benefit
from looping itself. Multiple seeds and a larger, independently justified
evaluation would be required for a stronger performance conclusion.

## Saving and restoring

`adapter_state_dict()` returns cloned CPU tensors for learned gates and adapters
only. Its keys match the full names of trainable parameters, for example:

```text
model.transformer_stack.refinements.0.gate.weight
model.transformer_stack.refinements.0.gate.bias
model.transformer_stack.refinements.0.adapter_down.weight
model.transformer_stack.refinements.0.adapter_up.weight
```

`load_adapter_state_dict(state)` rejects missing or additional keys, mismatched
shapes, nonfloating tensors, nonfinite values, and conversion overflow. It
validates the complete mapping before updating parameters. Save the constructor
configuration and checkpoint provenance alongside these tensors: tensor shapes
alone do not encode the loop's start/end positions or gate scale. Restoring
requires the same base checkpoint and loop configuration.

The backend exposes `checkpoint_path`, `checkpoint_config_sha256`,
`checkpoint_size_bytes`, `parameter_count`, `trainable_parameter_count`, and
`base_parameters()` for experiment provenance and optimizer checks. A
`baseline_forecast` call temporarily disables the refinements and restores their
previous enabled state even if inference raises an error.

Keep the original checkpoint and any learned adapter tensors outside version
control, such as under ignored `.data/` and `runs/` directories. The original
code is Apache-2.0, while the [checkpoint's TimesFM Non-Commercial License
v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE)
restricts model/derivative use to noncommercial, nonproduction purposes and
prohibits redistribution. The adapter-only format does not remove those
restrictions. This repository distributes source and numerical reports, not
trained model or adapter weights.

## Verification and scope of evidence

The small-stack tests run without the optional TimesFM package or downloaded
weights. They check exact identity initialization, shared layer identities,
unchanged mask/position propagation, causal prefix behavior, padding isolation,
finite gradients, frozen base parameters, strict adapter serialization, and
round-trip forecast equality. These tests validate control flow and training
contracts, not Google's forecasting quality.

A separate CPU smoke test with the actual 330M checkpoint verified bit-exact
initial forecast parity, two optimizer steps with finite adapter gradients,
absence of frozen-base gradients, and exact reset/reload behavior. It used
synthetic contexts to validate the gradient path, not to claim market
performance.

The subsequent [fixed financial pilot](../timesfm-loop-validation.md) trained
on 256 historical SPX examples for three epochs and selected among epoch zero
and all trained states using 128 validation examples. Epoch 2 had validation
RMSE 0.01587005, compared with 0.01600509 for the unchanged checkpoint. The
report contains every epoch, sample boundaries, source/checkpoint hashes, and
selected predictions. The test partition was unused. The original checkpoint
and historical test benchmark outputs were not changed.
