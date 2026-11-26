# TimesFM 3: small supervised model and official pretrained inference

This repository provides two deliberately separate model choices. `timesfm3_mini`
trains a small numeric architecture from random initialization using the Qlib
training split. `timesfm3_pretrained` loads Google's existing checkpoint and makes
zero-shot forecasts without fitting on this repository's labels. Their scores
have different provenance and should be reported separately.

## Architecture sources

The implementation was checked against the [Google Research architecture
description](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/),
the [official model card](https://huggingface.co/google/timesfm-3.0-pytorch),
and the [official source at commit
918df02](https://github.com/google-research/timesfm/tree/918df02c9d366b2145bc5876135fa2131d3ed2a6/src/timesfm3/torch).
The official [checkpoint configuration](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/config.json)
uses 20 layers, width 1,280, 16 heads, context patches of 32 and output patches of
64. The published checkpoint contains 330,710,976 parameters.

TimesFM 3 adds full attention across variates at each patch position alongside
causal temporal attention. Contiguous masked future patches allow the backbone
to predict the horizon in one pass. Its output statistic refinement uses
preceding predicted patches to update normalization moments. That refinement
does not rerun the attention stack or consume actual future observations. See
the upstream [model implementation](https://github.com/google-research/timesfm/blob/918df02c9d366b2145bc5876135fa2131d3ed2a6/src/timesfm3/torch/model.py)
and [refinement implementation](https://github.com/google-research/timesfm/blob/918df02c9d366b2145bc5876135fa2131d3ed2a6/src/timesfm3/torch/cpm_revin_refine.py).

## `timesfm3_mini`

`financial_ts.architectures.timesfm.TimesFMMiniEncoder` is an independently written
numeric adaptation. It contains no downloaded or distilled pretrained weights.
The default model has 110,253 parameters for 157 input factors.

| Mechanism | Small implementation |
| --- | --- |
| Variates | A learned projection compresses 157 factors to 8 latent variates. |
| Patches | Four sessions per context patch; left padding keeps the last observation aligned with the prediction cutoff. |
| Normalization | Each context patch uses only the moments of itself and preceding observed patches. Missing padding is excluded. |
| Mixing | Two layers with causal temporal attention, full attention across latent variates, and a feed-forward block. |
| Attention | RMSNorm, query/key RMS normalization, and temporal rotary positions; no rotary positions on the variate axis. |
| Masked horizon | Two contiguous masked future patches are processed in the same backbone pass as the context. |
| Refinement | Predicted earlier patches update the moments used to decode later predicted patches. |
| Supervision | A scalar regression head maps the estimated latent horizon to Qlib's return target. |

The factor projection and scalar head are specific to this financial experiment.
The model does not reproduce Google's pretrained task distribution, nine-quantile
head, detrending/stitching logic, or known future covariate interface. Its latent
forecast patches are internal representations trained through the final return
loss; they are not separately supervised price forecasts. The implementation
tests causal attention, reversible prefix normalization, padding isolation,
statistic refinement, singleton batches, and gradients through both attention
axes.

## `timesfm3_pretrained`

Install the optional official implementation using this project's extra:

```bash
python -m pip install -e '.[timesfm]'
```

This pins `timesfm[torch]==3.0.2`. The package exposes the `timesfm3` Python module.
The adapter requires an explicit local directory containing `config.json` and
`model.safetensors`; it never implicitly downloads weights. The public research
checkpoint is
[`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch/tree/43046b85ec22d584a13f8098c2ed39c889e129c2),
revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, approximately 1.32 GB of FP32
weights. Keep that directory outside version control (for example, the ignored
`.data/timesfm3-checkpoint` directory) and retain its license.

The verified checkpoint can be downloaded explicitly after installing the extra:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/timesfm-3.0-pytorch",
    revision="43046b85ec22d584a13f8098c2ed39c889e129c2",
    allow_patterns=["config.json", "model.safetensors", "LICENSE", "README.md"],
    local_dir=".data/timesfm3-checkpoint",
    token=False,
)
PY
```

Actual checkpoint inference was verified on Python 3.12 with CPU PyTorch. The
Python 3.10 CI job covers the small models and mocked adapter contract, without
downloading the official weights.

```python
from financial_ts.timesfm_pretrained import OfficialTimesFM3Model

model = OfficialTimesFM3Model(
    checkpoint_path="/absolute/path/to/timesfm-3.0-pytorch",
    context_length=256,
    batch_size=8,
    device="cpu",
)
model.fit(dataset)  # Loads fixed weights; does not read training labels.
scores = model.predict(dataset, segment="test")
```

For each `(date, instrument)` prediction row, the adapter obtains `$close`
prices directly from the Qlib provider, ending strictly at that row's date.
These are prices in the provider's units, including its upstream adjustments;
they are not unadjusted tape prices or normalized factor features. The repository's
Yahoo CSV importer adjusts OHLC using `Adj Close / Close`, so both this model's
input and its evaluation label use the same adjusted series. The adapter
forecasts the next two closes and calculates `forecast[t+2] / forecast[t+1] - 1`
to match the repository's return label. This wrapper uses the official model in
**univariate mode**: instruments are independent batch elements, not jointly
forecast variates. It preserves the dataset's exact index order and rejects
internal missing sessions, invalid prices, or nonfinite forecasts. Qlib's saved
adapter stores its configuration and checkpoint path, not a second copy of the
official weights. Restoring it requires the same checkpoint to remain available.

The official repository's [code license is
Apache-2.0](https://github.com/google-research/timesfm/blob/918df02c9d366b2145bc5876135fa2131d3ed2a6/LICENSE).
The checkpoint's separate [TimesFM Non-Commercial License
v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/43046b85ec22d584a13f8098c2ed39c889e129c2/LICENSE)
restricts use to noncommercial, nonproduction purposes and prohibits distributing
the model or derivatives. A repository code license does not grant commercial
rights to these weights. This repository does not redistribute them.

The official model's pretraining overlap with this repository's historical
market data is unknown. A chronological split here cannot establish that the
pretrained model has never seen those dates. Report pretrained results as an
inference/backtest experiment with unknown pretraining overlap; use fresh data
with established provenance for claims about unseen-data forecasting quality.
The mocked forecaster tests validate adapter behavior, not official checkpoint
quality. A real checkpoint inference run must be identified separately.
