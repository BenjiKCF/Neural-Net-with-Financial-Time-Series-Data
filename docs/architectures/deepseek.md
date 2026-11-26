# DeepSeek V4.1 Flash 1017: compact numeric adaptation

`DeepSeekMiniEncoder` is a small, randomly initialized financial forecasting network based on disclosed mechanisms from **DeepSeek-V4.1-Flash-1017**. It does not load DeepSeek weights or inherit the language model's training, knowledge, or benchmark performance.

## Verified source version

The [official 1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) describes an update to V4.1 Flash. Its [official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/config.json) specifies 40 backbone layers, 384 routed experts, 6 selected experts per token, 1 shared expert, four residual streams, compressed-attention schedules, and shared KV/index source layers. These mechanisms are documented in the [official V4.1 Flash architecture overview](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md) and [1017 inference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/inference/model.py).

V4.1's distinguishing attention design is **CSA2 with a causal encoder/decoder arrangement**. Substituting an older V3 MLA or V4 CSA/HCA implementation would not represent this requested version.

## Implemented mechanisms and explicit adaptations

| Published mechanism | Compact numeric implementation |
| --- | --- |
| Causal encoder followed by decoder | Two small stages, with `num_layers` layers **in each stage**. Every historical state is causal. |
| Decoder global KV derives from final encoder states | Encoder-derived KV is projected once and shared by every decoder layer, keeping one KV entry per encoder state (ratio 1). Each decoder keeps its own local-window KV projection. |
| CSA2 Full/Reindex/Reuse roles | The first decoder layer builds retrieval indices, alternate layers reuse them, and intervening layers recompute indices over the same shared memory. |
| Compressed global history plus local attention | Masked nonoverlapping compression and learned top-k retrieval supplement a short causal window. A compressed block is unavailable until its final session has been observed. |
| Single-Pass mHC | Four residual streams, token-specific read/write gates, Sinkhorn-normalized residual transport, and read gates passed to the next sublayer. The tiny model uses normalized read gates and bounded write gates for stability. |
| Routed and shared SwiGLU MoE | Four routed experts, top two active per token, and one always-on shared expert. Routing uses the disclosed square-root-softplus scoring family. |

The sparse indexer is trained with an auxiliary distillation loss from main attention; a small routing-balance term helps the few-expert model. These are explicit small-model training choices, not a reproduction of DeepSeek's full training recipe or auxiliary-loss-free balancing. `aux_loss` exposes their sum to the shared training loop; a coefficient such as 0.01 is suitable as a starting configuration.

For short financial windows the attention implementation materializes scores before masking, which favors inspectability and CPU portability. It does **not** reproduce the original sparse GPU kernels or claim their asymptotic runtime/cache savings. Encoder compression uses masked averages rather than the production learned compressor. Rotary positions, reduced-rank query/output projections, and a shared KV head provide a compact implementation, with dimensions scaled independently from the source checkpoint. The short decoder's alternating index reuse omits the production hierarchical candidate-pool stage; every reindex operation considers the already causal eligible memory.

Token-hash Engram memory, vision encoding, DSpark speculative text decoding, million-token serving, quantized FP4/FP8 kernels, distributed expert parallelism, and the original training corpus are omitted. These components do not have direct equivalents in this scalar-return experiment and are not silently approximated by unrelated layers.

## Interface and scale

```python
DeepSeekMiniEncoder(
    d_feat=157, d_model=64, nhead=4, num_layers=2, dropout=0.1,
    num_experts=4, top_k=2, compression=2, sparse_topk=4,
    window_size=4, hc_streams=4,
)
```

`forward(features, padding)` accepts `[batch, time, features]` numeric inputs and a boolean missing-session mask, returning `[batch]` scores. `num_layers=2` means two encoder plus two decoder layers. The default has **352,129 trainable parameters**; width 32 and one layer per stage has **54,817** with the same feature count. `compression` controls the encoder only; decoder KV retains ratio 1.

All values at masked sessions are excluded, and a fully masked history remains finite. No model state or cache is carried across unrelated instruments or batches. The retained unit tests verify causal outputs, closed compression blocks, shared decoder memory, reused indices, actual expert gradients, stochastic residual transport, singleton inference, and optimization on a numeric training signal.

This is an original PyTorch implementation of the disclosed ideas; no upstream inference kernels or pretrained parameters are vendored. The linked official DeepSeek model distribution declares an MIT license. Reusing its actual code or weights in a future extension should preserve its applicable license notices.
