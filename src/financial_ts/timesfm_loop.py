"""Parameter-efficient, shared-block looping of the official TimesFM3 checkpoint.

This experimental architecture is not an official Google model release. The
original backbone stays frozen; only independently initialized loop gates and
bottleneck adapters are trained. See docs/architectures/timesfm-loop.md.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(name: str, value: int, *, allow_zero: bool = False):
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _load_official_model(checkpoint: Path, device: torch.device):
    """Import the optional dependency only when a real checkpoint is requested."""
    try:
        installed = version("timesfm")
    except PackageNotFoundError as error:
        raise ImportError("Install the optional dependency timesfm[torch]==3.0.2.") from error
    if installed != "3.0.2":
        raise ImportError(
            f"Loop fine-tuning requires the verified timesfm==3.0.2 API; found {installed}."
        )
    try:
        from timesfm3 import ModelConfig, TimesFM3Forecaster
    except ImportError as error:
        raise ImportError("Install the optional dependency timesfm[torch]==3.0.2.") from error
    forecaster = TimesFM3Forecaster(
        ModelConfig(checkpoint_path=str(checkpoint), device=str(device), local_files_only=True)
    )
    return forecaster.model, forecaster.config.median_quantile_index


class _LoopRefinement(nn.Module):
    def __init__(self, width: int, rank: int, gate_scale: float):
        super().__init__()
        self.gate_scale = gate_scale
        self.norm = nn.RMSNorm(width, elementwise_affine=False)
        self.gate = nn.Linear(width, 1)
        self.adapter_down = nn.Linear(width, rank, bias=False)
        self.adapter_up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, hidden: torch.Tensor, replayed: torch.Tensor):
        normalized = self.norm(hidden)
        gate = self.gate_scale * torch.tanh(self.gate(normalized))
        adapter = self.adapter_up(F.gelu(self.adapter_down(normalized)))
        return hidden + gate * (replayed - hidden) + adapter


class _ProgressiveLoopStack(nn.Module):
    """Replay the same middle layer objects, preserving mask/position semantics."""

    def __init__(
        self,
        base: nn.Module,
        width: int,
        loop_start: int,
        loop_end: int,
        loop_steps: int,
        adapter_rank: int,
        gate_scale: float,
    ):
        super().__init__()
        if not isinstance(getattr(base, "layers", None), nn.ModuleList):
            raise ValueError("The official stack must expose an nn.ModuleList of layers.")
        for name, value in (
            ("width", width),
            ("loop_steps", loop_steps),
            ("adapter_rank", adapter_rank),
        ):
            _positive_integer(name, value)
        _positive_integer("loop_start", loop_start, allow_zero=True)
        _positive_integer("loop_end", loop_end)
        if not 0 <= loop_start < loop_end <= len(base.layers):
            raise ValueError("Loop bounds must satisfy 0 <= start < end <= layer count.")
        if adapter_rank > width:
            raise ValueError("adapter_rank must not exceed the hidden width.")
        if isinstance(gate_scale, bool) or not math.isfinite(gate_scale) or not 0 < gate_scale <= 1:
            raise ValueError("gate_scale must be finite and in (0,1].")
        self.base = base
        self.base.requires_grad_(False)
        self.width = width
        self.loop_start, self.loop_end = loop_start, loop_end
        self.refinements = nn.ModuleList(
            [_LoopRefinement(width, adapter_rank, gate_scale) for _ in range(loop_steps - 1)]
        )
        self.refinement_enabled = True

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        segment_pos: torch.Tensor | None = None,
        decode_cache=None,
        var_segment_pos: torch.Tensor | None = None,
    ):
        if input_embeddings.ndim != 4 or input_embeddings.shape[-1] != self.width:
            raise ValueError("Loop stack expects [batch,variate,patch,hidden] embeddings.")
        if patch_mask.shape != input_embeddings.shape[:-1] or patch_mask.dtype != torch.bool:
            raise ValueError("patch_mask must be boolean [batch,variate,patch].")
        if decode_cache is not None:
            if (
                not isinstance(decode_cache, (list, tuple))
                or len(decode_cache) != len(self.base.layers)
                or any(cache is not None for cache in decode_cache)
            ):
                raise ValueError(
                    "Looping supports full-sequence decoding, not populated KV caches."
                )
        if not self.refinement_enabled or not self.refinements:
            return self.base(
                input_embeddings,
                patch_mask,
                segment_ids,
                segment_pos,
                decode_cache,
                var_segment_pos,
            )
        hidden = input_embeddings
        caches, masks = [], []
        for layer_index, layer in enumerate(self.base.layers):
            hidden, cache, attention_mask = layer(
                hidden, patch_mask, segment_ids, segment_pos, None, var_segment_pos
            )
            caches.append(cache)
            masks.append(attention_mask)
            if layer_index + 1 == self.loop_end:
                for refinement in self.refinements:
                    replayed = hidden
                    for shared_layer in self.base.layers[self.loop_start : self.loop_end]:
                        replayed, _, _ = shared_layer(
                            replayed, patch_mask, segment_ids, segment_pos, None, var_segment_pos
                        )
                    hidden = refinement(hidden, replayed)
        # Preserve the official return contract for the original layer sequence.
        return hidden, caches, masks


class LoopTimesFM3(nn.Module):
    """Frozen official TimesFM3 with trainable gates/adapters on shared loops.

    ``loop_steps=2`` executes the chosen middle block once normally and once
    again with shared weights. Gates modulate its residual contribution; they
    do not implement adaptive halting or skip computation. Contexts contain
    positive provider close prices, including their upstream adjustments.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        num_threads: int = 2,
        loop_start: int = 8,
        loop_end: int = 12,
        loop_steps: int = 2,
        adapter_rank: int = 8,
        gate_scale: float = 0.1,
    ):
        super().__init__()
        _positive_integer("num_threads", num_threads)
        _positive_integer("loop_start", loop_start, allow_zero=True)
        for name, value in (
            ("loop_end", loop_end),
            ("loop_steps", loop_steps),
            ("adapter_rank", adapter_rank),
        ):
            _positive_integer(name, value)
        if loop_start >= loop_end:
            raise ValueError("loop_start must be strictly smaller than loop_end.")
        if isinstance(gate_scale, bool) or not math.isfinite(gate_scale) or not 0 < gate_scale <= 1:
            raise ValueError("gate_scale must be finite and in (0,1].")
        try:
            requested_device = torch.device(device)
        except (TypeError, RuntimeError, ValueError) as error:
            raise ValueError("device must be cpu, cuda, or an available cuda:N.") from error
        if requested_device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu, cuda, or an available cuda:N.")
        if requested_device.type == "cuda" and (
            not torch.cuda.is_available()
            or (
                requested_device.index is not None
                and requested_device.index >= torch.cuda.device_count()
            )
        ):
            raise ValueError("The requested CUDA device is unavailable.")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_dir() or not all(
            (checkpoint / name).is_file() for name in ("config.json", "model.safetensors")
        ):
            raise ValueError(
                "checkpoint_path must contain local config.json and model.safetensors files; "
                "the loop model does not download weights."
            )
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_config_sha256 = hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest()
        self.checkpoint_size_bytes = (checkpoint / "model.safetensors").stat().st_size
        self.loop_start, self.loop_end, self.loop_steps = loop_start, loop_end, loop_steps
        self.adapter_rank, self.gate_scale = adapter_rank, gate_scale
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)
        self.model, self.median_quantile_index = _load_official_model(checkpoint, requested_device)
        self.model.requires_grad_(False)
        self.model.eval()
        width = self.model.transformer_config.transformer.model_dims
        original_stack = self.model.transformer_stack
        stack = _ProgressiveLoopStack(
            original_stack, width, loop_start, loop_end, loop_steps, adapter_rank, gate_scale
        )
        reference_parameter = next(self.model.parameters())
        stack.refinements.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        self.model.transformer_stack = stack
        decode = type(self.model).decode
        if not hasattr(decode, "__wrapped__"):
            raise RuntimeError("The verified no-grad decode wrapper is missing from TimesFM3.")
        self._decode_with_grad = inspect.unwrap(decode)
        self.parameter_count = sum(parameter.numel() for parameter in self.parameters())
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        self.eval()

    @property
    def loop_stack(self) -> _ProgressiveLoopStack:
        return self.model.transformer_stack

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the pretrained inference backbone deterministic in training too.
        self.model.eval()
        self.loop_stack.refinements.train(mode)
        return self

    def base_parameters(self):
        adapter_ids = {id(parameter) for parameter in self.loop_stack.refinements.parameters()}
        return (
            parameter for parameter in self.model.parameters() if id(parameter) not in adapter_ids
        )

    def forecast(self, contexts: torch.Tensor) -> torch.Tensor:
        """Differentiable sorted-median forecasts of the next two provider closes."""
        if (
            not isinstance(contexts, torch.Tensor)
            or contexts.ndim != 2
            or contexts.shape[0] < 1
            or not 2 <= contexts.shape[1] <= 15360
            or not contexts.is_floating_point()
        ):
            raise ValueError("contexts must be a floating [positive_batch,2..15360] tensor.")
        if not torch.isfinite(contexts).all() or (contexts <= 0).any():
            raise ValueError("Each context must contain finite positive observed close prices.")
        reference = next(self.model.parameters())
        target = contexts.to(device=reference.device, dtype=reference.dtype).unsqueeze(1)
        # Explicitly remove the inference-only no_grad decorator of the pinned
        # API; retain the original decode body, including all preprocessing.
        horizon = 2
        iterative_cpm = getattr(self.model, "use_iterative_cpm_revin", False)
        unused_cpm = (
            iterative_cpm
            and horizon <= getattr(self.model, "input_patch_len", 0)
            and horizon <= getattr(self.model, "output_patch_len", 0)
        )
        # In the pinned decode, this horizon selects only the final observed
        # context patch, with or without output stitching. Iterative CPM RevIN
        # modifies only later, masked forecast patches. Its outputs therefore
        # cannot contribute to these two predictions, but its unused sqrt(0)
        # gradients can still propagate NaNs through torch.where. Skip only
        # that dead branch; keep every used operation and its dtype unchanged.
        if unused_cpm:
            self.model.use_iterative_cpm_revin = False
        try:
            logits = self._decode_with_grad(self.model, target=target, horizon=horizon)
        finally:
            if unused_cpm:
                self.model.use_iterative_cpm_revin = iterative_cpm
        if logits.ndim != 4 or logits.shape[:3] != (len(contexts), 1, 2):
            raise ValueError("Official decode must return [batch,1,2,quantiles].")
        if not 0 <= self.median_quantile_index < logits.shape[-1]:
            raise ValueError("The official median quantile index is invalid.")
        # Match the public forecaster's default sort_quantiles=True behavior.
        result = logits.sort(dim=-1).values[:, 0, :, self.median_quantile_index]
        if not torch.isfinite(result).all() or (result <= 0).any():
            raise ValueError("Loop forecasts must contain two finite positive close prices.")
        return result

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return self.forecast(contexts)

    def return_scores(self, contexts: torch.Tensor) -> torch.Tensor:
        prices = self.forecast(contexts).to(torch.float64)
        scores = prices[:, 1] / prices[:, 0] - 1
        if not torch.isfinite(scores).all():
            raise ValueError("Forecast close ratio overflowed.")
        return scores

    @torch.no_grad()
    def baseline_forecast(self, contexts: torch.Tensor) -> torch.Tensor:
        """Evaluate the unchanged original stack without modifying learned state."""
        enabled = self.loop_stack.refinement_enabled
        self.loop_stack.refinement_enabled = False
        try:
            return self.forecast(contexts)
        finally:
            self.loop_stack.refinement_enabled = enabled

    @torch.no_grad()
    def baseline_return_scores(self, contexts: torch.Tensor) -> torch.Tensor:
        prices = self.baseline_forecast(contexts).to(torch.float64)
        return prices[:, 1] / prices[:, 0] - 1

    def _adapter_parameters(self):
        prefix = "model.transformer_stack.refinements."
        return {
            prefix + name: parameter
            for name, parameter in self.loop_stack.refinements.named_parameters()
        }

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return cloned CPU adapter tensors only; never the frozen base weights."""
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self._adapter_parameters().items()
        }

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]):
        """Strictly load adapter tensors after validating the whole mapping."""
        parameters = self._adapter_parameters()
        if not isinstance(state, Mapping) or set(state) != set(parameters):
            raise ValueError("Adapter state keys must exactly match this loop configuration.")
        converted = {}
        for name, parameter in parameters.items():
            value = state[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid adapter tensor: {name}.")
            converted[name] = value.to(device=parameter.device, dtype=parameter.dtype)
            if not torch.isfinite(converted[name]).all():
                raise ValueError(f"Adapter tensor overflows the model dtype: {name}.")
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(converted[name])
