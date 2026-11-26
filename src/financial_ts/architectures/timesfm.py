"""Small, independently initialized TimesFM-3-style numerical encoder.

Mechanisms: per-variate patches, causal temporal / full variate attention,
masked forecast patches, and estimate-driven RevIN statistic refinement.
This is a supervised adaptation, not Google's pretrained TimesFM 3 checkpoint.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _patchify(values: torch.Tensor, missing: torch.Tensor, patch_len: int):
    """Left-pad so every context ends at the right edge of its last patch."""
    left = (-values.shape[-1]) % patch_len
    if left:
        values = F.pad(values, (left, 0))
        missing = F.pad(missing, (left, 0), value=True)
    return (
        values.reshape(*values.shape[:-1], -1, patch_len),
        missing.reshape(*missing.shape[:-1], -1, patch_len),
    )


def _running_revin(values: torch.Tensor, missing: torch.Tensor, eps: float = 1e-5):
    """Prefix-only moments; later patches never change an earlier patch's scale."""
    clean = values.masked_fill(missing, 0)
    count = (~missing).sum(dim=-1).to(values.dtype).cumsum(dim=-1)
    total = clean.sum(dim=-1).cumsum(dim=-1)
    square = clean.square().sum(dim=-1).cumsum(dim=-1)
    mean = total / count.clamp_min(1)
    variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
    scale = variance.clamp_min(eps).sqrt()
    scale = torch.where(count > 0, scale, torch.ones_like(scale))
    normalized = ((clean - mean.unsqueeze(-1)) / scale.unsqueeze(-1)).masked_fill(missing, 0)
    return normalized, count, mean, variance, scale


def _refine_forecast(
    raw: torch.Tensor,
    count: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    eps: float = 1e-5,
):
    """Update moments using only preceding estimated forecast patches.

    Attention is evaluated once. This recurrence adjusts forecast scale and
    location; it never reads ground-truth future values or reruns the backbone.
    """
    estimates = []
    for patch in raw.unbind(dim=2):
        scale = variance.clamp_min(eps).sqrt()
        scale = torch.where(count > 0, scale, torch.ones_like(scale))
        estimated = patch * scale.unsqueeze(-1) + mean.unsqueeze(-1)
        estimates.append(estimated)
        patch_count = estimated.shape[-1]
        patch_mean = estimated.mean(dim=-1)
        patch_variance = estimated.var(dim=-1, unbiased=False)
        delta = patch_mean - mean
        new_count = count + patch_count
        new_mean = mean + delta * patch_count / new_count
        new_variance = (
            count * variance
            + patch_count * patch_variance
            + delta.square() * count * patch_count / new_count
        ) / new_count
        count, mean, variance = new_count, new_mean, new_variance.clamp_min(0)
    return torch.stack(estimates, dim=2)


class _AxisAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, rotary: bool):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.rotary = rotary
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def _rope(self, value):
        length = value.shape[-2]
        positions = torch.arange(length, device=value.device, dtype=torch.float32)
        rates = torch.exp(
            torch.arange(0, self.head_dim, 2, device=value.device, dtype=torch.float32)
            * (-math.log(10000) / self.head_dim)
        )
        angles = positions[:, None] * rates[None, :]
        even, odd = value[..., 0::2], value[..., 1::2]
        cos, sin = angles.cos().to(value.dtype), angles.sin().to(value.dtype)
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, values: torch.Tensor, key_padding: torch.Tensor, causal: bool):
        batch, length, width = values.shape
        q, k, v = (
            self.qkv(values)
            .reshape(batch, length, 3, self.nhead, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rotary:
            q, k = self._rope(q), self._rope(k)
        allowed = (~key_padding)[:, None, None, :].expand(-1, 1, length, -1)
        if causal:
            allowed = (
                allowed
                & torch.ones(length, length, device=values.device, dtype=torch.bool).tril()[
                    None, None
                ]
            )
        # SDPA produces zero for a fully masked row, including leading padding.
        mixed = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(mixed.transpose(1, 2).reshape(batch, length, width))


class _MixingLayer(nn.Module):
    """Apply temporal attention, variate attention, then a local feed-forward block."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.temporal_norm = nn.RMSNorm(d_model)
        self.variate_norm = nn.RMSNorm(d_model)
        self.ff_norm = nn.RMSNorm(d_model)
        self.temporal_attention = _AxisAttention(d_model, nhead, dropout, rotary=True)
        self.variate_attention = _AxisAttention(d_model, nhead, dropout, rotary=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, key_padding: torch.Tensor):
        batch, variates, patches, width = tokens.shape
        temporal = self.temporal_norm(tokens).reshape(batch * variates, patches, width)
        update = self.temporal_attention(
            temporal, key_padding.reshape(batch * variates, patches), causal=True
        ).reshape(batch, variates, patches, width)
        tokens = tokens + self.dropout(update)
        cross = (
            self.variate_norm(tokens).permute(0, 2, 1, 3).reshape(batch * patches, variates, width)
        )
        cross_mask = key_padding.permute(0, 2, 1).reshape(batch * patches, variates)
        update = (
            self.variate_attention(cross, cross_mask, causal=False)
            .reshape(batch, patches, variates, width)
            .permute(0, 2, 1, 3)
        )
        tokens = tokens + self.dropout(update)
        tokens = tokens + self.dropout(self.ff(self.ff_norm(tokens)))
        return tokens.masked_fill(key_padding.unsqueeze(-1), 0)


class TimesFMMiniEncoder(nn.Module):
    """Small TimesFM-3-inspired return encoder with a contiguous masked horizon.

    Financial factors are projected into a small number of learned variates,
    then patched along time. A scalar regression head replaces the official
    multivariate quantile forecasting head. All parameters start from scratch.
    """

    def __init__(
        self,
        d_feat: int = 157,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        patch_len: int = 4,
        num_variates: int = 8,
        horizon_patches: int = 2,
    ):
        super().__init__()
        for name, value in (
            ("d_feat", d_feat),
            ("d_model", d_model),
            ("nhead", nhead),
            ("num_layers", num_layers),
            ("patch_len", patch_len),
            ("num_variates", num_variates),
            ("horizon_patches", horizon_patches),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if d_model % nhead or (d_model // nhead) % 2:
            raise ValueError("d_model must give an even per-head dimension for rotary attention.")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0,1).")
        self.d_feat = d_feat
        self.patch_len = patch_len
        self.num_variates = num_variates
        self.horizon_patches = horizon_patches
        self.variate_projection = nn.Linear(d_feat, num_variates, bias=False)
        self.patch_projection = nn.Linear(2 * patch_len, d_model, bias=False)
        self.patch_residual = nn.Sequential(
            nn.Linear(2 * patch_len, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.variate_embedding = nn.Parameter(torch.randn(1, num_variates, 1, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_MixingLayer(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.patch_head = nn.Linear(d_model, patch_len)
        self.readout = nn.Sequential(
            nn.Linear(num_variates * horizon_patches * patch_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, features: torch.Tensor, padding: torch.Tensor):
        if features.ndim != 3 or features.shape[-1] != self.d_feat or features.shape[1] == 0:
            raise ValueError(f"Expected a nonempty [batch,time,{self.d_feat}] feature tensor.")
        if padding.shape != features.shape[:2] or padding.dtype != torch.bool:
            raise ValueError("padding must be bool [batch,time].")
        clean = torch.where(padding.unsqueeze(-1) | ~torch.isfinite(features), 0.0, features)
        values = self.variate_projection(clean).transpose(1, 2)
        missing = padding[:, None, :].expand(-1, self.num_variates, -1)
        values, missing = _patchify(values, missing, self.patch_len)
        normalized, count, mean, variance, _ = _running_revin(values, missing)
        batch, variates, context_patches, _ = values.shape
        forecast_shape = (batch, variates, self.horizon_patches, self.patch_len)
        # These are placeholders, not future covariate values or teacher forcing.
        normalized = torch.cat((normalized, values.new_zeros(forecast_shape)), dim=2)
        mask = torch.cat(
            (missing, torch.ones(forecast_shape, device=features.device, dtype=torch.bool)), dim=2
        )
        inputs = torch.cat((normalized, mask.to(features.dtype)), dim=-1)
        tokens = (
            self.patch_projection(inputs) + self.patch_residual(inputs) + self.variate_embedding
        )
        key_padding = mask.all(dim=-1)
        # Masked horizon tokens remain visible as learned queries; absent context does not.
        key_padding[:, :, context_patches:] = False
        for layer in self.layers:
            tokens = layer(tokens, key_padding)
        raw = self.patch_head(self.final_norm(tokens[:, :, context_patches:]))
        estimates = _refine_forecast(raw, count[:, :, -1], mean[:, :, -1], variance[:, :, -1])
        return self.readout(estimates.flatten(1)).squeeze(-1)
