"""Small causal forecasters with restrained capacity for weak return signals.

These are local research designs, inspired by multiscale mixing and mixtures
of simple experts. They are not reproductions of a published flagship model.
The shared Qlib adapter maps an initial normalized prediction of zero back to
the training-label mean, not to a raw zero-return prediction.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

SCALES = (1, 4, 16, 32)


def _validate_dimensions(d_feat, d_model, nhead, num_layers, dropout):
    for name, value in (
        ("d_feat", d_feat),
        ("d_model", d_model),
        ("nhead", nhead),
        ("num_layers", num_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if d_model % nhead:
        raise ValueError("d_model must be divisible by nhead for the shared model interface.")
    if isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0,1).")


def _clean_inputs(features, padding, d_feat):
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 3
        or min(features.shape[:2]) < 1
        or features.shape[-1] != d_feat
        or not features.is_floating_point()
    ):
        raise ValueError(f"Expected floating [positive_batch,positive_time,{d_feat}] features.")
    if (
        not isinstance(padding, torch.Tensor)
        or padding.shape != features.shape[:2]
        or padding.dtype != torch.bool
        or padding.device != features.device
    ):
        raise ValueError("padding must be boolean [batch,time] on the features' device.")
    clean = features.masked_fill(padding.unsqueeze(-1), 0)
    if not torch.isfinite(clean).all():
        raise ValueError("Observed feature values must be finite.")
    return clean


def _trailing_moments(clean, padding):
    """Mean, population variance, and count in strictly trailing session windows.

    Missing rows contribute neither a value nor an observation count. Windows
    retain their session positions: a gap is not silently compressed out of time.
    """
    batch, length, width = clean.shape
    zero = clean.new_zeros(batch, 1, width)
    totals = torch.cat((zero, clean.cumsum(dim=1)), dim=1)
    squares = torch.cat((zero, clean.square().cumsum(dim=1)), dim=1)
    observed = (~padding).to(clean.dtype).unsqueeze(-1)
    counts = torch.cat((clean.new_zeros(batch, 1, 1), observed.cumsum(dim=1)), dim=1)
    end = torch.arange(1, length + 1, device=clean.device)
    means, variances, window_counts = [], [], []
    for window in SCALES:
        start = (end - window).clamp_min(0)
        count = counts[:, end] - counts[:, start]
        mean = (totals[:, end] - totals[:, start]) / count.clamp_min(1)
        second = (squares[:, end] - squares[:, start]) / count.clamp_min(1)
        means.append(mean)
        variances.append((second - mean.square()).clamp_min(0))
        window_counts.append(count)
    return (
        torch.stack(means, dim=2),
        torch.stack(variances, dim=2),
        torch.stack(window_counts, dim=2),
    )


class _CausalMixerBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.dilation = dilation
        self.temporal_norm = nn.RMSNorm(width)
        self.temporal = nn.Conv1d(
            width, width, kernel_size=3, dilation=dilation, groups=width, bias=False
        )
        self.feature_norm = nn.RMSNorm(width)
        self.feature = nn.Sequential(
            nn.Linear(width, 2 * width, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width, bias=False),
        )
        self.temporal_gain = nn.Parameter(torch.tensor(-2.0))
        self.feature_gain = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, padding):
        mask = padding.unsqueeze(-1)
        hidden = hidden.masked_fill(mask, 0)
        normalized = self.temporal_norm(hidden).masked_fill(mask, 0).transpose(1, 2)
        update = self.temporal(F.pad(normalized, (2 * self.dilation, 0))).transpose(1, 2)
        hidden = (hidden + self.temporal_gain.sigmoid() * self.dropout(update)).masked_fill(mask, 0)
        update = self.feature(self.feature_norm(hidden))
        return (hidden + self.feature_gain.sigmoid() * self.dropout(update)).masked_fill(mask, 0)


class MultiScaleResidualMixerEncoder(nn.Module):
    """Linear historical summary plus a modest causal multiscale residual.

    Four fixed trailing scales share a feature projection. A learned softmax
    blends their representations before causal temporal/feature mixer blocks.
    ``num_layers`` counts mixer blocks; ``nhead`` is accepted for compatibility
    with the shared training configuration but introduces no attention heads.
    """

    def __init__(
        self,
        d_feat: int = 157,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        _validate_dimensions(d_feat, d_model, nhead, num_layers, dropout)
        self.d_feat = d_feat
        self.scales = SCALES
        self.projection = nn.Linear(d_feat, d_model, bias=False)
        self.scale_embeddings = nn.Parameter(torch.randn(len(SCALES), d_model) * 0.02)
        self.scale_gate = nn.Linear(d_model, 1, bias=False)
        self.blocks = nn.ModuleList(
            [_CausalMixerBlock(d_model, 2**layer, dropout) for layer in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.linear_head = nn.Linear(len(SCALES) * d_feat, 1)
        self.residual_head = nn.Linear(d_model, 1)
        self.residual_gain = nn.Parameter(torch.tensor(-2.0))
        for head in (self.linear_head, self.residual_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _representations(self, features, padding):
        clean = _clean_inputs(features, padding, self.d_feat)
        means, _, counts = _trailing_moments(clean, padding)
        projected = self.projection(means).masked_fill(counts == 0, 0)
        logits = self.scale_gate(torch.tanh(projected + self.scale_embeddings)).squeeze(-1)
        logits = logits.masked_fill(counts.squeeze(-1) == 0, torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=-1)
        # An entirely missing prefix has zero projected values and remains finite.
        hidden = (weights.unsqueeze(-1) * projected).sum(dim=2)
        for block in self.blocks:
            hidden = block(hidden, padding)
        hidden = self.final_norm(hidden).masked_fill(padding.unsqueeze(-1), 0)
        return hidden, means.flatten(start_dim=2)

    def forward_features(self, features: torch.Tensor, padding: torch.Tensor):
        """Return causal [batch,time,width] hidden states for inspection."""
        return self._representations(features, padding)[0]

    def forward(self, features: torch.Tensor, padding: torch.Tensor):
        hidden, summaries = self._representations(features, padding)
        score = self.linear_head(summaries[:, -1])
        score = score + self.residual_gain.sigmoid() * self.residual_head(hidden[:, -1])
        return score.squeeze(-1)


class RegimeLinearMoEEncoder(nn.Module):
    """Three linear experts with a small historical feature-regime gate.

    The gate sees feature amplitude, changes across trailing means, dispersion,
    and history coverage. These describe normalized financial factors, not a
    separately estimated physical return-volatility process. ``num_layers``
    counts hidden gating layers; the expert count is fixed at three.
    """

    def __init__(
        self,
        d_feat: int = 157,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        _validate_dimensions(d_feat, d_model, nhead, num_layers, dropout)
        self.d_feat = d_feat
        self.scales = SCALES
        self.num_experts = 3
        layers = []
        for layer in range(num_layers):
            layers.extend(
                (nn.Linear(6 if layer == 0 else d_model, d_model), nn.GELU(), nn.Dropout(dropout))
            )
        layers.append(nn.Linear(d_model, self.num_experts))
        self.gate = nn.Sequential(*layers)
        self.experts = nn.Linear(len(SCALES) * d_feat, self.num_experts)
        nn.init.zeros_(self.experts.weight)
        nn.init.zeros_(self.experts.bias)
        # The gate retains independent random initialization. Uniform gates and
        # identical zero heads together would otherwise preserve expert symmetry.

    def _expert_state(self, features, padding):
        clean = _clean_inputs(features, padding, self.d_feat)
        means, variances, counts = _trailing_moments(clean, padding)
        descriptors = torch.stack(
            (
                means[:, :, 0].abs().mean(dim=-1),
                (means[:, :, 0] - means[:, :, 1]).abs().mean(dim=-1),
                (means[:, :, 1] - means[:, :, 2]).abs().mean(dim=-1),
                (means[:, :, 2] - means[:, :, 3]).abs().mean(dim=-1),
                (variances[:, :, 2] + 1e-6).sqrt().mean(dim=-1) * (counts[:, :, 2, 0] > 0),
            ),
            dim=-1,
        )
        coverage = counts[:, :, -1] / SCALES[-1]
        gate_input = torch.cat((descriptors.log1p(), coverage), dim=-1)
        weights = self.gate(gate_input).softmax(dim=-1)
        predictions = self.experts(means.flatten(start_dim=2))
        return weights, predictions

    def routing_weights(self, features: torch.Tensor, padding: torch.Tensor):
        """Return causal, normalized [batch,time,3] expert weights."""
        return self._expert_state(features, padding)[0]

    def forward_features(self, features: torch.Tensor, padding: torch.Tensor):
        """Return each expert's weighted contribution at every historical token."""
        weights, predictions = self._expert_state(features, padding)
        return weights * predictions

    def forward(self, features: torch.Tensor, padding: torch.Tensor):
        return self.forward_features(features, padding)[:, -1].sum(dim=-1)
