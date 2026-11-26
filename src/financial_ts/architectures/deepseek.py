"""Small numeric adaptation of disclosed DeepSeek-V4.1-Flash-1017 mechanisms.

This is a randomly initialized forecasting model, not a pretrained DeepSeek
checkpoint. See docs/architectures/deepseek.md for source and fidelity boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return x * inverse_rms.to(x.dtype) * self.weight


def _rotate(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """RoPE for [batch, head, time, channels] tensors."""
    rate = torch.exp(
        torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype)
        * (-math.log(10000.0) / x.shape[-1])
    )
    phase = positions.to(x.dtype)[:, None] * rate[None, :]
    sine, cosine = phase.sin()[None, None], phase.cos()[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


def _masked_softmax(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Zero mass, rather than NaNs or uniform attention, for an empty history."""
    logits = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = logits.softmax(-1) * allowed.to(scores.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


class _HyperMix(nn.Module):
    """Token-dependent read/write gates with doubly stochastic stream transport."""

    def __init__(self, width: int, streams: int, sinkhorn_iters: int = 20):
        super().__init__()
        self.streams, self.sinkhorn_iters = streams, sinkhorn_iters
        self.norm = _RMSNorm(width * streams)
        self.gates = nn.Linear(width * streams, streams * (streams + 2))
        nn.init.normal_(self.gates.weight, std=0.01)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[streams : 2 * streams].fill_(-2.0)
            self.gates.bias[2 * streams :].copy_((2 * torch.eye(streams)).flatten())

    def coefficients(self, states: torch.Tensor):
        raw = self.gates(self.norm(states.flatten(-2)))
        n = self.streams
        read = raw[..., :n].softmax(-1)
        write = raw[..., n : 2 * n].sigmoid()
        transport = raw[..., 2 * n :].reshape(*raw.shape[:-1], n, n).softmax(-1)
        for _ in range(self.sinkhorn_iters):
            transport = transport / transport.sum(-2, keepdim=True).clamp_min(1e-8)
            transport = transport / transport.sum(-1, keepdim=True).clamp_min(1e-8)
        return read, write, transport

    @staticmethod
    def read(states: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        return (states * gates.unsqueeze(-1)).sum(-2)

    @staticmethod
    def write(
        states: torch.Tensor, update: torch.Tensor, gates: torch.Tensor, transport: torch.Tensor
    ):
        residual = torch.einsum("btij,btjd->btid", transport, states)
        return residual + gates.unsqueeze(-1) * update.unsqueeze(-2)


class _SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _SparseMoE(nn.Module):
    """Execute top-k routed SwiGLU experts plus one always-on shared expert."""

    def __init__(self, width: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.router = nn.Linear(width, num_experts, bias=False)
        self.experts = nn.ModuleList([_SwiGLU(width, width) for _ in range(num_experts)])
        self.shared = _SwiGLU(width, width)
        self.last_counts: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, padding: torch.Tensor):
        flat = x.reshape(-1, x.shape[-1])
        active = (~padding).reshape(-1)
        # V4.1 uses sqrt(softplus), not a generic softmax router.
        scores = F.softplus(self.router(flat)).clamp_min(1e-12).sqrt()
        values, chosen = scores.topk(self.top_k, dim=-1)
        mixing = 1.5 * values / values.sum(-1, keepdim=True).clamp_min(1e-12)
        routed = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token, slot = torch.where((chosen == expert_id) & active[:, None])
            if len(token):
                contribution = expert(flat[token]) * mixing[token, slot, None]
                routed = routed.index_add(0, token, contribution)
        result = (routed + self.shared(flat)) * active[:, None]
        if active.any():
            probabilities = scores[active] / scores[active].sum(-1, keepdim=True)
            assignment = F.one_hot(chosen[active], self.num_experts).float().mean((0, 1))
            balance = self.num_experts * (probabilities.mean(0) * assignment.detach()).sum()
            self.last_counts = torch.bincount(
                chosen[active].flatten(), minlength=self.num_experts
            ).detach()
        else:
            balance = scores.sum() * 0
            self.last_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
        return result.reshape_as(x), balance


@dataclass
class _Memory:
    keys: torch.Tensor
    values: torch.Tensor
    index_keys: torch.Tensor
    valid: torch.Tensor
    ends: torch.Tensor


class _CompressedMemory(nn.Module):
    """Nonoverlapping masked compression; an unfinished block is never visible."""

    def __init__(self, width: int, head_dim: int, index_dim: int, compression: int):
        super().__init__()
        self.compression = compression
        self.kv = nn.Linear(width, head_dim, bias=False)
        self.index_key = nn.Linear(width, index_dim, bias=False)

    def forward(self, hidden: torch.Tensor, padding: torch.Tensor) -> _Memory:
        batch, length, width = hidden.shape
        ratio = self.compression
        groups = (length + ratio - 1) // ratio
        extension = groups * ratio - length
        available = (~padding).to(hidden.dtype)
        hidden = F.pad(hidden * available[..., None], (0, 0, 0, extension))
        available = F.pad(available, (0, extension))
        count = available.reshape(batch, groups, ratio).sum(-1)
        pooled = hidden.reshape(batch, groups, ratio, width).sum(-2) / count.clamp_min(1)[..., None]
        values = self.kv(pooled).unsqueeze(1)
        ends = (torch.arange(groups, device=hidden.device) + 1) * ratio - 1
        valid = (count > 0) & (ends[None, :] < length)
        return _Memory(_rotate(values, ends), values, self.index_key(pooled), valid, ends)


class _SparseAttention(nn.Module):
    """CSA2-inspired compressed global retrieval alongside a causal local window."""

    def __init__(
        self,
        width: int,
        nhead: int,
        dropout: float,
        compression: int,
        sparse_topk: int,
        window_size: int,
        *,
        owns_memory: bool,
        reindex: bool,
    ):
        super().__init__()
        self.nhead, self.head_dim = nhead, width // nhead
        self.sparse_topk, self.window_size = sparse_topk, window_size
        index_dim = max(4, self.head_dim // 2)
        rank = max(4, width // 2)
        self.query = nn.Sequential(
            nn.Linear(width, rank, bias=False), _RMSNorm(rank), nn.Linear(rank, width, bias=False)
        )
        # One joint KV head is shared across query heads.
        self.local_kv = nn.Linear(width, self.head_dim, bias=False)
        self.memory = (
            _CompressedMemory(width, self.head_dim, index_dim, compression) if owns_memory else None
        )
        self.index_query = nn.Linear(width, index_dim, bias=False) if reindex else None
        self.output = nn.Sequential(
            nn.Linear(width, rank, bias=False), nn.Linear(rank, width, bias=False)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding: torch.Tensor,
        memory: _Memory | None = None,
        selection: torch.Tensor | None = None,
    ):
        batch, length, width = x.shape
        positions = torch.arange(length, device=x.device)
        query = self.query(x).reshape(batch, length, self.nhead, self.head_dim).transpose(1, 2)
        query = _rotate(query, positions)
        local_values = self.local_kv(x).unsqueeze(1)
        local_keys = _rotate(local_values, positions)
        if memory is None:
            if self.memory is None:
                raise ValueError("Decoder attention requires encoder-derived memory.")
            memory = self.memory(x, padding)
        eligible = memory.valid[:, None, :] & (
            memory.ends[None, None, :] <= positions[None, :, None]
        )
        global_scores = torch.matmul(query, memory.keys.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        index_loss = x.sum() * 0
        if self.index_query is not None:
            index_logits = torch.matmul(self.index_query(x), memory.index_keys.transpose(-2, -1))
            index_logits = index_logits / math.sqrt(memory.index_keys.shape[-1])
            count = min(self.sparse_topk, memory.keys.shape[-2])
            indices = (
                index_logits.masked_fill(~eligible, torch.finfo(x.dtype).min)
                .topk(count, dim=-1)
                .indices
            )
            selection = torch.zeros_like(eligible).scatter(-1, indices, True) & eligible
            # Teach a hard indexer through the dense main-attention distribution.
            # This auxiliary training rule is an explicit small-model adaptation.
            teacher = _masked_softmax(global_scores.detach().mean(1), eligible)
            student = _masked_softmax(index_logits, eligible)
            rows = eligible.any(-1) & ~padding
            divergence = (
                teacher * (teacher.clamp_min(1e-9).log() - student.clamp_min(1e-9).log())
            ).sum(-1)
            if rows.any():
                index_loss = divergence[rows].mean()
        elif selection is None:
            raise ValueError("A Reuse attention layer requires prior selected indices.")
        global_allowed = selection & eligible
        local = (positions[:, None] >= positions[None, :]) & (
            positions[:, None] - positions[None, :] < self.window_size
        )
        local_allowed = local[None, :, :] & ~padding[:, None, :]
        local_scores = torch.matmul(query, local_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = torch.cat((local_allowed, global_allowed), -1)[:, None, :, :]
        weights = _masked_softmax(torch.cat((local_scores, global_scores), -1), allowed)
        values = torch.cat((local_values, memory.values), -2)
        attended = (
            torch.matmul(self.dropout(weights), values)
            .transpose(1, 2)
            .reshape(batch, length, width)
        )
        return self.output(attended).masked_fill(padding[..., None], 0), selection, index_loss


class _CEDBlock(nn.Module):
    def __init__(
        self,
        width,
        nhead,
        dropout,
        num_experts,
        top_k,
        compression,
        sparse_topk,
        window_size,
        hc_streams,
        *,
        encoder,
        reindex,
    ):
        super().__init__()
        self.attn_mix, self.ffn_mix = _HyperMix(width, hc_streams), _HyperMix(width, hc_streams)
        self.attn_norm, self.ffn_norm = _RMSNorm(width), _RMSNorm(width)
        self.attention = _SparseAttention(
            width,
            nhead,
            dropout,
            compression,
            sparse_topk,
            window_size,
            owns_memory=encoder,
            reindex=reindex,
        )
        self.moe = _SparseMoE(width, num_experts, top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, streams, prior_read, padding, memory=None, selection=None):
        next_read, write, transport = self.attn_mix.coefficients(streams)
        value = self.attn_norm(self.attn_mix.read(streams, prior_read))
        update, selection, index_loss = self.attention(value, padding, memory, selection)
        streams = self.attn_mix.write(streams, self.dropout(update), write, transport)
        final_read, write, transport = self.ffn_mix.coefficients(streams)
        value = self.ffn_norm(self.ffn_mix.read(streams, next_read))
        update, balance = self.moe(value, padding)
        streams = self.ffn_mix.write(streams, self.dropout(update), write, transport)
        streams = streams.masked_fill(padding[..., None, None], 0)
        return streams, final_read, selection, balance + index_loss


class DeepSeekMiniEncoder(nn.Module):
    """CED + compressed sparse attention + mHC + sparse/shared MoE for numeric bars.

    ``num_layers`` is the number of layers in each of the encoder and decoder.
    The decoder shares one compressed encoder-derived KV memory. Its first layer
    indexes it, alternate layers reuse those selections, and intervening layers
    reindex. ``aux_loss`` is a differentiable routing/indexer training objective;
    the surrounding trainer may add a small coefficient such as 0.01.
    """

    def __init__(
        self,
        d_feat: int = 157,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
        compression: int = 2,
        sparse_topk: int = 4,
        window_size: int = 4,
        hc_streams: int = 4,
    ):
        super().__init__()
        for name, value in locals().copy().items():
            if name in {
                "d_feat",
                "d_model",
                "nhead",
                "num_layers",
                "num_experts",
                "top_k",
                "compression",
                "sparse_topk",
                "window_size",
                "hc_streams",
            }:
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"{name} must be a positive integer.")
        if d_model % nhead or (d_model // nhead) % 2:
            raise ValueError("d_model must be divisible by nhead with an even head dimension.")
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        self.d_feat, self.hc_streams = d_feat, hc_streams
        self.input_projection = nn.Linear(d_feat, d_model)
        args = (
            d_model,
            nhead,
            dropout,
            num_experts,
            top_k,
            compression,
            sparse_topk,
            window_size,
            hc_streams,
        )
        self.encoder = nn.ModuleList(
            [_CEDBlock(*args, encoder=True, reindex=True) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [
                _CEDBlock(*args, encoder=False, reindex=(layer % 2 == 0))
                for layer in range(num_layers)
            ]
        )
        self.memory_norm = _RMSNorm(d_model)
        # The disclosed V4.1 schedule uses ratio 2 in the encoder and ratio 1
        # in the decoder: decoder memory keeps one KV entry per encoder state.
        self.decoder_memory = _CompressedMemory(
            d_model, d_model // nhead, max(4, d_model // nhead // 2), compression=1
        )
        self.final_norm = _RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.aux_loss = torch.tensor(0.0)

    def forward_features(self, features: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.d_feat or features.shape[1] < 1:
            raise ValueError(f"Expected features [batch, positive_time, {self.d_feat}].")
        if padding.shape != features.shape[:2] or padding.dtype != torch.bool:
            raise ValueError("padding must be a boolean [batch, time] tensor.")
        values = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        hidden = self.input_projection(values.masked_fill(padding[..., None], 0))
        hidden = hidden.masked_fill(padding[..., None], 0)
        streams = hidden.unsqueeze(-2).expand(-1, -1, self.hc_streams, -1)
        prior_read = hidden.new_zeros(*hidden.shape[:2], self.hc_streams)
        prior_read[..., 0] = 1
        penalties = []
        for block in self.encoder:
            streams, prior_read, _, penalty = block(streams, prior_read, padding)
            penalties.append(penalty)
        encoder_states = _HyperMix.read(streams, prior_read)
        memory = self.decoder_memory(self.memory_norm(encoder_states), padding)
        selection = None
        for block in self.decoder:
            streams, prior_read, selection, penalty = block(
                streams, prior_read, padding, memory, selection
            )
            penalties.append(penalty)
        self.aux_loss = torch.stack(penalties).mean()
        return self.final_norm(_HyperMix.read(streams, prior_read)).masked_fill(
            padding[..., None], 0
        )

    def forward(self, features: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(features, padding)[:, -1]).squeeze(-1)
