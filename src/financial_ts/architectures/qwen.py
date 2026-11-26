"""Compact numeric Qwen4-inspired hybrid, independently implemented.

The model keeps the disclosed 3:1 gated-delta/sparse-attention pattern, four
gated residual streams, and sparse experts. It replaces lexical N-gram lookup
with a causal numeric convolution. No flagship weights or exact reproduction
are claimed; see docs/architectures/glm_qwen.md.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .glm import (
    SparseMoE,
    _clean_input,
    _masked_softmax,
    _position_encoding,
    _validate_dimensions,
)


class GatedDeltaMixer(nn.Module):
    """Explicit recurrent gated delta rule with fixed-size per-head state."""

    def __init__(self, width, nhead):
        super().__init__()
        self.nhead, self.head_dim = nhead, width // nhead
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.decay = nn.Linear(width, nhead)
        self.beta = nn.Linear(width, nhead)
        self.output_gate = nn.Linear(width, width)
        self.norm = nn.RMSNorm(width)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x, padding):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.nhead, self.head_dim).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        decay = torch.exp(-F.softplus(self.decay(x)))
        beta = torch.sigmoid(self.beta(x))
        state = x.new_zeros(batch, self.nhead, self.head_dim, self.head_dim)
        outputs = []
        for step in range(length):
            old = state
            decayed = state * decay[:, step, :, None, None]
            expected = torch.einsum("bhd,bhdv->bhv", k[:, step], decayed)
            correction = v[:, step] - expected
            updated = decayed + beta[:, step, :, None, None] * (
                k[:, step, :, :, None] * correction[:, :, None, :]
            )
            # Missing sessions neither decay nor write to the recurrent state.
            state = torch.where(padding[:, step, None, None, None], old, updated)
            result = torch.einsum("bhd,bhdv->bhv", q[:, step], state)
            outputs.append(result.masked_fill(padding[:, step, None, None], 0))
        mixed = torch.stack(outputs, dim=1).reshape(batch, length, width)
        mixed = self.norm(mixed) * F.silu(self.output_gate(x))
        return self.output(mixed).masked_fill(padding.unsqueeze(-1), 0)


class BlockSparseAttention(nn.Module):
    """Select completed historical micro-blocks and retain the current block.

    Top-k block scores influence selected attention logits, allowing the indexer
    to learn with the forecasting objective. This lightweight training recipe is
    an adaptation; it does not claim Qwen4's unpublished exact indexer objective.
    """

    def __init__(self, width, nhead, block_size=4, selected_blocks=2, dropout=0.1):
        super().__init__()
        self.nhead, self.head_dim = nhead, width // nhead
        self.block_size, self.selected_blocks = block_size, selected_blocks
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        index_dim = max(4, width // 8)
        self.index_q = nn.Linear(width, index_dim, bias=False)
        self.index_k = nn.Linear(width, index_dim, bias=False)
        self.index_scale = index_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.last_selected_blocks = None
        self.last_token_indices = None
        self.last_allowed = None

    def forward(self, x, padding):
        batch, length, width = x.shape
        size = self.block_size
        nblocks = (length + size - 1) // size
        extension = nblocks * size - length
        valid = (~padding).to(x.dtype)
        pooled_x = F.pad(x * valid.unsqueeze(-1), (0, 0, 0, extension))
        pooled_valid = F.pad(valid, (0, extension)).reshape(batch, nblocks, size)
        means = pooled_x.reshape(batch, nblocks, size, width).sum(2) / pooled_valid.sum(
            2
        ).clamp_min(1).unsqueeze(-1)
        block_scores = self.index_q(x) @ self.index_k(means).transpose(-1, -2) * self.index_scale
        positions = torch.arange(length, device=x.device)
        completed = torch.arange(nblocks, device=x.device)[None, :] < (positions // size)[:, None]
        possible = completed[None] & (pooled_valid.sum(2)[:, None] > 0)
        selected_scores, selected = block_scores.masked_fill(~possible, -torch.inf).topk(
            min(self.selected_blocks, nblocks), dim=-1
        )
        block_allowed = torch.isfinite(selected_scores)
        offsets = torch.arange(size, device=x.device)
        global_indices = (selected.unsqueeze(-1) * size + offsets).flatten(-2).clamp_max(length - 1)
        global_allowed = block_allowed.repeat_interleave(size, dim=-1)
        raw_local = (positions // size * size)[:, None] + offsets[None]
        local_indices = raw_local.clamp_max(length - 1)[None].expand(batch, -1, -1)
        # The incomplete final micro-block must not repeat its last observation.
        local_allowed = ((raw_local <= positions[:, None]) & (raw_local < length))[None]
        local_allowed = local_allowed.expand(batch, -1, -1)
        indices = torch.cat((global_indices, local_indices), dim=-1)
        allowed = torch.cat((global_allowed, local_allowed), dim=-1)
        batch_indices = torch.arange(batch, device=x.device)[:, None, None]
        allowed = allowed & ~padding[batch_indices, indices]
        self.last_selected_blocks = selected.detach()
        self.last_token_indices = indices.detach()
        self.last_allowed = allowed.detach()
        q, k, v = self.qkv(x).reshape(batch, length, 3, self.nhead, self.head_dim).unbind(2)
        selected_k = k[batch_indices, indices].permute(0, 3, 1, 2, 4)
        selected_v = v[batch_indices, indices].permute(0, 3, 1, 2, 4)
        q = q.transpose(1, 2)
        scores = torch.einsum("bhtd,bhtkd->bhtk", q, selected_k) * self.head_dim**-0.5
        index_bias = selected_scores.masked_fill(~block_allowed, 0).repeat_interleave(size, dim=-1)
        index_bias = torch.cat((index_bias, x.new_zeros(batch, length, size)), dim=-1)
        scores = scores + 0.1 * index_bias[:, None]
        weights = self.dropout(_masked_softmax(scores, allowed[:, None]))
        output = torch.einsum("bhtk,bhtkd->bhtd", weights, selected_v)
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.output(output).masked_fill(padding.unsqueeze(-1), 0)


class _StreamGates(nn.Module):
    def __init__(self, width, streams):
        super().__init__()
        rank = max(4, width // 8)
        self.streams, self.width = streams, width
        self.gate = nn.Sequential(
            nn.Linear(width, rank), nn.SiLU(), nn.Linear(rank, streams * width)
        )

    def forward(self, states):
        summary = states.mean(dim=-2)
        return self.gate(summary).reshape(*summary.shape[:-1], self.streams, self.width)


class _GatedResidualBlock(nn.Module):
    def __init__(self, width, mixer, num_experts, dropout, streams=4):
        super().__init__()
        self.mixer = mixer
        self.moe = SparseMoE(
            width,
            num_experts=num_experts,
            top_k=min(2, num_experts),
            hidden=max(8, width // 2),
            shared_expert=False,
        )
        self.norms = nn.ModuleList(nn.RMSNorm(width) for _ in range(2))
        self.read_gates = nn.ModuleList(_StreamGates(width, streams) for _ in range(2))
        self.write_gates = nn.ModuleList(_StreamGates(width, streams) for _ in range(2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, padding):
        for step, operation in enumerate((self.mixer, self.moe)):
            weights = self.read_gates[step](states).softmax(dim=-2)
            read = self.norms[step]((weights * states).sum(dim=-2))
            delta = operation(read, padding)
            write = torch.sigmoid(self.write_gates[step](states))
            states = states + write * self.dropout(delta).unsqueeze(-2)
            states = states.masked_fill(padding[:, :, None, None], 0)
        return states


class QwenMiniEncoder(nn.Module):
    """Four-stream hybrid with three gated-delta layers per sparse-attention layer.

    ``num_layers`` counts hybrid groups, each containing four distinct mixers.
    No token lookup table is used: a causal numeric local mixer at the second
    layer substitutes for lexical N-grams, explicitly documented as an adaptation.
    """

    def __init__(
        self,
        d_feat=157,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        block_size=4,
        selected_blocks=2,
        num_experts=4,
        local_kernel=3,
    ):
        super().__init__()
        _validate_dimensions(d_feat, d_model, nhead, num_layers, dropout)
        for name, value in {
            "block_size": block_size,
            "selected_blocks": selected_blocks,
            "local_kernel": local_kernel,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.d_feat, self.d_model, self.local_kernel = d_feat, d_model, local_kernel
        self.projection = nn.Linear(d_feat, d_model)
        self.stream_embeddings = nn.Parameter(torch.empty(4, d_model))
        nn.init.normal_(self.stream_embeddings, std=0.02)
        self.local_mixer = nn.Conv1d(d_model, d_model, local_kernel, groups=d_model, bias=False)
        self.local_gate = nn.Parameter(torch.tensor(-2.0))
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            for _ in range(3):
                self.blocks.append(
                    _GatedResidualBlock(
                        d_model, GatedDeltaMixer(d_model, nhead), num_experts, dropout
                    )
                )
            self.blocks.append(
                _GatedResidualBlock(
                    d_model,
                    BlockSparseAttention(d_model, nhead, block_size, selected_blocks, dropout),
                    num_experts,
                    dropout,
                )
            )
        self.readout_logits = nn.Parameter(torch.zeros(4, d_model))
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.aux_loss = None

    def forward_features(self, features, padding):
        x = self.projection(_clean_input(features, padding, self.d_feat))
        x = x + _position_encoding(x.shape[1], self.d_model, x.device, x.dtype)
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        local = self.local_mixer(F.pad(x.transpose(1, 2), (self.local_kernel - 1, 0))).transpose(
            1, 2
        )
        local = local.masked_fill(padding.unsqueeze(-1), 0)
        states = x.unsqueeze(-2) + self.stream_embeddings[None, None]
        states = states.masked_fill(padding[:, :, None, None], 0)
        losses = []
        for layer, block in enumerate(self.blocks):
            if layer == 1:
                states = states + torch.sigmoid(self.local_gate) * local.unsqueeze(-2)
            states = block(states, padding)
            losses.append(block.moe.aux_loss)
        self.aux_loss = torch.stack(losses).mean()
        readout = (states * self.readout_logits.softmax(dim=0)[None, None]).sum(dim=-2)
        return self.norm(readout).masked_fill(padding.unsqueeze(-1), 0)

    def forward(self, features, padding):
        return self.head(self.forward_features(features, padding)[:, -1]).squeeze(-1)
