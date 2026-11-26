"""Small numeric GLM-5.4-inspired loop model, independently implemented.

This implements the disclosed loop, per-pass gates, hybrid KV, and index-reuse
ideas. It is not the flagship model, pretrained weights, or a production kernel
reimplementation. See docs/architectures/glm_qwen.md for the adaptation boundary.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_dimensions(d_feat, d_model, nhead, num_layers, dropout):
    for name, value in {
        "d_feat": d_feat,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if d_model % nhead:
        raise ValueError("d_model must be divisible by nhead")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def _clean_input(features, padding, d_feat):
    if features.ndim != 3 or features.shape[-1] != d_feat or features.shape[1] < 1:
        raise ValueError(f"Expected nonempty [batch, time, {d_feat}] features")
    if padding.shape != features.shape[:2] or padding.dtype != torch.bool:
        raise ValueError("padding must be a bool [batch, time] tensor")
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).masked_fill(
        padding.unsqueeze(-1), 0
    )


def _position_encoding(length, width, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    rates = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10_000.0) / width)
    )
    values = torch.zeros(length, width, device=device, dtype=dtype)
    values[:, 0::2] = torch.sin(positions * rates)
    values[:, 1::2] = torch.cos(positions * rates[: width // 2])
    return values


def _masked_softmax(scores, allowed):
    """Zero probability for wholly masked rows, without NaNs or fabricated keys."""
    weights = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(dim=-1)
    weights = weights * allowed.to(weights.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)


class _SwiGLU(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class SparseMoE(nn.Module):
    """Actual top-k token dispatch with a trainable router and optional shared FFN.

    The compact expert count and balancing auxiliary loss are explicit research
    choices, not an assertion about a flagship model's exact routing recipe.
    """

    def __init__(self, width, num_experts=4, top_k=2, hidden=None, shared_expert=True):
        super().__init__()
        if (
            not isinstance(num_experts, int)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= num_experts
        ):
            raise ValueError("Require 1 <= top_k <= num_experts")
        self.num_experts, self.top_k = num_experts, top_k
        hidden = width if hidden is None else hidden
        self.router = nn.Linear(width, num_experts, bias=False)
        self.experts = nn.ModuleList(_SwiGLU(width, hidden) for _ in range(num_experts))
        self.shared = _SwiGLU(width, hidden) if shared_expert else None
        self.aux_loss = None
        self.expert_counts = None

    def forward(self, x, padding):
        flat = x.reshape(-1, x.shape[-1])
        valid = (~padding).reshape(-1)
        active = flat[valid]
        if active.shape[0] == 0:
            self.aux_loss = flat.sum() * 0
            self.expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)
            return torch.zeros_like(x)
        probabilities = self.router(active).softmax(dim=-1)
        weights, assignment = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        output = torch.zeros_like(active)
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(assignment == expert_id)
            if rows.numel():
                contribution = expert(active[rows]) * weights[rows, slots].unsqueeze(-1)
                output = output.index_add(0, rows, contribution)
        if self.shared is not None:
            output = output + self.shared(active)
        dispatch = F.one_hot(assignment, self.num_experts).float().mean(dim=(0, 1))
        self.aux_loss = self.num_experts * (probabilities.mean(dim=0) * dispatch).sum()
        self.expert_counts = F.one_hot(assignment, self.num_experts).sum(dim=(0, 1)).detach()
        complete = torch.zeros_like(flat)
        complete[valid] = output
        return complete.reshape_as(x)


class _SparseLoopAttention(nn.Module):
    """Reuse first-pass sparse global KV; later passes add current local KV."""

    def __init__(self, width, nhead, sparse_k, local_window, dropout):
        super().__init__()
        self.nhead, self.head_dim = nhead, width // nhead
        self.sparse_k, self.local_window = sparse_k, local_window
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        index_dim = max(4, width // 8)
        self.index_q = nn.Linear(width, index_dim, bias=False)
        self.index_k = nn.Linear(width, index_dim, bias=False)
        self.index_scale = index_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.indexer_calls = 0
        self.last_indices = None
        self.last_memory_tokens = 0

    def _gather(self, values, indices):
        batch, length, width = values.shape
        values = values.reshape(batch, length, self.nhead, self.head_dim)
        selected = values[torch.arange(batch, device=values.device)[:, None, None], indices]
        return selected.permute(0, 3, 1, 2, 4)

    def forward(self, x, padding, cache=None):
        batch, length, width = x.shape
        q = self.q(x).reshape(batch, length, self.nhead, self.head_dim).transpose(1, 2)
        if cache is None:
            self.indexer_calls += 1
            index_scores = self.index_q(x) @ self.index_k(x).transpose(-1, -2) * self.index_scale
            causal = (
                torch.arange(length, device=x.device)[None, :]
                <= torch.arange(length, device=x.device)[:, None]
            )
            possible = causal[None] & ~padding[:, None, :]
            selected_scores, indices = index_scores.masked_fill(~possible, -torch.inf).topk(
                min(self.sparse_k, length), dim=-1
            )
            allowed = torch.isfinite(selected_scores)
            cache = {
                "indices": indices,
                "allowed": allowed,
                "index_bias": selected_scores.masked_fill(~allowed, 0),
                "keys": self.k(x),
                "values": self.v(x),
            }
        self.last_indices = cache["indices"].detach()
        keys = self._gather(cache["keys"], cache["indices"])
        values = self._gather(cache["values"], cache["indices"])
        allowed = cache["allowed"]
        # Selected index scores participate in learning; top-k positions remain
        # discrete. No full dense attention is used after selection.
        index_bias = cache["index_bias"]
        if cache.get("consumed", False):
            raw_local = (
                torch.arange(length, device=x.device)[:, None]
                - torch.arange(self.local_window - 1, -1, -1, device=x.device)[None, :]
            )
            local_indices = raw_local.clamp_min(0)[None].expand(batch, -1, -1)
            local_allowed = (raw_local >= 0)[None] & ~padding[
                torch.arange(batch, device=x.device)[:, None, None], local_indices
            ]
            keys = torch.cat((keys, self._gather(self.k(x), local_indices)), dim=-2)
            values = torch.cat((values, self._gather(self.v(x), local_indices)), dim=-2)
            allowed = torch.cat((allowed, local_allowed), dim=-1)
            index_bias = torch.cat(
                (index_bias, torch.zeros_like(local_indices, dtype=x.dtype)), dim=-1
            )
        scores = torch.einsum("bhtd,bhtkd->bhtk", q, keys) * self.head_dim**-0.5
        self.last_memory_tokens = keys.shape[-2]
        scores = scores + 0.1 * index_bias[:, None]
        weights = self.dropout(_masked_softmax(scores, allowed[:, None]))
        output = torch.einsum("bhtk,bhtkd->bhtd", weights, values)
        output = output.transpose(1, 2).reshape(batch, length, width)
        cache["consumed"] = True
        return self.out(output).masked_fill(padding.unsqueeze(-1), 0), cache


class _LoopBlock(nn.Module):
    def __init__(self, width, nhead, loop_steps, sparse_k, local_window, dropout, num_experts):
        super().__init__()
        self.attention_norm = nn.RMSNorm(width)
        self.ffn_norm = nn.RMSNorm(width)
        self.attention = _SparseLoopAttention(width, nhead, sparse_k, local_window, dropout)
        self.moe = SparseMoE(width, num_experts=num_experts, top_k=min(2, num_experts))
        self.update_gates = nn.ModuleList(nn.Linear(2 * width, 1) for _ in range(loop_steps))
        self.dropout = nn.Dropout(dropout)
        for step, gate in enumerate(self.update_gates):
            nn.init.constant_(gate.bias, -1.5 if step == 0 else 0.0)

    def forward(self, x, padding, step, cache):
        attention, cache = self.attention(self.attention_norm(x), padding, cache)
        delta = attention + self.moe(self.ffn_norm(x + attention), padding)
        gate = torch.sigmoid(self.update_gates[step](torch.cat((x, delta), dim=-1)))
        x = x + gate * self.dropout(delta)
        return x.masked_fill(padding.unsqueeze(-1), 0), cache


class GLMMiniEncoder(nn.Module):
    """Shared sparse-MoE body with two gated refinement passes by default.

    ``num_layers`` counts distinct shared blocks; increasing ``loop_steps`` adds
    per-pass gates and compute, not copies of the attention/expert body.
    """

    def __init__(
        self,
        d_feat=157,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        loop_steps=2,
        sparse_k=8,
        local_window=4,
        num_experts=4,
    ):
        super().__init__()
        _validate_dimensions(d_feat, d_model, nhead, num_layers, dropout)
        for name, value in {
            "loop_steps": loop_steps,
            "sparse_k": sparse_k,
            "local_window": local_window,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.d_feat, self.d_model, self.loop_steps = d_feat, d_model, loop_steps
        self.projection = nn.Linear(d_feat, d_model)
        self.blocks = nn.ModuleList(
            _LoopBlock(d_model, nhead, loop_steps, sparse_k, local_window, dropout, num_experts)
            for _ in range(num_layers)
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.aux_loss = None
        self.last_indexer_calls = 0

    def forward_features(self, features, padding):
        x = self.projection(_clean_input(features, padding, self.d_feat))
        x = x + _position_encoding(x.shape[1], self.d_model, x.device, x.dtype)
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        caches, losses = [None] * len(self.blocks), []
        for block in self.blocks:
            block.attention.indexer_calls = 0
        for step in range(self.loop_steps):
            for layer, block in enumerate(self.blocks):
                x, caches[layer] = block(x, padding, step, caches[layer])
                losses.append(block.moe.aux_loss)
        self.aux_loss = torch.stack(losses).mean()
        self.last_indexer_calls = sum(block.attention.indexer_calls for block in self.blocks)
        return self.norm(x).masked_fill(padding.unsqueeze(-1), 0)

    def forward(self, features, padding):
        return self.head(self.forward_features(features, padding)[:, -1]).squeeze(-1)
