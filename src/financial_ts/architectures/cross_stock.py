"""Own-data MASTER-style temporal/cross-stock encoders and regime expert heads.

Inspired by https://arxiv.org/abs/2312.15235, with deliberate differences:
market context is observable moments of this date's eligible Alpha157 panel,
not the authors' 63 index descriptors. No upstream weights/data/code are used.
Every forward call represents one decision date. Stock order carries no signal.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CrossStockEncoder(nn.Module):
    """Alternate intra-stock time attention and inter-stock attention per time.

    ``features`` is ``[stock, history, feature]``. ``padding=True`` excludes an
    observation. ``stock_valid=False`` excludes a padded instrument from both
    market descriptors and attention. The returned default head is ``[stock]``;
    ``forward_heads`` exposes optional horizon and monotonic quantile heads.
    """

    def __init__(
        self,
        d_feat: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_stock_attention: bool = True,
        market_context: bool = True,
        num_experts: int = 1,
        uniform_routing: bool = False,
        horizons: tuple[int, ...] = (1,),
        quantiles: tuple[float, ...] = (),
    ):
        super().__init__()
        for key, value in {
            "d_feat": d_feat,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "num_experts": num_experts,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if d_model % nhead or not 0 <= dropout < 1:
            raise ValueError("d_model must divide nhead; dropout must be in [0,1)")
        if any(
            type(flag) is not bool
            for flag in (cross_stock_attention, market_context, uniform_routing)
        ):
            raise ValueError("Ablation switches must be boolean")
        if (
            not horizons
            or any(type(h) is not int or h < 1 for h in horizons)
            or len(set(horizons)) != len(horizons)
        ):
            raise ValueError("horizons must be distinct positive session counts")
        if quantiles and (
            any(not 0 < q < 1 for q in quantiles) or sorted(set(quantiles)) != list(quantiles)
        ):
            raise ValueError("quantiles must be increasing unique probabilities")
        self.d_feat, self.d_model = d_feat, d_model
        self.cross_stock_attention, self.market_context = cross_stock_attention, market_context
        self.num_experts, self.uniform_routing = num_experts, uniform_routing
        self.horizons, self.quantiles = tuple(horizons), tuple(quantiles)
        self.projection = nn.Linear(d_feat, d_model)
        self.feature_gate = (
            nn.Sequential(nn.Linear(2 * d_feat, d_model), nn.GELU(), nn.Linear(d_model, d_feat))
            if market_context
            else None
        )
        self.temporal_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    nhead,
                    2 * d_model,
                    dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.stock_layers = (
            nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model,
                        nhead,
                        2 * d_model,
                        dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            if cross_stock_attention
            else nn.ModuleList()
        )
        self.pool_query = nn.Linear(d_model, d_model, bias=False)
        self.normalization = nn.LayerNorm(d_model)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, len(horizons))
                )
                for _ in range(num_experts)
            ]
        )
        self.router = (
            nn.Sequential(
                nn.Linear(2 * d_feat, d_model), nn.GELU(), nn.Linear(d_model, num_experts)
            )
            if num_experts > 1
            else None
        )
        self.quantile_head = (
            nn.Linear(d_model, len(horizons) * len(quantiles)) if quantiles else None
        )
        self.reconstruction_head = nn.Linear(d_model, d_feat)
        self.aux_loss: Tensor | float = 0.0
        self.routing_diagnostics_: dict = {}

    def _validate(self, features, padding, stock_valid):
        if features.ndim != 3 or features.shape[-1] != self.d_feat or min(features.shape[:2]) < 1:
            raise ValueError("Features must have nonempty [stock, history, d_feat] shape")
        if padding.shape != features.shape[:2] or padding.dtype != torch.bool:
            raise ValueError("padding must be boolean [stock, history]")
        if stock_valid is None:
            stock_valid = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
        if stock_valid.shape != features.shape[:1] or stock_valid.dtype != torch.bool:
            raise ValueError("stock_valid must be boolean [stock]")
        padding = padding | ~stock_valid[:, None]
        if not torch.isfinite(features[~padding]).all():
            raise ValueError("Unmasked feature values must be finite")
        return padding, stock_valid

    def encode(self, features: Tensor, padding: Tensor, stock_valid: Tensor | None = None):
        padding, stock_valid = self._validate(features, padding, stock_valid)
        clean = features.masked_fill(padding[..., None], 0)
        # All descriptors use the current date only and exclude instrument padding.
        current_valid = (~padding[:, -1]) & stock_valid
        values = clean[:, -1]
        count = current_valid.sum().clamp_min(1).to(features.dtype)
        mean = (values * current_valid[:, None]).sum(0) / count
        variance = ((values - mean).square() * current_valid[:, None]).sum(0) / count
        context = torch.cat([mean, (variance + 1e-8).sqrt()])
        if self.feature_gate is not None:
            clean = clean * (2 * torch.sigmoid(self.feature_gate(context)))[None, None]
        encoded = self.projection(clean)
        length = features.shape[1]
        positions = torch.arange(length, device=features.device, dtype=features.dtype)[:, None]
        rates = torch.exp(
            torch.arange(0, self.d_model, 2, device=features.device, dtype=features.dtype)
            * (-math.log(10000) / self.d_model)
        )
        positional = torch.zeros(length, self.d_model, device=features.device, dtype=features.dtype)
        positional[:, 0::2] = torch.sin(positions * rates)
        positional[:, 1::2] = torch.cos(positions * rates[: self.d_model // 2])
        encoded = encoded + positional
        # MHA cannot consume an entirely masked row; expose a zero temporary key
        # and discard that row immediately after attention.
        temporal_padding = padding.clone()
        temporal_padding[padding.all(1), -1] = False
        for i, temporal in enumerate(self.temporal_layers):
            encoded = temporal(encoded, src_key_padding_mask=temporal_padding)
            encoded = encoded.masked_fill(padding[..., None], 0)
            if self.cross_stock_attention:
                cross_padding = padding.T.clone()
                cross_padding[cross_padding.all(1), 0] = False
                encoded = self.stock_layers[i](
                    encoded.transpose(0, 1), src_key_padding_mask=cross_padding
                ).transpose(0, 1)
                encoded = encoded.masked_fill(padding[..., None], 0)
        query = self.pool_query(encoded[:, -1])
        logits = (encoded * query[:, None]).sum(-1) / math.sqrt(self.d_model)
        logits = logits.masked_fill(temporal_padding, -torch.inf)
        pooled = (torch.softmax(logits, dim=1)[..., None] * encoded).sum(1)
        pooled = self.normalization(pooled).masked_fill(~stock_valid[:, None], 0)
        return pooled, encoded, context, stock_valid

    def forward_heads(self, features: Tensor, padding: Tensor, stock_valid: Tensor | None = None):
        pooled, encoded, context, stock_valid = self.encode(features, padding, stock_valid)
        if self.router is None or self.uniform_routing:
            routing = pooled.new_full((self.num_experts,), 1 / self.num_experts)
        else:
            routing = torch.softmax(self.router(context), dim=-1)
        expert_outputs = torch.stack([expert(pooled) for expert in self.experts], dim=-1)
        returns = (expert_outputs * routing).sum(-1).masked_fill(~stock_valid[:, None], 0)
        self.aux_loss = self.num_experts * routing.square().sum() - 1
        entropy = -(routing * routing.clamp_min(1e-12).log()).sum()
        self.routing_diagnostics_ = {
            "weights": routing.detach().cpu().tolist(),
            "entropy": float(entropy.detach()),
            "effective_experts": float(entropy.detach().exp()),
            "uniform_control": self.uniform_routing,
        }
        result = {"returns": returns, "routing": routing, "representation": pooled}
        if self.quantile_head is not None:
            raw = self.quantile_head(pooled).reshape(
                len(pooled), len(self.horizons), len(self.quantiles)
            )
            # A freely located first quantile and positive increments prohibit
            # crossing; all levels are fitted with their actual pinball scores.
            quantiles = torch.cat(
                [raw[..., :1], raw[..., :1] + F.softplus(raw[..., 1:]).cumsum(-1)], -1
            )
            result["quantiles"] = quantiles.masked_fill(~stock_valid[:, None, None], 0)
        return result

    def forward(self, features: Tensor, padding: Tensor, stock_valid: Tensor | None = None):
        return self.forward_heads(features, padding, stock_valid)["returns"][:, 0]

    def reconstruct(self, features: Tensor, padding: Tensor, stock_valid: Tensor | None = None):
        _, encoded, _, _ = self.encode(features, padding, stock_valid)
        return self.reconstruction_head(encoded)
