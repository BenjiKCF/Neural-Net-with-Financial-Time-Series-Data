"""Differentiable, equally weighted daily objectives and portfolio surrogates.

The optimized Pearson objective is distinct from evaluation Spearman Rank IC.
Missing/constant targets are counted; constant predictions are never discarded.
No raw-score smoothing term is described as a realized transaction cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class DailyLoss:
    loss: Tensor
    counts: dict[str, int]
    terms: dict[str, float]


def daily_objective(
    prediction: Tensor,
    target: Tensor,
    *,
    valid: Tensor | None = None,
    objective: str = "huber",
    huber_delta: float = 1.0,
    epsilon: float = 1e-4,
    ranking_temperature: float = 1.0,
    regression_weight: float = 1.0,
    ic_weight: float = 1.0,
    ranking_weight: float = 1.0,
    ic_auxiliary_weight: float = 0.05,
) -> DailyLoss:
    """Optimize one or more complete daily cross sections, each with equal weight.

    Inputs have shape ``[stock]`` or ``[date, stock]``. ``valid`` masks target
    availability only; callers must retain every eligible stock in the model's
    context. IC-only mode includes a declared small Huber signal for singleton
    or constant-label days. Pairwise ranking ignores tied target pairs, never
    tied predictions. Counts include every skipped/undefined correlation day.
    """
    if objective not in {"mse", "huber", "ic", "ranking", "combined"}:
        raise ValueError("objective must be mse, huber, ic, ranking or combined")
    settings = {
        "huber_delta": huber_delta,
        "epsilon": epsilon,
        "ranking_temperature": ranking_temperature,
        "regression_weight": regression_weight,
        "ic_weight": ic_weight,
        "ranking_weight": ranking_weight,
        "ic_auxiliary_weight": ic_auxiliary_weight,
    }
    if any(
        not isinstance(x, (int, float)) or not torch.isfinite(torch.tensor(x))
        for x in settings.values()
    ):
        raise ValueError("Objective settings must be finite numbers")
    if min(huber_delta, epsilon, ranking_temperature) <= 0:
        raise ValueError("Huber delta, epsilon and ranking temperature must be positive")
    if min(regression_weight, ic_weight, ranking_weight, ic_auxiliary_weight) < 0:
        raise ValueError("Objective weights must be nonnegative")
    if prediction.shape != target.shape or prediction.ndim not in {1, 2}:
        raise ValueError("prediction and target must share [stock] or [date, stock] shape")
    if valid is None:
        valid = torch.isfinite(target)
    if valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask matching targets")
    valid = valid & torch.isfinite(target)
    if not torch.isfinite(prediction[valid]).all():
        raise ValueError("Predictions at usable target positions must be finite")
    if prediction.ndim == 1:
        prediction, target, valid = prediction[None], target[None], valid[None]
    counts = {
        "dates": len(prediction),
        "used_dates": 0,
        "empty_dates": 0,
        "undefined_ic_dates": 0,
        "no_ranking_pairs_dates": 0,
        "valid_pairs": 0,
    }
    losses, sums = [], {"mse": 0.0, "huber": 0.0, "negative_pearson_ic": 0.0, "ranking": 0.0}
    for p, y, mask in zip(prediction, target, valid):
        p, y = p[mask], y[mask]
        if not len(p):
            counts["empty_dates"] += 1
            continue
        counts["used_dates"] += 1
        mse = F.mse_loss(p, y)
        huber = F.huber_loss(p, y, delta=huber_delta)
        # This zero retains a gradient path even when IC/ranking is undefined.
        zero = p.sum() * 0
        centered_p, centered_y = p - p.mean(), y - y.mean()
        if len(p) < 2 or bool((centered_y.square().mean() <= epsilon**2).detach()):
            ic = zero
            counts["undefined_ic_dates"] += 1
        else:
            covariance = (centered_p * centered_y).mean()
            denominator = (centered_p.square().mean() + epsilon**2).sqrt()
            denominator = denominator * (centered_y.square().mean() + epsilon**2).sqrt()
            ic = -covariance / denominator
        rows, cols = torch.triu_indices(len(p), len(p), offset=1, device=p.device)
        directions = torch.sign(y[rows] - y[cols])
        untied = directions != 0
        if bool(untied.any()):
            margins = (p[rows[untied]] - p[cols[untied]]) * directions[untied]
            ranking = F.softplus(-margins / ranking_temperature).mean()
            counts["valid_pairs"] += int(untied.sum().detach())
        else:
            ranking = zero
            counts["no_ranking_pairs_dates"] += 1
        if objective == "mse":
            loss = mse
        elif objective == "huber":
            loss = huber
        elif objective == "ic":
            loss = ic + ic_auxiliary_weight * huber
        elif objective == "ranking":
            loss = ranking + ic_auxiliary_weight * huber
        else:
            loss = regression_weight * huber + ic_weight * ic + ranking_weight * ranking
        losses.append(loss)
        for name, value in zip(sums, (mse, huber, ic, ranking)):
            sums[name] += float(value.detach())
    if not losses:
        raise ValueError("No finite, unmasked targets in this batch")
    return DailyLoss(
        torch.stack(losses).mean(),
        counts,
        {name: value / counts["used_dates"] for name, value in sums.items()},
    )


def portfolio_turnover_surrogate(
    scores: Tensor,
    previous_weights: Tensor,
    *,
    tradable: Tensor | None = None,
    temperature: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Long-only weight turnover on an explicitly aligned instrument union.

    Nontradable holdings remain frozen at their previous weights. The remaining
    budget is allocated to tradable names by softmax. With no tradable names,
    unused capital stays cash. Turnover includes the cash weight, i.e. buying a
    fully invested portfolio from cash has turnover 1. A caller may multiply by
    an assumed proportional fee, but that is still a surrogate, not a fill model.
    """
    if scores.ndim != 1 or previous_weights.shape != scores.shape or not scores.numel():
        raise ValueError("scores and previous_weights must share a nonempty instrument axis")
    if not torch.isfinite(scores).all() or not torch.isfinite(previous_weights).all():
        raise ValueError("Scores and previous holdings must be finite")
    if (previous_weights < 0).any() or previous_weights.sum() > 1 + 1e-6:
        raise ValueError("Previous weights must be long-only and sum to at most one")
    if not isinstance(temperature, (int, float)) or not 0 < temperature < float("inf"):
        raise ValueError("temperature must be finite and positive")
    if tradable is None:
        tradable = torch.ones_like(scores, dtype=torch.bool)
    if tradable.shape != scores.shape or tradable.dtype != torch.bool:
        raise ValueError("tradable must be a boolean mask on the instrument union")
    weights = previous_weights.detach().clone()
    if tradable.any():
        budget = (1 - previous_weights[~tradable].sum()).clamp_min(0)
        weights[tradable] = torch.softmax(scores[tradable] / temperature, dim=0) * budget
    cash_change = ((1 - weights.sum()) - (1 - previous_weights.sum())).abs()
    turnover = 0.5 * ((weights - previous_weights).abs().sum() + cash_change)
    return turnover, weights


def quantile_loss(prediction: Tensor, target: Tensor, quantiles: tuple[float, ...]) -> Tensor:
    """Masked pinball loss for ``[..., quantile]`` heads and raw target units."""
    if prediction.shape[:-1] != target.shape or prediction.shape[-1] != len(quantiles):
        raise ValueError("Quantile head shape does not match targets/levels")
    if (
        not quantiles
        or any(not 0 < q < 1 for q in quantiles)
        or sorted(set(quantiles)) != list(quantiles)
    ):
        raise ValueError("quantiles must be distinct increasing levels in (0,1)")
    valid = torch.isfinite(target)
    if not valid.any() or not torch.isfinite(prediction[valid]).all():
        raise ValueError("Quantile loss needs finite predictions and at least one finite target")
    residual = target[valid, None] - prediction[valid]
    levels = prediction.new_tensor(quantiles)
    return torch.maximum(levels * residual, (levels - 1) * residual).mean()
