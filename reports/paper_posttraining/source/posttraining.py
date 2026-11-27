"""Bounded, validation-selected weight experiments for small forecasting models.

The Gaussian optimizer follows the simple full-parameter ES update in
https://arxiv.org/html/2509.24372v3. Directional editing borrows an affine
output attenuation from https://arxiv.org/html/2611.21885v2; the error diagnosis
here is external code, not a model reasoning about or rewriting itself.

These primitives know nothing about test data. Callers must provide separate,
past-only fitting and validation callbacks and record their chronological scope.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class ESConfig:
    population: int = 12
    generations: int = 8
    sigma: float = 0.001
    learning_rate: float = 1e-6
    seed: int = 42

    def __post_init__(self):
        if type(self.population) is not int or self.population < 2:
            raise ValueError("population must be an integer of at least two.")
        if type(self.generations) is not int or self.generations < 0:
            raise ValueError("generations must be a nonnegative integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        for name in ("sigma", "learning_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")


@dataclass
class ESResult:
    selected_state: dict[str, torch.Tensor]
    selected_generation: int
    selection_loss: float
    history: list[dict]
    candidate_evaluations: int
    status: str = "completed"


def _clone_state(model):
    state = model.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("ES supports tensor-only model state dictionaries.")
    return {name: value.detach().clone() for name, value in state.items()}


def _parameter_vector(parameters):
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters]).clone()


def _set_vector(parameters, vector):
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(vector[offset : offset + count].view_as(parameter))
            offset += count


def _noise(seed, reference):
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    return torch.randn(
        reference.shape, generator=generator, device=reference.device, dtype=reference.dtype
    )


def _finite_scalar(value, name):
    if isinstance(value, torch.Tensor) and value.numel() != 1:
        raise ValueError(f"{name} callback must return a scalar.")
    try:
        scalar = float(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} callback must return a scalar.") from error
    return scalar if math.isfinite(scalar) else None


def evolution_strategy(
    model: nn.Module,
    fit_reward: Callable[[nn.Module], float],
    validation_loss: Callable[[nn.Module], float],
    *,
    config: ESConfig = ESConfig(),
) -> ESResult:
    """Mutate ``model`` to its best validation-selected Gaussian ES generation.

    Every parameter, including frozen parameters, belongs to the search vector.
    A generation samples independent N(0, I) perturbations and updates the mean
    by ``learning_rate / (population * sigma) * sum(z_reward * noise)``.
    This uses no gradients, mirrored samples, rank rewards, or Adam. Noise is
    reconstructed from local seeds instead of retaining the population vectors.

    Only generation means (and the initial state) are evaluated for validation
    selection. Strict loss improvement wins; ties retain the earlier generation.
    All callbacks receive the module in eval mode under ``torch.no_grad()`` and
    should be deterministic, read-only functions. Parameters and buffers are
    restored before each call. The supplied model finishes in eval mode.

    Nonfinite proposal rewards are rejected, with the remaining finite proposals
    recombined using their count. Fewer than two finite rewards or a flat reward
    population makes no update. Nonfinite mean updates are rejected. These
    safeguards do not change the ordinary all-finite ES formula. Callback
    exceptions restore the original state and propagate. Random number generation
    is local and does not modify Python's or PyTorch's global RNG state.
    """
    if not isinstance(config, ESConfig):
        raise TypeError("config must be an ESConfig.")
    parameters = list(model.parameters())
    if not parameters or any(not parameter.is_floating_point() for parameter in parameters):
        raise ValueError("ES requires a nonempty set of floating-point parameters.")
    if len({(parameter.device, parameter.dtype) for parameter in parameters}) != 1:
        raise ValueError("All ES parameters must share a device and floating-point dtype.")
    initial_state = _clone_state(model)
    if any(not torch.isfinite(value).all() for value in initial_state.values()):
        raise ValueError("The initial model state must be finite.")
    current = _parameter_vector(parameters)
    initial_vector = current.clone()
    best_state = _clone_state(model)
    selected_generation, candidate_evaluations = 0, 0
    random_seeds = random.Random(config.seed)
    history = []
    completed = False

    def restore(vector):
        model.load_state_dict(initial_state)
        _set_vector(parameters, vector)
        model.eval()

    try:
        restore(current)
        with torch.no_grad():
            best_loss = _finite_scalar(validation_loss(model), "validation_loss")
        if best_loss is None:
            raise ValueError("The initial validation loss must be finite.")
        history.append(
            {
                "generation": 0,
                "status": "initial",
                "validation_loss": best_loss,
                "selected_on_generation": True,
                "candidate_seeds": [],
                "candidate_rewards": [],
                "valid_candidates": 0,
                "update_rms": 0.0,
                "distance_from_initial": 0.0,
            }
        )
        for generation in range(1, config.generations + 1):
            seeds = [random_seeds.randrange(2**63) for _ in range(config.population)]
            rewards = []
            for seed in seeds:
                proposal = current + config.sigma * _noise(seed, current)
                if not torch.isfinite(proposal).all():
                    rewards.append(None)
                    continue
                restore(proposal)
                with torch.no_grad():
                    rewards.append(_finite_scalar(fit_reward(model), "fit_reward"))
                candidate_evaluations += 1
            finite = [(seed, reward) for seed, reward in zip(seeds, rewards) if reward is not None]
            status = "updated"
            update = torch.zeros_like(current)
            reward_mean, reward_std = None, None
            if len(finite) >= 2:
                # Scale before moments so finite, very large rewards cannot overflow.
                values = torch.tensor([reward for _, reward in finite], dtype=torch.float64)
                scale = max(float(values.abs().max()), 1.0)
                scaled = values / scale
                mean, std = scaled.mean(), scaled.std(unbiased=False)
                reward_mean, reward_std = float(mean) * scale, float(std) * scale
                if float(std) > 0:
                    normalized = (scaled - mean) / std
                    for (seed, _), weight in zip(finite, normalized.tolist()):
                        update.add_(_noise(seed, current), alpha=weight)
                    update.mul_(config.learning_rate / (len(finite) * config.sigma))
                else:
                    status = "flat_rewards"
            else:
                status = "insufficient_finite_rewards"
            proposal_mean = current + update
            if not torch.isfinite(proposal_mean).all():
                status = "rejected_nonfinite_update"
                update.zero_()
            else:
                current = proposal_mean
            restore(current)
            candidate_state = _clone_state(model)
            with torch.no_grad():
                loss = _finite_scalar(validation_loss(model), "validation_loss")
            improved = loss is not None and loss < best_loss
            if improved:
                best_loss = loss
                selected_generation = generation
                best_state = candidate_state
            history.append(
                {
                    "generation": generation,
                    "status": status if loss is not None else "rejected_nonfinite_validation",
                    "validation_loss": loss,
                    "selected_on_generation": improved,
                    "candidate_seeds": seeds,
                    "candidate_rewards": rewards,
                    "valid_candidates": len(finite),
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "update_rms": float(update.double().square().mean().sqrt()),
                    "distance_from_initial": float(
                        torch.linalg.vector_norm((current - initial_vector).double())
                    ),
                }
            )
        result = ESResult(
            selected_state={name: value.clone() for name, value in best_state.items()},
            selected_generation=selected_generation,
            selection_loss=best_loss,
            history=history,
            candidate_evaluations=candidate_evaluations,
        )
        completed = True
        return result
    finally:
        model.load_state_dict(best_state if completed else initial_state)
        model.eval()


def diagnose_error_direction(
    hidden: torch.Tensor, losses: torch.Tensor, *, quantile: float = 0.25
) -> torch.Tensor:
    """Unit direction from low-error to high-error hidden-output means.

    ``hidden`` must already contain one vector per fitting example, with any
    temporal/token pooling fixed by the caller. ``losses`` contains corresponding
    past-only, nonnegative example losses. The highest/lowest ``floor(N*q)``
    examples are used (at least one per group); stable ordering breaks ties.
    Identical losses or indistinguishable group means return a zero vector, which
    ``apply_directional_edit`` treats as an explicit no-op.
    """
    if isinstance(quantile, bool) or not math.isfinite(quantile) or not 0 < quantile <= 0.5:
        raise ValueError("quantile must be finite and in (0, 0.5].")
    if not isinstance(hidden, torch.Tensor) or not isinstance(losses, torch.Tensor):
        raise TypeError("hidden and losses must be tensors.")
    if (
        hidden.ndim != 2
        or hidden.shape[0] < 2
        or hidden.shape[1] < 1
        or not hidden.is_floating_point()
        or losses.ndim != 1
        or losses.shape[0] != hidden.shape[0]
    ):
        raise ValueError("Expected floating hidden [N,width] and losses [N], with N >= 2.")
    if not torch.isfinite(hidden).all() or not torch.isfinite(losses).all() or (losses < 0).any():
        raise ValueError("Hidden states and nonnegative losses must be finite.")
    zero = hidden.new_zeros(hidden.shape[1])
    if bool(losses.max() == losses.min()):
        return zero
    count = max(1, int(hidden.shape[0] * quantile))
    order = torch.argsort(losses.detach().to(hidden.device), stable=True)
    values = hidden.detach().double()
    low, high = values[order[:count]].mean(dim=0), values[order[-count:]].mean(dim=0)
    delta = high - low
    norm = torch.linalg.vector_norm(delta)
    if not torch.isfinite(norm):
        raise ValueError("The error-direction calculation overflowed.")
    scale = max(float(values.abs().max()), 1.0)
    if float(norm) <= torch.finfo(hidden.dtype).eps * scale:
        return zero
    return (delta / norm).to(dtype=hidden.dtype)


def apply_directional_edit(
    model: nn.Module, layer_name: str, direction: torch.Tensor, alpha: float
) -> dict:
    """Attenuate one fixed Linear layer's entire affine output along a direction.

    Applies ``W <- W - alpha*u*(u.T@W)`` and the analogous bias edit, normalizing
    a nonzero supplied direction to unit length. ``alpha`` is restricted to [0,1].
    The operation is in-place and does not select a strength: each candidate must
    start from the same restored base checkpoint. A zero direction is a no-op.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0,1].")
    if not isinstance(layer_name, str):
        raise TypeError("layer_name must be a module path string.")
    try:
        layer = model.get_submodule(layer_name)
    except AttributeError as error:
        raise ValueError(f"Unknown module path: {layer_name!r}.") from error
    if not isinstance(layer, nn.Linear):
        raise ValueError("The selected layer must be torch.nn.Linear.")
    if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
        raise ValueError("direction must be a one-dimensional tensor.")
    if direction.numel() != layer.out_features or not torch.isfinite(direction).all():
        raise ValueError("A finite direction must match the Linear layer's output width.")
    weight = layer.weight.detach().clone()
    bias = layer.bias.detach().clone() if layer.bias is not None else None
    if not torch.isfinite(weight).all() or (bias is not None and not torch.isfinite(bias).all()):
        raise ValueError("The selected affine layer must be finite before editing.")
    value = direction.detach().to(device=weight.device, dtype=torch.float64)
    norm = torch.linalg.vector_norm(value)
    if not torch.isfinite(norm):
        raise ValueError("Direction norm must be finite.")
    degenerate = float(norm) == 0.0
    unit = (value / norm).to(weight.dtype) if not degenerate else weight.new_zeros(value.shape)
    changed_weight = weight - alpha * torch.outer(unit, unit @ weight)
    changed_bias = bias - alpha * unit * torch.dot(unit, bias) if bias is not None else None
    if not torch.isfinite(changed_weight).all() or (
        changed_bias is not None and not torch.isfinite(changed_bias).all()
    ):
        raise ValueError("The directional edit produced nonfinite parameters.")
    change = float(torch.linalg.vector_norm((changed_weight - weight).double()))
    baseline = float(torch.linalg.vector_norm(weight.double()))
    with torch.no_grad():
        layer.weight.copy_(changed_weight)
        if changed_bias is not None:
            layer.bias.copy_(changed_bias)
    return {
        "layer_name": layer_name,
        "alpha": float(alpha),
        "direction_norm": float(norm),
        "degenerate": degenerate,
        "weight_change_norm": change,
        "relative_weight_change": change / baseline if baseline else 0.0,
        "bias_change_norm": (
            float(torch.linalg.vector_norm((changed_bias - bias).double()))
            if bias is not None
            else 0.0
        ),
    }
