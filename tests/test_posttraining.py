"""Check selection discipline and exact algebra for the paper-inspired primitives."""

import copy
import json
import random

import pytest
import torch
from torch import nn

from financial_ts.posttraining import (
    ESConfig,
    apply_directional_edit,
    diagnose_error_direction,
    evolution_strategy,
)


def _model(weight=0.0):
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def test_es_moves_toward_fit_optimum_and_selects_only_generation_means():
    model = _model()
    validation_weights = []

    def validation(module):
        assert not module.training and not torch.is_grad_enabled()
        validation_weights.append(float(module.weight.item()))
        return (module.weight.item() - 1.0) ** 2

    result = evolution_strategy(
        model,
        lambda module: -((module.weight.item() - 1.0) ** 2),
        validation,
        config=ESConfig(population=16, generations=5, sigma=0.1, learning_rate=0.01),
    )
    assert len(validation_weights) == 6
    assert result.candidate_evaluations == 80
    assert result.selected_generation > 0
    assert result.selection_loss < 0.5
    assert model.weight.item() == validation_weights[result.selected_generation]
    assert result.selection_loss == min(row["validation_loss"] for row in result.history)
    json.dumps(result.history, allow_nan=False)


def test_validation_can_reject_every_fit_improvement_and_restore_exact_baseline():
    model = _model()
    original = copy.deepcopy(model.state_dict())
    result = evolution_strategy(
        model,
        lambda module: -((module.weight.item() - 1.0) ** 2),
        lambda module: module.weight.item() ** 2,
        config=ESConfig(population=8, generations=4, sigma=0.1, learning_rate=0.01),
    )
    assert result.selected_generation == 0
    assert result.selection_loss == 0
    assert torch.equal(model.weight, original["weight"])
    assert torch.equal(result.selected_state["weight"], original["weight"])
    assert all(row["validation_loss"] > 0 for row in result.history[1:])


def test_seeded_es_is_reproducible_without_changing_global_random_states():
    model = _model(0.125)
    rng_python, rng_torch = random.getstate(), torch.random.get_rng_state().clone()
    config = ESConfig(population=4, generations=2, sigma=0.1, learning_rate=0.01, seed=7)
    results = []
    for _ in range(2):
        run = copy.deepcopy(model)
        results.append(
            evolution_strategy(
                run,
                lambda module: -((module.weight.item() - 1.0) ** 2),
                lambda module: (module.weight.item() - 1.0) ** 2,
                config=config,
            )
        )
    assert results[0].history == results[1].history
    assert torch.equal(results[0].selected_state["weight"], results[1].selected_state["weight"])
    assert random.getstate() == rng_python
    assert torch.equal(torch.random.get_rng_state(), rng_torch)


def test_zero_generations_has_only_an_initial_validation_evaluation():
    model = _model(0.333)

    def forbidden(_):
        raise AssertionError("No proposal should be evaluated.")

    result = evolution_strategy(model, forbidden, lambda _: 1.0, config=ESConfig(generations=0))
    assert result.selected_generation == result.candidate_evaluations == 0
    assert len(result.history) == 1
    assert model.weight.item() == 0.333


def test_flat_population_keeps_exact_state_and_ties_keep_epoch_zero():
    model = _model(0.333)
    original = model.weight.detach().clone()
    result = evolution_strategy(
        model,
        lambda _: 5.0,
        lambda _: 1.0,
        config=ESConfig(population=4, generations=3, sigma=0.1, learning_rate=0.01),
    )
    assert result.selected_generation == 0
    assert torch.equal(model.weight, original)
    assert all(row["status"] == "flat_rewards" for row in result.history[1:])
    assert all(row["distance_from_initial"] == 0 for row in result.history)


def test_proposals_restore_exact_parameters_and_buffers_even_after_impure_callback():
    model = _model(0.333)
    model.register_buffer("counter", torch.tensor(0))
    originals = copy.deepcopy(model.state_dict())
    seen = []

    def reward(module):
        assert module.counter.item() == 0
        seen.append(module.weight.item())
        with torch.no_grad():
            module.counter.add_(1)
            module.weight.add_(1000)
        return 0.0  # A flat generation must remain exactly at its original center.

    result = evolution_strategy(
        model,
        reward,
        lambda module: float(module.weight.square().sum()),
        config=ESConfig(population=4, generations=2, sigma=0.1, learning_rate=0.01),
    )
    assert len(seen) == 8
    assert max(abs(value - 0.333) for value in seen) < 0.5
    assert torch.equal(model.weight, originals["weight"])
    assert torch.equal(model.counter, originals["counter"])
    assert result.selected_generation == 0


def test_callback_exception_restores_original_state():
    model = _model(0.333)
    original = model.weight.detach().clone()

    def broken(module):
        with torch.no_grad():
            module.weight.add_(123)
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        evolution_strategy(model, broken, lambda _: 1.0)
    assert torch.equal(model.weight, original)
    assert not model.training


def test_nonfinite_rewards_and_validation_are_rejected_and_json_stays_finite():
    model = _model(0.125)
    validations = iter([1.0, float("nan"), 2.0])
    result = evolution_strategy(
        model,
        lambda _: float("inf"),
        lambda _: next(validations),
        config=ESConfig(population=4, generations=2),
    )
    assert result.selected_generation == 0
    assert result.history[1]["status"] == "rejected_nonfinite_validation"
    assert result.history[2]["status"] == "insufficient_finite_rewards"
    assert all(value is None for value in result.history[1]["candidate_rewards"])
    assert model.weight.item() == 0.125
    json.dumps(result.history, allow_nan=False)
    with pytest.raises(ValueError, match="initial validation"):
        evolution_strategy(model, lambda _: 1, lambda _: float("nan"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population": 1},
        {"population": True},
        {"generations": -1},
        {"sigma": 0},
        {"learning_rate": float("nan")},
        {"seed": -1},
    ],
)
def test_invalid_es_settings_fail_before_any_evaluation(kwargs):
    with pytest.raises(ValueError):
        ESConfig(**kwargs)


def test_error_direction_uses_extreme_losses_and_handles_degeneracy():
    hidden = torch.tensor([[0.0, 1.0], [7.0, 7.0], [8.0, 8.0], [2.0, 1.0]])
    losses = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert torch.equal(diagnose_error_direction(hidden, losses), torch.tensor([1.0, 0.0]))
    assert torch.equal(diagnose_error_direction(hidden, torch.ones(4)), torch.zeros(2))
    assert torch.equal(diagnose_error_direction(torch.ones(4, 2), losses), torch.zeros(2))


def test_directional_edit_matches_affine_output_attenuation_and_preserves_other_layer():
    model = nn.Sequential(nn.Linear(3, 2, dtype=torch.float64), nn.Linear(2, 1))
    original = copy.deepcopy(model.state_dict())
    inputs = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], dtype=torch.float64)
    before = model[0](inputs).detach()
    direction = torch.tensor([3.0, 4.0], dtype=torch.float64)
    unit = direction / 5
    expected = before - 0.25 * (before @ unit).unsqueeze(-1) * unit
    report = apply_directional_edit(model, "0", direction, 0.25)
    assert torch.allclose(model[0](inputs), expected, atol=1e-15, rtol=1e-15)
    assert torch.equal(model[1].weight, original["1.weight"])
    assert torch.equal(model[1].bias, original["1.bias"])
    assert 0 < report["relative_weight_change"] <= 0.25 + 1e-15
    assert report["direction_norm"] == 5
    assert report["degenerate"] is False
    json.dumps(report, allow_nan=False)


def test_zero_direction_and_zero_alpha_are_exact_noops():
    model = nn.Sequential(nn.Linear(3, 2))
    original = copy.deepcopy(model.state_dict())
    for direction, alpha in [(torch.zeros(2), 0.5), (torch.ones(2), 0.0)]:
        report = apply_directional_edit(model, "0", direction, alpha)
        assert report["relative_weight_change"] == 0
        assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize(
    "hidden,losses",
    [
        (torch.ones(1, 2), torch.ones(1)),
        (torch.ones(2, 2), torch.ones(3)),
        (torch.ones(2, 2), torch.tensor([0.0, float("nan")])),
        (torch.ones(2, 2), torch.tensor([0.0, -1.0])),
    ],
)
def test_invalid_diagnostics_are_rejected(hidden, losses):
    with pytest.raises(ValueError):
        diagnose_error_direction(hidden, losses)


def test_invalid_edit_does_not_mutate_parameters():
    model = nn.Sequential(nn.Linear(3, 2))
    original = copy.deepcopy(model.state_dict())
    for direction, alpha in [(torch.ones(3), 0.1), (torch.ones(2), -0.1), (torch.ones(2), 1.1)]:
        with pytest.raises(ValueError):
            apply_directional_edit(model, "0", direction, alpha)
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
