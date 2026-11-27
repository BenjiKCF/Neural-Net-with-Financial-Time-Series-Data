import pytest
import torch

from financial_ts.objectives import daily_objective, portfolio_turnover_surrogate, quantile_loss


@pytest.mark.parametrize("objective", ["mse", "huber", "ic", "ranking", "combined"])
def test_constant_predictions_have_finite_nonzero_gradient(objective):
    p = torch.zeros(4, requires_grad=True)
    result = daily_objective(p, torch.tensor([-1.0, -1.0, 0.0, 2.0]), objective=objective)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert result.counts["undefined_ic_dates"] == 0
    assert result.counts["valid_pairs"] == 5


def test_equal_date_weighting_masks_missing_targets_and_counts_constant_days():
    p = torch.tensor([[1.0, 1.0, 1.0], [3.0, 0.0, 0.0]], requires_grad=True)
    y = torch.tensor([[0.0, 0.0, 0.0], [0.0, float("nan"), float("nan")]])
    result = daily_objective(p, y, objective="mse")
    assert result.loss.item() == 5  # mean of daily MSE 1 and 9, not row-weighted 3
    assert result.counts["undefined_ic_dates"] == 2
    assert result.counts["no_ranking_pairs_dates"] == 2
    result.loss.backward()
    assert p.grad[1, 1:].eq(0).all()


def test_empty_date_is_counted_and_all_empty_is_rejected():
    result = daily_objective(
        torch.zeros(2, 2), torch.tensor([[float("nan"), float("nan")], [0.0, 1.0]])
    )
    assert result.counts["empty_dates"] == 1
    with pytest.raises(ValueError, match="No finite"):
        daily_objective(torch.zeros(2), torch.full((2,), float("nan")))


def test_ranking_ties_use_auxiliary_signal_instead_of_nan():
    p = torch.zeros(3, requires_grad=True)
    result = daily_objective(p, torch.ones(3), objective="ranking")
    result.loss.backward()
    assert result.counts["valid_pairs"] == 0
    assert p.grad.isfinite().all() and p.grad.abs().sum() > 0


def test_invalid_prediction_is_not_silently_dropped():
    with pytest.raises(ValueError, match="Predictions"):
        daily_objective(torch.tensor([float("nan"), 0.0]), torch.tensor([1.0, 2.0]))


def test_turnover_tracks_weights_cash_and_untradeable_holdings():
    scores = torch.tensor([0.0, 2.0, -1.0], requires_grad=True)
    prior = torch.tensor([0.3, 0.1, 0.1])
    turnover, weights = portfolio_turnover_surrogate(
        scores, prior, tradable=torch.tensor([False, True, True])
    )
    assert weights[0] == prior[0]
    assert weights.sum().item() == pytest.approx(1.0)
    expected = 0.5 * ((weights - prior).abs().sum() + 0.5)
    assert turnover.item() == pytest.approx(expected.item())
    turnover.backward()
    assert scores.grad[0] == 0 and scores.grad.isfinite().all()
    from_cash, _ = portfolio_turnover_surrogate(scores.detach(), torch.zeros(3))
    assert from_cash.item() == pytest.approx(1.0)


def test_pinball_uses_raw_quantile_levels_and_missing_mask():
    p = torch.tensor([[[-1.0, 1.0]], [[2.0, 3.0]]], requires_grad=True)
    loss = quantile_loss(p, torch.tensor([[0.0], [float("nan")]]), (0.1, 0.9))
    assert loss.item() == pytest.approx(0.1)
    loss.backward()
    assert p.grad[1].eq(0).all()
