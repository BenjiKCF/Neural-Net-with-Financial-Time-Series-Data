import pytest
import torch

from financial_ts.architectures.cross_stock import CrossStockEncoder


def model(**kwargs):
    torch.manual_seed(12)
    return CrossStockEncoder(
        d_feat=3, d_model=8, nhead=2, num_layers=1, dropout=0.0, **kwargs
    ).eval()


@pytest.mark.parametrize("num_experts", [1, 3])
def test_instrument_permutation_equivariance_and_padding(num_experts):
    net = model(num_experts=num_experts)
    x = torch.randn(4, 5, 3)
    padding = torch.zeros(4, 5, dtype=torch.bool)
    padding[1, :2] = True
    order = torch.tensor([2, 0, 3, 1])
    expected = net(x, padding)
    actual = net(x[order], padding[order])
    torch.testing.assert_close(actual, expected[order], atol=1e-6, rtol=1e-5)
    poisoned = x.clone()
    poisoned[padding] = float("nan")
    torch.testing.assert_close(net(poisoned, padding), expected)


def test_added_padded_instruments_cannot_change_real_scores_or_regimes():
    net = model(num_experts=3)
    x = torch.randn(3, 4, 3)
    mask = torch.zeros(3, 4, dtype=torch.bool)
    expected = net(x, mask)
    routing = net.routing_diagnostics_["weights"]
    expanded = torch.cat([x, torch.full((2, 4, 3), float("nan"))])
    padding = torch.cat([mask, torch.ones(2, 4, dtype=torch.bool)])
    actual = net(expanded, padding, torch.tensor([True, True, True, False, False]))
    torch.testing.assert_close(actual[:3], expected, atol=1e-6, rtol=1e-5)
    assert actual[3:].eq(0).all()
    assert net.routing_diagnostics_["weights"] == pytest.approx(routing)


def test_temporal_control_does_not_use_other_instruments():
    net = model(cross_stock_attention=False, market_context=False)
    x, mask = torch.randn(3, 4, 3), torch.zeros(3, 4, dtype=torch.bool)
    first = net(x, mask)[0].detach()
    x[1:] += 100
    torch.testing.assert_close(net(x, mask)[0], first)


def test_uniform_expert_control_and_monotonic_quantile_heads():
    net = model(num_experts=3, uniform_routing=True, horizons=(1, 5, 20), quantiles=(0.1, 0.5, 0.9))
    outputs = net.forward_heads(torch.randn(4, 6, 3), torch.zeros(4, 6, dtype=torch.bool))
    assert outputs["returns"].shape == (4, 3)
    assert outputs["quantiles"].shape == (4, 3, 3)
    assert (outputs["quantiles"].diff(dim=-1) > 0).all()
    torch.testing.assert_close(outputs["routing"], torch.full((3,), 1 / 3))
    assert net.routing_diagnostics_["effective_experts"] == pytest.approx(3)


def test_all_historical_tokens_missing_is_finite():
    net = model()
    scores = net(torch.full((2, 4, 3), float("nan")), torch.ones(2, 4, dtype=torch.bool))
    assert scores.isfinite().all()


def test_cross_stock_and_regime_parameters_receive_gradients():
    net = model(num_experts=3).train()
    output = net(torch.randn(4, 4, 3), torch.zeros(4, 4, dtype=torch.bool))
    (output.square().sum() + 0.01 * net.aux_loss).backward()
    for parameter in [
        net.feature_gate[0].weight,
        net.stock_layers[0].self_attn.in_proj_weight,
        net.router[0].weight,
        net.experts[0][0].weight,
    ]:
        assert parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0
