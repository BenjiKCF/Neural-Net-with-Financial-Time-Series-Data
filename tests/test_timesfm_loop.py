"""Loop mechanics using a tiny fake checkpoint; no TimesFM download/import needed."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from financial_ts import timesfm_loop
from financial_ts.timesfm_loop import LoopTimesFM3, _ProgressiveLoopStack


class TinyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        self.calls = 0
        self.metadata = None

    def forward(
        self,
        hidden,
        patch_mask,
        segment_ids=None,
        segment_pos=None,
        decode_cache=None,
        var_segment_pos=None,
    ):
        self.calls += 1
        self.metadata = (patch_mask, segment_ids, segment_pos, var_segment_pos)
        valid = (~patch_mask).unsqueeze(-1)
        clean = hidden.masked_fill(~valid, 0)
        history = clean.cumsum(dim=2) / valid.cumsum(dim=2).clamp_min(1)
        result = hidden + 0.1 * self.projection(torch.tanh(history))
        return result.masked_fill(~valid, 0), None, patch_mask


class TinyStack(nn.Module):
    def __init__(self, width=8, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(width) for _ in range(depth)])

    def forward(
        self,
        hidden,
        patch_mask,
        segment_ids=None,
        segment_pos=None,
        decode_cache=None,
        var_segment_pos=None,
    ):
        masks = []
        for layer in self.layers:
            hidden, _, mask = layer(
                hidden, patch_mask, segment_ids, segment_pos, None, var_segment_pos
            )
            masks.append(mask)
        return hidden, [None] * len(self.layers), masks


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_config = SimpleNamespace(transformer=SimpleNamespace(model_dims=8))
        self.input_projection = nn.Linear(1, 8)
        self.transformer_stack = TinyStack()
        self.output_head = nn.Linear(8, 2)

    @torch.no_grad()
    def decode(self, target, horizon):
        assert horizon == 2
        tokens = self.input_projection(target.unsqueeze(-1) / 100)
        hidden, _, _ = self.transformer_stack(
            tokens, torch.zeros(tokens.shape[:-1], dtype=torch.bool, device=target.device)
        )
        median = target[..., -1:] + self.output_head(hidden[:, :, -1])
        # Deliberate quantile crossing checks the public forecaster's sorting rule.
        return torch.stack((median + 0.3, median - 0.2, median), dim=-1)


@pytest.fixture
def backend_factory(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"test_only":true}')
    (tmp_path / "model.safetensors").write_bytes(b"fake; never loaded")
    torch.manual_seed(73)
    original = TinyModel()
    monkeypatch.setattr(
        timesfm_loop,
        "_load_official_model",
        lambda checkpoint, device: (copy.deepcopy(original), 1),
    )

    def make(**kwargs):
        defaults = {"loop_start": 1, "loop_end": 3, "adapter_rank": 2}
        defaults.update(kwargs)
        model_class = defaults.pop("model_class", LoopTimesFM3)
        return model_class(tmp_path, **defaults)

    return make


def contexts():
    return torch.tensor([[100.0, 101.0, 99.0, 102.0], [120.0, 118.0, 121.0, 122.0]])


def test_initial_conversion_preserves_the_original_forecast_exactly(backend_factory):
    model = backend_factory()
    baseline = model.baseline_forecast(contexts())
    converted = model.forecast(contexts())
    assert converted.requires_grad
    assert torch.equal(converted, baseline)
    assert torch.equal(model.baseline_return_scores(contexts()), model.return_scores(contexts()))
    assert converted.shape == (2, 2)
    expected_scores = converted.double()[:, 1] / converted.double()[:, 0] - 1
    torch.testing.assert_close(model.return_scores(contexts()), expected_scores, rtol=0, atol=0)
    assert model.trainable_parameter_count == 41
    assert model.parameter_count == sum(parameter.numel() for parameter in model.parameters())
    assert model.checkpoint_size_bytes == 18
    assert len(model.checkpoint_config_sha256) == 64


def test_loop_reuses_layer_objects_and_preserves_positions_and_masks(backend_factory):
    model = backend_factory(loop_steps=3)
    stack = model.loop_stack
    original_ids = [id(layer) for layer in stack.base.layers]
    tokens = torch.randn(2, 1, 7, 8)
    padding = torch.zeros(2, 1, 7, dtype=torch.bool)
    padding[:, :, :2] = True
    segment_ids = torch.zeros(2, 7, dtype=torch.long)
    positions = torch.arange(7).expand(2, -1)
    variate_positions = torch.zeros(14, 1, dtype=torch.long)
    stack(tokens, padding, segment_ids, positions, None, variate_positions)
    assert [layer.calls for layer in stack.base.layers] == [1, 3, 3, 1]
    assert [id(layer) for layer in stack.base.layers] == original_ids
    for layer in stack.base.layers:
        assert all(
            actual is expected
            for actual, expected in zip(
                layer.metadata, (padding, segment_ids, positions, variate_positions)
            )
        )
    with pytest.raises(ValueError, match="KV caches"):
        stack(tokens, padding, decode_cache=[object()] * 4)


def test_trained_gates_and_adapters_preserve_causality_and_padding(backend_factory):
    model = backend_factory(loop_steps=3)
    stack = model.loop_stack
    with torch.no_grad():
        for refinement in stack.refinements:
            refinement.gate.weight.fill_(0.1)
            refinement.adapter_up.weight.fill_(0.05)
    tokens = torch.randn(2, 1, 9, 8)
    padding = torch.zeros(2, 1, 9, dtype=torch.bool)
    padding[0, :, :2] = True
    expected = stack(tokens, padding)[0]
    changed = tokens.clone()
    changed[:, :, 6:] += 100
    actual = stack(changed, padding)[0]
    torch.testing.assert_close(actual[:, :, :6], expected[:, :, :6], rtol=0, atol=0)
    changed = tokens.clone()
    changed[padding] = 1e6
    actual = stack(changed, padding)[0]
    torch.testing.assert_close(actual[~padding], expected[~padding], rtol=0, atol=0)


def test_training_updates_only_adapters_and_all_paths_receive_finite_gradients(backend_factory):
    model = backend_factory().train()
    base = list(model.base_parameters())
    before = [parameter.detach().clone() for parameter in base]
    learned = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in base}.isdisjoint(
        {id(parameter) for parameter in learned}
    )
    assert not model.model.training
    assert model.loop_stack.refinements.training
    optimizer = torch.optim.Adam(learned, lr=0.01)
    initial = model.forecast(contexts()).detach()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (
            ((model.return_scores(contexts()) - torch.tensor([0.02, -0.01])) / 0.01).square().mean()
        )
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in learned
        )
        optimizer.step()
    refinement = model.loop_stack.refinements[0]
    assert refinement.gate.weight.grad.abs().sum() > 0
    assert refinement.adapter_up.weight.grad.abs().sum() > 0
    assert refinement.adapter_down.weight.grad.abs().sum() > 0
    for expected, parameter in zip(before, base):
        assert not parameter.requires_grad and parameter.grad is None
        assert torch.equal(parameter, expected)
    assert not torch.equal(model.forecast(contexts()), initial)
    assert torch.equal(model.baseline_forecast(contexts()), initial)


def test_adapter_only_reload_is_strict_and_reproduces_forecasts(backend_factory):
    first = backend_factory()
    with torch.no_grad():
        first.loop_stack.refinements[0].gate.bias.fill_(0.2)
        first.loop_stack.refinements[0].adapter_up.weight.fill_(0.05)
    state = first.adapter_state_dict()
    expected_keys = {
        name for name, parameter in first.named_parameters() if parameter.requires_grad
    }
    assert set(state) == expected_keys
    assert all(".refinements." in name for name in state)
    assert sum(value.numel() for value in state.values()) == first.trainable_parameter_count
    second = backend_factory()
    second.load_adapter_state_dict(state)
    torch.testing.assert_close(
        first.forecast(contexts()), second.forecast(contexts()), rtol=0, atol=0
    )
    before = second.adapter_state_dict()
    bad = dict(state)
    bad[next(iter(bad))] = torch.full_like(bad[next(iter(bad))], torch.nan)
    with pytest.raises(ValueError, match="Invalid adapter"):
        second.load_adapter_state_dict(bad)
    for name, value in before.items():
        assert torch.equal(second.adapter_state_dict()[name], value)
    with pytest.raises(ValueError, match="keys"):
        second.load_adapter_state_dict({**state, "base.weight": torch.ones(1)})
    incompatible = backend_factory(adapter_rank=1)
    with pytest.raises(ValueError, match="Invalid adapter"):
        incompatible.load_adapter_state_dict(state)


def test_baseline_mode_is_restored_when_forecast_fails(backend_factory):
    model = backend_factory()
    with pytest.raises(ValueError, match="positive"):
        model.baseline_forecast(torch.zeros(2, 4))
    assert model.loop_stack.refinement_enabled


@pytest.mark.parametrize("mode", ["loop", "adapter"])
@pytest.mark.parametrize("constant_context", [False, True])
@pytest.mark.parametrize("zero_unused_variance", [False, True])
def test_unused_cpm_refinement_preserves_forecasts_and_finite_gradients(
    backend_factory, mode, constant_context, zero_unused_variance
):
    from financial_ts.timesfm_comparison import GatedAdapterTimesFM3

    model = backend_factory(model_class=LoopTimesFM3 if mode == "loop" else GatedAdapterTimesFM3)
    model.model.input_patch_len = 32
    model.model.output_patch_len = 64
    model.model.use_iterative_cpm_revin = True
    original_decode = model._decode_with_grad

    def decode_with_unused_refinement(official, *, target, horizon):
        logits = original_decode(official, target=target, horizon=horizon)
        if official.use_iterative_cpm_revin:
            # Model the pinned implementation's masked, unused future-patch
            # statistics. sqrt(0) has an infinite derivative even when the
            # downstream where selects the observed context's statistics.
            variance_input = logits * (0 if zero_unused_variance else 1)
            unused_std = variance_input.square().sum().sqrt()
            logits = logits + torch.where(
                torch.tensor(False), unused_std, torch.zeros_like(unused_std)
            )
        return logits

    model._decode_with_grad = decode_with_unused_refinement
    values = torch.full_like(contexts(), 100) if constant_context else contexts()
    unrepaired = (
        decode_with_unused_refinement(model.model, target=values.unsqueeze(1), horizon=2)
        .sort(dim=-1)
        .values[:, 0, :, model.median_quantile_index]
    )
    assert torch.isfinite(unrepaired).all()
    unrepaired.square().mean().backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    old_gradients = [parameter.grad.clone() for parameter in parameters]
    assert all(torch.isfinite(gradient).all() for gradient in old_gradients) != zero_unused_variance
    model.zero_grad(set_to_none=True)

    repaired = model.forecast(values)
    assert torch.equal(unrepaired, repaired)
    repaired.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )
    if not zero_unused_variance:
        for expected, parameter in zip(old_gradients, parameters):
            assert torch.equal(expected, parameter.grad)
    assert all(parameter.grad is None for parameter in model.base_parameters())
    assert model.model.use_iterative_cpm_revin is True


@pytest.mark.parametrize("initial_flag", [False, True])
def test_cpm_configuration_is_restored_after_decode_error(backend_factory, initial_flag):
    model = backend_factory()
    model.model.input_patch_len = 32
    model.model.output_patch_len = 64
    model.model.use_iterative_cpm_revin = initial_flag

    def failed_decode(official, *, target, horizon):
        assert official.use_iterative_cpm_revin is False
        raise RuntimeError("decode failed")

    model._decode_with_grad = failed_decode
    with pytest.raises(RuntimeError, match="decode failed"):
        model.forecast(contexts())
    assert model.model.use_iterative_cpm_revin is initial_flag


@pytest.mark.parametrize("patch_sizes", [(1, 64), (32, 1)])
def test_cpm_bypass_requires_horizon_within_one_context_patch(backend_factory, patch_sizes):
    model = backend_factory()
    model.model.input_patch_len, model.model.output_patch_len = patch_sizes
    model.model.use_iterative_cpm_revin = True
    original_decode = model._decode_with_grad

    def checked_decode(official, *, target, horizon):
        assert official.use_iterative_cpm_revin is True
        return original_decode(official, target=target, horizon=horizon)

    model._decode_with_grad = checked_decode
    assert torch.isfinite(model.forecast(contexts())).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"loop_start": 2, "loop_end": 2},
        {"loop_end": 5},
        {"loop_steps": 0},
        {"adapter_rank": 9},
        {"gate_scale": float("nan")},
        {"gate_scale": 0},
    ],
)
def test_invalid_loop_configuration_fails(backend_factory, kwargs):
    with pytest.raises(ValueError):
        backend_factory(**kwargs)


def test_single_pass_remains_the_original_model(backend_factory):
    model = backend_factory(loop_steps=1)
    assert model.trainable_parameter_count == 0
    assert model.adapter_state_dict() == {}
    assert torch.equal(model.forecast(contexts()), model.baseline_forecast(contexts()))


def test_invalid_patch_shape_is_rejected():
    stack = _ProgressiveLoopStack(TinyStack(), 8, 1, 3, 2, 2, 0.1)
    with pytest.raises(ValueError, match="patch_mask"):
        stack(torch.randn(2, 1, 4, 8), torch.zeros(2, 4, dtype=torch.bool))
