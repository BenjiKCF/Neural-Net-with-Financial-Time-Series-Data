"""Behavioral checks for the DeepSeek V4.1-inspired numeric adaptation."""

import pytest
import torch

from financial_ts.architectures.deepseek import (
    DeepSeekMiniEncoder,
    _CompressedMemory,
    _HyperMix,
    _SparseMoE,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    defaults = dict(d_feat=6, d_model=16, nhead=2, num_layers=1, dropout=0.0)
    defaults.update(kwargs)
    return DeepSeekMiniEncoder(**defaults)


def test_singleton_and_entirely_padded_windows_are_finite():
    model = small_model().eval()
    values = torch.randn(1, 7, 6)
    for padding in (torch.zeros(1, 7, dtype=torch.bool), torch.ones(1, 7, dtype=torch.bool)):
        output = model(values, padding)
        assert output.shape == (1,)
        assert torch.isfinite(output).all()
        assert torch.isfinite(model.aux_loss)


def test_masked_values_cannot_change_predictions():
    torch.manual_seed(14)
    model = small_model().eval()
    values = torch.randn(2, 9, 6)
    padding = torch.zeros(2, 9, dtype=torch.bool)
    padding[0, :4] = True
    padding[1, :2] = True
    altered = values.clone()
    altered[padding] = 1e8
    with torch.no_grad():
        expected = model(values, padding)
        actual = model(altered, padding)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_encoder_and_decoder_are_causal_even_with_compressed_memory():
    torch.manual_seed(21)
    model = small_model(num_layers=2).eval()
    values = torch.randn(2, 11, 6)
    padding = torch.zeros(2, 11, dtype=torch.bool)
    changed = values.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 100
    with torch.no_grad():
        before = model.forward_features(values, padding)
        after = model.forward_features(changed, padding)
    torch.testing.assert_close(before[:, :6], after[:, :6], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_partial_compression_block_is_never_exposed_as_finished_history():
    memory = _CompressedMemory(2, 2, 2, compression=2)
    with torch.no_grad():
        memory.kv.weight.copy_(torch.eye(2))
    values = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [90.0, 100.0]]])
    result = memory(values, torch.zeros(1, 3, dtype=torch.bool))
    torch.testing.assert_close(result.values[0, 0, 0], torch.tensor([2.0, 4.0]))
    assert result.ends.tolist() == [1, 3]
    assert result.valid.tolist() == [[True, False]]


def test_decoder_reuses_encoder_kv_and_sparse_indices():
    model = small_model(num_layers=2).eval()
    memories, selections = [], []
    handles = [
        model.decoder_memory.register_forward_hook(
            lambda module, inputs, output: memories.append(output)
        )
    ]
    for block in model.decoder:
        handles.append(
            block.attention.register_forward_hook(
                lambda module, inputs, output: selections.append(output[1])
            )
        )
    model(torch.randn(2, 12, 6), torch.zeros(2, 12, dtype=torch.bool))
    for handle in handles:
        handle.remove()
    assert len(memories) == 1
    assert all(block.attention.memory is None for block in model.decoder)
    assert model.decoder[1].attention.index_query is None
    torch.testing.assert_close(selections[0], selections[1], rtol=0, atol=0)
    # No selected compressed block ends after its query timestamp.
    future = memories[0].ends[None, None, :] > torch.arange(12)[None, :, None]
    assert not (selections[0] & future).any()


def test_topk_routing_executes_multiple_experts_and_backpropagates():
    torch.manual_seed(5)
    model = _SparseMoE(4, num_experts=4, top_k=2)
    with torch.no_grad():
        model.router.weight.copy_(torch.eye(4) * 4)
    values = torch.eye(4)[None, :, :]
    predictions, penalty = model(values, torch.zeros(1, 4, dtype=torch.bool))
    (predictions.square().sum() + 0.01 * penalty).backward()
    assert model.last_counts.sum().item() == 8
    assert (model.last_counts > 0).all()
    for expert in [*model.experts, model.shared]:
        assert expert.down.weight.grad is not None
        assert expert.down.weight.grad.abs().sum() > 0
    assert model.router.weight.grad.abs().sum() > 0


def test_hyperconnections_preserve_stochastic_transport_and_gradients():
    torch.manual_seed(8)
    mixer = _HyperMix(8, streams=4)
    states = torch.randn(2, 5, 4, 8, requires_grad=True)
    read, write, transport = mixer.coefficients(states)
    torch.testing.assert_close(transport.sum(-1), torch.ones(2, 5, 4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(transport.sum(-2), torch.ones(2, 5, 4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(read.sum(-1), torch.ones(2, 5), atol=1e-6, rtol=1e-6)
    updated = mixer.write(states, mixer.read(states, read), write, transport)
    updated.square().mean().backward()
    assert torch.isfinite(states.grad).all()
    assert mixer.gates.weight.grad.abs().sum() > 0


def test_sparse_indexer_receives_auxiliary_learning_signal():
    torch.manual_seed(9)
    model = small_model()
    output = model(torch.randn(4, 12, 6), torch.zeros(4, 12, dtype=torch.bool))
    (output.square().mean() + 0.01 * model.aux_loss).backward()
    for block in [*model.encoder, *model.decoder]:
        indexer = block.attention.index_query
        if indexer is not None:
            assert indexer.weight.grad is not None
            assert indexer.weight.grad.abs().sum() > 0


def test_tiny_model_can_fit_a_numeric_training_signal():
    torch.manual_seed(33)
    model = small_model(num_experts=2, top_k=2)
    values = torch.randn(16, 8, 6)
    padding = torch.zeros(16, 8, dtype=torch.bool)
    target = values[:, -1, 0] * 0.4 - values[:, -2, 1] * 0.2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = torch.nn.functional.mse_loss(model(values, padding), target).item()
    for _ in range(12):
        optimizer.zero_grad()
        output = model(values, padding)
        loss = torch.nn.functional.mse_loss(output, target) + 0.01 * model.aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = torch.nn.functional.mse_loss(model(values, padding), target).item()
    assert final < initial * 0.6


def test_parameter_budget_scales_with_width_and_depth():
    small = sum(parameter.numel() for parameter in small_model().parameters())
    default = sum(parameter.numel() for parameter in DeepSeekMiniEncoder().parameters())
    assert 10000 < small < default < 500000


@pytest.mark.parametrize("kwargs", [{"top_k": 5}, {"d_model": 15}, {"num_layers": 0}])
def test_invalid_architecture_sizes_fail_early(kwargs):
    with pytest.raises(ValueError):
        small_model(**kwargs)
