"""Behavioral checks for the independently implemented GLM/Qwen adaptations."""

import pytest

torch = pytest.importorskip("torch")

from financial_ts.architectures.glm import GLMMiniEncoder, SparseMoE, _SparseLoopAttention  # noqa: E402
from financial_ts.architectures.qwen import BlockSparseAttention, GatedDeltaMixer, QwenMiniEncoder  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _model(cls, **kwargs):
    torch.manual_seed(17)
    return cls(d_feat=5, d_model=16, nhead=2, num_layers=1, dropout=0, **kwargs)


@pytest.mark.parametrize("cls", [GLMMiniEncoder, QwenMiniEncoder])
def test_singleton_all_missing_and_padding_invariance(cls):
    model = _model(cls).eval()
    values = torch.randn(1, 13, 5)
    padding = torch.zeros(1, 13, dtype=torch.bool)
    padding[:, [0, 1, 4]] = True
    altered = values.clone()
    altered[padding] = torch.nan
    expected = model(values, padding)
    actual = model(altered, padding)
    assert expected.shape == (1,)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    empty = model(altered, torch.ones_like(padding))
    assert empty.shape == (1,) and torch.isfinite(empty).all()


@pytest.mark.parametrize("cls", [GLMMiniEncoder, QwenMiniEncoder])
def test_future_features_cannot_change_earlier_hidden_states(cls):
    model = _model(cls).eval()
    values = torch.randn(2, 21, 5)
    padding = torch.zeros(2, 21, dtype=torch.bool)
    padding[0, :3] = True
    altered = values.clone()
    altered[:, 10:] = 20 * torch.randn_like(altered[:, 10:])
    original = model.forward_features(values, padding)
    changed = model.forward_features(altered, padding)
    torch.testing.assert_close(changed[:, :10], original[:, :10], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(changed[:, -1], original[:, -1])


def test_glm_reuses_the_body_indices_and_first_pass_kv():
    model = _model(GLMMiniEncoder, loop_steps=3)
    calls = []
    hook = model.blocks[0].register_forward_hook(lambda block, args, out: calls.append(id(block)))
    values = torch.randn(2, 14, 5)
    padding = torch.zeros(2, 14, dtype=torch.bool)
    model(values, padding)
    hook.remove()
    assert calls == [id(model.blocks[0])] * 3
    assert model.last_indexer_calls == 1
    assert model.blocks[0].attention.last_memory_tokens == 8 + 4

    one_pass = _model(GLMMiniEncoder, loop_steps=1)
    one_pass.load_state_dict(model.state_dict(), strict=False)
    # Only two additional scalar token gates, not duplicate attention/expert bodies.
    assert sum(p.numel() for p in model.parameters()) - sum(
        p.numel() for p in one_pass.parameters()
    ) == 2 * (2 * 16 + 1)
    assert not torch.allclose(model(values, padding), one_pass(values, padding))

    attention = _SparseLoopAttention(16, 2, sparse_k=3, local_window=2, dropout=0)
    hidden = torch.randn(2, 14, 16)
    _, cache = attention(hidden, padding)
    old_keys, old_indices = cache["keys"].clone(), cache["indices"].clone()
    _, next_cache = attention(hidden + 0.5, padding, cache)
    assert next_cache is cache and attention.indexer_calls == 1
    torch.testing.assert_close(cache["keys"], old_keys, rtol=0, atol=0)
    torch.testing.assert_close(cache["indices"], old_indices, rtol=0, atol=0)
    assert attention.last_memory_tokens == 5


def test_delta_mixer_skips_missing_observations_without_state_decay():
    torch.manual_seed(9)
    mixer = GatedDeltaMixer(16, 2).eval()
    values = torch.randn(1, 9, 16)
    padding = torch.tensor([[True, False, False, True, False, False, True, False, False]])
    full = mixer(values, padding)
    compact = mixer(values[:, ~padding[0]], torch.zeros(1, 6, dtype=torch.bool))
    torch.testing.assert_close(full[:, ~padding[0]], compact, rtol=1e-6, atol=1e-6)
    assert not full[:, padding[0]].any()


def test_sparse_block_attention_respects_causality_and_partial_block_length():
    torch.manual_seed(13)
    attention = BlockSparseAttention(16, 2, block_size=4, selected_blocks=2, dropout=0)
    values = torch.randn(2, 33, 16, requires_grad=True)
    padding = torch.zeros(2, 33, dtype=torch.bool)
    padding[0, :2] = True
    output = attention(values, padding)
    positions = torch.arange(33)[None, :, None].expand_as(attention.last_token_indices)
    assert (
        attention.last_token_indices[attention.last_allowed] <= positions[attention.last_allowed]
    ).all()
    # Two completed four-token blocks + exactly one current token, never four
    # duplicate copies of the incomplete final block's sole observation.
    assert attention.last_allowed[1, -1].sum().item() == 9
    assert attention.last_allowed[:, -1].sum(-1).max().item() < 33
    output[:, -1].square().sum().backward()
    assert attention.index_q.weight.grad.abs().sum() > 0
    assert attention.index_k.weight.grad.abs().sum() > 0


def test_qwen_has_three_delta_mixers_per_sparse_layer_and_trainable_stream_gates():
    model = _model(QwenMiniEncoder)
    assert [type(block.mixer) for block in model.blocks] == [GatedDeltaMixer] * 3 + [
        BlockSparseAttention
    ]
    assert tuple(model.stream_embeddings.shape) == (4, 16)
    output = model(torch.randn(4, 17, 5), torch.zeros(4, 17, dtype=torch.bool))
    (output.square().mean() + 0.01 * model.aux_loss).backward()
    assert model.local_mixer.weight.grad.abs().sum() > 0
    assert model.blocks[0].mixer.beta.weight.grad.abs().sum() > 0
    assert model.blocks[0].mixer.decay.weight.grad.abs().sum() > 0
    for gates in (model.blocks[0].read_gates, model.blocks[0].write_gates):
        assert all(gate.gate[-1].weight.grad.abs().sum() > 0 for gate in gates)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_sparse_expert_dispatch_excludes_padding_and_trains_router_and_experts():
    torch.manual_seed(37)
    moe = SparseMoE(16, num_experts=4, top_k=2)
    values = torch.randn(4, 17, 16)
    padding = torch.zeros(4, 17, dtype=torch.bool)
    padding[:, :3] = True
    output = moe(values, padding)
    (output.square().mean() + 0.01 * moe.aux_loss).backward()
    assert moe.expert_counts.sum().item() == int((~padding).sum()) * 2
    assert (moe.expert_counts > 0).all()
    assert moe.router.weight.grad.abs().sum() > 0
    assert all(expert.down.weight.grad.abs().sum() > 0 for expert in moe.experts)
    assert not output[padding].any()


@pytest.mark.parametrize("cls", [GLMMiniEncoder, QwenMiniEncoder])
def test_training_reduces_known_numeric_target_loss_and_state_dict_replays(cls):
    model = _model(cls)
    values = torch.randn(8, 12, 5)
    padding = torch.zeros(8, 12, dtype=torch.bool)
    target = 0.4 * values[:, -1, 0] - 0.2 * values[:, -2, 1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0)
    initial = ((model(values, padding) - target) ** 2).mean().item()
    for _ in range(12):
        optimizer.zero_grad()
        loss = ((model(values, padding) - target) ** 2).mean() + 0.001 * model.aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = ((model(values, padding) - target) ** 2).mean().item()
    assert final < initial * 0.8
    restored = _model(cls).eval()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        restored(values, padding), model.eval()(values, padding), rtol=0, atol=0
    )


@pytest.mark.parametrize("cls", [GLMMiniEncoder, QwenMiniEncoder])
def test_default_models_are_compact_and_dimension_errors_are_actionable(cls):
    model = cls()
    assert 50_000 < sum(parameter.numel() for parameter in model.parameters()) < 500_000
    with pytest.raises(ValueError, match="divisible"):
        cls(d_model=17, nhead=4)
    with pytest.raises(ValueError, match="padding"):
        _model(cls)(torch.randn(1, 3, 5), torch.zeros(1, 3))
