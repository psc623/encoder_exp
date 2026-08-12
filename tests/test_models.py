import pytest


torch = pytest.importorskip("torch")

from encoderbench.models import AttentionPoolHead, fixed_3d_position_encoding


def test_attention_pool_head_shape():
    tokens = torch.randn(2, 64, 32)
    head = AttentionPoolHead(32, torch.zeros(32), torch.ones(32))
    assert head(tokens).shape == (2, 2)


def test_fixed_position_encoding_is_deterministic():
    first = fixed_3d_position_encoding((4, 4, 4), 32)
    second = fixed_3d_position_encoding((4, 4, 4), 32)
    assert first.shape == (64, 32)
    assert torch.equal(first, second)
