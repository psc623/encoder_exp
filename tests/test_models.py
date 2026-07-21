import pytest


torch = pytest.importorskip("torch")

from encoderbench.models import (AttentionPoolHead, FactorizedLinearBridge, ResamplerBridge,
                                 fixed_3d_position_encoding)


def test_model_shapes_and_linear_bridge_has_no_activation():
    tokens = torch.randn(2, 64, 32)
    head = AttentionPoolHead(32, torch.zeros(32), torch.ones(32))
    assert head(tokens).shape == (2, 2)
    linear = FactorizedLinearBridge(32, output_width=64, rank=16)
    assert linear(tokens).shape == (2, 64, 64)
    assert not any(isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh))
                   for module in linear.modules())
    resampler = ResamplerBridge(32, output_width=64, width=16, query_count=64,
                                layers=2, heads=4, ffn_size=32)
    assert resampler(tokens).shape == (2, 64, 64)


def test_fixed_position_encoding_is_deterministic():
    first = fixed_3d_position_encoding((4, 4, 4), 32)
    second = fixed_3d_position_encoding((4, 4, 4), 32)
    assert first.shape == (64, 32)
    assert torch.equal(first, second)
