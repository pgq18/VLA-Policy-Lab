"""Scoped KV batching retains parameter layouts, bias algebra and autograd."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pressb.models.walloss import _npu_kv_projection_option

SOURCE = Path(__file__).resolve().parents[1] / 'src/pressb/model_sources/walloss/wall_x/model/npu_projection.py'
spec = importlib.util.spec_from_file_location('pressb_npu_projection_test', SOURCE)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


@pytest.mark.parametrize('shape', [(5,), (3, 5), (2, 3, 5), (2, 2, 3, 5)])
@pytest.mark.parametrize('with_bias', [False, True])
def test_batched_linear_matches_linear_with_input_weight_and_bias_gradients(shape, with_bias):
    torch.manual_seed(27)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(7, dtype=torch.float64, requires_grad=True) if with_bias else None
    actual = projection.batched_linear(x, weight, bias)
    expected = torch.nn.functional.linear(x, weight, bias)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    parameters = (x, weight, bias) if with_bias else (x, weight)
    left = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), parameters)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


def test_instance_configuration_keeps_state_names_objects_and_non_npu_forward():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(5, 5)
            self.k_proj = nn.Linear(5, 3)
            self.v_proj = nn.Linear(5, 3)
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([Block(), Block()])
    before = {k:v.clone() for k,v in model.state_dict().items()}
    identities = {k:id(v) for k,v in model.named_parameters()}
    x = torch.randn(2, 4, 5)
    outputs = [layer.self_attn.k_proj(x) for layer in model.model.layers]
    assert projection.configure_npu_kv_projection(model) == 4
    assert {k:id(v) for k,v in model.named_parameters()} == identities
    assert model.state_dict().keys() == before.keys()
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    for layer, expected in zip(model.model.layers, outputs):
        torch.testing.assert_close(layer.self_attn.k_proj(x), expected, rtol=0, atol=0)
        assert '_pressb_original_kv_forward' not in layer.self_attn.q_proj.__dict__
    cloned = copy.deepcopy(model)
    for layer in cloned.model.layers:
        assert layer.self_attn.k_proj.forward.__self__ is layer.self_attn.k_proj
        assert layer.self_attn.k_proj._pressb_original_kv_forward.__self__ is layer.self_attn.k_proj
    projection.configure_npu_kv_projection(model, 'linear')
    for layer in model.model.layers:
        assert '_pressb_original_kv_forward' not in layer.self_attn.k_proj.__dict__


@pytest.mark.parametrize('bad', ['fp32', 'auto', True, None, {}])
def test_projection_mode_is_explicit_and_validated(bad):
    with pytest.raises(ValueError, match='npu_kv_projection'):
        _npu_kv_projection_option({'npu_kv_projection':bad})


def test_unsupported_module_cannot_silently_drop_custom_forward():
    layer=SimpleNamespace(self_attn=SimpleNamespace(k_proj=nn.Identity(),v_proj=nn.Linear(2,2)))
    with pytest.raises(TypeError,match='native nn.Linear'):
        projection.configure_npu_kv_projection(SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    assert '_pressb_original_kv_forward' not in layer.self_attn.v_proj.__dict__
