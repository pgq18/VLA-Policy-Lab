"""Batch-preserving KV projections for native Ascend BF16 execution.

On the qualified 910C/Torch-NPU stack, flattening [B,T,2048] into one matrix
for a 256-wide KV projection made results depend on the observation's batch
slot. Baddbmm keeps the batch axis and fuses the original bias addition. It
does not change parameters, dtype, or the CUDA/CPU computation path.
"""
from __future__ import annotations

from types import MethodType
import math

import torch
from torch import nn


def batched_linear(value, weight, bias=None):
    """Linear algebra with an explicit batch axis and a fused bias addition."""
    if value.ndim < 1 or weight.ndim != 2:
        raise ValueError('Linear input must have a feature axis and a matrix weight')
    shape = value.shape
    if value.ndim == 1:
        inputs = value.reshape(1, 1, shape[-1])
    elif value.ndim == 2:
        inputs = value.unsqueeze(0)
    else:
        inputs = value.reshape(math.prod(shape[:-2]), shape[-2], shape[-1])
    batch, length, _ = inputs.shape
    weights = weight.T.unsqueeze(0).expand(batch, -1, -1)
    if bias is None:
        result = torch.bmm(inputs, weights)
    else:
        biases = bias[None, None, :].expand(batch, length, -1)
        result = torch.baddbmm(biases, inputs, weights)
    return result.reshape(*shape[:-1], weight.shape[0])


def _projection_forward(self, value):
    if value.device.type == 'npu':
        return batched_linear(value, self.weight, self.bias)
    return self._pressb_original_kv_forward(value)


def configure_npu_kv_projection(model, mode='baddbmm'):
    """Scope the NPU implementation to this model's existing K/V modules."""
    if mode not in ('linear', 'baddbmm'):
        raise ValueError('npu_kv_projection must be linear or baddbmm')
    projections = [getattr(layer.self_attn, name)
                   for layer in model.model.layers for name in ('k_proj', 'v_proj')]
    if any(type(module) is not nn.Linear for module in projections):
        raise TypeError('NPU KV projection adaptation requires native nn.Linear modules')
    for module in projections:
        original = getattr(module, '_pressb_original_kv_forward', None)
        if mode == 'baddbmm':
            if original is None:
                module._pressb_original_kv_forward = module.forward
            module.forward = MethodType(_projection_forward, module)
        elif original is not None:
            module.forward = original
            del module._pressb_original_kv_forward
    return len(projections)
