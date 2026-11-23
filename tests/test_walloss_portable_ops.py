"""CPU numerical/backward coverage for the same ATen path used on Ascend."""
import importlib
import sys
from unittest.mock import patch

import pytest
import torch

from pressb import model_sources


@pytest.fixture
def ops(monkeypatch):
    # Source-selection tests intentionally exercise alternate roots. Do not
    # leave bundled import names in their process after these numeric checks.
    previous = set(sys.modules)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.setattr(model_sources, '_INSERTED', dict(model_sources._INSERTED))
    model_sources.activate_source('walloss')
    result = importlib.import_module('wall_x.fusions.ops')
    yield result
    for name in set(sys.modules) - previous:
        if name == 'wall_x' or name.startswith('wall_x.'):
            del sys.modules[name]


@pytest.mark.parametrize('trans_b', [False, True])
def test_dual_experts_support_different_widths_and_gradients(trans_b, ops):
    torch.manual_seed(41)
    a = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    wa = torch.randn((5, 6) if trans_b else (6, 5), dtype=torch.float64, requires_grad=True)
    wb = torch.randn((7, 6) if trans_b else (6, 7), dtype=torch.float64, requires_grad=True)
    actual = ops.asym_dual_gmm(a, b, wa, wb, trans_b)
    expected = (torch.nn.functional.linear(a, wa if trans_b else wa.T),
                torch.nn.functional.linear(b, wb if trans_b else wb.T))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.autograd.gradcheck(lambda *xs: ops.asym_dual_gmm(*xs, trans_b), (a, b, wa, wb))


def test_routed_experts_roundtrip_and_backpropagate_without_cuda_binary(ops):
    torch.manual_seed(11)
    value = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    weights = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
    routes = torch.tensor([1, 0, 1, 0, 0, 1, 1])
    real_import = importlib.import_module

    def reject_cuda(name, *args, **kwargs):
        assert name != 'wallx_csrc', 'CPU/NPU path must not load a CUDA binary'
        return real_import(name, *args, **kwargs)

    with patch('importlib.import_module', reject_cuda):
        permuted, rowmap = ops.permute(value, routes)
        # Expert 0 has three rows and expert 1 has four, each in source order.
        outputs = torch.cat((permuted[:3] @ weights[0], permuted[3:] @ weights[1]))
        result = ops.unpermute(outputs, rowmap, torch.ones(7, 1))
    expected = torch.stack([value[i] @ weights[int(route)] for i, route in enumerate(routes)])
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(result.square().sum(), (value, weights), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (value, weights))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


def test_topk_row_map_drop_and_probability_gradients(ops):
    x = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    routes = torch.tensor([[1, 0], [0, 1], [1, 0]])
    probs = torch.tensor([[.2, .8], [.4, .6], [.7, .3]], dtype=torch.float64, requires_grad=True)
    permuted, mapping = ops.permute(x, routes, num_out_tokens=4)
    torch.testing.assert_close(mapping, torch.tensor([3, 1, -1, 0, -1, 2], dtype=torch.int32))
    output = ops.unpermute(permuted, mapping, probs)
    expected = torch.stack((x[0], x[1] * probs[1, 0], x[2] * probs[2, 1]))
    # Row 0 retains both assignments, so explicitly include its probability sum.
    expected = torch.stack((x[0] * probs[0].sum(), expected[1], expected[2]))
    torch.testing.assert_close(output, expected)
    assert torch.autograd.gradcheck(lambda v, p: ops.unpermute(ops.permute(v, routes, 4)[0], mapping, p), (x, probs))


def test_rotary_forward_and_backward_preserve_deployed_cuda_bands(ops):
    torch.manual_seed(9)
    q = torch.randn(2, 3, 4, 12, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 1, 4, 12, dtype=torch.float64, requires_grad=True)
    c, s = [torch.randn(3, 2, 4, 12, dtype=torch.float64) for _ in range(2)]
    actual = ops.multimodal_rope(q, k, c, s, [1, 2, 3])
    sections = [0] * 2 + [1] * 4 + [2] * 6
    expected = []
    for value in (q, k):
        columns = []
        for col, section in enumerate(sections):
            paired = col + 6 if col < 6 else col - 6
            sign = -1 if col < 6 else 1
            columns.append(value[..., col] * c[section, :, :, col][:, None]
                           + sign * value[..., paired] * s[section, :, :, col][:, None])
        expected.append(torch.stack(columns, -1))
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.autograd.gradcheck(lambda a, b: ops.multimodal_rope(a, b, c, s, [1, 2, 3]), (q, k))


def test_vision_rotary_positions_have_block_then_pixel_order(ops):
    inv = torch.tensor([1., .1])
    grid = torch.tensor([[1, 4, 6], [2, 2, 2]])
    result = ops.rot_pos_emb(inv, grid, 2)
    expected = []
    for t, h, w in grid.tolist():
        for _ in range(t):
            for row in range(0, h, 2):
                for col in range(0, w, 2):
                    for dy in range(2):
                        for dx in range(2):
                            expected.append(torch.cat(((row + dy) * inv, (col + dx) * inv)))
    torch.testing.assert_close(result, torch.stack(expected), rtol=0, atol=0)


def test_vision_windows_multiple_images_frames_and_exact_multiples(ops):
    grid = torch.tensor([[1, 4, 6], [2, 4, 4]])
    indices, bounds = ops.get_window_index(grid, 8, 2, 2, 4)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 3, 4, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=torch.int32))
    torch.testing.assert_close(bounds, torch.tensor([0, 16, 24, 40, 56], dtype=torch.int32))


def test_multimodal_rope_index_keeps_physical_padding_until_model_correction(ops):
    ids = torch.tensor([[0, 0, 10, 11, 11, 11, 11, 9]])
    pos, delta = ops.get_rope_index(ids, 2, 11, 12, 10, 4.,
                                   image_grid_thw=torch.tensor([[1, 4, 4]]), attention_mask=ids.ne(0).long())
    expected = torch.tensor([[1, 1, 2, 3, 3, 3, 3, 5],
                             [1, 1, 2, 3, 3, 4, 4, 5],
                             [1, 1, 2, 3, 4, 3, 4, 5]])[:, None]
    torch.testing.assert_close(pos, expected)
    torch.testing.assert_close(delta, torch.tensor([[-2]]))


def test_text_only_and_video_position_indices(ops):
    ids = torch.tensor([[0, 0, 9, 8]])
    pos, delta = ops.get_rope_index(ids, 2, 11, 12, 10, 4., attention_mask=ids.ne(0).long())
    torch.testing.assert_close(pos, torch.tensor([1, 1, 0, 1])[None, None].expand(3, 1, -1))
    torch.testing.assert_close(delta, torch.tensor([[-2]]))
    video = torch.tensor([[10] + [12] * 8 + [9]])
    pos, delta = ops.get_rope_index(video, 2, 11, 12, 10, 4.,
                                   video_grid_thw=torch.tensor([[2, 4, 4]]), second_per_grid_ts=torch.tensor([.5]))
    torch.testing.assert_close(pos[0, 0], torch.tensor([0, 1, 1, 1, 1, 3, 3, 3, 3, 4]))
    torch.testing.assert_close(delta, torch.tensor([[-5]]))


def test_empty_grids_and_routing(ops):
    grid = torch.empty((0, 3), dtype=torch.int64)
    a, b = ops.get_window_index(grid, 8, 2, 2, 4)
    assert a.numel() == 0 and b.tolist() == [0]
    assert ops.rot_pos_emb(torch.ones(3), grid, 2).shape == (0, 6)
    x = torch.empty(0, 4)
    y, mapping = ops.permute(x, torch.empty(0, dtype=torch.long))
    assert y is x and mapping is None
