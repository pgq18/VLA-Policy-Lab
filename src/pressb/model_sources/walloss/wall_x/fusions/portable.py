"""ATen implementations of the bundled Wall CUDA operators for CPU and NPU.

Learned activations and gradients stay on their input device. Small integer
vision-grid/token metadata is assembled on the CPU, then copied to that device.
The multimodal positions deliberately preserve the deployed CUDA kernel's
physical left-padding offsets; the model already corrects them when batching.
"""
from __future__ import annotations

import torch


def asym_dual_gmm(input0, input1, weight0, weight1, trans_b=False):
    weights = (weight0.T, weight1.T) if trans_b else (weight0, weight1)
    return input0 @ weights[0], input1 @ weights[1]


def permute(input_act, indices, num_out_tokens=0, max_token_num=0):
    if input_act.ndim != 2 or indices.ndim not in (1, 2):
        raise ValueError('MoE permutation requires [tokens,width] activations and token routes')
    if indices.shape[0] != input_act.shape[0]:
        raise ValueError('MoE routes must have one row per input token')
    if not input_act.numel():
        return input_act, None
    routes = indices.reshape(len(indices), -1).to(device=input_act.device)
    topk = routes.shape[1]
    count = routes.numel() if not num_out_tokens else int(num_out_tokens)
    if not 0 <= count <= routes.numel():
        raise ValueError('num_out_tokens exceeds the expanded token count')
    # CUB's original radix sort is stable; ties retain token/top-K order.
    order = torch.argsort(routes.flatten(), stable=True)
    selected = order[:count]
    output = input_act.index_select(0, torch.div(selected, topk, rounding_mode='floor'))
    inverse = torch.full((routes.numel(),), -1, dtype=torch.int64, device=input_act.device)
    inverse.scatter_(0, selected, torch.arange(count, device=input_act.device))
    # CUDA stores row_id_map[top_k_index, source_token], flattened.
    row_map = inverse.reshape(-1, topk).transpose(0, 1).reshape(-1).to(torch.int32)
    return output, row_map


def unpermute(input_act, row_id_map, probs=None):
    if not input_act.numel():
        return input_act
    if probs is None:
        tokens, topk = row_id_map.numel(), 1
    else:
        if probs.ndim != 2 or row_id_map.numel() != probs.numel():
            raise ValueError('MoE probability and row-map shapes differ')
        tokens, topk = probs.shape
    mapping = row_id_map.to(device=input_act.device, dtype=torch.long).reshape(topk, tokens).T
    rows = input_act.index_select(0, mapping.clamp_min(0).flatten()).reshape(tokens, topk, -1)
    rows = torch.where(mapping[..., None] >= 0, rows, torch.zeros_like(rows))
    if topk == 1:
        # The original top-1 CUDA path has no probability weighting.
        return rows[:, 0]
    # Weighted top-K reductions accumulate in FP32 in the CUDA implementation.
    work_dtype = torch.float64 if input_act.dtype == torch.float64 else torch.float32
    rows = rows.to(work_dtype)
    if probs is not None:
        rows = rows * probs.to(device=input_act.device, dtype=work_dtype)[..., None]
    return rows.sum(1).to(input_act.dtype)


def multimodal_rope(q, k, cos, sin, mrope_section):
    if len(mrope_section) != 3 or sum(mrope_section) * 2 != q.shape[-1]:
        raise ValueError('mRoPE requires three sections covering half the head width')
    if q.shape[-1] != k.shape[-1] or cos.shape[0] != 3 or sin.shape != cos.shape:
        raise ValueError('Incompatible multimodal rotary shapes')
    # This is the deployed CUDA convention (three contiguous doubled bands),
    # not the six repeated bands of the separate Qwen eager-attention helper.
    sizes = [int(size) * 2 for size in mrope_section]
    c = torch.cat([part[i] for i, part in enumerate(cos.split(sizes, dim=-1))], -1).unsqueeze(1).detach()
    s = torch.cat([part[i] for i, part in enumerate(sin.split(sizes, dim=-1))], -1).unsqueeze(1).detach()

    def rotate(value):
        left, right = value.chunk(2, dim=-1)
        return value * c + torch.cat((-right, left), -1) * s

    return rotate(q), rotate(k)


def _grids(value, merge):
    if value.ndim != 2 or value.shape[1] != 3 or type(merge) is not int or merge < 1:
        raise ValueError('Expected [images,3] grids and a positive spatial merge size')
    rows = value.detach().cpu().tolist()
    for t, h, w in rows:
        if min(t, h, w) < 1 or h % merge or w % merge:
            raise ValueError('Vision grids must be positive and spatially divisible by merge size')
    return rows


def rot_pos_emb(inv_freq, grid_thw, spatial_merge_size):
    if inv_freq.ndim != 1 or inv_freq.device != grid_thw.device:
        raise ValueError('Inverse frequencies and grids must have compatible shapes/devices')
    positions = []
    merge = spatial_merge_size
    for t, h, w in _grids(grid_thw, merge):
        hp = torch.arange(h)[:, None].expand(h, w)
        wp = torch.arange(w)[None, :].expand(h, w)
        ordered = [v.reshape(h // merge, merge, w // merge, merge).permute(0, 2, 1, 3).flatten()
                   for v in (hp, wp)]
        positions.append(torch.stack(ordered, -1).repeat(t, 1))
    pos = (torch.cat(positions) if positions else torch.empty((0, 2), dtype=torch.long)).to(inv_freq.device)
    return (pos.float()[..., None] * inv_freq.float()[None, None, :]).flatten(1)


def get_window_index(grid_thw, window_size, spatial_merge_size, patch_size, spatial_merge_unit=1):
    window = window_size // spatial_merge_size // patch_size
    if window < 1 or spatial_merge_unit < 1:
        raise ValueError('Window width and spatial merge unit must be positive')
    indices, boundaries, offset = [], [0], 0
    for frames, height, width in _grids(grid_thw, spatial_merge_size):
        height, width = height // spatial_merge_size, width // spatial_merge_size
        for frame in range(frames):
            for row in range(0, height, window):
                for col in range(0, width, window):
                    values = [offset + frame * height * width + y * width + x
                              for y in range(row, min(row + window, height))
                              for x in range(col, min(col + window, width))]
                    indices.extend(values)
                    boundaries.append(boundaries[-1] + len(values) * spatial_merge_unit)
        offset += frames * height * width
    return (torch.tensor(indices, dtype=torch.int32, device=grid_thw.device),
            torch.tensor(boundaries, dtype=torch.int32, device=grid_thw.device))


def get_rope_index(input_ids, spatial_merge_size, image_token_id, video_token_id,
                   vision_start_token_id, tokens_per_second, image_grid_thw=None,
                   video_grid_thw=None, second_per_grid_ts=None, attention_mask=None):
    if input_ids.ndim != 2 or spatial_merge_size < 1 or tokens_per_second <= 0:
        raise ValueError('Invalid multimodal position-index inputs')
    batch, length = input_ids.shape
    ids = input_ids.detach().cpu()
    mask = torch.ones_like(ids) if attention_mask is None else attention_mask.detach().cpu()
    if mask.shape != ids.shape:
        raise ValueError('Position attention mask must match input IDs')
    if image_grid_thw is None and video_grid_thw is None:
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        delta = positions.max(-1, keepdim=True).values + 1 - length
        return positions[None].expand(3, -1, -1).to(input_ids.device), delta.to(input_ids.device)
    images = [] if image_grid_thw is None else _grids(image_grid_thw, spatial_merge_size)
    videos = [] if video_grid_thw is None else _grids(video_grid_thw, spatial_merge_size)
    seconds = None if second_per_grid_ts is None else second_per_grid_ts.detach().float().cpu().tolist()
    positions = torch.ones((3, batch, length), dtype=torch.long)
    deltas = torch.empty((batch, 1), dtype=torch.long)
    image_idx = video_idx = 0
    for row, tokens in enumerate(ids.tolist()):
        cursor = offset = scan = 0
        valid = mask[row].bool()
        while scan + 1 < length:
            if not valid[scan] or tokens[scan] != vision_start_token_id or tokens[scan + 1] not in (image_token_id, video_token_id):
                scan += 1
                continue
            start = scan + 1
            is_video = tokens[start] == video_token_id
            index, grids = (video_idx, videos) if is_video else (image_idx, images)
            if index >= len(grids):
                raise ValueError('Vision tokens exceed supplied grid metadata')
            frames, height, width = grids[index]
            height, width = height // spatial_merge_size, width // spatial_merge_size
            count = frames * height * width
            if start + count > length:
                raise ValueError('Vision grid token span exceeds sequence length')
            text = torch.arange(start - cursor) + offset
            positions[:, row, cursor:start] = text[None]
            offset += start - cursor
            interval = (seconds[index] if seconds is not None else 1.) if is_video else 0.
            temporal = (torch.arange(frames, dtype=torch.float32) * interval * float(tokens_per_second)).long()
            t = temporal[:, None, None].expand(frames, height, width)
            h = torch.arange(height)[None, :, None].expand(frames, height, width)
            w = torch.arange(width)[None, None, :].expand(frames, height, width)
            positions[:, row, start:start + count] = torch.stack((t.flatten(), h.flatten(), w.flatten())) + offset
            offset += max(height, width, int(temporal[-1]) + 1) if is_video else max(height, width)
            cursor = scan = start + count
            if is_video:
                video_idx += 1
            else:
                image_idx += 1
        positions[:, row, cursor:] = (torch.arange(length - cursor) + offset)[None]
        positions[:, row, ~valid] = 1
        maximum = int(positions[:, row, valid].max()) if valid.any() else -1
        deltas[row, 0] = maximum + 1 - length
    return positions.to(input_ids.device), deltas.to(input_ids.device)
