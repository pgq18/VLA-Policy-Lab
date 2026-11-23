"""Explicit accelerator selection without replacing another backend's APIs."""
from __future__ import annotations

import importlib

import torch


def resolve_device(device):
    """Register Ascend before parsing its device string; CPU/CUDA stay native."""
    if str(device).split(":", 1)[0] == "npu":
        try:
            importlib.import_module("torch_npu")
        except ImportError as error:
            raise RuntimeError("NPU execution requires a matching torch_npu installation") from error
    return torch.device(device)


def initialize_device(device):
    """Bind the explicitly selected NPU; never redirect CUDA or CPU operations."""
    device = resolve_device(device)
    if device.type == "npu":
        backend = torch.npu
        if not backend.is_available():
            raise RuntimeError("NPU execution requested, but no Ascend device is available")
        index = backend.current_device() if device.index is None else device.index
        if index < 0 or index >= backend.device_count():
            raise ValueError(f"Requested NPU device {device} is unavailable")
        device = torch.device(f"npu:{index}")
        backend.set_device(device)
    return device


def device_module(device):
    device = resolve_device(device)
    return getattr(torch, device.type) if device.type in ("cuda", "npu") else None


def seed_device(seed, device):
    """Seed CPU initialization and this accelerator, without seeding other cards."""
    device = resolve_device(device)
    torch.random.default_generator.manual_seed(seed)
    backend = device_module(device)
    if backend is not None:
        with backend.device(device):
            backend.manual_seed(seed)


def device_rng_state(device):
    device = resolve_device(device)
    backend = device_module(device)
    return backend.get_rng_state(device) if backend is not None else None


def device_rng_payload(device):
    device = resolve_device(device)
    state = device_rng_state(device)
    # Preserve the CUDA checkpoint key for existing readers. Ascend has its own
    # RNG format and must never be serialized or restored as a CUDA generator.
    return {f"{device.type}_rng_device": state} if state is not None else {}


def restore_device_rng(payload, device):
    device = resolve_device(device)
    backend = device_module(device)
    if backend is None:
        return
    key = f"{device.type}_rng_device"
    if key in payload:
        backend.set_rng_state(payload[key].cpu(), device=device)
    elif device.type == "cuda" and "cuda_rng" in payload:
        # Legacy all-visible-device checkpoints: restore only the selected card.
        index = device.index if device.index is not None else backend.current_device()
        backend.set_rng_state(payload["cuda_rng"][index].cpu(), device=device)
    elif any(name.endswith("_rng_device") or name == "cuda_rng" for name in payload):
        raise ValueError(f"Checkpoint accelerator RNG does not match {device.type}; cannot resume its random stream")
