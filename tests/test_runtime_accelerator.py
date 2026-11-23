from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from pressb.runtime import accelerator


def test_cpu_registration_and_rng_do_not_import_an_accelerator(monkeypatch):
    monkeypatch.setattr(accelerator.importlib, "import_module", lambda name: pytest.fail(name))
    assert accelerator.initialize_device("cpu") == torch.device("cpu")
    assert accelerator.device_rng_payload("cpu") == {}
    accelerator.seed_device(43, "cpu")
    first = torch.rand(3)
    accelerator.seed_device(43, "cpu")
    assert torch.equal(torch.rand(3), first)


class Device:
    def __init__(self, value):
        if isinstance(value, Device):
            self.type, self.index = value.type, value.index
        else:
            parts = str(value).split(":")
            self.type = parts[0]
            self.index = int(parts[1]) if len(parts) > 1 else None

    def __str__(self):
        return self.type + (f":{self.index}" if self.index is not None else "")


def npu_fixture(monkeypatch):
    calls, states = [], {i: torch.tensor([i, 19], dtype=torch.uint8) for i in range(3)}
    current = [0]

    @contextmanager
    def device(value):
        old, current[0] = current[0], value.index
        try:
            yield
        finally:
            current[0] = old

    def get(value):
        calls.append(("get", str(value)))
        return states[value.index].clone()

    def put(value, *, device):
        calls.append(("set", str(device)))
        states[device.index] = value.clone()

    backend = SimpleNamespace(is_available=lambda: True, device_count=lambda: 3,
        current_device=lambda: current[0], set_device=lambda value: calls.append(("bind", str(value))),
        get_rng_state=get, set_rng_state=put, device=device,
        manual_seed=lambda seed: calls.append(("seed", current[0], seed)))
    fake_torch = SimpleNamespace(device=Device, npu=backend, random=torch.random)
    monkeypatch.setattr(accelerator, "torch", fake_torch)
    monkeypatch.setattr(accelerator.importlib, "import_module", lambda name: calls.append(("import", name)))
    return backend, calls, states


def test_npu_selection_and_rng_restore_only_touch_requested_device(monkeypatch):
    backend, calls, states = npu_fixture(monkeypatch)
    selected = accelerator.initialize_device("npu:1")
    assert str(selected) == "npu:1" and ("bind", "npu:1") in calls
    payload = accelerator.device_rng_payload(selected)
    assert set(payload) == {"npu_rng_device"}
    original_zero = states[0].clone()
    accelerator.restore_device_rng(payload, "npu:2")
    assert torch.equal(states[2], states[1]) and torch.equal(states[0], original_zero)
    accelerator.seed_device(42, "npu:2")
    assert ("seed", 2, 42) in calls
    assert not any(call[:1] in (("get_all",), ("set_all",)) for call in calls)


def test_npu_does_not_accept_cuda_rng_bytes(monkeypatch):
    _, _, states = npu_fixture(monkeypatch)
    with pytest.raises(ValueError, match="does not match npu"):
        accelerator.restore_device_rng({"cuda_rng_device": states[0]}, "npu:1")


def test_npu_unavailable_device_is_reported_before_binding(monkeypatch):
    backend, calls, _ = npu_fixture(monkeypatch)
    with pytest.raises(ValueError, match="unavailable"):
        accelerator.initialize_device("npu:3")
    assert not any(c[0] == "bind" for c in calls)
    backend.is_available = lambda: False
    with pytest.raises(RuntimeError, match="no Ascend"):
        accelerator.initialize_device("npu:0")
