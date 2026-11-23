"""Common CPU/GPU solver configuration without opening an Isaac application."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from pressb.runtime.physics import (GPU_CAPACITY_DEFAULTS, PhysXErrorGuard, configure_physx,
                                    gpu_dynamics_memory_config, inspect_physx, normalize_physics_config)
from pressb.runtime.parallel import execution_config, SharedWorldBackend
from pressb.runtime.isaac_parallel import IsaacParallelWorld


class Settings(dict):
    def set_bool(self, key, value):
        self[key] = value


class Context:
    def __init__(self):
        self.gpu, self.broadphase, self.ccd = False, "MBP", True
        self.capacities = {}
        self.calls = []

    def set_broadphase_type(self, value):
        self.calls.append(("broadphase", value))
        self.broadphase = value

    def get_broadphase_type(self):
        return self.broadphase

    def enable_gpu_dynamics(self, value):
        self.calls.append(("gpu", value))
        self.gpu = value

    def is_gpu_dynamics_enabled(self):
        return self.gpu

    def enable_ccd(self, value):
        self.calls.append(("ccd", value))
        self.ccd = value

    def is_ccd_enabled(self):
        return self.ccd

    def __getattr__(self, name):
        if name.startswith("set_gpu_"):
            return lambda value: self.capacities.__setitem__(name.removeprefix("set_gpu_"), value)
        if name.startswith("get_gpu_"):
            return lambda: self.capacities[name.removeprefix("get_gpu_")]
        raise AttributeError(name)


@pytest.mark.parametrize("raw", [{}, {"physics_backend": "cpu"}, {"gpu_dynamics": False},
                                  {"physics_backend": "cpu", "gpu_dynamics": False}])
def test_cpu_is_default_and_legacy_false_alias_is_equivalent(raw):
    assert normalize_physics_config(raw) == {
        "physics_backend": "cpu", "gpu_dynamics": False, "gpu_memory_capacities": {}}
    cfg = execution_config(raw)
    assert cfg["physics_backend"] == "cpu" and cfg["gpu_dynamics"] is False


@pytest.mark.parametrize("raw", [{"physics_backend": "gpu"}, {"gpu_dynamics": True},
                                  {"physics_backend": "gpu", "gpu_dynamics": True}])
def test_gpu_alias_and_canonical_settings_match(raw):
    cfg = execution_config({"num_envs": 64, **raw})
    assert cfg["physics_backend"] == "gpu" and cfg["gpu_dynamics"] is True
    assert cfg["gpu_memory_capacities"] == gpu_dynamics_memory_config(64)


@pytest.mark.parametrize("raw", [
    {"physics_backend": None}, {"physics_backend": "cuda"}, {"physics_backend": 1},
    {"gpu_dynamics": 1}, {"gpu_dynamics": "true"},
    {"physics_backend": "cpu", "gpu_dynamics": True},
    {"physics_backend": "gpu", "gpu_dynamics": False},
    {"physics_backend": "gpu", "gpu_memory_capacities": None},
    {"physics_backend": "gpu", "gpu_memory_capacities": {"typo": 1024}},
    {"physics_backend": "gpu", "gpu_memory_capacities": {"heap_capacity": True}},
    {"physics_backend": "gpu", "gpu_memory_capacities": {"heap_capacity": 0}},
    {"physics_backend": "gpu", "gpu_memory_capacities": {"heap_capacity": -1}},
    {"physics_backend": "gpu", "gpu_memory_capacities": {"heap_capacity": 1.5}},
    {"physics_backend": "gpu", "gpu_memory_capacities": {"heap_capacity": 2**32}},
    {"physics_backend": "cpu", "gpu_memory_capacities": {"heap_capacity": 1024}},
    {"physics_backend": "gpu", "num_envs": 4096},
])
def test_invalid_solver_options_fail_before_opening_isaac(raw):
    with pytest.raises(ValueError):
        execution_config(raw)


def test_capacity_scaling_matches_old_64_environment_elevator_and_accepts_overrides():
    expected = dict(max_rigid_contact_count=1048576, max_rigid_patch_count=65536,
        found_lost_pairs_capacity=262144, found_lost_aggregate_pairs_capacity=65536,
        total_aggregate_pairs_capacity=262144, collision_stack_size=67108864,
        heap_capacity=67108864, temp_buffer_capacity=16777216)
    assert gpu_dynamics_memory_config(64) == expected
    grown = gpu_dynamics_memory_config(65)
    assert grown["max_rigid_contact_count"] == 2 * expected["max_rigid_contact_count"]
    assert all(value >= expected[key] for key, value in grown.items())
    raw = {"num_envs": 64, "physics_backend": "gpu", "gpu_memory_capacities": {"heap_capacity": 2**28}}
    before = deepcopy(raw)
    cfg = normalize_physics_config(raw)
    assert raw == before
    assert cfg["gpu_memory_capacities"] == {**expected, "heap_capacity": 2**28}
    assert normalize_physics_config({"num_envs": 64, **cfg}) == cfg


def test_cpu_keeps_template_ccd_and_publishes_actual_solver():
    context, settings = Context(), Settings({"/physics/suppressReadback": True})
    metadata = configure_physx(context, {}, settings)
    assert context.calls == [("broadphase", "MBP"), ("gpu", False)]
    assert metadata["physics_backend"] == "CPU PhysX tensor views"
    assert metadata["physics_backend_verified"] is True
    assert metadata["gpu_dynamics"] is False
    assert metadata["ccd_enabled"] is True
    assert metadata["gpu_memory_capacities"] == {}
    assert settings["/physics/suppressReadback"] is False


def test_gpu_initialization_uses_host_readback_and_all_qualified_buffers():
    context, settings = Context(), Settings()
    cfg = execution_config({"num_envs": 64, "physics_backend": "gpu"})
    before = deepcopy(cfg)
    metadata = configure_physx(context, cfg, settings)
    assert cfg == before
    assert context.calls == [("broadphase", "GPU"), ("gpu", True), ("ccd", False)]
    assert context.capacities == cfg["gpu_memory_capacities"]
    assert metadata["physics_backend_mode"] == "gpu"
    assert metadata["physics_backend_verified"] is True
    assert metadata["gpu_dynamics"] is True
    assert metadata["gpu_readback"] == "enabled_for_numpy"
    assert metadata["physics_tensor_frontend"] == "numpy"
    assert metadata["gpu_memory_capacities"] == context.capacities
    assert inspect_physx(context, cfg, settings) == metadata


@pytest.mark.parametrize("change,match", [
    (lambda c, s: setattr(c, "gpu", False), "backend"),
    (lambda c, s: setattr(c, "broadphase", "MBP"), "backend"),
    (lambda c, s: setattr(c, "ccd", True), "CCD"),
    (lambda c, s: c.capacities.update(heap_capacity=1), "capacity"),
    (lambda c, s: s.update({"/physics/suppressReadback": True}), "readback"),
])
def test_post_reset_configuration_drift_or_silent_fallback_is_rejected(change, match):
    cfg = execution_config({"physics_backend": "gpu"})
    context, settings = Context(), Settings()
    configure_physx(context, cfg, settings)
    change(context, settings)
    with pytest.raises(RuntimeError, match=match):
        inspect_physx(context, cfg, settings)


def test_backend_and_buffer_settings_are_part_of_task_identity():
    from test_parallel_backend import DynamicsEngine, LiftTask
    cpu = SharedWorldBackend(LiftTask(), engine_factory=DynamicsEngine)
    explicit_cpu = SharedWorldBackend(LiftTask(physics_backend="cpu"), engine_factory=DynamicsEngine)
    gpu = SharedWorldBackend(LiftTask(physics_backend="gpu"), engine_factory=DynamicsEngine)
    larger = SharedWorldBackend(LiftTask(physics_backend="gpu", gpu_memory_capacities={"heap_capacity": 2**28}),
                                engine_factory=DynamicsEngine)
    try:
        assert cpu.describe()["task_config_sha256"] == explicit_cpu.describe()["task_config_sha256"]
        assert len({v.describe()["task_config_sha256"] for v in (cpu, gpu, larger)}) == 3
    finally:
        for backend in (cpu, explicit_cpu, gpu, larger):
            backend.close()


@pytest.mark.parametrize("warning", [
    "[Warning] [omni.physx.plugin] GPU dispatcher failed; falling back to CPU",
    "[Warning] [omni.physx.plugin] GPU dynamics is disabled after CUDA initialization failure",
    "[Warning] [omni.physx.plugin] Not enough GPU memory available to create a PhysicsScene, falling back to CPU simulation.",
    "[Error] [omni.physx.plugin] failed to create a CUDA context",
])
def test_required_gpu_native_errors_or_fallback_warnings_are_fatal(tmp_path, warning):
    path = tmp_path / "kit.log"
    path.write_text(warning + "\n")
    guard = PhysXErrorGuard(path, require_gpu=True)
    try:
        with pytest.raises(RuntimeError, match="PhysX invalidated"):
            guard.check()
    finally:
        guard.close()


def test_numpy_host_pipeline_is_not_misclassified_as_gpu_dynamics_fallback(tmp_path):
    path = tmp_path / "kit.log"
    path.write_text("[Warning] [omni.physx.plugin] GPU tensor pipeline disabled; host readback enabled\n")
    guard = PhysXErrorGuard(path, require_gpu=True)
    try:
        guard.check()
    finally:
        guard.close()


def test_individual_convex_cpu_collision_does_not_imply_scene_cpu_fallback(tmp_path):
    path = tmp_path / "kit.log"
    path.write_text("[Warning] [omni.physx.plugin] ConvexMesh not GPU compatible, fall back to CPU, Object path: /World/Robot\n")
    guard = PhysXErrorGuard(path, require_gpu=True)
    try:
        guard.check()
    finally:
        guard.close()


def test_engine_metadata_carries_effective_not_just_requested_solver():
    cfg = execution_config({"physics_backend": "gpu"})
    context, settings = Context(), Settings()
    engine = IsaacParallelWorld(SimpleNamespace(), SimpleNamespace(execution=cfg, num_envs=1))
    engine.physics_metadata = configure_physx(context, cfg, settings)
    engine.camera_names, engine.camera_color_responses = [], []
    engine.image_preprocessing, engine.renderer_settings, engine.directory = None, {}, "unused"
    metadata = engine.execution_metadata()
    assert metadata["gpu_dynamics"] is True and metadata["physics_backend_verified"] is True
    assert metadata["gpu_memory_capacities"] == cfg["gpu_memory_capacities"]
    metadata["gpu_memory_capacities"]["heap_capacity"] = 1
    assert engine.physics_metadata["gpu_memory_capacities"]["heap_capacity"] != 1
