"""Task-independent PhysX tensor adapters and native error monitoring.

Isaac imports stay inside adapter construction so configuration, recording, and
CPU tests can import this module without starting a simulation application.
"""
from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np


GPU_CAPACITY_DEFAULTS = {
    "max_rigid_contact_count": (2**20, 2**14),
    "max_rigid_patch_count": (2**16, 2**10),
    "found_lost_pairs_capacity": (2**18, 2**12),
    "found_lost_aggregate_pairs_capacity": (2**16, 2**8),
    "total_aggregate_pairs_capacity": (2**18, 2**12),
    "collision_stack_size": (2**26, 2**20),
    "heap_capacity": (2**26, 2**20),
    "temp_buffer_capacity": (2**24, 2**18),
}


def gpu_dynamics_memory_config(num_envs):
    """The qualified elevator GPU capacities, generalized to any task's copies.

    Values are PhysX counts/bytes in the units of the corresponding SDK setter.
    Task-specific scene complexity can require explicit larger overrides; lost
    contacts or exhausted buffers remain fatal through ``PhysXErrorGuard``.
    """
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    result = {}
    for name, (minimum, per_env) in GPU_CAPACITY_DEFAULTS.items():
        required = max(minimum, num_envs * per_env)
        result[name] = 1 << (required - 1).bit_length()
    return result


def normalize_physics_config(values):
    """Resolve CPU/GPU selection before task defaults inject a legacy alias.

    ``gpu_dynamics`` remains a supported boolean alias. Other execution/task
    fields are ignored here, so plugins share the same CPU-only validation.
    """
    if not isinstance(values, dict):
        raise ValueError("Physics options must be a configuration object")
    mode = values.get("physics_backend")
    alias = values.get("gpu_dynamics")
    if "physics_backend" in values and mode not in ("cpu", "gpu"):
        raise ValueError("physics_backend must be cpu or gpu")
    if "gpu_dynamics" in values and type(alias) is not bool:
        raise ValueError("gpu_dynamics must be boolean")
    if mode is not None and alias is not None and (mode == "gpu") != alias:
        raise ValueError("physics_backend conflicts with gpu_dynamics")
    mode = mode or ("gpu" if alias else "cpu")
    overrides = values.get("gpu_memory_capacities", {})
    if not isinstance(overrides, dict):
        raise ValueError("gpu_memory_capacities must be an object")
    if set(overrides) - set(GPU_CAPACITY_DEFAULTS):
        raise ValueError(f"Unknown GPU memory capacities: {sorted(set(overrides)-set(GPU_CAPACITY_DEFAULTS))}")
    for name, value in overrides.items():
        if type(value) is not int or not 1 <= value < 2**32:
            raise ValueError(f"gpu_memory_capacities.{name} must be a positive 32-bit integer")
    if mode == "cpu" and overrides:
        raise ValueError("gpu_memory_capacities requires physics_backend=gpu")
    capacities = ({**gpu_dynamics_memory_config(values.get("num_envs", 1)), **overrides}
                  if mode == "gpu" else {})
    if any(value >= 2**32 for value in capacities.values()):
        raise ValueError("Automatic GPU memory capacities exceed 32-bit SDK limits")
    return {"physics_backend": mode, "gpu_dynamics": mode == "gpu",
            "gpu_memory_capacities": capacities}


def inspect_physx(context, config, settings):
    """Check the effective SDK scene, including after Isaac's world reset.

    A CUDA device selection alone does not establish GPU dynamics. These
    checks are paired with native-log fallback/error monitoring during steps.
    """
    options = normalize_physics_config(config)
    requested = options["gpu_dynamics"]
    actual = bool(context.is_gpu_dynamics_enabled())
    broadphase = context.get_broadphase_type()
    if actual != requested or (requested and broadphase != "GPU"):
        raise RuntimeError("PhysX did not apply the requested physics backend")
    if settings.get("/physics/suppressReadback") is not False:
        raise RuntimeError("Shared NumPy physics requires host readback enabled")
    ccd = bool(context.is_ccd_enabled())
    if requested and ccd:
        raise RuntimeError("GPU PhysX must use CCD disabled, matching the qualified legacy path")
    capacities = {}
    for name, desired in options["gpu_memory_capacities"].items():
        value = int(getattr(context, "get_gpu_" + name)())
        if value != desired:
            raise RuntimeError(f"PhysX GPU capacity {name} was not applied: {value} != {desired}")
        capacities[name] = value
    return dict(physics_backend=f"{'GPU' if actual else 'CPU'} PhysX tensor views",
                physics_backend_mode=options["physics_backend"], physics_backend_verified=True,
                gpu_dynamics=actual, broadphase=broadphase, ccd_enabled=ccd,
                physics_tensor_frontend="numpy", gpu_readback="enabled_for_numpy" if actual else "not_applicable",
                gpu_memory_capacities=capacities)


def configure_physx(context, config, settings):
    """Configure the shared scene before reset, cloning or tensor-view creation.

    GPU simulation uses NumPy/CPU views through host readback, exactly as the
    old elevator GPU implementation. This does not enable a CUDA tensor pipeline.
    CPU mode preserves the template's CCD choice and uses CPU broadphase.
    """
    options = normalize_physics_config(config)
    settings.set_bool("/physics/suppressReadback", False)
    context.set_broadphase_type("GPU" if options["gpu_dynamics"] else "MBP")
    context.enable_gpu_dynamics(options["gpu_dynamics"])
    if options["gpu_dynamics"]:
        context.enable_ccd(False)
        for name, value in options["gpu_memory_capacities"].items():
            getattr(context, "set_gpu_" + name)(value)
    return inspect_physx(context, options, settings)


class PhysXErrorGuard:
    """Read the flushed native log on the Kit thread, never via a Python callback.

    Native PhysX workers can log while the Kit thread holds the GIL and waits
    for those workers. Registering a Python logger therefore risks deadlock.
    The synchronous native file sink supplies the same error evidence without
    crossing that thread boundary. Missing/replaced/truncated logs fail closed.
    """

    def __init__(self, path, *, require_gpu=False):
        self.path = Path(path)
        self.require_gpu = require_gpu
        self.errors = []
        self._stream = self.path.open("rb")
        stat = self.path.stat()
        self._identity = (stat.st_dev, stat.st_ino)
        self._pending = b""

    def _inspect(self, message):
        lower = message.lower()
        lost_physics = any(marker in lower for marker in
            ("miss interactions", "buffer overflow", "capacity exceeded", "pxgpudynamicsmemoryconfig"))
        lost_physics |= any(word in lower for word in ("dropped", "discarded")) and any(
            kind in lower for kind in ("contact", "constraint", "pair", "interaction"))
        fallback = any(word in lower for word in (
            "falling back to cpu", "fall back to cpu", "fallback to cpu", "switching to cpu",
            "using cpu instead", "gpu dynamics disabled", "gpu dynamics is disabled"))
        # PhysX can use CPU collision processing for an individual non-GPU
        # convex hull while the scene still runs GPU dynamics. Reject a solver
        # fallback, not that supported per-shape hybrid path. The first marker
        # matches this SDK's insufficient-memory PhysicsScene warning.
        gpu_fallback = self.require_gpu and fallback and any(word in lower for word in (
            "physicsscene", "physics scene", "gpu dynamics", "gpu dispatcher", "cuda context"))
        if "physx" in lower and ("[error]" in lower or "physx error:" in lower or
                (("[warning]" in lower or "[warn]" in lower) and (lost_physics or gpu_fallback))):
            if len(self.errors) < 16:
                self.errors.append(dict(source="native_physx_log", message=message[:4096],
                    file=str(self.path)))

    def check(self):
        if not self.errors:
            try:
                stat = self.path.stat()
                if (stat.st_dev, stat.st_ino) != self._identity or stat.st_size < self._stream.tell():
                    raise OSError("Native physics log was replaced or truncated")
                data = self._stream.read()
                if data:
                    lines = (self._pending + data).split(b"\n")
                    self._pending = lines.pop()[-8192:]
                    for line in lines:
                        self._inspect(line.decode("utf-8", errors="replace"))
                    # A flushed error need not wait for its trailing newline.
                    self._inspect(self._pending.decode("utf-8", errors="replace"))
            except (OSError, ValueError) as error:
                self.errors.append(dict(source="native_physx_log", message=str(error), file=str(self.path)))
        if self.errors:
            error = self.errors[0]
            raise RuntimeError(f"PhysX invalidated simulation: {error['source']}: {error['message']}")

    def close(self):
        self._stream.close()



def view_env_order(paths, count):
    """Explicitly map PhysX view order, which is not guaranteed numeric order."""
    ids = []
    for path in paths:
        match = re.search(r"/env_(\d+)(?:/|$)", str(path))
        if not match:
            raise ValueError(f"Unexpected environment prim path: {path}")
        ids.append(int(match.group(1)))
    if sorted(ids) != list(range(count)):
        raise ValueError("Articulation view omitted or duplicated an environment")
    return np.argsort(ids)



class TensorArms:
    """Small NumPy adapter over PhysX's public batched tensor API.

    Avoids Isaac 5 Articulation wrapper's unrelated-prim-deletion callback,
    which deletes the physics view when Replicator changes its render graph.
    """
    def __init__(self, simulation_view, paths):
        from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
        self.view = simulation_view.create_articulation_view(
            [get_articulation_root_api_prim_path(path) for path in paths])
        self.prim_paths = self.view.prim_paths

    def get_dof_index(self, name):
        return self.view.shared_metatype.dof_names.index(name)

    def get_joint_positions(self):
        return self.view.get_dof_positions()

    def get_joint_velocities(self):
        return self.view.get_dof_velocities()

    def _set(self, getter, setter, value, indices):
        data = getter().copy()
        data[indices] = value
        setter(data, np.asarray(indices, dtype=np.uint32))

    def set_joint_positions(self, value, indices):
        self._set(self.view.get_dof_positions, self.view.set_dof_positions, value, indices)

    def set_joint_velocities(self, value, indices):
        self._set(self.view.get_dof_velocities, self.view.set_dof_velocities, value, indices)

    def set_joint_position_targets(self, value, indices):
        self._set(self.view.get_dof_position_targets, self.view.set_dof_position_targets, value, indices)



class TensorRigidBodies:
    def __init__(self, simulation_view, paths):
        self.view = simulation_view.create_rigid_body_view(paths)
        self.prim_paths = self.view.prim_paths

    def get_world_poses(self):
        poses = self.view.get_transforms()
        return poses[:, :3].copy(), poses[:, [6, 3, 4, 5]].copy()

    def get_velocities(self):
        return self.view.get_velocities()

    def set_world_poses(self, positions, orientations, indices):
        poses = self.view.get_transforms().copy()
        poses[indices, :3] = positions
        poses[indices, 3:] = orientations[:, [1, 2, 3, 0]]
        self.view.set_transforms(poses, np.asarray(indices, dtype=np.uint32))

    def set_velocities(self, velocities, indices):
        data = self.view.get_velocities().copy()
        data[indices] = velocities
        self.view.set_velocities(data, np.asarray(indices, dtype=np.uint32))



def path_rows(paths, requested):
    mapping = {str(path): i for i, path in enumerate(paths)}
    if len(mapping) != len(requested) or set(mapping) != set(requested):
        raise RuntimeError("PhysX view contains missing, duplicated, or unexpected bodies")
    return np.array([mapping[path] for path in requested], dtype=np.int64)



class ContactPathClassifier:
    """Cache immutable scene-topology matches, never slot activity or outcomes.

    Native routing supplies template-local actor/collider paths. All cloned
    environments therefore share this small mapping; recreate it on bind when
    the scene topology changes.
    """
    def __init__(self, roots):
        self.roots = tuple(roots)
        self._cache = {}

    def __call__(self, actor_paths):
        paths = tuple(actor_paths)
        result = self._cache.get(paths)
        if result is None:
            result = tuple(any(path == root or path.startswith(root+"/") for path in paths)
                           for root in self.roots)
            self._cache[paths] = result
        return result


def _contact_xyz(value, error):
    """Read native carb.Float3 without allocating a NumPy normal vector.

    Non-native inputs retain the original (3,) shape check. In particular, a
    tuple of one-element arrays must not become an accidentally valid vector.
    """
    try:
        if len(value) != 3:
            raise RuntimeError(error)
        x, y, z = value[0], value[1], value[2]
        if type(x) is float and type(y) is float and type(z) is float:
            return x, y, z
        array = np.asarray(value, dtype=float)
        if array.shape != (3,):
            raise RuntimeError(error)
        return float(array[0]), float(array[1]), float(array[2])
    except (TypeError, ValueError, IndexError) as exception:
        raise RuntimeError(error) from exception


class ContactForceReducer:
    """Reduce native points without changing impulse norms or summation order.

    Keep NumPy's impulse norm bit-for-bit, because accumulated finger forces
    control grasp/release predicates. Only the normal's three-scalar norm has
    a fast path; near either support decision boundary it uses the original
    NumPy expression. Shape, finite, normal, and separation checks remain.
    """
    def __init__(self, physics_dt, max_support_normal_tilt_deg, separation_tolerance_m,
                 error="Malformed measured contact data"):
        self.dt = physics_dt
        self.cosine = math.cos(math.radians(max_support_normal_tilt_deg))
        self.tolerance = separation_tolerance_m
        self.error = error

    def __call__(self, contacts):
        force, vertical = 0., 0.
        dt, cosine, tolerance, error = self.dt, self.cosine, self.tolerance, self.error
        for contact in contacts:
            impulse = np.asarray(contact.impulse, dtype=float)
            if impulse.shape != (3,):
                raise RuntimeError(error)
            ix, iy, iz = float(impulse[0]), float(impulse[1]), float(impulse[2])
            raw_normal = contact.normal
            nx, ny, nz = _contact_xyz(raw_normal, error)
            separation = float(contact.separation)
            if not (math.isfinite(ix) and math.isfinite(iy) and math.isfinite(iz)
                    and math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)
                    and math.isfinite(separation)):
                raise RuntimeError(error)
            # Exact NumPy 1.26 norm fast path for a validated real float64
            # vector: keep the K-order flatten, ndarray.dot, and np.sqrt.
            # This removes only generic norm argument/dtype/shape dispatch.
            flat = impulse.ravel(order="K")
            force += float(np.sqrt(flat.dot(flat)))/dt
            normal_norm = math.sqrt(nx*nx+ny*ny+nz*nz)
            # Three positive products have no cancellation. These guards are
            # deliberately much wider than scalar-vs-dot rounding error so a
            # normal at, or nextafter either threshold takes the exact old path.
            if (abs(normal_norm-1e-8) <= 1e-20 or
                    (normal_norm > 1e-8 and abs(abs(nz)/normal_norm-cosine) <= 1e-12)):
                normal_norm = float(np.linalg.norm(np.asarray(raw_normal, dtype=float)))
            if (normal_norm > 1e-8 and abs(nz)/normal_norm >= cosine
                    and abs(separation) <= tolerance):
                vertical += abs(iz)/dt
        return force, vertical
