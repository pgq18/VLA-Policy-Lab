"""Versioned on-policy DPPO across inference, environment and learner nodes.

One iteration samples a single immutable expert version, computes GAE, updates
only the action expert and a local value network, then waits for a verified
atomic weight publication. There is deliberately no SAC/off-policy replay.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch
from torch import nn

from .client import NodeClient
from .checkpoint_reference import resolve_checkpoint
from .contract import finite_array, fingerprint, negotiate, validate_observation
from .dppo import (DPPOConfig, clipped_policy_loss, clipped_value_loss, generalized_advantages,
                   reduce_logprobs, metrics_to_floats)
from .trainer import EndpointConfig, RolloutConfig, CheckpointConfig, WandbConfig, _atomic_json, _WandbMonitor
from .checkpoint_sink import CheckpointSinkConfig, save_checkpoint_stream, load_checkpoint_stream
from .accelerator import (initialize_device, seed_device,
                          device_rng_state as _device_rng_state,
                          device_rng_payload as _device_rng_payload,
                          restore_device_rng as _restore_device_rng)


def _algorithm_identity(config):
    """Preserve old fingerprints only when new optional behavior is disabled."""
    result = asdict(config)
    if result["critic_warmup_iterations"] == 0:
        del result["critic_warmup_iterations"]
    if result["critic_continue_after_kl"] is False:
        del result["critic_continue_after_kl"]
    return result


def _terminal_transport(rollout, algorithm):
    mode = rollout.terminal_observation_mode
    if mode not in ("full", "state_only_if_no_bootstrap"):
        raise ValueError("Unknown rollout.terminal_observation_mode")
    if mode != "full" and algorithm.bootstrap_truncated is not False:
        raise ValueError("State-only DPPO terminals require bootstrap_truncated=false")
    if rollout.fused_step_reset:
        raise ValueError("DPPO requires separate indexed reset; fused_step_reset is not supported")
    return mode


def _checkpoint_contract_identity(contract):
    """Omitted unused terminal RGB is transport, not a different PPO problem.

    The bootstrap rule remains in the algorithm identity. Every model/task,
    action, discount and observation-schema field stays in the contract.
    """
    result = deepcopy(contract)
    if result.get("terminal_observation_mode") == "state_only_if_no_bootstrap":
        result.pop("terminal_observation_mode")
        result["fingerprint"] = fingerprint({key: value for key, value in result.items()
                                              if key not in ("num_envs", "fingerprint")})
    return result


class MasterAdamW:
    """FP32 optimizer state/master weights for native BF16/FP16 experts."""

    def __init__(self, parameters, *, lr, weight_decay):
        self.parameters = list(parameters)
        self.masters = [parameter if parameter.dtype == torch.float32 else
                        nn.Parameter(parameter.detach().float().clone()) for parameter in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr, weight_decay=weight_decay)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.parameters:
            parameter.grad = None

    def step(self, max_grad_norm):
        for parameter, master in zip(self.parameters, self.masters, strict=True):
            if master is not parameter:
                master.grad = None if parameter.grad is None else parameter.grad.detach().float()
        norm = torch.nn.utils.clip_grad_norm_(self.masters, max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for parameter, master in zip(self.parameters, self.masters, strict=True):
                if master is not parameter:
                    parameter.copy_(master)
        return norm

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "masters": [master.detach().clone() for master in self.masters]}

    def load_state_dict(self, state):
        if len(state["masters"]) != len(self.masters):
            raise ValueError("Master parameter count differs")
        with torch.no_grad():
            for parameter, master, saved in zip(self.parameters, self.masters, state["masters"], strict=True):
                if master.shape != saved.shape or not torch.isfinite(saved).all():
                    raise ValueError("Invalid FP32 master parameter")
                master.copy_(saved)
                parameter.copy_(master)
        self.optimizer.load_state_dict(state["optimizer"])


@dataclass
class DPPOTrainerConfig:
    method: str = "dppo"
    mode: str = "train"
    model: dict = field(default_factory=dict)
    action_expert: dict = field(default_factory=dict)
    device: str = "cpu"
    seed: int = 42
    endpoints: EndpointConfig = field(default_factory=EndpointConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    dppo: DPPOConfig = field(default_factory=DPPOConfig)
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig(include_replay=False))
    checkpoint_sink: CheckpointSinkConfig | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "outputs/dppo_run"

    def __post_init__(self):
        for name, cls in (("endpoints", EndpointConfig), ("rollout", RolloutConfig), ("dppo", DPPOConfig),
                          ("checkpoint", CheckpointConfig), ("wandb", WandbConfig)):
            value = getattr(self, name)
            if isinstance(value, dict):
                setattr(self, name, cls(**value))
            elif not isinstance(value, cls):
                raise ValueError(f"{name} must be an object")
        self.dppo.__post_init__()
        if isinstance(self.checkpoint_sink, dict):
            self.checkpoint_sink = CheckpointSinkConfig(**self.checkpoint_sink)
        elif self.checkpoint_sink is not None and not isinstance(self.checkpoint_sink, CheckpointSinkConfig):
            raise ValueError("checkpoint_sink must be an object or null")
        if self.method != "dppo" or self.mode not in ("train", "eval"):
            raise ValueError("DPPO runner requires method=dppo, mode=train/eval")
        if not isinstance(self.model, dict) or ("plugin" in self.model) == ("factory" in self.model):
            raise ValueError("DPPO model requires exactly one plugin/factory and config")
        if not isinstance(self.model.get("config", {}), dict) or not isinstance(self.action_expert, dict):
            raise ValueError("model.config and action_expert must be objects")
        _terminal_transport(self.rollout, self.dppo)
        if self.rollout.action_bounds not in ("error", "clip"):
            raise ValueError("action_bounds must be error or clip")
        for key, value in (("seed", self.seed), ("max_transitions", self.rollout.max_transitions),
                           ("eval_episodes", self.rollout.eval_episodes), ("recent_window", self.rollout.recent_window),
                           ("checkpoint.interval", self.checkpoint.interval)):
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid integer {key}")
        for key in ("num_envs", "execution_horizon"):
            value = getattr(self.rollout, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be positive or null")
        if self.checkpoint.keep is not None:
            raise ValueError("DPPO retains every interval checkpoint; checkpoint.keep must be null")
        if self.checkpoint.include_replay:
            raise ValueError("DPPO has no off-policy replay; set include_replay=false")
        if not isinstance(self.rollout.reset_options, dict):
            raise ValueError("reset_options must be an object")
        if type(self.rollout.deterministic_eval) is not bool:
            raise ValueError("deterministic_eval must be boolean")


class DPPOTrainer:
    def __init__(self, config, *, inference_client=None, simulation_client=None, expert=None, observer=None):
        self.config = deepcopy(config if isinstance(config, DPPOTrainerConfig) else DPPOTrainerConfig(**config))
        self.config.__post_init__()
        cfg = self.config
        self.inference = inference_client or self._client(cfg.endpoints.inference)
        self.simulation = simulation_client or self._client(cfg.endpoints.simulation)
        self.expert, self.backend, self.policy, self.value = expert, None, None, None
        self.actor_optimizer = self.value_optimizer = None
        self.observer = observer
        self.output = Path(cfg.output_dir)
        self.run_id = uuid.uuid4().hex
        self.stop_event = threading.Event()
        self.rng = np.random.default_rng(cfg.seed)
        self.slots = {}
        self._seen_episode_ids = set()
        self.episode_serial = defaultdict(int)
        self.episode_seed = {}
        self.eval_quotas = {}
        self.recent = deque(maxlen=cfg.rollout.recent_window)
        self.transitions = self.episodes = self.successes = self.updates = self.iterations = 0
        self.actor_updates = self.critic_updates = 0
        self.initial_transitions = 0
        self.policy_version = 0
        self.latest_metrics = {}
        self._status = {"state": "created"}
        self._claimed = False
        self._physics_in_flight = False
        self._pending_physics_envs = set()
        self._pending_physics_kind = None
        # No tensor copies are made between updates. Outside optimizer mutation,
        # the live learner is a complete, resumable state even if an RPC fails.
        self._learner_state_consistent = False
        self._last_optimized_transitions = 0
        self._unoptimized_rollout_transitions = 0
        self.discarded_rollout_transitions = 0
        self._publication_state = "not_initialized"
        self._wandb = None
        self._last_checkpoint_bucket = 0
        self._last_saved_coordinates = None
        self.last_checkpoint = None
        self.started = None

    def _client(self, endpoint):
        return NodeClient(endpoint, **{
            key: value for key, value in asdict(self.config.endpoints).items() if key not in ("inference", "simulation")})

    def stop(self, _payload=None):
        self.stop_event.set()
        return self.health()

    def health(self, _payload=None):
        return deepcopy(self._status)

    def _log(self, name, record):
        with (self.output / name).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def _status_write(self, state, **extra):
        elapsed = time.monotonic() - self.started if self.started else 0.
        self._status = {"state": state, "run_id": self.run_id, "method": "dppo", "mode": self.config.mode,
            "transitions": self.transitions, "updates": self.updates, "iterations": self.iterations,
            "actor_updates": self.actor_updates, "critic_updates": self.critic_updates,
            "discarded_rollout_transitions": self.discarded_rollout_transitions,
            "critic_warmup_iterations_remaining": (
                max(0, self.config.dppo.critic_warmup_iterations - self.iterations)
                if self.config.mode == "train" else None),
            "policy_version": self.policy_version, "episodes": self.episodes, "successes": self.successes,
            "success_rate": self.successes / self.episodes if self.episodes else None,
            "recent_sr": sum(self.recent) / len(self.recent) if self.recent else None,
            "recent_episodes": len(self.recent), "elapsed_seconds": elapsed,
            "transitions_per_second": (self.transitions - self.initial_transitions) / elapsed if elapsed else 0.,
            "transition_budget_overshoot": max(0, self.transitions - self.config.rollout.max_transitions),
            "last_update_metrics": self.latest_metrics, "last_checkpoint": self.last_checkpoint, **extra}
        _atomic_json(self.output / "status.json", self._status)

    def _call(self, client, route, payload=None):
        start = time.monotonic()
        result = client.call(route, payload)
        self._log("rpc_timing.jsonl", {"node": "inference" if client is self.inference else "simulation",
            "path": route, "seconds": time.monotonic() - start, "transitions": self.transitions})
        return result

    def _negotiate(self):
        from .denoising import GaussianChainPolicy
        from .expert_sync import state_sha256, expert_spec
        initialize_device(self.config.device)
        self.inference_metadata = self._call(self.inference, "/health")
        self.simulation_metadata = self._call(self.simulation, "/health")
        if self.simulation_metadata.get("environment_type") == "real_robot":
            raise ValueError("DPPO currently supports simulation only; physical on-policy execution is not enabled")
        terminal_mode = _terminal_transport(self.config.rollout, self.config.dppo)
        self.contract = negotiate(self.inference_metadata, self.simulation_metadata, "dppo",
            self.config.rollout.execution_horizon, terminal_observation_mode=terminal_mode)
        self.num_envs = self.config.rollout.num_envs or self.simulation_metadata["num_envs"]
        if self.num_envs > self.simulation_metadata["num_envs"]:
            raise ValueError("Configured num_envs exceeds environment capacity")
        if self.simulation_metadata.get("initial_reset") == "all_envs" and self.num_envs != self.simulation_metadata["num_envs"]:
            raise ValueError("Environment requires all environments in initial reset")
        if self.config.mode == "eval" and self.config.rollout.eval_episodes < self.num_envs:
            raise ValueError("Paired evaluation requires eval_episodes >= num_envs")
        for eid in range(self.num_envs):
            self.eval_quotas[eid] = self.config.rollout.eval_episodes // self.num_envs + (eid < self.config.rollout.eval_episodes % self.num_envs)
        remote = self.inference_metadata.get("action_expert", {})
        if not remote or type(remote.get("policy_version")) is not int:
            raise ValueError("Inference must advertise a versioned action expert")
        self.policy_version = remote["policy_version"]
        seed_device(self.config.seed, self.config.device)
        if self.expert is None:
            from .server import load_backend
            spec = deepcopy(self.config.model)
            spec.setdefault("config", {})["device"] = self.config.device
            self.backend = load_backend(spec, kind="model")
            self._check_backbone_identity(self.backend.describe())
            self.expert = self.backend.create_action_expert(self.config.action_expert)
        self.expert.to(self.config.device).eval()
        if expert_spec(self.expert) != remote.get("spec"):
            raise ValueError("Local/inference action expert specification or implementation differs")
        self.policy = GaussianChainPolicy(self.expert)
        self.parameters = [parameter for parameter in self.expert.parameters() if parameter.requires_grad]
        if not self.parameters:
            raise ValueError("Action expert exposes no trainable parameters")
        self.total_steps = self.expert.denoising_steps
        self.ft_steps = self.config.dppo.ft_denoising_steps or self.total_steps
        if self.ft_steps != self.total_steps:
            raise ValueError("Partial denoising fine-tuning requires a frozen prefix sampler; currently ft_denoising_steps must equal all steps")
        width = self.contract["obs_dim"]
        hidden = self.config.dppo.critic_hidden_dim
        self.value = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(self.config.device)
        self.actor_optimizer = MasterAdamW(self.parameters, lr=self.config.dppo.actor_lr, weight_decay=self.config.dppo.weight_decay)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.config.dppo.critic_lr)
        if remote.get("config", {}) != self.config.action_expert:
            raise ValueError("Inference and trainer action_expert configurations differ")
        self.identity = {"contract": _checkpoint_contract_identity(self.contract),
                         "action_expert": remote.get("spec", {}), "action_expert_config": remote.get("config", {}),
                         "dppo": _algorithm_identity(self.config.dppo), "action_bounds": self.config.rollout.action_bounds}
        self.schema_fingerprint = fingerprint(self.identity)
        _atomic_json(self.output / "negotiated.json", {**self.identity, "rollout_seed_schedule": self._seed_schedule()})
        if self.config.checkpoint.resume:
            self._restore()
            self._publish()
        elif state_sha256(self.expert.state_dict()) != remote.get("weights_sha256"):
            raise ValueError("Local action expert weights differ from inference; both nodes must load the same SFT checkpoint")
        self._log("events.jsonl", {"event": "initialized", "trainable_expert_parameters": sum(p.numel() for p in self.parameters),
            "policy_version": self.policy_version, "expert_sha256": state_sha256(self.expert.state_dict()),
            "sampler": remote.get("spec", {}), "replay": False})

    def _check_backbone_identity(self, local):
        # Equal trainable heads are insufficient: a different frozen encoder,
        # normalizer, adapter source or preprocessing changes the policy too.
        keys = ("model_id", "checkpoint_sha256", "weights_sha256", "source_revision",
                "normalization_sha256", "model_config_sha256", "image_preprocessing",
                "observation", "action", "noise", "feature_dim")
        for key in keys:
            if local.get(key) != self.inference_metadata.get(key):
                raise ValueError(f"Local/inference frozen model identity differs: {key}")

    def _seed(self, env_id, episode, stream, step=0):
        payload = f"{self.config.seed}:{env_id}:{episode}:{stream}:{step}".encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little") & 0x7fffffff

    def _seed_schedule(self):
        batched = "per_env_reset_seeds" in self.simulation_metadata.get("capabilities", [])
        return {"algorithm": "blake2b", "digest_size": 4, "encoding": "utf-8", "byteorder": "little",
                "mask": 0x7fffffff, "preimage": "{seed}:{env_id}:{episode_index}:{stream}:{step}",
                "base_seed": self.config.seed, "streams": ["reset", "policy", "bootstrap"],
                "initial_all_envs_reset": "shared seed derived from env_id=0, episode_index=0, stream=reset, step=0",
                "subsequent_reset": (
                    "one request for the reset set with env_seeds keyed by env_id; each environment uses its own episode_index, stream=reset, step=0"
                    if batched else
                    "one environment per request, its own episode_index, stream=reset, step=0"),
                "policy": "each observation uses its environment, episode index, stream=policy and actual step_id",
                "trainer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def _reset(self, ids, *, initial=False):
        if not ids:
            return
        # Keep initial shared-seed semantics and each subsequent episode's RNG.
        # Capable backends can reset all done environments with one render; old
        # backends still receive exactly the previous singleton requests.
        shared_initial = initial and self.simulation_metadata.get("initial_reset") == "all_envs"
        batched = not initial and "per_env_reset_seeds" in self.simulation_metadata.get("capabilities", [])
        groups = [ids] if shared_initial or batched else [[eid] for eid in ids]
        for group in groups:
            seed = self._seed(group[0], self.episode_serial[group[0]], "reset")
            seeds = {eid: self._seed(eid, self.episode_serial[eid], "reset") if batched else seed
                     for eid in group}
            payload = {"run_id": self.run_id, "request_id": uuid.uuid4().hex,
                       "env_ids": group, "seed": seed, "options": self.config.rollout.reset_options}
            if batched:
                payload["env_seeds"] = {str(eid): value for eid, value in seeds.items()}
            self._claimed = self._physics_in_flight = True
            self._pending_physics_envs = set(group)
            self._pending_physics_kind = "reset"
            rows = self._call(self.simulation, "/reset", payload).get("items", [])
            if len(rows) != len(group) or {row.get("env_id") for row in rows} != set(group):
                raise ValueError("Reset returned incorrect environment ids")
            for row in rows:
                if (not isinstance(row.get("episode_id"), str) or not row["episode_id"]
                        or row["episode_id"] in self._seen_episode_ids or type(row.get("step_id")) is not int
                        or row["step_id"] != 0):
                    raise ValueError("Reset requires unique episode_id and step zero")
                self._seen_episode_ids.add(row["episode_id"])
                validate_observation(row["observation"], self.simulation_metadata["observation"])
                eid = row["env_id"]
                self.episode_seed[eid] = seeds[eid]
                self.slots[eid] = row
                self._log("resets.jsonl", {"env_id": eid, "episode_index": self.episode_serial[eid], "seed": seeds[eid],
                    "policy_seed_at_step0": self._seed(eid, self.episode_serial[eid], "policy", 0),
                    "episode_id": row["episode_id"], "info": row.get("info", {})})
            self._physics_in_flight = False
            self._pending_physics_envs.clear()
            self._pending_physics_kind = None
            if self.observer:
                self.observer.on_reset(deepcopy(rows))

    def _vector(self, observation, features):
        parts = [finite_array(features, (self.contract["feature_dim"],), "features")]
        for key in ("state", "control_state", "task_features"):
            parts.append(finite_array(observation.get(key, []), (self.contract[key + "_dim"],), key))
        return torch.from_numpy(np.concatenate(parts))

    def _values(self, vectors):
        with torch.no_grad():
            return self.value(torch.stack(vectors).to(self.config.device)).squeeze(-1).cpu()

    def _bootstrap(self, rows):
        if not rows:
            return []
        if any(row.get("terminal_observation_omitted", False) for row in rows):
            raise ValueError("DPPO bootstrap requires a full observation; omitted terminal RGB cannot be encoded")
        response = self._call(self.inference, "/encode", {"observations": [row["observation"] for row in rows],
            "seeds": [self._seed(row["env_id"], self.episode_serial[row["env_id"]], "bootstrap", row["step_id"]) for row in rows],
            "include_base_actions": False})
        items = response.get("items", [])
        contexts = [item["context_id"] for item in items if isinstance(item.get("context_id"), str)]
        try:
            if len(items) != len(rows):
                raise ValueError("Bootstrap encoder returned wrong batch size")
            return self._values([self._vector(row["observation"], item["features"]) for row, item in zip(rows, items, strict=True)]).tolist()
        finally:
            if contexts:
                self._call(self.inference, "/release", {"context_ids": contexts})

    def _sample(self, rows):
        from .expert_sync import tree_from_wire
        seeds = [self._seed(row["env_id"], self.episode_serial[row["env_id"]], "policy", row["step_id"]) for row in rows]
        response = self._call(self.inference, "/sample_chain", {"observations": [row["observation"] for row in rows],
            "seeds": seeds, "expected_policy_version": self.policy_version,
            "deterministic": self.config.mode == "eval" and self.config.rollout.deterministic_eval})
        items = response.get("items", [])
        if len(items) != len(rows) or response.get("policy_version") != self.policy_version:
            raise ValueError("Stale policy version or incorrect chain batch size")
        samples = []
        for row, item, seed in zip(rows, items, seeds, strict=True):
            if item.get("policy_version") != self.policy_version:
                raise ValueError("Mixed policy versions in on-policy rollout")
            sample = {"vector": self._vector(row["observation"], item["features"]), "policy_version": self.policy_version,
                      "seed": seed, "env_id": row["env_id"], "episode_id": row["episode_id"], "step_id": row["step_id"]}
            sample["actions"] = finite_array(item["actions"], (self.contract["action_horizon"], self.contract["action_dim"]), "actions")
            if self.config.mode == "train":
                sample["conditioning"] = tree_from_wire(item["conditioning"], device="cpu")
                for key, expected in (("chain", (self.total_steps + 1, self.expert.horizon, self.expert.latent_dim)),
                                      ("logprobs", (self.total_steps, self.expert.horizon, self.expert.latent_dim))):
                    tensor = tree_from_wire(item[key], device="cpu")
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.as_tensor(tensor)
                    if tuple(tensor.shape) != expected or not torch.isfinite(tensor).all():
                        raise ValueError(f"Invalid sampled {key} geometry or values")
                    sample[key] = tensor.detach()
            samples.append(sample)
        values = self._values([sample["vector"] for sample in samples])
        for sample, value in zip(samples, values, strict=True):
            sample["value"] = float(value)
        return samples

    def _actions(self, samples):
        actions = np.stack([sample["actions"] for sample in samples])
        lower, upper = self.simulation_metadata["action"].get("lower"), self.simulation_metadata["action"].get("upper")
        prefix = self.contract["execution_horizon"]
        if self.config.rollout.action_bounds == "clip":
            if lower is None or upper is None:
                raise ValueError("Clipping requires declared environment action bounds")
            actions = np.clip(actions, lower, upper)
        elif ((lower is not None and np.any(actions[:, :prefix] < np.asarray(lower) - 1e-6))
              or (upper is not None and np.any(actions[:, :prefix] > np.asarray(upper) + 1e-6))):
            raise ValueError("Executable actions exceed declared limits")
        return actions[:, :prefix]

    def _step(self, rows, samples):
        actions = self._actions(samples)
        self._physics_in_flight = True
        self._pending_physics_envs = {row["env_id"] for row in rows}
        self._pending_physics_kind = "step"
        payload = {"run_id": self.run_id, "request_id": uuid.uuid4().hex,
            "items": [{"env_id": row["env_id"], "episode_id": row["episode_id"], "step_id": row["step_id"],
                       "actions": action.tolist()} for row, action in zip(rows, actions, strict=True)]}
        if self.config.rollout.terminal_observation_mode != "full":
            payload["terminal_observation_mode"] = self.config.rollout.terminal_observation_mode
        result = self._call(self.simulation, "/step", payload)
        returned = result.get("items", [])
        if len(returned) != len(rows) or {row.get("env_id") for row in returned} != {row["env_id"] for row in rows}:
            raise ValueError("Step returned incorrect environment ids")
        by_id = {row["env_id"]: row for row in returned}
        ordered = [by_id[row["env_id"]] for row in rows]
        for prior, current in zip(rows, ordered, strict=True):
            if (current.get("episode_id") != prior["episode_id"] or type(current.get("step_id")) is not int
                    or current["step_id"] != prior["step_id"] + 1):
                raise ValueError("Stale episode or step identity")
            terminated, truncated = current.get("terminated"), current.get("truncated")
            if type(terminated) is not bool or type(truncated) is not bool or terminated and truncated:
                raise ValueError("Exclusive boolean termination flags required")
            omitted = current.get("terminal_observation_omitted", False)
            expected_omission = self.config.rollout.terminal_observation_mode != "full" and (terminated or truncated)
            if type(omitted) is not bool or omitted != expected_omission:
                raise ValueError("Simulation terminal_observation_omitted disagrees with negotiated mode")
            validate_observation(current["observation"], self.simulation_metadata["observation"],
                                 allow_missing_images=omitted)
            if omitted:
                for key in ("control_state", "task_features"):
                    finite_array(current["observation"].get(key, []), (self.contract[key + "_dim"],), key)
            finite_array([current.get("reward"), current.get("discount")], (2,), "reward/discount")
            info = current.get("info", {})
            duration = info.get("elapsed_control_steps")
            if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= self.contract["execution_horizon"] + 1e-6:
                raise ValueError("Invalid actual chunk duration")
            if type(info.get("success")) is not bool or type(info.get("physics_steps")) is not int or info["physics_steps"] <= 0:
                raise ValueError("Missing success or physics step count")
            discount = self.contract["discount"]
            expected = discount["gamma"] ** (duration * discount["timebase_hz"] / self.contract["action"]["control_hz"])
            if not math.isclose(current["discount"], expected, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError("Environment discount disagrees with actual chunk duration")
        self._physics_in_flight = False
        self._pending_physics_envs.clear()
        self._pending_physics_kind = None
        if self.observer:
            self.observer.on_transition(deepcopy(rows), actions.copy(), deepcopy(ordered))
        return ordered

    def _remaining(self):
        return self.config.rollout.max_transitions - self.transitions

    def _collect(self):
        records, unresolved = [], {}
        for _ in range(self.config.dppo.rollout_steps):
            if not self.slots or self._remaining() <= 0 or self.stop_event.is_set():
                break
            ids = sorted(self.slots)
            if self.simulation_metadata.get("scheduling") != "all_active":
                ids = ids[:self._remaining()]
            rows = [self.slots[eid] for eid in ids]
            samples = self._sample(rows)
            for sample in samples:
                if sample["env_id"] in unresolved:
                    previous = unresolved.pop(sample["env_id"])
                    if previous["episode_id"] != sample["episode_id"] or previous["step_id"] + 1 != sample["step_id"]:
                        raise ValueError("GAE continuation crosses episode identity")
                    previous["next_value"] = sample["value"]
            next_rows = self._step(rows, samples)
            truncated_pairs = []
            reset_ids = []
            for sample, row in zip(samples, next_rows, strict=True):
                sample.update({key: row[key] for key in ("reward", "discount", "terminated", "truncated")})
                sample["next_value"] = 0.
                records.append(sample)
                self.transitions += 1
                self._unoptimized_rollout_transitions += 1
                eid = row["env_id"]
                self.slots[eid] = row
                if row["terminated"] or row["truncated"]:
                    success = row["info"]["success"]
                    self.episodes += 1
                    self.successes += int(success)
                    self.recent.append(int(success))
                    self._log("episodes.jsonl", {"transition": self.transitions, "env_id": eid,
                        "episode_index": self.episode_serial[eid], "reset_seed": self.episode_seed[eid],
                        "policy_version": self.policy_version, "success": success, "info": row["info"],
                        "episode_id": row["episode_id"], "step_id": row["step_id"],
                        "terminated": row["terminated"], "truncated": row["truncated"]})
                    self.episode_serial[eid] += 1
                    if self.config.mode == "train" and row["truncated"] and self.config.dppo.bootstrap_truncated:
                        truncated_pairs.append((sample, row))
                    del self.slots[eid]
                    if self.config.mode == "train" or self.episode_serial[eid] < self.eval_quotas[eid]:
                        reset_ids.append(eid)
                elif self.config.mode == "train":
                    unresolved[eid] = sample
            if truncated_pairs:
                values = self._bootstrap([row for _, row in truncated_pairs])
                for (sample, _), value in zip(truncated_pairs, values, strict=True):
                    sample["next_value"] = value
            if self._remaining() > 0 and not self.stop_event.is_set():
                self._reset(reset_ids)
            self._status_write("collecting")
            self._log("metrics.jsonl", self.health())
        if unresolved:
            ids = sorted(unresolved)
            values = self._bootstrap([self.slots[eid] for eid in ids])
            for eid, value in zip(ids, values, strict=True):
                unresolved[eid]["next_value"] = value
        return records

    def _advantages(self, records):
        grouped = defaultdict(list)
        for index, record in enumerate(records):
            grouped[record["env_id"]].append(index)
        for indices in grouped.values():
            tensors = [torch.tensor([records[index][key] for index in indices], dtype=dtype) for key, dtype in (
                ("reward", torch.float32), ("value", torch.float32), ("next_value", torch.float32),
                ("discount", torch.float32), ("terminated", torch.bool), ("truncated", torch.bool))]
            advantage, returns = generalized_advantages(*tensors, gae_lambda=self.config.dppo.gae_lambda,
                                                       bootstrap_truncated=self.config.dppo.bootstrap_truncated)
            for index, adv, target in zip(indices, advantage, returns, strict=True):
                records[index]["advantage"], records[index]["return"] = float(adv), float(target)
        advantages = torch.tensor([record["advantage"] for record in records])
        if self.config.dppo.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return advantages

    def _conditioning(self, records):
        conditions = [record["conditioning"] for record in records]
        if callable(getattr(self.expert, "collate_conditions", None)):
            result = self.expert.collate_conditions(conditions)
        else:
            def collate(values):
                if isinstance(values[0], dict):
                    if any(set(value) != set(values[0]) for value in values):
                        raise ValueError("Conditioning keys differ")
                    return {key: collate([value[key] for value in values]) for key in values[0]}
                if not all(isinstance(value, torch.Tensor) for value in values):
                    raise ValueError("Conditioning must contain only tensors")
                return torch.cat(values, dim=0)
            result = collate(conditions)
        def move(value):
            return {key: move(item) for key, item in value.items()} if isinstance(value, dict) else value.to(self.config.device)
        return move(result)

    def _pair_batch(self, records, selected):
        chunk_indices = selected // self.ft_steps
        local_step_indices = selected % self.ft_steps
        step_indices = local_step_indices + self.total_steps - self.ft_steps
        rows = [records[int(index)] for index in chunk_indices]
        # Rollout tensors live on CPU. Index them with the CPU step integers,
        # rather than synchronizing one accelerator scalar per row/tensor.
        before = torch.stack([row["chain"][int(step)] for row, step in zip(rows, step_indices, strict=True)]).to(self.config.device)
        after = torch.stack([row["chain"][int(step) + 1] for row, step in zip(rows, step_indices, strict=True)]).to(self.config.device)
        old = torch.stack([row["logprobs"][int(step)] for row, step in zip(rows, step_indices, strict=True)]).to(self.config.device)
        local_steps = torch.tensor(local_step_indices, device=self.config.device)
        steps = local_steps + self.total_steps - self.ft_steps
        current = self.policy.log_prob(self._conditioning(rows), before, after, steps)
        return chunk_indices, local_steps, rows, old, current

    def _minibatch_selection(self, selected):
        """Single-learner default; distributed learners partition this global batch."""
        return selected, len(selected)

    def _shuffle_update_indices(self, indices):
        self.rng.shuffle(indices)

    def _aggregate_minibatch_metrics(self, metrics, sample_count):
        return metrics_to_floats(metrics)

    def _synchronize_gradients(self, parameters, sample_count):
        """Called after backward and before clipping/step; one learner needs no reduction."""
        return None

    @torch.no_grad()
    def _verify_rollout(self, records):
        """Re-score every sampled pair before the first optimizer mutation.

        Clipping/averaging alone can conceal wrong likelihoods: compare raw
        elementwise errors as well as the actual reduced PPO log ratio.
        """
        prepare = getattr(self.expert, "prepare_condition_cache", None)
        if callable(prepare):
            prepared = prepare([record["conditioning"] for record in records])
            if prepared is not None:
                self._log("events.jsonl", {"event": "frozen_prefix_prepared", "policy_version": self.policy_version,
                                          "iteration": self.iterations, **prepared})
        cfg = self.config.dppo
        indices = np.arange(len(records) * self.ft_steps)
        maximum = reduced_maximum = square_sum = 0.
        count = 0
        for offset in range(0, len(indices), cfg.minibatch_size):
            *_, old, current = self._pair_batch(records, indices[offset:offset + cfg.minibatch_size])
            difference = current.float() - old.float()
            maximum = max(maximum, float(difference.abs().max()))
            square_sum += float(difference.square().sum())
            count += difference.numel()
            reduced = reduce_logprobs(current, cfg, execution_horizon=self.contract["execution_horizon"])
            reduced -= reduce_logprobs(old, cfg, execution_horizon=self.contract["execution_horizon"])
            reduced_maximum = max(reduced_maximum, float(reduced.abs().max()))
        measured = {"raw_logprob_max_error": maximum, "raw_logprob_rms_error": math.sqrt(square_sum / count),
                    "reduced_logprob_max_error": reduced_maximum, "verified_pairs": len(indices)}
        self._log("events.jsonl", {"event": "rollout_logprob_verified", "policy_version": self.policy_version, **measured})
        if maximum > cfg.initial_logprob_atol or reduced_maximum > cfg.initial_reduced_logprob_atol:
            raise ValueError(f"Rollout/trainer log-probability mismatch: raw_max={maximum:.6g}, reduced_max={reduced_maximum:.6g}")
        return measured

    def _update(self, records):
        if not records:
            return
        if any(record["policy_version"] != self.policy_version for record in records):
            raise ValueError("Stale samples cannot enter on-policy PPO update")
        self.expert.eval()  # Disable stochastic dropout; do not disable gradients.
        cfg = self.config.dppo
        # Warmup still verifies the actual on-policy sampling distribution, but
        # _verify_rollout runs under no_grad and does not evaluate actor KL.
        verified = self._verify_rollout(records)
        advantages = self._advantages(records)
        raw_advantages = np.asarray([row["advantage"] for row in records], dtype=np.float64)
        returns = np.asarray([row["return"] for row in records], dtype=np.float64)
        diagnostics = {"raw_advantage_std": float(raw_advantages.std()),
            "nonzero_reward_count": sum(row["reward"] != 0 for row in records),
            "return_min": float(returns.min()), "return_max": float(returns.max())}
        warmup = self.iterations < cfg.critic_warmup_iterations
        indices = np.arange(len(records) * self.ft_steps)
        planned = cfg.epochs * math.ceil(len(indices) / cfg.minibatch_size)
        metrics = []
        actor_steps = critic_steps = rejected = 0
        rejected_kl = None
        stop_actor = stop_iteration = False
        if warmup:
            self.actor_optimizer.zero_grad()
        for epoch in range(cfg.epochs):
            self._shuffle_update_indices(indices)
            for offset in range(0, len(indices), cfg.minibatch_size):
                selected, sample_count = self._minibatch_selection(indices[offset:offset + cfg.minibatch_size])
                row_metrics = {}
                if not warmup and not stop_actor:
                    chunk_indices, local_steps, rows, old, current = self._pair_batch(records, selected)
                    policy_loss, row_metrics = clipped_policy_loss(current, old,
                        advantages[chunk_indices].to(self.config.device), local_steps, self.ft_steps, cfg,
                        execution_horizon=self.contract["execution_horizon"], tensor_metrics=True)
                    row_metrics = self._aggregate_minibatch_metrics(row_metrics, sample_count)
                    if cfg.target_kl is not None and row_metrics["approx_kl"] > cfg.target_kl:
                        rejected_kl = row_metrics["approx_kl"]
                        rejected += 1
                        stop_actor = True
                        # Rejected statistics are reported separately from the
                        # accepted-minibatch means, including the triggering KL.
                        row_metrics = {}
                        if not cfg.critic_continue_after_kl:
                            stop_iteration = True
                            break
                    else:
                        self.actor_optimizer.zero_grad()
                        policy_loss.backward()
                        self._synchronize_gradients(self.parameters, sample_count)
                        self._learner_state_consistent = False
                        actor_grad = self.actor_optimizer.step(cfg.max_grad_norm)
                        actor_steps += 1
                        self.actor_updates += 1
                        row_metrics["actor_grad_norm"] = float(actor_grad)
                else:
                    # No actor forward graph is constructed during warmup or
                    # after an actor KL stop. Retain upstream critic pair counts.
                    rows = [records[int(index)] for index in selected // self.ft_steps]
                # Critic uses chunk states, not denoising states. Repeated chunk
                # indices follow the upstream chunk/denoising pair minibatches.
                vectors = torch.stack([row["vector"] for row in rows]).to(self.config.device)
                prediction = self.value(vectors).squeeze(-1)
                targets = torch.tensor([row["return"] for row in rows], device=self.config.device)
                previous = torch.tensor([row["value"] for row in rows], device=self.config.device)
                value_loss = clipped_value_loss(prediction, previous, targets, cfg.value_clip)
                self.value_optimizer.zero_grad(set_to_none=True)
                (cfg.value_coefficient * value_loss).backward()
                self._synchronize_gradients(list(self.value.parameters()), sample_count)
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
                self._learner_state_consistent = False
                self.value_optimizer.step()
                self.updates += 1  # Historical completed optimization minibatches.
                self.critic_updates += 1
                critic_steps += 1
                row_metrics.update(self._aggregate_minibatch_metrics(
                    {"value_loss": value_loss.detach()}, sample_count))
                metrics.append(row_metrics)
            if stop_iteration:
                break
        self.iterations += 1
        keys = set().union(*(row.keys() for row in metrics))
        self.latest_metrics = {key: float(np.mean([row[key] for row in metrics if key in row])) for key in keys}
        self.latest_metrics.update(early_stop_kl=stop_actor, rejected_approx_kl=rejected_kl,
            critic_warmup=warmup, critic_warmup_iterations_completed=min(self.iterations, cfg.critic_warmup_iterations),
            actor_minibatches_accepted=actor_steps, actor_minibatches_rejected=rejected,
            actor_minibatches_skipped=planned - actor_steps - rejected,
            critic_minibatches=critic_steps, planned_minibatches=planned,
            rollout_transitions=len(records), **verified, **diagnostics)
        self._last_optimized_transitions = self.transitions
        self._unoptimized_rollout_transitions = 0
        self._learner_state_consistent = True
        self._publication_state = "acknowledged" if warmup else "pending"
        self._log("events.jsonl", {"event": "iteration_optimized", "iteration": self.iterations,
            "policy_version": self.policy_version, **self.latest_metrics})
        # Warmup changes no served weights. Preserve normal-phase publication
        # semantics, including a KL rejection before any actor step.
        if not warmup:
            self._publish()

    def _publish(self):
        from .expert_sync import publish_expert
        expected, new = self.policy_version, self.policy_version + 1
        self._publication_state = "pending"
        acknowledged = publish_expert(self.inference, self.expert.state_dict(), expected_version=expected, new_version=new)
        if acknowledged.get("policy_version") != new:
            raise ValueError("Inference did not acknowledge the new policy version")
        self.policy_version = new
        self._publication_state = "acknowledged"
        self._log("events.jsonl", {"event": "expert_published", "policy_version": new,
            "transitions": self.transitions, "updates": self.updates, "weights_sha256": acknowledged.get("weights_sha256")})

    def _save(self, *, final=False, recovery=None):
        if self.config.mode != "train" or self.expert is None:
            return
        bucket = self.transitions // self.config.checkpoint.interval
        coordinates = (self.transitions, self.updates)
        if recovery is None and (self._last_saved_coordinates == coordinates
                                 or not final and bucket <= self._last_checkpoint_bucket):
            return
        folder = self.output / "checkpoints"
        folder.mkdir(exist_ok=True)
        name = (f"step_{self.transitions:09d}.pt" if recovery is None else
                f"recovery_step_{self.transitions:09d}_iter_{self.iterations:06d}_{self.run_id[:8]}.pt")
        path = folder / name
        interrupted_envs = set(self.slots) | self._pending_physics_envs
        payload = {"format": "pressb_dppo_v1", "schema_fingerprint": self.schema_fingerprint,
            "expert": {key: value.detach().cpu() for key, value in self.expert.state_dict().items()},
            "value": self.value.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(), "policy_version": self.policy_version,
            "transitions": self.transitions, "updates": self.updates, "iterations": self.iterations,
            "actor_updates": self.actor_updates, "critic_updates": self.critic_updates,
            "discarded_rollout_transitions": self.discarded_rollout_transitions,
            "episodes": self.episodes, "successes": self.successes, "recent": list(self.recent),
            "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "episode_serial": dict(self.episode_serial), "config": asdict(self.config),
            "resume_episode_serial": {eid: serial + int(eid in interrupted_envs) for eid, serial in self.episode_serial.items()},
            "last_optimized_transitions": self._last_optimized_transitions,
            "checkpoint_interval_bucket": bucket if recovery is None else self._last_checkpoint_bucket,
            "simulation_continuation": "fresh episodes; no physics state or on-policy rollout is resumed"}
        payload.update(_device_rng_payload(self.config.device))
        if recovery is not None:
            payload["recovery"] = deepcopy(recovery)
        payload["config"]["endpoints"]["token"] = None
        if self.config.checkpoint_sink is None:
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            self.last_checkpoint = str(path)
            receipt = {"path": path.name}
        else:
            receipt = save_checkpoint_stream(payload, path.name, self.config.checkpoint_sink)
            self.last_checkpoint = receipt["url"]
            self._log("events.jsonl", {"event": "checkpoint_streamed", **receipt})
        if recovery is None:
            self._last_checkpoint_bucket = bucket
            self._last_saved_coordinates = coordinates
        marker = "latest.json" if recovery is None else "recovery_latest.json"
        _atomic_json(folder / marker, {**receipt, "transitions": self.transitions, "policy_version": self.policy_version})
        return receipt

    def _save_failure_checkpoint(self, error):
        """Persist complete learner state, without replaying an unknown RPC.

        Collected-but-unoptimized records are intentionally discarded. If an
        optimizer failed midway, only an earlier durable checkpoint is valid;
        the live parameters must not be labelled a complete update. A fully
        optimized learner whose publication acknowledgement was lost *is*
        resumable: restore negotiates the receiver's version and republishes.
        """
        if self.config.mode != "train" or not self._learner_state_consistent:
            return {"state": "skipped", "reason": "learner_not_at_complete_update_state",
                    "previous_checkpoint": self.last_checkpoint}
        discarded = self._unoptimized_rollout_transitions
        self.discarded_rollout_transitions += discarded
        self._unoptimized_rollout_transitions = 0
        recovery = {
            "error_type": type(error).__name__, "error": str(error),
            "learner_state": "complete_iteration" if self.iterations else "initialized",
            "last_optimized_transitions": self._last_optimized_transitions,
            "confirmed_discarded_rollout_transitions": discarded,
            "physics_request_in_flight": self._physics_in_flight,
            "pending_physics_kind": self._pending_physics_kind,
            "pending_physics_env_ids": sorted(self._pending_physics_envs),
            "unconfirmed_transitions": None if self._physics_in_flight else 0,
            "publication_state": self._publication_state,
            "on_policy_rollout_discarded": True,
            "simulation_continuation": "fresh episodes; never retry the interrupted action request",
        }
        try:
            receipt = self._save(final=True, recovery=recovery)
            if receipt is None:
                return {"state": "skipped", "reason": "checkpoint_writer_did_not_write"}
            result = {"state": "saved", "checkpoint": self.last_checkpoint, **recovery}
        except BaseException as saving_error:
            # Preserve the original transport/update failure when storage also
            # fails. In particular do not retry a physical action to save state.
            result = {"state": "failed", "error_type": type(saving_error).__name__,
                      "error": str(saving_error), "previous_checkpoint": self.last_checkpoint}
        try:
            self._log("events.jsonl", {"event": "failure_checkpoint", **result})
        except Exception:
            pass
        return result

    def _restore(self):
        location = self.config.checkpoint.resume
        if isinstance(location, str) and location.startswith(("http://", "https://")):
            options = self.config.checkpoint_sink
            payload = load_checkpoint_stream(location, map_location=self.config.device,
                **({key: getattr(options, key) for key in ("timeout", "max_bytes", "token_env")} if options else {}))
        else:
            payload = torch.load(location, map_location=self.config.device, weights_only=True)
        if payload.get("format") != "pressb_dppo_v1" or payload.get("schema_fingerprint") != self.schema_fingerprint:
            raise ValueError("DPPO checkpoint model/task/sampler/algorithm identity differs")
        self.expert.load_state_dict(payload["expert"], strict=True)
        self.value.load_state_dict(payload["value"], strict=True)
        if self.config.mode == "train":
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.value_optimizer.load_state_dict(payload["value_optimizer"])
            for key in ("transitions", "updates", "iterations", "episodes", "successes"):
                setattr(self, key, payload[key])
            # Legacy checkpoints completed actor and critic together.
            self.actor_updates = payload.get("actor_updates", payload["updates"])
            self.critic_updates = payload.get("critic_updates", payload["updates"])
            self.initial_transitions = self.transitions
            self._last_optimized_transitions = payload.get("last_optimized_transitions", self.transitions)
            self.discarded_rollout_transitions = payload.get("discarded_rollout_transitions", 0)
            self._unoptimized_rollout_transitions = 0
            self.recent.extend(payload["recent"])
            self.rng.bit_generator.state = payload["rng"]
            torch.set_rng_state(payload["torch_rng"].cpu())
            _restore_device_rng(payload, self.config.device)
            self.episode_serial.update(payload.get("resume_episode_serial", payload["episode_serial"]))
            interval = self.config.checkpoint.interval
            saved_interval = payload.get("config", {}).get("checkpoint", {}).get("interval")
            if "recovery" in payload:
                # A recovery snapshot may be ahead of the last regular save.
                # The old bucket's lower bound is in transitions. Mapping it
                # to the new interval may save early, but cannot skip a save.
                regular_boundary = payload.get("checkpoint_interval_bucket", 0) * (saved_interval or 0)
            else:
                regular_boundary = self.transitions
            self._last_checkpoint_bucket = regular_boundary // interval
            self._last_saved_coordinates = None if "recovery" in payload else (self.transitions, self.updates)
            self.last_checkpoint = str(location)
        self._log("events.jsonl", {"event": "restored", "checkpoint": str(self.config.checkpoint.resume),
            "saved_policy_version": payload["policy_version"], "simulation_continuation": "fresh episodes"})

    def run(self):
        if self.started is not None:
            raise RuntimeError("A DPPO trainer can run only once")
        # Pin the checkpoint before this run can replace its own prior status.
        self.config.checkpoint.resume = resolve_checkpoint(self.config.checkpoint.resume, "dppo")
        if self.output.exists() and any(self.output.iterdir()) and not self.config.checkpoint.resume:
            raise FileExistsError(f"Output directory is not empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        saved_config = asdict(self.config)
        saved_config["endpoints"]["token"] = None
        config_path = self.output / "config.json"
        if config_path.exists():
            config_path = self.output / f"config_resume_{self.run_id}.json"
        _atomic_json(config_path, saved_config)
        if self.config.wandb.enabled:
            self._wandb = _WandbMonitor(self.output, self.config.wandb, saved_config, self.run_id)
            self._wandb.start()
        failure = None
        try:
            self._status_write("negotiating")
            self._negotiate()
            self._learner_state_consistent = True
            self._publication_state = "acknowledged"
            if self._remaining() > 0:
                self._reset(list(range(self.num_envs)), initial=True)
            while self.slots and self._remaining() > 0 and not self.stop_event.is_set():
                records = self._collect()
                if self.config.mode == "train" and records:
                    self._status_write("updating")
                    self._update(records)
                    self._save()
                del records  # All samples are discarded before the next version.
                self._status_write("running")
                self._log("metrics.jsonl", self.health())
            self._save(final=True)
            state = "stopped" if self.stop_event.is_set() else "completed"
            if self.config.mode == "eval" and self.episodes < self.config.rollout.eval_episodes and state == "completed":
                state = "transition_limit"
            self._status_write(state)
        except BaseException as error:
            failure = error
            recovery = self._save_failure_checkpoint(error)
            self._status_write("failed", error_type=type(error).__name__, error=str(error),
                execution_state="unknown; request was not retried" if self._physics_in_flight else "no automatic continuation",
                recovery_checkpoint=recovery)
            raise
        finally:
            if self._claimed:
                try:
                    self._call(self.simulation, "/close_run", {"run_id": self.run_id})
                except Exception as error:
                    self._log("events.jsonl", {"event": "close_failed", "error_type": type(error).__name__})
            for client in (self.inference, self.simulation):
                close = getattr(client, "close", None)
                if close:
                    close()
            if self.backend:
                self.backend.close()
            if self._wandb:
                self._wandb.close()
            if self.observer:
                try:
                    self.observer.close(self.health())
                except BaseException:
                    if failure is None:
                        raise
        return self.health()
