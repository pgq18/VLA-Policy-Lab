"""Residual SAC learning from DSRL behavior, without residual execution.

This component owns no inference/environment client. The trainer supplies the
decoded behavior chunks and resolves continuing rows on the next DSRL decode.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import copy
import math
import random
import time

import numpy as np
import torch

from .learner import ReplayBuffer, ResidualWarmupLossConfig, SACLearner


@dataclass
class ResidualWarmupConfig:
    learner: dict = field(default_factory=dict)
    replay_capacity: int = 100_000
    learning_starts: int = 2_000
    utd: float = 1.
    max_backlog: int = 1_000
    max_updates_per_wait: int = 16
    loss: ResidualWarmupLossConfig = field(default_factory=ResidualWarmupLossConfig)

    def __post_init__(self):
        if not isinstance(self.learner, dict) or "residual_scale" not in self.learner:
            raise ValueError("residual_warmup.learner requires explicit residual_scale in executable units")
        if isinstance(self.loss, dict):
            self.loss = ResidualWarmupLossConfig(**self.loss)
        if not isinstance(self.loss, ResidualWarmupLossConfig):
            raise ValueError("residual_warmup.loss must be an object")
        self.loss.__post_init__()
        for name, minimum in (("replay_capacity", 1), ("learning_starts", 0),
                              ("max_backlog", 1), ("max_updates_per_wait", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"residual_warmup.{name} must be an integer >= {minimum}")
        if type(self.utd) not in (int, float) or not math.isfinite(self.utd) or self.utd < 0:
            raise ValueError("residual_warmup.utd must be finite and nonnegative")


class IsolatedRNG:
    """Keep auxiliary network sampling off the primary learner's generators.

    All learners run on the trainer thread. Only its owned CUDA device is
    touched, never the simulator's GPU. Also protects load's legacy global RNGs.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                        if self.device.type == "cuda" else [])
        self.state = None

    @contextmanager
    def use(self):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        with torch.random.fork_rng(devices=self.devices):
            try:
                if self.state is not None:
                    torch.set_rng_state(self.state["torch"].cpu())
                    if self.devices:
                        torch.cuda.set_rng_state(self.state["cuda"].cpu(), self.device)
                    np.random.set_state(self.state["numpy"])
                    random.setstate(self.state["python"])
                yield
            finally:
                self.state = {"torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state(self.device) if self.devices else None,
                              "numpy": np.random.get_state(), "python": random.getstate()}
                np.random.set_state(numpy_state)
                random.setstate(python_state)


class ResidualWarmup:
    def __init__(self, config, learner_config, device):
        self.config = copy.deepcopy(config)
        if config.replay_capacity < learner_config.batch_size:
            raise ValueError("residual_warmup.replay_capacity must be >= learner.batch_size")
        self.rng = IsolatedRNG(device)
        with self.rng.use():
            self.learner = SACLearner(learner_config, device=device)
        self.replay = ReplayBuffer(config.replay_capacity, learner_config.obs_dim,
                                   learner_config.action_dim, seed=learner_config.seed,
                                   base_action_shape=learner_config.base_action_shape)
        self.pending = {}
        self.transitions = 0
        self.update_credit = self.dropped_update_credit = 0.
        self.learner_seconds = 0.
        self.metrics = {}

    def _add(self, row, next_base, transitions):
        self.replay.add(row["obs"], np.zeros(self.learner.action_dim, np.float32),
                        row["reward"], row["discount"], row["next_obs"],
                        row["terminated"], row["truncated"], row["base"], next_base)
        self.transitions += 1
        if transitions >= self.config.learning_starts:
            credit = self.update_credit + self.config.utd
            self.update_credit = min(credit, self.config.max_backlog)
            self.dropped_update_credit += credit - self.update_credit

    def observe(self, old, new, base, transitions):
        eid = new["env_id"]
        if eid in self.pending:
            raise RuntimeError("Unresolved residual warmup transition before next execution")
        row = {"obs": old["vector"].copy(), "next_obs": new["vector"].copy(),
               "base": base.copy(), "reward": new["reward"], "discount": new["discount"],
               "terminated": new["terminated"], "truncated": new["truncated"],
               "episode_id": new["episode_id"], "step_id": new["step_id"]}
        no_bootstrap = (row["terminated"] or (row["truncated"] and not self.learner.config.bootstrap_truncated)
                        or row["discount"] == 0)
        if no_bootstrap:
            self._add(row, np.zeros_like(base), transitions)
        else:
            self.pending[eid] = row

    def resolve(self, slots, bases, transitions):
        for slot, base in zip(slots, bases, strict=True):
            eid = slot["env_id"]
            row = self.pending.get(eid)
            if row is None:
                continue
            if (row["episode_id"] != slot["episode_id"] or row["step_id"] != slot["step_id"]
                    or not np.array_equal(row["next_obs"], slot["vector"])):
                raise ValueError("Residual warmup next action does not match its transition observation")
            self._add(row, base, transitions)
            del self.pending[eid]

    def can_update(self, transitions):
        return (transitions >= self.config.learning_starts and self.update_credit >= 1
                and len(self.replay) >= self.learner.config.batch_size)

    def update(self):
        started = time.monotonic()
        with self.rng.use():
            self.metrics.update(self.learner.update(self.replay.sample(self.learner.config.batch_size),
                                                   warmup=self.config.loss))
        self.update_credit -= 1.
        self.learner_seconds += time.monotonic() - started

    def status(self):
        return {"executed": False, "transitions": self.transitions, "updates": self.learner.updates,
                "replay_size": len(self.replay), "pending_transitions": len(self.pending),
                "update_backlog": self.update_credit, "dropped_update_credit": self.dropped_update_credit,
                "learner_seconds": self.learner_seconds, "last_update_metrics": self.metrics.copy()}

    def checkpoint_state(self, include_replay):
        if self.pending:
            raise RuntimeError("Resolve residual warmup next actions before checkpointing")
        with self.rng.use():
            learner = self.learner.checkpoint_state()
        result = {"version": 1, "config": asdict(self.config), "learner": learner,
                  "transitions": self.transitions, "update_credit": self.update_credit,
                  "dropped_update_credit": self.dropped_update_credit, "learner_seconds": self.learner_seconds,
                  "last_update_metrics": self.metrics.copy()}
        if include_replay:
            result["replay"] = self.replay.state_dict()
        return result

    def restore(self, state):
        if state.get("version") != 1 or state.get("config") != asdict(self.config):
            raise ValueError("Checkpoint residual_warmup configuration differs")
        desired = asdict(self.learner.config)
        with self.rng.use():
            learner = SACLearner.from_checkpoint_state(state["learner"], device=self.learner.device)
        if asdict(learner.config) != desired:
            raise ValueError("Checkpoint residual_warmup learner configuration differs")
        if "replay" not in state:
            raise ValueError("Residual warmup resume requires include_replay=true")
        self.replay.load_state_dict(state["replay"])
        self.learner = learner
        self.transitions = int(state["transitions"])
        self.update_credit = float(state["update_credit"])
        self.dropped_update_credit = float(state["dropped_update_credit"])
        self.learner_seconds = float(state["learner_seconds"])
        self.metrics = state["last_update_metrics"].copy()
