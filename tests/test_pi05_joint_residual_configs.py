"""The inherited PiPER and FR3 recipes must retain their own robot and model geometry."""
from pathlib import Path

import pytest

from pressb.experiments.config import PROJECT_ROOT, load_experiment
from pressb.runtime.learner import LearnerConfig
from pressb.runtime.trainer import TrainerConfig


@pytest.mark.parametrize(("task", "joints"), [("pap_plus", 7), ("pap_plus_plus", 7), ("fr3_pap", 8)])
def test_pi05_joint_residual_recipe(task, joints):
    name = f"{task}_pi05_joint_residual"
    nodes = load_experiment(PROJECT_ROOT / "configs" / "experiments" / f"{name}.json")["nodes"]
    simulator = nodes["simulation"]["backend"]
    assert simulator["plugin"] == task
    assert nodes["inference"]["backend"]["plugin"] == "pi05"
    assert simulator["config"].get("image_preprocessing") is None

    run = TrainerConfig(**nodes["trainer"])
    assert run.method == "joint_residual" and run.mode == "train"
    # A held-out rollout seed must not change the learner config bound to its checkpoint.
    assert run.learner["seed"] == 42
    assert run.observation_state_encoding == "raw" and not run.endpoints.preprocess_images
    assert run.frozen_dsrl is None and run.residual_warmup is None and run.checkpoint.resume is None
    assert name in Path(run.output_dir).parts
    horizon = simulator["config"]["horizon"]
    assert run.rollout.execution_horizon == horizon == 16
    # The learner concatenates 2048 visual features, joints and controller state.
    geometry = dict(run.learner, method=run.method, obs_dim=2048 + 2 * joints,
                    action_horizon=horizon, model_action_dim=joints)
    learner = LearnerConfig(**geometry)
    assert learner.noise_dim == 32 and learner.noise_steps == 1
    assert learner.residual_mode == "all" and learner.residual_indices == tuple(range(joints))
    assert learner.action_dim == horizon * joints + 32
