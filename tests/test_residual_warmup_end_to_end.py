"""Real HTTP transport smoke tests with synthetic flow/plant, not Isaac benchmarks."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from pressb.runtime.inference import InferenceService
from pressb.runtime.simulation import SimulationService
from pressb.runtime.trainer import GenericTrainer
from test_rl_nodes_end_to_end import Flow, Plant, endpoint


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RecordingPlant(Plant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []

    def step(self, items, *, terminal_observation_mode="full"):
        self.executed.extend(deepcopy(items))
        return super().step(items, terminal_observation_mode=terminal_observation_mode)


@pytest.mark.parametrize("fused", [False, True])
def test_http_dsrl_execution_residual_updates_checkpoint_and_resume(tmp_path, fused):
    horizon, action_dim, noise_dim = 4, 3, 3
    inference = InferenceService(Flow(horizon, action_dim, noise_dim), batch_size=2, cache_size=8)
    plant = RecordingPlant(horizon, action_dim, noise_dim, fused=fused)
    simulation = SimulationService(plant)
    learner = dict(hidden_dim=8, batch_size=2, num_qs=2, num_subset=2)
    with endpoint(inference) as inf, endpoint(simulation) as sim:
        settings = dict(
            method="dsrl", output_dir=str(tmp_path / "first"),
            endpoints=dict(inference=inf, simulation=sim),
            learner={**learner, "noise_steps": horizon},
            rollout=dict(max_transitions=10, execution_horizon=2),
            update=dict(learning_starts=0, replay_capacity=32, utd=1., max_updates_per_wait=2),
            residual_warmup=dict(
                learner={**learner, "residual_scale": [.02] * action_dim},
                learning_starts=0, replay_capacity=32, utd=1., max_updates_per_wait=2),
            checkpoint=dict(interval=4, keep=2, include_replay=True),
        )
        if fused:
            settings["rollout"].update(
                terminal_observation_mode="state_only_if_no_bootstrap", fused_step_reset=True)
        runner = GenericTrainer(settings)
        report = runner.run()
        shadow = runner.residual_warmup
        assert report["state"] == "completed" and report["transitions"] == 10
        assert runner.learner.updates > 0 and shadow.learner.updates > 0
        assert shadow.learner.actor_optimizer.state and shadow.learner.q_optimizer.state
        assert len(shadow.replay) == plant.steps == len(plant.executed) == 10
        assert np.all(shadow.replay.arrays["action"][:10] == 0)
        assert not shadow.pending
        # Reconstruct commands from actual noise replay and the synthetic flow.
        # This would fail if any warmup actor output entered the execution path.
        behavior = runner.replay.arrays
        state = behavior["obs"][:10, :action_dim]
        noise = behavior["action"][:10].reshape(10, horizon, noise_dim)
        expected = .5 * state[:, None] + .05 * noise * runner.learner.config.noise_scale
        executed = np.asarray([item["actions"] for item in plant.executed])
        np.testing.assert_allclose(executed, expected[:, :2], atol=1e-7, rtol=1e-6)
        assert not inference.contexts and simulation.run_id is None
        if fused:
            assert report["rpc_calls"]["simulation/reset"] == 1
            assert report["rpc_calls"]["simulation/step_and_reset"] == 5
            assert "simulation/step" not in report["rpc_calls"]
        else:
            assert report["rpc_calls"]["simulation/step"] == 5
        saved = torch.load(report["last_checkpoint"], weights_only=False)
        saved_shadow = saved["extra"]["residual_warmup"]
        assert saved_shadow["replay"]["size"] == 10
        prior_arrays = {name: value.copy() for name, value in saved_shadow["replay"]["arrays"].items()}
        settings["output_dir"] = str(tmp_path / "resumed")
        settings["rollout"]["max_transitions"] = 14
        settings["checkpoint"]["resume"] = report["last_checkpoint"]
        resumed = GenericTrainer(settings)
        resumed_report = resumed.run()
        assert resumed_report["transitions"] == plant.steps == len(plant.executed) == 14
        assert resumed.learner.updates >= runner.learner.updates
        assert resumed.residual_warmup.learner.updates > shadow.learner.updates
        assert len(resumed.residual_warmup.replay) == 14
        for name, array in prior_arrays.items():
            np.testing.assert_array_equal(resumed.residual_warmup.replay.arrays[name][:10], array)
        assert not resumed.residual_warmup.pending
        assert not inference.contexts and simulation.run_id is None
