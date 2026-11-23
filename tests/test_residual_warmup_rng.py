"""A shadow learner resumes independently of the DSRL random stream."""
from copy import deepcopy
import random

import numpy as np
import torch

from pressb.runtime.learner import LearnerConfig
from pressb.runtime.residual_warmup import ResidualWarmup, ResidualWarmupConfig


def test_shadow_checkpoint_continuation_is_independent_of_primary_random_draws():
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        learner_config = LearnerConfig(
            method="action_residual", obs_dim=4, action_horizon=2,
            model_action_dim=3, residual_scale=(.1, .1, .1), residual_mode="all",
            hidden_dim=8, batch_size=2, policy_freq=1, residual_q_bounds=None,
        )
        config = ResidualWarmupConfig(
            learner={"residual_scale": [.1, .1, .1]},
            replay_capacity=8, learning_starts=0,
        )
        original = ResidualWarmup(config, learner_config, "cpu")
        for index in range(6):
            original.replay.add(
                np.full(4, index), np.zeros(6), float(index % 2), .99,
                np.full(4, index + 1), False, False,
                np.full((2, 3), index * .01), np.full((2, 3), (index + 1) * .01),
            )
        original.update_credit = 10.
        original.transitions = 6
        original.update()
        checkpoint = deepcopy(original.checkpoint_state(include_replay=True))

        expected = []
        for _ in range(3):
            torch.rand(51)
            np.random.random(23)
            random.random()
            original.update()
            expected.append(original.metrics.copy())

        primary = (torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
        restored = ResidualWarmup(config, learner_config, "cpu")
        restored.restore(checkpoint)
        assert torch.equal(torch.get_rng_state(), primary[0])
        assert np.array_equal(np.random.get_state()[1], primary[1][1])
        assert np.random.get_state()[2:] == primary[1][2:]
        assert random.getstate() == primary[2]

        actual = []
        for _ in range(3):
            # Simulate a different amount of DSRL work between warmup updates.
            torch.rand(95)
            np.random.random(37)
            for _ in range(7):
                random.random()
            restored.update()
            actual.append(restored.metrics.copy())

        assert actual == expected
        for name in ("actor", "qs", "q_targets"):
            original_state = getattr(original.learner, name).state_dict()
            restored_state = getattr(restored.learner, name).state_dict()
            assert all(torch.equal(value, restored_state[key]) for key, value in original_state.items())
        assert original.update_credit == restored.update_credit
        assert original.learner.updates == restored.learner.updates
    finally:
        torch.set_num_threads(prior_threads)
