"""The joint SAC action is [action residual, noise residual] in every update."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pressb.runtime.learner import LearnerConfig, ReplayBuffer, SACLearner


def test_joint_action_reaches_all_critics_and_both_actor_outputs(tmp_path):
    cfg = LearnerConfig(method="joint_residual", obs_dim=4, hidden_dim=8,
                        action_horizon=4, model_action_dim=3, residual_indices=(2, 0),
                        residual_scale=(.2, .1), noise_dim=5, noise_steps=2, policy_freq=1)
    learner = SACLearner(cfg)
    rng = np.random.default_rng(7)
    obs = rng.normal(size=(3, cfg.obs_dim)).astype(np.float32)
    assert cfg.action_residual_dim == 8 and cfg.action_dim == 18
    action = learner.act(obs)
    assert action.shape == (3, 18)
    assert not np.any(learner.act(obs, warmup=True))
    assert not np.any(learner.act(obs, exploration_probability=0))
    replay = ReplayBuffer(3, cfg.obs_dim, cfg.action_dim, base_action_shape=cfg.base_action_shape)
    for i in range(3):
        replay.add(obs[i], action[i], float(i == 0), .99, obs[(i+1) % 3], False, False)
    batch = replay.sample(3)
    inputs = []

    def capture_critic_action(model, args):
        inputs.append(args[1].detach().clone())

    handles = [network.register_forward_pre_hook(capture_critic_action)
               for network in (learner.qs, learner.q_targets)]
    try:
        metrics = learner.update(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert len(inputs) == 3  # next sampled action, replay action, actor loss
    assert all(tuple(value.shape) == (3, cfg.action_dim) for value in inputs)
    torch.testing.assert_close(inputs[1], torch.from_numpy(batch["action"]))
    assert metrics["actor_updated"] == 1 and metrics["target_updated"] == 1
    assert all(np.isfinite(list(metrics.values())))

    # Isolate the policy gradient through Q from the entropy gradient.
    sampled_action = learner.actor.sample(torch.from_numpy(obs))[0]
    value = learner.qs(torch.from_numpy(obs), learner._critic_action(sampled_action, None)).mean()
    grad, = torch.autograd.grad(value, learner.actor.mean.bias)
    assert grad[:cfg.action_residual_dim].abs().sum() > 0
    assert grad[cfg.action_residual_dim:].abs().sum() > 0

    path = tmp_path / "joint.pt"
    learner.save(path, extra={"replay": replay.state_dict()})
    next_action = learner.act(obs)
    next_update = learner.update(batch)
    restored = SACLearner.load(path)
    np.testing.assert_array_equal(restored.act(obs), next_action)
    assert restored.update(batch) == next_update
    np.testing.assert_array_equal(restored.checkpoint_extra["replay"]["arrays"]["action"], action)


def test_joint_composes_only_action_block_with_decoded_base_and_bounds():
    cfg = LearnerConfig(method="joint_residual", obs_dim=4, hidden_dim=8,
                        action_horizon=2, model_action_dim=4, residual_mode="xyz",
                        residual_scale=(.1, .2, .3), action_lower=(-1.,)*4, action_upper=(1.,)*4)
    learner = SACLearner(cfg)
    base = np.array([[[.98, 0., 0., .75], [0., 0., 0., .25]]], np.float32)
    block = np.array([[1, -1, 0, 0, .5, -1]], np.float32)
    expected = [[[1., -.2, 0., .75], [0., .1, -.3, .25]]]
    np.testing.assert_allclose(learner.compose_actions(block, base), expected, atol=1e-7)
