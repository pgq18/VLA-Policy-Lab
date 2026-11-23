"""Residual networks can learn from DSRL behavior without executing a residual."""
from dataclasses import asdict
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pressb.runtime.learner import (
    LearnerConfig, ResidualWarmupLossConfig, SACLearner,
)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_learner(**overrides):
    cfg = dict(method="action_residual", obs_dim=4, hidden_dim=8,
               action_horizon=2, model_action_dim=3, residual_mode="all",
               residual_scale=(.1, .2, .3), batch_size=3,
               policy_freq=1, target_freq=1, residual_q_bounds=None)
    cfg.update(overrides)
    return SACLearner(LearnerConfig(**cfg))


def make_batch(learner):
    cfg = learner.config
    rng = np.random.default_rng(51)
    return {
        "obs": rng.normal(size=(3, cfg.obs_dim)).astype(np.float32),
        "next_obs": rng.normal(size=(3, cfg.obs_dim)).astype(np.float32),
        "action": np.zeros((3, cfg.action_dim), np.float32),
        "base_action": rng.normal(size=(3, *cfg.base_action_shape)).astype(np.float32),
        "next_base_action": rng.normal(size=(3, *cfg.base_action_shape)).astype(np.float32),
        "reward": np.array([[1.], [0.], [-.1]], np.float32),
        "discount": np.full((3, 1), .95, np.float32),
        "terminated": np.array([[True], [False], [False]]),
        "truncated": np.array([[False], [True], [False]]),
    }


def clone_parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def changed(before, module):
    return any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))


@pytest.mark.parametrize("bootstrap_truncated", [False, True])
def test_behavior_target_uses_next_dsrl_action_and_real_termination(monkeypatch, bootstrap_truncated):
    learner = make_learner(bootstrap_truncated=bootstrap_truncated,
                           action_lower=(-1.,) * 3, action_upper=(1.,) * 3)
    batch = learner._prepare_batch(make_batch(learner))
    evaluated = []

    def no_actor(*args, **kwargs):
        raise AssertionError("The behavior critic target must not sample an unexecuted residual")

    def value_of_action(obs, action):
        evaluated.append(action.detach().clone())
        return action.sum(-1, keepdim=True).unsqueeze(0).expand(2, -1, -1)

    monkeypatch.setattr(learner.actor, "sample", no_actor)
    monkeypatch.setattr(learner.q_targets, "forward", value_of_action)
    target = learner._bellman_target(batch, warmup=ResidualWarmupLossConfig())
    expected_action = batch["next_base_action"].clamp(-1., 1.)
    torch.testing.assert_close(evaluated[0], expected_action)
    bootstrap = 1 - batch["terminated"]
    if not bootstrap_truncated:
        bootstrap *= 1 - batch["truncated"]
    expected = batch["reward"] + bootstrap * batch["discount"] * expected_action.sum(-1, keepdim=True)
    torch.testing.assert_close(target, expected)


def test_actor_target_mode_is_explicitly_available(monkeypatch):
    learner = make_learner()
    batch = learner._prepare_batch(make_batch(learner))
    captured = []
    monkeypatch.setattr(learner.actor, "sample", lambda obs: (
        torch.ones((3, learner.action_dim)), torch.zeros((3, 1)), torch.zeros((3, learner.action_dim))))
    hook = learner.q_targets.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[1].detach().clone()))
    try:
        learner._bellman_target(batch, warmup=ResidualWarmupLossConfig(target_mode="actor"))
    finally:
        hook.remove()
    torch.testing.assert_close(captured[0], batch["next_base_action"] + learner.scale)


@pytest.mark.parametrize("l2_weight", [0., 1.])
def test_actor_and_critic_really_update_and_alpha_stays_frozen(l2_weight):
    learner = make_learner()
    batch = make_batch(learner)
    actor, critic, targets = map(clone_parameters, (learner.actor, learner.qs, learner.q_targets))
    log_alpha = learner.log_alpha.detach().clone()
    result = learner.update(batch, warmup=ResidualWarmupLossConfig(actor_l2_weight=l2_weight))
    assert changed(actor, learner.actor), "Q-only warmup must train the actor even without L2"
    assert changed(critic, learner.qs)
    assert changed(targets, learner.q_targets)
    assert result["actor_updated"] == 1.
    assert result["actor_grad_norm"] > 0.
    assert result["warmup_behavior_target"] == 1.
    assert result["warmup_residual_l2"] >= 0.
    assert "alpha_loss" not in result
    assert torch.equal(log_alpha, learner.log_alpha)
    assert not learner.alpha_optimizer.state
    assert all(np.isfinite(value) for value in result.values())
    # A phase switch needs no reinitialization: ordinary SAC updates resume,
    # including automatic temperature adaptation and pre-existing optimizers.
    ordinary = learner.update(batch)
    assert "alpha_loss" in ordinary
    assert learner.alpha_optimizer.state
    assert learner.updates == 2


def test_actor_anchor_penalizes_sampled_variance_and_keeps_caps(monkeypatch):
    learner = make_learner(lambda_s=2., lambda_t=3.)
    batch = learner._prepare_batch(make_batch(learner))
    monkeypatch.setattr(learner.actor, "sample", lambda obs: (
        torch.full((3, learner.action_dim), .5),
        torch.full((3, 1), 2.), torch.zeros((3, learner.action_dim))))
    monkeypatch.setattr(learner.qs, "forward", lambda obs, action: torch.full((2, 3, 1), 4.))
    cfg = ResidualWarmupLossConfig(actor_q_weight=2., actor_l2_weight=3., entropy_weight=.5)
    loss, metrics = learner._actor_loss(batch, warmup=cfg)
    assert metrics["warmup_residual_l2"] == .25  # Despite the zero mean.
    expected_rl = .5 * learner.alpha * 2. - 2. * 4.
    expected = (expected_rl + 3. * .25 + 2. * metrics["spatial_smoothness_loss"]
                + 3. * metrics["temporal_smoothness_loss"])
    assert float(loss) == pytest.approx(expected)
    assert metrics["actor_rl_loss"] == pytest.approx(expected_rl)


def test_warmup_rejects_noise_learner_and_unexecuted_action_labels():
    noise = make_learner(method="initial_noise")
    with pytest.raises(ValueError, match="action_residual"):
        noise.update(make_batch(noise), warmup=ResidualWarmupLossConfig())
    residual = make_learner()
    batch = make_batch(residual)
    batch["action"][0, 0] = .1
    with pytest.raises(ValueError, match="zero executed residual"):
        residual.update(batch, warmup=ResidualWarmupLossConfig())
    assert residual.updates == 0
    with pytest.raises(ValueError, match="ResidualWarmupLossConfig"):
        residual.update(make_batch(residual), warmup={})


@pytest.mark.parametrize("overrides", [
    {"target_mode": "invalid"}, {"actor_q_weight": 0.}, {"actor_q_weight": -1.},
    {"actor_q_weight": True}, {"actor_l2_weight": -1.}, {"actor_l2_weight": "1"},
    {"actor_l2_weight": float("nan")}, {"entropy_weight": float("inf")},
    {"entropy_weight": False}, {"entropy_weight": -1.},
])
def test_invalid_warmup_objectives_fail_before_training(overrides):
    with pytest.raises(ValueError):
        ResidualWarmupLossConfig(**overrides)


def test_nested_checkpoint_roundtrip_preserves_warmup_and_online_continuation(tmp_path):
    learner = make_learner()
    batch = make_batch(learner)
    warmup = ResidualWarmupLossConfig()
    learner.update(batch, warmup=warmup)
    extra = {"warmup": asdict(warmup), "behavior": "dsrl_only"}
    nested = copy.deepcopy(learner.checkpoint_state(extra))
    path = tmp_path / "residual.pt"
    learner.save(path, extra)
    expected = learner.update(batch, warmup=warmup)
    restored = SACLearner.from_checkpoint_state(nested)
    assert restored.checkpoint_extra == extra
    assert restored.update(batch, warmup=warmup) == expected
    from_file = SACLearner.load(path)
    assert from_file.update(batch, warmup=warmup) == expected
    # Explicit None remains identical to an omitted warmup argument, including
    # random draws, optimization, and alpha updates.
    snapshot = copy.deepcopy(restored.checkpoint_state())
    expected_online = restored.update(batch)
    restored_again = SACLearner.from_checkpoint_state(snapshot)
    assert restored_again.update(batch, warmup=None) == expected_online


def test_nested_checkpoint_identity_is_validated():
    payload = make_learner().checkpoint_state()
    payload["version"] = -1
    with pytest.raises(ValueError, match="checkpoint identity"):
        SACLearner.from_checkpoint_state(payload)
