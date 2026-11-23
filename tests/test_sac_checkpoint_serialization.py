"""Binary NumPy checkpoint storage retains SAC/replay continuation semantics."""
from copy import deepcopy
import hashlib
import zipfile

import numpy as np
import pytest
import torch

from pressb.runtime.learner import CHECKPOINT_VERSION, LearnerConfig, ReplayBuffer, SACLearner


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_exact(a, b)
    else:
        assert left == right


def trained_fixture(method):
    cfg = LearnerConfig(method=method, obs_dim=6, hidden_dim=8, batch_size=4,
                        action_horizon=4, model_action_dim=3, noise_dim=5,
                        noise_steps=4, residual_mode='all', residual_scale=(.1, .2, .3),
                        residual_q_bounds=None, policy_freq=1, target_freq=1, seed=37)
    learner = SACLearner(cfg, device='cpu')
    replay = ReplayBuffer(8, cfg.obs_dim, cfg.action_dim, seed=13,
                          base_action_shape=cfg.base_action_shape)
    rng = np.random.default_rng(11)
    for i in range(12):
        base = rng.normal(size=cfg.base_action_shape).astype(np.float32) if learner.residual else None
        replay.add(rng.normal(size=cfg.obs_dim), rng.uniform(-1, 1, cfg.action_dim),
                   float(i % 3 == 0), .95, rng.normal(size=cfg.obs_dim),
                   i % 5 == 0, i % 5 == 1, base, None if base is None else base + .01)
    learner.update(replay.sample(cfg.batch_size))
    learner.update(replay.sample(cfg.batch_size))
    return learner, replay


@pytest.mark.parametrize('method', ['initial_noise', 'action_residual'])
def test_binary_checkpoint_matches_legacy_payload_and_exact_resume(tmp_path, method):
    learner, replay = trained_fixture(method)
    # Include the same nested learner/replay structure used by DSRL shadow warmup.
    extra = {'replay': replay.state_dict(), 'residual_warmup': {
        'version': 1, 'learner': deepcopy(learner.checkpoint_state()),
        'replay': replay.state_dict()}, 'provenance': {'schema': 'fixture'}}
    legacy = tmp_path / 'legacy_protocol2.pt'
    binary = tmp_path / 'binary.pt'
    torch.save(learner.checkpoint_state(extra), legacy, pickle_protocol=2)
    learner.save(binary, extra=extra)

    # Both ordinary and mmap trusted loading paths must retain NumPy replay.
    expected = torch.load(legacy, map_location='cpu', weights_only=False, mmap=True)
    actual = torch.load(binary, map_location='cpu', weights_only=False, mmap=True)
    assert expected['version'] == actual['version'] == CHECKPOINT_VERSION
    assert_exact(expected, actual)
    assert type(actual['extra']['replay']['arrays']['obs']) is np.ndarray
    assert type(actual['extra']['residual_warmup']['replay']['arrays']['action']) is np.ndarray
    with zipfile.ZipFile(binary) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        with archive.open(name) as stream:
            assert stream.read(2) == b'\x80\x05'

    def continue_from(path):
        restored = SACLearner.load(path, device='cpu')
        replay2 = ReplayBuffer(8, restored.config.obs_dim, restored.action_dim,
                               base_action_shape=restored.config.base_action_shape)
        replay2.load_state_dict(restored.checkpoint_extra['replay'])
        assert all(array.flags.writeable for array in replay2.arrays.values())
        batch = replay2.sample(4)
        action = restored.act(batch['obs'], batch.get('base_action'), exploration_probability=.4)
        metrics = restored.update(batch)
        # Exercise write-after-resume, including a wrapped replay ring.
        replay2.add(batch['obs'][0], batch['action'][0], float(batch['reward'][0, 0]),
                    float(batch['discount'][0, 0]), batch['next_obs'][0],
                    bool(batch['terminated'][0, 0]), bool(batch['truncated'][0, 0]),
                    batch.get('base_action', [None])[0], batch.get('next_base_action', [None])[0])
        return action, metrics, restored.checkpoint_state(extra={'replay': replay2.state_dict()})

    old_action, old_metrics, old_state = continue_from(legacy)
    new_action, new_metrics, new_state = continue_from(binary)
    np.testing.assert_array_equal(new_action, old_action)
    assert new_metrics == old_metrics
    assert_exact(old_state, new_state)


def test_binary_numpy_checkpoint_avoids_utf8_size_expansion(tmp_path):
    learner, _ = trained_fixture('initial_noise')
    # Arbitrary byte values model binary replay storage without a large test.
    data = np.tile(np.arange(256, dtype=np.uint8), 4096)
    binary, legacy = tmp_path / 'binary.pt', tmp_path / 'legacy.pt'
    learner.save(binary, extra={'binary_replay_fixture': data})
    torch.save(learner.checkpoint_state({'binary_replay_fixture': data}), legacy, pickle_protocol=2)
    assert legacy.stat().st_size - binary.stat().st_size > data.nbytes * .45
    restored = torch.load(binary, map_location='cpu', weights_only=False, mmap=True)
    np.testing.assert_array_equal(restored['extra']['binary_replay_fixture'], data)


def test_failed_save_preserves_previous_checkpoint_and_removes_temporary_file(tmp_path, monkeypatch):
    learner, replay = trained_fixture('action_residual')
    path = tmp_path / 'checkpoint.pt'
    learner.save(path, {'replay': replay.state_dict()})
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def fail(payload, stream, **kwargs):
        assert kwargs['pickle_protocol'] == 5
        stream.write(b'incomplete checkpoint')
        raise OSError('fixture disk write failed')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='fixture disk write failed'):
        learner.save(path, {'replay': replay.state_dict()})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob('.checkpoint.pt.*'))
