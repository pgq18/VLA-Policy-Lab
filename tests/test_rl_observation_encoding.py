"""RL state encoding changes neither model inputs nor simulator observations."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from pressb.control.pose_protocol import pose8_to_pose9
from pressb.runtime.contract import negotiate
from pressb.runtime.observation_encoding import (POSE8_NAMES, POSE8_UNITS,
                                                 configure_rl_observation, encode_rl_state)
from pressb.runtime.trainer import GenericTrainer, TrainerConfig, _schema_fingerprint
from test_rl_nodes_trainer import Inference, StateOnlySimulation, metadata, config


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pose_metadata():
    inference, simulation = metadata(7, 9, 9, 8)
    for item in (inference, simulation):
        item['observation']['state'] = {'names': POSE8_NAMES.copy(), 'units': POSE8_UNITS.copy()}
        item['quaternion_order'] = 'wxyz'
    inference['feature_dim'] = 2048
    simulation.update(control_state_dim=37, task_features_dim=12)
    return inference, simulation


class PoseInference(Inference):
    def call(self, path, payload=None):
        result = super().call(path, payload)
        if path == '/encode':
            for item in result['items']:
                item['features'] += [0.] * (2048 - len(item['features']))
        return result


class PoseSimulation(StateOnlySimulation):
    def observation(self, eid):
        result = super().observation(eid)
        angle = .4 + .001 * self.slots[eid]['step_id']
        q = Rotation.from_euler('xyz', [angle, -.3, .7]).as_quat()[[3, 0, 1, 2]]
        result['state'] = [result['state'][0], .2, .3, *q.tolist(), .008]
        result['control_state'] += [0.] * 35
        result['task_features'] = np.eye(12)[eid].tolist()
        return result


def build(output, *, encoding='pose9', fast=False, options=None):
    inf_meta, sim_meta = pose_metadata()
    inf, sim = PoseInference(inf_meta), PoseSimulation(sim_meta)
    cfg = config(output, steps=8) if options is None else deepcopy(options)
    if options is None:
        cfg['observation_state_encoding'] = encoding
        cfg['checkpoint'].update(keep=None, interval=100_000)
        cfg['residual_warmup'] = dict(learner=dict(hidden_dim=8, batch_size=2,
            num_qs=2, num_subset=2, residual_mode='xyz', residual_scale=[.02] * 3),
            replay_capacity=32, learning_starts=0, max_updates_per_wait=2)
        if fast:
            cfg['rollout'].update(terminal_observation_mode='state_only_if_no_bootstrap', fused_step_reset=True)
    return GenericTrainer(cfg, inference_client=inf, simulation_client=sim), inf, sim


def test_raw_encoding_leaves_contract_and_resume_identity_unchanged():
    inf, sim = pose_metadata()
    wire = negotiate(inf, sim, 'dsrl', 7)
    before = deepcopy(wire)
    raw = configure_rl_observation(wire, 'raw', inf, sim)
    assert wire == before == raw
    identity = dict(contract=wire, action_projection={'mode': 'error', 'lower': None, 'upper': None})
    assert _schema_fingerprint(identity) == _schema_fingerprint({**identity, 'contract': raw})
    assert 'learner_state_dim' not in raw and 'learner_observation' not in raw


def test_pose9_schema_keeps_wire_state_and_binds_the_encoded_identity():
    inf, sim = pose_metadata()
    wire = negotiate(inf, sim, 'dsrl', 7)
    encoded = configure_rl_observation(wire, 'pose9', inf, sim)
    assert wire['obs_dim'] == 2105 and encoded['obs_dim'] == 2106
    assert encoded['state_dim'] == 8 and encoded['learner_state_dim'] == 9
    assert encoded['observation'] == wire['observation']
    assert encoded['fingerprint'] != wire['fingerprint']
    assert encoded['learner_observation']['rotation6d'] == 'first_two_matrix_rows_row_major'
    assert _schema_fingerprint({'contract': wire}) != _schema_fingerprint({'contract': encoded})


@pytest.mark.parametrize('sign', [1., -1.])
def test_pose9_values_match_legacy_row_convention_and_ignore_gripper(sign):
    rotation = Rotation.from_euler('xyz', [.8, -.2, 1.2])
    quaternion = rotation.as_quat()[[3, 0, 1, 2]] * sign
    state = np.array([.123456789, -.2, .3, *quaternion, .008], np.float32)
    encoded = encode_rl_state(state, 'pose9')
    assert encoded.dtype == np.float32 and encoded.shape == (9,)
    np.testing.assert_array_equal(encoded, pose8_to_pose9(state).astype(np.float32))
    np.testing.assert_allclose(encoded[3:], rotation.as_matrix()[:2].reshape(-1), atol=1e-7)
    assert not np.allclose(encoded[3:], rotation.as_matrix()[:, :2].reshape(-1))
    changed = state.copy(); changed[-1] = .1
    np.testing.assert_array_equal(encoded, encode_rl_state(changed, 'pose9'))
    opposite = state.copy(); opposite[3:7] *= -1
    np.testing.assert_array_equal(encoded, encode_rl_state(opposite, 'pose9'))


def test_pose9_rejects_nonunit_quaternion_instead_of_hiding_bad_state():
    with pytest.raises(ValueError, match='unit wxyz'):
        encode_rl_state(np.array([0, 0, 0, 2, 0, 0, 0, .008], np.float32), 'pose9')


@pytest.mark.parametrize('mode', ['xyz', 'rotation6d', None, 9])
def test_invalid_config_rejected(mode):
    with pytest.raises(ValueError, match='observation_state_encoding'):
        TrainerConfig(observation_state_encoding=mode)


@pytest.mark.parametrize('change', ['names', 'units', 'quaternion'])
def test_pose9_rejects_ambiguous_schema_before_reset(tmp_path, change):
    runner, inf, sim = build(tmp_path)
    if change == 'names':
        for meta in (inf.meta, sim.meta): meta['observation']['state']['names'][3] = 'joint_4'
    elif change == 'units':
        for meta in (inf.meta, sim.meta): meta['observation']['state']['units'][0] = 'mm'
    else:
        sim.meta['quaternion_order'] = 'xyzw'
    with pytest.raises(ValueError, match='pose9'):
        runner.run()
    assert not sim.resets and not inf.encode_calls


@pytest.mark.parametrize('fast', [False, True])
def test_both_learners_use_pose9_while_wire_and_terminal_states_remain_pose8(tmp_path, fast):
    runner, inf, sim = build(tmp_path, fast=fast)
    result = runner.run()
    assert result['state'] == 'completed' and runner.learner.updates > 0
    assert runner.residual_warmup.learner.updates > 0
    assert runner.learner.config.obs_dim == runner.residual_warmup.learner.config.obs_dim == 2106
    assert runner.contract['state_dim'] == 8 and runner.contract['learner_state_dim'] == 9
    for request in inf.encode_calls:
        for observation in request['observations']:
            state = observation['state']
            assert len(state) == 8 and state[-1] == .008
            assert np.linalg.norm(state[3:7]) == pytest.approx(1.)
    replay = runner.replay.arrays
    for vector in replay['obs'][:len(runner.replay)]:
        matrix_rows = vector[2051:2057].reshape(2, 3)
        np.testing.assert_allclose(matrix_rows @ matrix_rows.T, np.eye(2), atol=1e-6)
        np.testing.assert_array_equal(vector[-12:].sum(), 1.)
    done = (replay['terminated'][:len(runner.replay)] | replay['truncated'][:len(runner.replay)])[:, 0]
    assert done.any()
    for vector in replay['next_obs'][:len(runner.replay)][done]:
        matrix_rows = vector[2051:2057].reshape(2, 3)
        np.testing.assert_allclose(matrix_rows @ matrix_rows.T, np.eye(2), atol=1e-6)
        if fast: assert np.all(vector[:2048] == 0)
    saved = torch.load(result['last_checkpoint'], weights_only=False)
    assert saved['extra']['contract']['learner_observation']['state_encoding'] == 'pose9'
    assert saved['extra']['residual_warmup']['learner']['config']['obs_dim'] == 2106
    assert saved['extra']['replay']['obs_dim'] == 2106
    assert not inf.contexts


def test_raw_default_still_runs_with_original_geometry(tmp_path):
    runner, _, _ = build(tmp_path, encoding='raw')
    runner.run()
    assert runner.learner.config.obs_dim == runner.residual_warmup.learner.config.obs_dim == 2105
    assert runner.replay.arrays['obs'].shape[1] == 2105


def test_pose9_resume_restores_both_replays_and_wrong_encoding_fails_before_reset(tmp_path):
    first, _, _ = build(tmp_path/'first', fast=True)
    checkpoint = first.run()['last_checkpoint']
    options = asdict(first.config)
    options['output_dir'] = str(tmp_path/'resumed')
    options['checkpoint']['resume'] = checkpoint
    options['rollout']['max_transitions'] = 12
    resumed, _, _ = build(tmp_path/'resumed', options=options)
    assert resumed.run()['transitions'] == 12
    np.testing.assert_array_equal(resumed.replay.arrays['obs'][:8], first.replay.arrays['obs'][:8])
    assert len(resumed.residual_warmup.replay) == 12
    options['output_dir'] = str(tmp_path/'wrong')
    options['observation_state_encoding'] = 'raw'
    wrong, inf, sim = build(tmp_path/'wrong', options=options)
    with pytest.raises(ValueError, match='fingerprint'):
        wrong.run()
    assert not sim.resets and not inf.encode_calls


def test_frozen_dsrl_and_residual_handoff_use_identical_pose9_identity(tmp_path):
    source, _, _ = build(tmp_path/'source')
    checkpoint = source.run()['last_checkpoint']
    options = config(tmp_path/'residual', method='action_residual', steps=4)
    options.update(observation_state_encoding='pose9',
        frozen_dsrl=dict(checkpoint=checkpoint, deterministic=True), residual_initialize_from=checkpoint)
    options['learner'] = asdict(source.residual_warmup.learner.config)
    options['update'].update(utd=0., replay_capacity=32, learning_starts=100)
    residual, _, _ = build(tmp_path/'residual', options=options)
    assert residual.run()['state'] == 'completed'
    assert residual.frozen_dsrl.config.obs_dim == 2106
    assert residual.learner.config.obs_dim == 2106
    for key, value in source.residual_warmup.learner.actor.state_dict().items():
        torch.testing.assert_close(residual.learner.actor.state_dict()[key], value, rtol=0, atol=0)
    options['output_dir'] = str(tmp_path/'wrong')
    options['observation_state_encoding'] = 'raw'
    wrong, inf, sim = build(tmp_path/'wrong', options=options)
    with pytest.raises(ValueError, match='fingerprint'):
        wrong.run()
    assert not sim.resets and not inf.encode_calls
