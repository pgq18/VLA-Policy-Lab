"""Metric XYZ policy outputs on joint7: same differentiable map for SAC and execution."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from pressb.robots.piper.cartesian_residual import build_piper_xyz_spec
from pressb.robots.piper.kinematics import PiperKinematics
from pressb.runtime.learner import LearnerConfig, SACLearner
from pressb.runtime.trainer import (FrozenDSRLConfig, GenericTrainer, _FrozenDSRLPolicy,
                                    _cartesian_composition_identity, _module_sha256, _wandb_payload,
                                    _schema_fingerprint)
from test_rl_nodes_trainer import Inference, Simulation, config, metadata

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT/'vendor/robot_lab/source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf'
NOMINAL = np.array([.1, 1., -.8, .1, .7, .1, .04], np.float32)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def spec():
    return build_piper_xyz_spec(URDF)


def learner_config(spec, **changes):
    values = dict(method='action_residual', obs_dim=8, action_horizon=2, model_action_dim=7,
                  noise_dim=20, hidden_dim=8, batch_size=2, residual_mode='cartesian_xyz',
                  residual_scale=[.02]*3, cartesian_residual=spec,
                  action_lower=spec['lower']+[0.], action_upper=spec['upper']+[.1],
                  residual_q_bounds=None)
    values.update(changes)
    return LearnerConfig(**values)


def test_cartesian_policy_has_three_outputs_but_full_joint7_critic(spec):
    learner = SACLearner(learner_config(spec, action_horizon=32))
    assert learner.action_dim == 96
    assert learner.config.base_action_shape == (32, 7)
    assert learner.actor.net[0].in_features == 8+224
    assert learner.qs.net[0].weight.shape[-2] == 8+224
    assert learner.config.residual_indices == (0, 1, 2)


@pytest.mark.parametrize('override,match', [
    ({'cartesian_residual': None}, 'specification'),
    ({'model_action_dim': 9}, 'six PiPER'),
    ({'residual_indices': [0, 2, 4]}, 'residual_indices'),
    ({'action_lower': None, 'action_upper': None}, 'joint action bounds'),
    ({'residual_mode': 'all'}, 'only defined'),
    ({'method': 'initial_noise'}, 'only defined'),
])
def test_invalid_cartesian_configuration_is_explicit(spec, override, match):
    with pytest.raises(ValueError, match=match):
        learner_config(spec, **override)


def test_zero_residual_is_projected_base_and_inputs_are_untouched(spec):
    learner = SACLearner(learner_config(spec))
    base = np.tile(NOMINAL, (2, 2, 1))
    base[0, 0, 0] = spec['lower'][0]-.1
    base[0, 1, 6] = -.001
    before = base.copy()
    actual = learner.compose_actions(np.zeros((2, 6), np.float32), base)
    expected = np.clip(base, np.asarray(learner.config.action_lower, np.float32),
                        np.asarray(learner.config.action_upper, np.float32))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(base, before)


def test_execution_and_critic_use_identical_map_with_physical_xyz_rotation_gripper(spec):
    learner = SACLearner(learner_config(spec))
    base = np.tile(NOMINAL, (2, 2, 1))
    normalized = np.tile([.2, -.3, .4, -.2, .15, -.1], (2, 1)).astype(np.float32)
    before = base.copy()
    actual = learner.compose_actions(normalized, base)
    metrics = deepcopy(learner.composition_metrics)
    assert metrics['calls'] == 1 and metrics['poses'] == 4
    assert metrics['full_scale_poses'] == 4 and metrics['fallback_poses'] == 0
    tensor = torch.tensor(normalized, requires_grad=True)
    critic_action = learner._critic_action(tensor, torch.as_tensor(base.reshape(2, -1)))
    assert learner.composition_metrics == metrics  # Optimizer forwards do not count as robot execution.
    np.testing.assert_array_equal(actual, critic_action.detach().numpy().reshape(2, 2, 7))
    np.testing.assert_array_equal(actual[..., 6], base[..., 6])
    np.testing.assert_array_equal(base, before)
    kin = PiperKinematics(URDF, tip_offset=spec['tip_offset_m'])
    desired = normalized.reshape(2, 2, 3)*.02
    for index in np.ndindex(2, 2):
        initial, final = kin.fk(base[index][:6]), kin.fk(actual[index][:6])
        np.testing.assert_allclose(final[:3, 3]-initial[:3, 3], desired[index], atol=5e-5, rtol=0)
        assert Rotation.from_matrix(final[:3, :3]@initial[:3, :3].T).magnitude() < 1e-3
    assert np.any(np.abs(actual[..., 3:6]-base[..., 3:6]) > 1e-4), 'XYZ IK must also compensate wrist joints'
    values = learner.qs(torch.ones(2, 8), critic_action)
    values.sum().backward()
    assert torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad) > 0


def test_rollout_composition_metrics_report_fallback_and_wandb(spec):
    spec['solver']['max_joint_delta_rad'] = 1e-7
    learner = SACLearner(learner_config(spec))
    base = np.tile(NOMINAL, (2, 2, 1))
    actual = learner.compose_actions(np.full((2, 6), .5, np.float32), base)
    np.testing.assert_array_equal(actual, base)
    metrics = learner.composition_metrics
    assert metrics['poses'] == metrics['fallback_poses'] == 4
    assert metrics['backoff_poses'] == metrics['full_scale_poses'] == 0
    assert metrics['last']['fallback_rate'] == 1.
    assert metrics['seconds'] > 0
    payload = _wandb_payload({'cartesian_composition': metrics})
    assert payload['cartesian_composition/fallback_poses'] == 4
    assert payload['cartesian_composition/last_fallback_rate'] == 1.


def test_cartesian_checkpoint_restores_numeric_composition_without_urdf(spec, tmp_path):
    learner = SACLearner(learner_config(spec))
    path = tmp_path/'cartesian.pt'
    learner.save(path)
    restored = SACLearner.load(path)
    assert asdict(restored.config) == asdict(learner.config)
    base = np.tile(NOMINAL, (2, 2, 1))
    actions = np.full((2, 6), .1, np.float32)
    np.testing.assert_array_equal(restored.compose_actions(actions, base), learner.compose_actions(actions, base))
    # Caller mutation cannot modify the saved/constructed kinematic recipe.
    spec['lower'][0] -= 1
    assert restored.config.cartesian_residual['lower'][0] != spec['lower'][0]


def test_legacy_frozen_noise_identity_does_not_gain_optional_none_field(tmp_path):
    settings = LearnerConfig(method='initial_noise', obs_dim=3, action_horizon=2,
                             model_action_dim=7, noise_dim=20, hidden_dim=8)
    learner = SACLearner(settings)
    payload = learner.checkpoint_state({'schema_fingerprint': 'unchanged-schema', 'transitions': 4})
    legacy_config = deepcopy(payload['config'])
    legacy_config.pop('cartesian_residual')
    payload['config'] = legacy_config
    path = tmp_path/'legacy_noise.pt'
    torch.save(payload, path)
    contract = dict(obs_dim=3, action_horizon=2, action_dim=7, noise_shape=[2, 20])
    policy = _FrozenDSRLPolicy(FrozenDSRLConfig(str(path), deterministic=False),
                              'unchanged-schema', contract, 'cpu')
    assert policy.identity['learner_config'] == legacy_config
    assert 'cartesian_residual' not in policy.identity['learner_config']
    assert policy.actor_sha256 == _module_sha256(learner.actor)


def joint_metadata(spec):
    inf, sim = metadata(horizon=2, action_dim=7, noise_dim=20, state_dim=7)
    for item in (inf, sim):
        item['action'].update(names=[f'joint{i}' for i in range(1, 7)]+['gripper_width'],
                              representation='joint_position', frame='robot_base',
                              lower=spec['lower']+[0.], upper=spec['upper']+[.1])
    sim['asset_sha256'] = {'robot_urdf': spec['urdf_sha256']}
    return inf, sim


class JointInference(Inference):
    def call(self, path, payload=None):
        result = super().call(path, payload)
        if path == '/encode' and payload['include_base_actions']:
            for row in result['items']:
                row['base_actions'] = np.tile(NOMINAL, (2, 1)).tolist()
        if path == '/decode':
            for row, request in zip(result['items'], payload['items'], strict=True):
                noise = np.asarray(request['initial_noise'])[:, :7]
                row['actions'] = (NOMINAL[None]+noise*.001).tolist()
        return result


def test_frozen_old_dsrl_then_cartesian_training_wire_replay_and_resume(tmp_path, spec):
    inf_meta, sim_meta = joint_metadata(spec)
    noise_cfg = config(tmp_path/'noise', steps=4)
    noise_cfg['rollout']['action_bounds'] = 'clip'
    source = GenericTrainer(noise_cfg, inference_client=JointInference(inf_meta), simulation_client=Simulation(sim_meta))
    source_result = source.run()
    path = Path(source_result['last_checkpoint'])
    payload = torch.load(path, weights_only=False)
    payload['config'].pop('cartesian_residual')  # The concurrently running DSRL writes this older shape.
    torch.save(payload, path)
    frozen_actor = _module_sha256(source.learner.actor)
    settings = config(tmp_path/'residual', method='action_residual', steps=8)
    settings['rollout']['action_bounds'] = 'clip'
    settings['learner'].update(residual_mode='cartesian_xyz', residual_scale=[.02]*3,
                                cartesian_residual=spec, residual_q_bounds=None)
    settings['frozen_dsrl'] = dict(checkpoint=str(path), deterministic=False)
    inference, simulation = JointInference(inf_meta), Simulation(sim_meta)
    runner = GenericTrainer(settings, inference_client=inference, simulation_client=simulation)
    result = runner.run()
    assert result['state'] == 'completed' and result['transitions'] == 8 and result['updates'] > 0
    assert runner.learner.action_dim == 6 and runner.learner.config.base_action_shape == (2, 7)
    assert result['cartesian_composition']['chunks'] == result['transitions']
    assert result['cartesian_composition']['poses'] == result['transitions']*2
    assert runner.frozen_dsrl.actor_sha256 == frozen_actor and runner.frozen_dsrl.verify()['matches_initial']
    assert all(not row['include_base_actions'] for row in inference.encode_calls)
    rows = runner.replay.arrays
    base = rows['base_action'][:8].reshape(8, 2, 7)
    expected = runner.learner.compose_actions(rows['action'][:8], base)
    executed = np.asarray([item['actions'] for request in simulation.steps for item in request['items']], np.float32)
    np.testing.assert_array_equal(executed, expected)
    np.testing.assert_array_equal(executed[..., 6], base[..., 6])
    assert np.count_nonzero(rows['action'][:2]) == 0
    negotiated = __import__('json').loads((runner.output/'negotiated.json').read_text())
    assert negotiated['action_composition']['kinematics'] == spec
    source_identity = __import__('json').loads((source.output/'negotiated.json').read_text())
    assert 'action_composition' not in source_identity
    from pressb.runtime.checkpoint_migration import _noise_identity
    assert _schema_fingerprint(_noise_identity(negotiated)) == source.schema_fingerprint
    resumed_cfg = deepcopy(runner.config)
    resumed_cfg.output_dir = str(tmp_path/'resume')
    resumed_cfg.checkpoint.resume = result['last_checkpoint']
    resumed_cfg.rollout.max_transitions = 10
    resumed = GenericTrainer(resumed_cfg, inference_client=JointInference(inf_meta), simulation_client=Simulation(sim_meta))
    assert resumed.run()['transitions'] == 10
    wrong_cfg = deepcopy(resumed_cfg)
    wrong_cfg.output_dir = str(tmp_path/'wrong')
    wrong_cfg.learner['cartesian_residual']['solver']['damping'] *= 2
    wrong = GenericTrainer(wrong_cfg, inference_client=JointInference(inf_meta), simulation_client=Simulation(sim_meta))
    with pytest.raises(ValueError, match='fingerprint'):
        wrong.run()
    assert not wrong.simulation.resets


def test_cartesian_shadow_settings_are_validated_before_any_dsrl_rollout(tmp_path, spec):
    inf_meta, sim_meta = joint_metadata(spec)
    wrong_spec = deepcopy(spec)
    wrong_spec['urdf_sha256'] = 'f'*64
    settings = config(tmp_path/'invalid-shadow', steps=4)
    settings['rollout']['action_bounds'] = 'clip'
    settings['residual_warmup'] = {'learner': {'residual_mode': 'cartesian_xyz',
        'residual_scale': [.02]*3, 'cartesian_residual': wrong_spec, 'batch_size': 2}}
    runner = GenericTrainer(settings, inference_client=JointInference(inf_meta), simulation_client=Simulation(sim_meta))
    with pytest.raises(ValueError, match='URDF'):
        runner.run()
    assert not runner.simulation.resets


@pytest.mark.parametrize('change,match', [('units', 'joint7'), ('names', 'joint7'),
    ('frame', 'joint7'), ('urdf', 'URDF'), ('bounds', 'joint limits'), ('projection', 'action_bounds')])
def test_metadata_mismatch_rejected_before_execution(spec, change, match):
    inf, sim = joint_metadata(spec)
    mode = 'clip'
    if change == 'units': inf['action']['units'][0] = 'm'
    if change == 'names': inf['action']['names'][0] = 'x'
    if change == 'frame': inf['action']['frame'] = 'world'
    if change == 'urdf': sim['asset_sha256']['robot_urdf'] = '0'*64
    if change == 'bounds': sim['action']['lower'][0] -= .1
    if change == 'projection': mode = 'error'
    with pytest.raises(ValueError, match=match):
        _cartesian_composition_identity({'residual_mode':'cartesian_xyz', 'cartesian_residual':spec}, inf, sim, mode)
