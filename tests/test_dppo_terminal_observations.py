"""DPPO terminal transport through real node contracts; no Isaac/model benchmark."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from pressb.runtime.contract import decode_image, encode_image, fingerprint
from pressb.runtime.dppo_trainer import DPPOTrainer, DPPOTrainerConfig
from pressb.runtime.inference import InferenceService
from pressb.runtime.simulation import SimulationService


COMPACT = 'state_only_if_no_bootstrap'


def metadata(kind):
    observation = {'state': {'names': ['joint', 'gripper'], 'units': ['rad', 'm']},
                   'cameras': ['up', 'wrist']}
    action = {'representation': 'joint_absolute', 'frame': 'robot_base',
              'names': ['joint', 'gripper'], 'units': ['rad', 'm'], 'horizon': 4, 'control_hz': 30.}
    result = {'schema_version': 2, 'node_type': kind, 'observation': observation, 'action': action}
    if kind == 'inference':
        result.update(model_id='terminal-transport-fixture', feature_dim=3, noise={'shape': [4, 2]},
            capabilities=['encode', 'decode', 'features', 'context_cache', 'external_initial_noise'])
    else:
        result.update(task_id='terminal-transport-fixture', num_envs=2, control_state_dim=2,
            task_features_dim=1, reset_mode='indexed', discount={'gamma': .95, 'timebase_hz': 30.},
            capabilities=['terminal_state_only'], terminal_observation_modes=['full', COMPACT])
    return result


class Expert(nn.Module):
    horizon, latent_dim, denoising_steps = 4, 2, 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))

    @property
    def device(self):
        return self.weight.device

    def describe(self):
        return {'sampler_kind': 'terminal-test-gaussian', 'horizon': 4, 'latent_dim': 2, 'denoising_steps': 2}

    def pack_contexts(self, contexts):
        return {'state': torch.stack(contexts)}

    def transition(self, x, steps, condition):
        return x*.1 + self.weight + condition['state'].sum(1)[:, None, None]*.01, torch.full_like(x, .2)

    def actions_from_latent(self, x):
        return x


class Model:
    def __init__(self):
        self.metadata = metadata('inference')
        self.expert = Expert()
        self.encoded = []

    def describe(self):
        return deepcopy(self.metadata)

    def encode_batch(self, observations, seeds):
        result = []
        for obs in observations:
            assert set(obs['images']) == {'up', 'wrist'}
            image = np.asarray(decode_image(obs['images']['up']))
            feature = float(image.mean()) / 255.
            self.encoded.append(deepcopy(obs))
            result.append((torch.tensor([obs['state'][0], feature]),
                           np.asarray([feature, obs['state'][0], obs['state'][1]], dtype=np.float32)))
        return result

    def decode_batch(self, contexts, noises):
        return [np.zeros((4, 2), dtype=np.float32) for _ in contexts]

    def create_action_expert(self, config):
        return self.expert

    def close(self):
        pass


class Plant:
    def __init__(self, *, all_terminal=False):
        self.metadata = metadata('simulation')
        self.state = np.zeros((2, 2))
        self.ticks = [0, 0]
        self.serial = [0, 0]
        self.resets, self.steps, self.rows = [], [], []
        self.terminal_captures = 0
        self.all_terminal = all_terminal

    def describe(self):
        return deepcopy(self.metadata)

    def observation(self, eid, *, images=True):
        result = {'state': self.state[eid].tolist(), 'control_state': self.state[eid].tolist(),
                  'task_features': [float(eid)], 'images': {}, 'task': 'Complete the task.'}
        if images:
            pixel = int(abs(self.state[eid, 0]) * 1000) % 200
            array = (np.arange(27).reshape(3, 3, 3) + pixel).astype(np.uint8)
            result['images'] = {'up': encode_image(array), 'wrist': encode_image(array[:, ::-1])}
        return result

    def reset(self, env_ids, seed, options):
        self.resets.append({'env_ids': list(env_ids), 'seed': seed, 'options': deepcopy(options)})
        for eid in env_ids:
            self.ticks[eid] = 0
            self.serial[eid] += 1
            self.state[eid] = [.2 + (seed % 17)*.001 + eid*.1, .02]
        return [{'env_id': eid, 'observation': self.observation(eid), 'info': {}} for eid in env_ids]

    def step(self, items, *, terminal_observation_mode='full'):
        self.steps.append({'items': deepcopy(items), 'mode': terminal_observation_mode})
        rows = []
        for item in items:
            eid = item['env_id']
            self.ticks[eid] += 1
            self.state[eid] += np.asarray(item['actions'][-1])*.001
            terminal = self.all_terminal or eid == 0 and self.ticks[eid] == 2
            timeout = not terminal and eid == 1 and self.ticks[eid] == 3
            done = terminal or timeout
            omitted = done and terminal_observation_mode == COMPACT
            duration = .5 if terminal else 1.25 if timeout else len(item['actions'])
            self.terminal_captures += int(done and not omitted)
            row = {'env_id': eid, 'observation': self.observation(eid, images=not omitted),
                   'reward': 1. if terminal else 0. if timeout else -.05,
                   'discount': .95**duration, 'terminated': terminal, 'truncated': timeout,
                   'info': {'success': terminal, 'elapsed_control_steps': duration,
                            'physics_steps': int(duration*4), 'marker': [eid, self.serial[eid], self.ticks[eid]]}}
            if omitted:
                row['terminal_observation_omitted'] = True
            rows.append(row)
        self.rows.extend(deepcopy(rows))
        return rows


class Client:
    def __init__(self, service):
        self.service, self.calls, self.responses = service, [], []
        self.mutate = None

    def call(self, route, payload=None):
        self.calls.append((route, deepcopy(payload)))
        result = self.service.handlers[route](payload or {})
        if self.mutate is not None:
            self.mutate(route, result)
        self.responses.append((route, deepcopy(result)))
        return result


class CapturingTrainer(DPPOTrainer):
    def _advantages(self, records):
        advantages = super()._advantages(records)
        self.collected.append(deepcopy(records))
        self.normalized_advantages.append(advantages.clone())
        return advantages


def runner(output, terminal_mode='full', *, mode='train', steps=12, bootstrap=False, checkpoint=None,
           all_terminal=False, capability=True):
    model, plant = Model(), Plant(all_terminal=all_terminal)
    if not capability:
        plant.metadata['capabilities'] = []
        plant.metadata['terminal_observation_modes'] = ['full']
    inference = Client(InferenceService(model, batch_size=2, action_expert={}))
    simulation = Client(SimulationService(plant))
    config = dict(model={'factory': 'unused:create'}, output_dir=str(output), mode=mode,
        rollout={'max_transitions': steps, 'execution_horizon': 2, 'eval_episodes': 4,
                 'terminal_observation_mode': terminal_mode},
        dppo={'rollout_steps': 2, 'epochs': 1, 'minibatch_size': 4, 'critic_hidden_dim': 8,
              'actor_lr': .003, 'target_kl': None, 'bootstrap_truncated': bootstrap, 'initial_logprob_atol': .001},
        checkpoint={'interval': 4, 'include_replay': False, 'resume': str(checkpoint) if checkpoint else None})
    instance = CapturingTrainer(config, inference_client=inference, simulation_client=simulation, expert=Expert())
    instance.collected, instance.normalized_advantages = [], []
    return instance, inference, simulation, plant


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_compact_collect_gae_and_parameter_updates_equal_full(tmp_path):
    full, inf1, sim1, plant1 = runner(tmp_path/'full')
    compact, inf2, sim2, plant2 = runner(tmp_path/'compact', COMPACT)
    a, b = full.run(), compact.run()
    assert a['state'] == b['state'] == 'completed'
    for key in ('transitions', 'episodes', 'successes', 'updates', 'actor_updates', 'critic_updates', 'policy_version'):
        assert a[key] == b[key]
    assert full.schema_fingerprint == compact.schema_fingerprint
    assert full.identity['contract'] == full.contract
    assert full.schema_fingerprint == fingerprint(full.identity)
    assert compact.contract['terminal_observation_mode'] == COMPACT
    assert compact.contract['fingerprint'] != full.contract['fingerprint']
    equal_tree(full.normalized_advantages, compact.normalized_advantages)
    for batch1, batch2 in zip(full.collected, compact.collected, strict=True):
        for row1, row2 in zip(batch1, batch2, strict=True):
            # UUIDs identify separate node runs; logical episode/step ordering
            # and all data used by PPO are otherwise exactly the same.
            equal_tree({k:v for k,v in row1.items() if k != 'episode_id'},
                       {k:v for k,v in row2.items() if k != 'episode_id'})
            if row2['terminated'] or row2['truncated']:
                assert row2['next_value'] == 0.
                assert row2['return'] == pytest.approx(row2['reward'], abs=1e-7)
    records = [row for batch in compact.collected for row in batch]
    assert any(row['terminated'] for row in records) and any(row['truncated'] for row in records)
    assert any(not (row['terminated'] or row['truncated']) and row['next_value'] != 0 for row in records)
    for left,right in [(full.expert.state_dict(),compact.expert.state_dict()),
                       (full.value.state_dict(),compact.value.state_dict()),
                       (full.actor_optimizer.state_dict(),compact.actor_optimizer.state_dict()),
                       (full.value_optimizer.state_dict(),compact.value_optimizer.state_dict())]:
        equal_tree(left,right)
    assert plant1.terminal_captures > 0 and plant2.terminal_captures == 0
    equal_tree(plant1.resets,plant2.resets)
    assert [row['items'] for row in plant1.steps] == [row['items'] for row in plant2.steps]
    assert all(payload.get('terminal_observation_mode') == COMPACT for route,payload in sim2.calls if route == '/step')
    assert all('terminal_observation_mode' not in payload for route,payload in sim1.calls if route == '/step')
    assert len(inf1.service.backend.encoded) == len(inf2.service.backend.encoded)
    for route,payload in inf2.calls:
        if route in ('/encode','/sample_chain'):
            assert all(set(obs['images']) == {'up','wrist'} for obs in payload['observations'])
    assert any(route == '/encode' for route,_ in inf2.calls)  # Continuing rollout-tail bootstrap still runs.


def test_all_terminal_batches_never_bootstrap_and_eval_quotas_match(tmp_path):
    train, inference, _, _ = runner(tmp_path/'train', COMPACT, all_terminal=True, steps=4)
    train.run()
    assert not any(route == '/encode' for route,_ in inference.calls)
    results = []
    for terminal_mode in ('full', COMPACT):
        task, inference, _, plant = runner(tmp_path/terminal_mode, terminal_mode, mode='eval', steps=40)
        status = task.run(); results.append(status)
        assert status['episodes'] == 4 and status['updates'] == 0
        assert not any(route == '/encode' for route,_ in inference.calls)
    assert results[0]['successes'] == results[1]['successes']
    assert results[0]['transitions'] == results[1]['transitions']


@pytest.mark.parametrize('mode',['train','eval'])
def test_compact_requires_explicit_no_timeout_bootstrap(tmp_path, mode):
    with pytest.raises(ValueError,match='bootstrap_truncated=false'):
        runner(tmp_path,COMPACT,mode=mode,bootstrap=True)


def test_missing_capability_and_mutated_unsafe_config_fail_before_reset(tmp_path):
    task, _, _, plant = runner(tmp_path/'cap',COMPACT,capability=False)
    with pytest.raises(ValueError,match='terminal_state_only'):task.run()
    assert not plant.resets and not plant.steps
    task, _, _, plant = runner(tmp_path/'bootstrap',COMPACT)
    task.config.dppo.bootstrap_truncated=True
    with pytest.raises(ValueError,match='bootstrap_truncated=false'):task.run()
    assert not plant.resets and not plant.steps


@pytest.mark.parametrize('fault',['missing_flag','nonboolean_flag','live_flag','live_missing_image',
                                'nonempty_omitted_images','bad_control','bad_features','bad_step_id'])
def test_malformed_compact_rows_are_rejected_without_replaying_physics(tmp_path,fault):
    task, _, client, plant = runner(tmp_path, COMPACT)
    def mutate(route,response):
        if route != '/step':return
        terminal = next((r for r in response['items'] if r['terminated'] or r['truncated']),None)
        live = next((r for r in response['items'] if not (r['terminated'] or r['truncated'])),None)
        if fault == 'live_flag' and live:live['terminal_observation_omitted']=True
        elif fault == 'live_missing_image' and live:live['observation']['images']={}
        elif terminal:
            if fault == 'missing_flag':terminal.pop('terminal_observation_omitted')
            elif fault == 'nonboolean_flag':terminal['terminal_observation_omitted']=1
            elif fault == 'nonempty_omitted_images':terminal['observation']['images']={'up':encode_image(np.zeros((2,2,3),np.uint8))}
            elif fault == 'bad_control':terminal['observation']['control_state']=[float('nan'),0.]
            elif fault == 'bad_features':terminal['observation']['task_features']=[]
            elif fault == 'bad_step_id':terminal['step_id']+=1
    client.mutate=mutate
    with pytest.raises(ValueError):task.run()
    assert len(plant.steps) in (1,2)
    assert task.updates==0
    assert task.health()['execution_state'].startswith('unknown')


def test_full_mode_rejects_unexpected_omission_and_bootstrap_rejects_omitted_row(tmp_path):
    task, _, client, _ = runner(tmp_path)
    def mutate(route,response):
        if route=='/step':response['items'][0]['terminal_observation_omitted']=True
    client.mutate=mutate
    with pytest.raises(ValueError,match='terminal_observation_omitted'):task.run()
    with pytest.raises(ValueError,match='bootstrap requires a full observation'):
        task._bootstrap([{'terminal_observation_omitted':True}])


def test_full_timeout_bootstrap_remains_enabled_when_requested(tmp_path):
    task, inference, _, _ = runner(tmp_path,bootstrap=True)
    task.run()
    timeouts=[row for batch in task.collected for row in batch if row['truncated']]
    assert timeouts and any(row['next_value'] != 0 for row in timeouts)
    assert any(route=='/encode' for route,_ in inference.calls)


def test_old_full_checkpoint_resumes_compact_with_all_other_identity_checks(tmp_path):
    first, _, _, _ = runner(tmp_path/'first',steps=4)
    checkpoint=first.run()['last_checkpoint']
    saved=torch.load(checkpoint,weights_only=True)
    resumed,inference,_,plant=runner(tmp_path/'resumed',COMPACT,checkpoint=checkpoint,steps=12)
    result=resumed.run()
    assert result['state']=='completed' and result['transitions']==12
    assert resumed.schema_fingerprint==saved['schema_fingerprint']
    assert all(request['mode']==COMPACT for request in plant.steps)
    routes=[route for route,_ in inference.calls]
    assert routes.index('/weights_commit') < routes.index('/sample_chain')
    for change in ('discount','task','model'):
        invalid,client,_,backend=runner(tmp_path/change,COMPACT,checkpoint=checkpoint,steps=12)
        if change=='discount':
            invalid.simulation.service.metadata['discount']['gamma']=.9
        elif change=='task':
            invalid.simulation.service.metadata['task_id']='another-task'
        else:
            client.service.metadata['model_id']='another-model'
        with pytest.raises(ValueError,match='identity differs'):invalid.run()
        assert not backend.resets and not backend.steps
