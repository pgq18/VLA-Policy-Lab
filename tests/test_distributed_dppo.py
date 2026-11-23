"""Two actual Gloo learners versus one learner, through real DPPO math/RPCs."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import time
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pressb.runtime.distributed_dppo import DistributedDPPOTrainer, LearnerCollectives
from pressb.runtime.dppo_trainer import DPPOTrainer
from pressb.runtime.expert_sync import state_sha256
from pressb.runtime.inference import InferenceService
from test_dppo_training import LocalClient, ToyBackend, ToyExpert
from test_rl_nodes_trainer import Simulation


@pytest.mark.parametrize('dtypes', [(torch.bfloat16,), (torch.float32,),
                                   (torch.bfloat16, torch.float32, torch.float16),
                                   (torch.float64, torch.bfloat16)])
@pytest.mark.parametrize('weight', [0., .5, .25, 3/5])
def test_coalesced_gradient_packing_matches_old_values_without_mutation(dtypes, weight):
    from pressb.runtime.distributed_dppo import _pack_gradient_bucket
    parameters = [torch.nn.Parameter(torch.arange(12, dtype=dtype).reshape(3,4).t())
                  for dtype in dtypes]
    for index, parameter in enumerate(parameters):
        if index != 1:
            parameter.grad = (torch.arange(12, dtype=parameter.dtype).reshape(3,4).t() / 7).clone()
    before = [p.grad.clone() if p.grad is not None else None for p in parameters]
    dtype = torch.float64 if torch.float64 in dtypes else torch.float32
    expected = torch.cat([(p.grad.detach().to(dtype) if p.grad is not None else
                           torch.zeros_like(p,dtype=dtype)).reshape(-1)*weight for p in parameters])
    actual = _pack_gradient_bucket(parameters, weight)
    assert torch.equal(actual, expected) and not actual.requires_grad
    actual.add_(1)  # The communication buffer must not alias original gradients.
    for p, saved in zip(parameters, before, strict=True):
        assert p.grad is None if saved is None else torch.equal(p.grad, saved)


@pytest.mark.parametrize('global_count,count', [(32,16),(32,8),(32,0),(5,3),(7,4)])
def test_tensor_metric_payload_preserves_original_weighting(monkeypatch, global_count, count):
    context = LearnerCollectives.__new__(LearnerCollectives)
    context.device = torch.device('cpu')
    context.gradient_group = None
    context.global_batch_count = global_count
    values = {'policy_loss':torch.tensor(.000132765),'approx_kl':torch.tensor(.000032459),
              'clip_fraction':torch.tensor(.619),'mean_ratio':torch.tensor(.9975)}
    captured=[]
    monkeypatch.setattr(dist,'all_reduce',lambda data, **kw:captured.append(data.clone()))
    actual=context.metrics(values,count)
    keys=sorted(values)
    expected=torch.tensor([float(values[k])*count/global_count for k in keys],dtype=torch.float32)
    assert torch.equal(captured[0],expected)
    assert actual==dict(zip(keys,expected.tolist(),strict=True))


@pytest.mark.parametrize('kind', ['cuda', 'npu'])
def test_allocator_observation_reads_only_selected_device_without_resets(monkeypatch, kind):
    from pressb.runtime.distributed_dppo import _allocator_memory
    calls = []
    backend = SimpleNamespace()
    for name, value in [('memory_allocated', 123), ('memory_reserved', 456),
                        ('max_memory_allocated', 234), ('max_memory_reserved', 567)]:
        setattr(backend, name, lambda device, name=name, value=value:
                calls.append((name, device)) or value)
    backend.synchronize = lambda *a, **k: pytest.fail('Telemetry must not synchronize')
    backend.reset_peak_memory_stats = lambda *a, **k: pytest.fail('Telemetry must not reset peaks')
    monkeypatch.setattr(torch, kind, backend, raising=False)
    device = SimpleNamespace(type=kind, index=3)
    report = _allocator_memory(device)
    assert report['supported'] and report['unit'] == 'bytes'
    assert [report[k] for k in ('allocated_bytes', 'reserved_bytes',
                               'peak_allocated_bytes', 'peak_reserved_bytes')] == [123, 456, 234, 567]
    assert len(calls) == 4 and all(d is device for _, d in calls)


def test_cpu_allocator_observation_is_explicitly_unsupported():
    from pressb.runtime.distributed_dppo import _allocator_memory
    result = _allocator_memory(torch.device('cpu'))
    assert result['supported'] is False
    assert all(result[k] is None for k in ('allocated_bytes', 'reserved_bytes',
                                         'peak_allocated_bytes', 'peak_reserved_bytes'))


def test_replica_allocator_differences_are_observations_not_state_mismatches(tmp_path, monkeypatch):
    learner = DistributedDPPOTrainer.__new__(DistributedDPPOTrainer)
    learner.rank = 0
    learner.expert, learner.value = ToyExpert(), torch.nn.Linear(1, 1)
    learner.iterations = learner.actor_updates = learner.critic_updates = learner.policy_version = 0
    learner.transitions = 0
    learner._replicas_ready = False
    learner._learner_state_consistent = True
    learner.config = SimpleNamespace(device='cpu', dppo=SimpleNamespace(minibatch_size=32),
                                      action_expert={'transition_microbatch_size': 16})
    learner.output = tmp_path
    def gather(row):
        other = deepcopy(row)
        other.update(rank=1, memory={'supported': True, 'allocated_bytes': 99999})
        return [row, other]
    learner.collectives = SimpleNamespace(gather=gather, world_size=2, gradient_group=None)
    monkeypatch.setattr(dist, 'get_backend', lambda group: 'gloo')
    identity = learner._assert_replicas()
    assert 'memory' not in identity and learner._learner_state_consistent
    result = json.loads((tmp_path/'distributed.json').read_text())
    assert result['passed']
    assert result['identities'][0]['state'] == result['identities'][1]['state']
    assert result['identities'][0]['memory'] != result['identities'][1]['memory']


def configuration(output, batch, *, warmup=0, checkpoint=None, steps=12, target_kl=None):
    return dict(model={'factory': 'unused:create', 'config': {}}, output_dir=str(output),
        rollout=dict(max_transitions=steps, execution_horizon=2, eval_episodes=4),
        dppo=dict(rollout_steps=2, epochs=1, minibatch_size=batch, critic_hidden_dim=8,
                  actor_lr=.003, target_kl=target_kl, initial_logprob_atol=.001,
                  critic_warmup_iterations=warmup, critic_continue_after_kl=True),
        checkpoint=dict(interval=4, include_replay=False,
                        resume=str(checkpoint) if checkpoint else None))


def clients():
    backend = ToyBackend()
    service = InferenceService(backend, batch_size=2, action_expert={})
    return LocalClient(service), Simulation(backend.simulation_metadata)


def snapshot(runner, result, client=None):
    return {'status': result, 'expert': runner.expert.state_dict(), 'value': runner.value.state_dict(),
            'actor_optimizer': runner.actor_optimizer.state_dict(),
            'value_optimizer': runner.value_optimizer.state_dict(),
            'expert_sha': state_sha256(runner.expert.state_dict()),
            'value_sha': state_sha256(runner.value.state_dict()),
            'sample_versions': [payload['expected_policy_version'] for route, payload in client.calls
                                if route == '/sample_chain'] if client else [],
            'published': client.service.action_expert.weights.version if client else None}


def worker(rank, path, config, result_root, fail_publication=False):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(path), rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    collective = LearnerCollectives('cpu', gradient_bucket_bytes=64)
    inference, simulation = clients() if rank == 0 else (None, None)
    runner = DistributedDPPOTrainer(config, collectives=collective, expert=ToyExpert(),
        inference_client=inference, simulation_client=simulation)
    if rank == 0 and fail_publication:
        original = inference.call
        def lose_ack(route, payload=None):
            result = original(route, payload)
            if route == '/weights_commit':
                raise TimeoutError('commit succeeded but acknowledgement lost')
            return result
        inference.call = lose_ack
        save_failure = runner._save_failure_checkpoint
        def slow_save(error):
            # An early worker failure makes mp.spawn kill its peers, just as
            # torchrun would. Keep the leader here while the worker must wait.
            time.sleep(.7)
            return save_failure(error)
        runner._save_failure_checkpoint = slow_save
    try:
        if rank == 0:
            if fail_publication:
                try:
                    runner.run()
                except TimeoutError as error:
                    assert 'acknowledgement lost' in str(error)
                else:
                    raise AssertionError('Expected acknowledgement loss')
                result = runner.health()
                assert result['recovery_checkpoint']['state'] == 'saved'
                assert Path(result['last_checkpoint']).is_file()
                (result_root / 'recovery_saved.json').write_text(json.dumps({'path': result['last_checkpoint']}))
            else:
                result = runner.run()
            torch.save(snapshot(runner, result, inference), result_root / 'leader.pt')
            runner.shutdown_workers()
        else:
            runner.serve_worker()
            if fail_publication:
                assert (result_root / 'recovery_saved.json').exists()
            torch.save(snapshot(runner, runner.health()), result_root / 'worker.pt')
    finally:
        dist.destroy_process_group()


def assert_tree_close(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_close(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_tree_close(a, e)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-5, abs=2e-7)
    else:
        assert actual == expected


def assert_tree_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_tree_exact(a, e)
    else:
        assert actual == expected


@pytest.mark.parametrize('batch,warmup,target_kl', [(5, 0, None), (7, 1, None), (3, 0, 1e-12)])
def test_two_learners_match_global_minibatch_with_tail_and_kl(tmp_path, batch, warmup, target_kl):
    torch.set_num_threads(1)
    client, simulation = clients()
    config = configuration(tmp_path / 'reference', batch, warmup=warmup, target_kl=target_kl)
    reference = DPPOTrainer(config, inference_client=client, simulation_client=simulation, expert=ToyExpert())
    expected = snapshot(reference, reference.run(), client)
    config = configuration(tmp_path / 'distributed', batch, warmup=warmup, target_kl=target_kl)
    mp.spawn(worker, args=(tmp_path / 'rendezvous', config, tmp_path), nprocs=2, join=True)
    leader = torch.load(tmp_path / 'leader.pt', weights_only=True)
    follower = torch.load(tmp_path / 'worker.pt', weights_only=True)
    for key in ('expert', 'value', 'actor_optimizer', 'value_optimizer'):
        assert_tree_close(leader[key], expected[key])
        assert_tree_exact(follower[key], leader[key])
    for key in ('transitions', 'iterations', 'updates', 'actor_updates', 'critic_updates', 'policy_version'):
        assert leader['status'][key] == expected['status'][key]
    assert leader['sample_versions'] == expected['sample_versions']
    assert follower['expert_sha'] == leader['expert_sha']
    assert follower['value_sha'] == leader['value_sha']
    assert leader['published'] == expected['published']
    metrics = leader['status']['last_update_metrics']
    assert metrics['raw_logprob_max_error'] == 0
    assert metrics['reduced_logprob_max_error'] == 0
    if target_kl is not None:
        assert metrics['early_stop_kl']
    assert len(list((tmp_path / 'distributed/checkpoints').glob('*.pt'))) == 3
    assert not list((tmp_path / 'distributed_workers').rglob('checkpoints'))
    evidence = json.loads((tmp_path/'distributed/distributed.json').read_text())
    assert evidence['passed'] and evidence['backend'] == 'gloo'
    assert evidence['world_size'] == 2 and sum(evidence['local_minibatch_sizes']) == batch
    assert evidence['identities'][0]['state'] == evidence['identities'][1]['state']


def test_workers_cannot_issue_environment_or_publication_requests():
    from pressb.runtime.distributed_dppo import _MetadataClient
    client = _MetadataClient({'test': True})
    assert client.call('/health') == {'test': True}
    for route in ('/step', '/reset', '/sample_chain', '/weights_commit'):
        with pytest.raises(RuntimeError, match='Only the leader'):
            client.call(route)


def test_lost_publication_ack_keeps_worker_alive_until_recovery_then_resumes(tmp_path):
    failed = tmp_path / 'failure'; failed.mkdir()
    config = configuration(failed / 'training', 7, steps=8)
    config['checkpoint']['interval'] = 100000
    mp.spawn(worker, args=(failed/'rendezvous', config, failed, True), nprocs=2, join=True)
    leader = torch.load(failed/'leader.pt', weights_only=True)
    path = Path(json.loads((failed/'recovery_saved.json').read_text())['path'])
    payload = torch.load(path, weights_only=True)
    assert payload['recovery']['publication_state'] == 'pending'
    assert payload['transitions'] == payload['last_optimized_transitions'] == 4
    assert payload['policy_version'] == 0 and leader['published'] == 1
    for key in ('expert', 'value', 'actor_optimizer', 'value_optimizer'):
        assert_tree_close(payload[key], leader[key])
    resumed = tmp_path/'resumption'; resumed.mkdir()
    config = configuration(resumed/'training', 7, checkpoint=path, steps=8)
    config['checkpoint']['interval'] = 100000
    mp.spawn(worker, args=(resumed/'rendezvous', config, resumed), nprocs=2, join=True)
    result = torch.load(resumed/'leader.pt', weights_only=True)
    follower = torch.load(resumed/'worker.pt', weights_only=True)
    assert result['status']['state'] == 'completed'
    assert result['status']['transitions'] == 8 and result['status']['iterations'] == 2
    assert result['sample_versions'] == [1, 1]
    assert result['published'] == 2 and result['expert_sha'] == follower['expert_sha']


def test_launcher_check_only_accepts_npu_without_loading_accelerator(tmp_path):
    root = Path(__file__).parents[1]
    cfg = configuration(tmp_path/'training', 32)
    cfg['device'] = 'npu:0'
    path = tmp_path/'config.json';path.write_text(json.dumps(cfg))
    result = subprocess.run([sys.executable, str(root/'scripts/run_distributed_dppo.py'),
        '--source-root', str(root), '--config', str(path), '--backend', 'hccl',
        '--collective-timeout', '900', '--control-timeout', '3600',
        '--stop-file', str(tmp_path/'stop'), '--leader-pid-file', str(tmp_path/'leader.json'), '--check-only'],
        capture_output=True,text=True,check=True,timeout=30)
    evidence=json.loads(result.stdout)
    assert evidence['passed'] and evidence['processes_started'] is False
    assert evidence['backend']=='hccl' and evidence['global_minibatch_size']==32
    assert not (tmp_path/'training').exists() and not (tmp_path/'leader.json').exists()
