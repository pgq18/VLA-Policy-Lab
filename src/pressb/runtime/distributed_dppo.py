"""Synchronous learner replicas for one versioned, on-policy DPPO trainer.

Only rank zero talks to simulation/inference, owns episode counters and writes
checkpoints. Workers execute the same global minibatch schedule on disjoint
rows. Gradients are weighted by real row counts, reduced before clipping, and
applied on every replica. Publication remains a barrier between rollouts.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist

from .dppo import reduce_logprobs, metrics_to_floats
from .dppo_trainer import DPPOTrainer, _atomic_json
from .expert_sync import state_sha256


class _PublicationFailure(RuntimeError):
    """A completed optimizer boundary whose remote publication was not acknowledged."""


def _allocator_memory(device):
    """Read this device's allocator counters without synchronization or resets."""
    methods = {'allocated_bytes': 'memory_allocated', 'reserved_bytes': 'memory_reserved',
               'peak_allocated_bytes': 'max_memory_allocated', 'peak_reserved_bytes': 'max_memory_reserved'}
    result = {'supported': False, 'device': str(device), 'unit': 'bytes',
              'scope': 'process_caching_allocator_not_total_device_HBM',
              **{key: None for key in methods}}
    if device.type not in ('cuda', 'npu'):
        return result
    backend = getattr(torch, device.type, None)
    if backend is None:
        return result
    errors = []
    for key, name in methods.items():
        reader = getattr(backend, name, None)
        if not callable(reader):
            errors.append(f'{name}: unavailable')
            continue
        try:
            result[key] = int(reader(device))
        except (RuntimeError, TypeError, ValueError) as error:
            errors.append(f'{name}: {type(error).__name__}: {error}')
    result['supported'] = not errors
    if errors:
        result['errors'] = errors
    return result


def _pack_gradient_bucket(bucket, weight):
    """Keep the same FP32 weighted values with fewer casts/copies per bucket."""
    dtype = torch.float64 if any(p.dtype == torch.float64 for p in bucket) else torch.float32
    if len(bucket) == 1:
        parameter = bucket[0]
        flat = (parameter.grad.detach().to(dtype=dtype, copy=True) if parameter.grad is not None
                else torch.zeros_like(parameter, dtype=dtype)).reshape(-1)
    else:
        pieces = [(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                  for p in bucket]
        flat = torch.cat(pieces).to(dtype=dtype)
    return flat.mul_(weight)


class LearnerCollectives:
    """CPU control messages and accelerator-native numeric reductions."""

    def __init__(self, device, *, gradient_group=None, control_group=None,
                 gradient_bucket_bytes=32 * 1024**2):
        if not dist.is_initialized():
            raise RuntimeError("Initialize the learner process group before creating replicas")
        self.device = torch.device(device)
        self.gradient_group, self.control_group = gradient_group, control_group
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        if type(gradient_bucket_bytes) is not int or gradient_bucket_bytes < 4:
            raise ValueError("gradient_bucket_bytes must be at least four")
        self.gradient_bucket_bytes = gradient_bucket_bytes
        self.global_batch_count = None

    def broadcast(self, value=None):
        objects = [value]
        dist.broadcast_object_list(objects, src=0, group=self.control_group)
        return objects[0]

    def gather(self, value):
        result = [None] * self.world_size
        dist.all_gather_object(result, value, group=self.control_group)
        return result

    def select(self, selected):
        selected = np.asarray(selected)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("Expected a nonempty global denoising-pair minibatch")
        self.global_batch_count = len(selected)
        local = np.array_split(selected, self.world_size)[self.rank]
        # Empty tail ranks still participate in the same forward/collectives.
        # Their dummy row has zero weight in every metric and gradient.
        return (local if len(local) else selected[-1:]), len(local)

    def metrics(self, values, count):
        if self.global_batch_count is None or not 0 <= count <= self.global_batch_count:
            raise ValueError("Select a global batch before reducing its local metrics")
        keys = sorted(values)
        power_of_two = self.global_batch_count & (self.global_batch_count - 1) == 0
        same_device_fp32 = all(isinstance(values[key], torch.Tensor)
            and values[key].dtype == torch.float32 and values[key].device == self.device for key in keys)
        if power_of_two and same_device_fp32:
            # n / 2**k is exactly representable: this preserves the previous
            # Python-weighted-then-FP32 result for normal global batch 32.
            data = torch.stack([values[key].detach().reshape(()) for key in keys])
            data.mul_(count / self.global_batch_count)
        else:
            # Odd tail denominators retain the original double-precision
            # Python weighting before its FP32 cast (one batched host read).
            host_values = metrics_to_floats(values)
            data = torch.tensor([host_values[key] * count / self.global_batch_count for key in keys],
                                dtype=torch.float32, device=self.device)
        dist.all_reduce(data, group=self.gradient_group)
        return dict(zip(keys, data.cpu().tolist(), strict=True))

    def gradients(self, parameters, count):
        """Global mean gradients, including missing/unused parameters and tails.

        Reduce in FP32 for BF16/FP16 parameters, then return native gradients to
        MasterAdamW. No rank clips independently before this collective.
        """
        parameters = list(parameters)
        if self.global_batch_count is None or not 0 <= count <= self.global_batch_count:
            raise ValueError("Invalid local gradient batch weight")
        present = torch.tensor([int(p.grad is not None and count > 0) for p in parameters],
                               dtype=torch.int32, device=self.device)
        dist.all_reduce(present, group=self.gradient_group)
        present = present.cpu().tolist()
        participating = []
        for parameter, active in zip(parameters, present, strict=True):
            if not active:
                parameter.grad = None  # AdamW must also skip its weight decay.
            else:
                participating.append(parameter)
        weight = count / self.global_batch_count
        bucket, size = [], 0

        def flush():
            if not bucket:
                return
            flat = _pack_gradient_bucket(bucket, weight)
            dist.all_reduce(flat, group=self.gradient_group)
            offset = 0
            for parameter in bucket:
                reduced = flat[offset:offset + parameter.numel()].view_as(parameter).to(parameter.dtype)
                if parameter.grad is None:
                    parameter.grad = reduced.clone()
                else:
                    parameter.grad.copy_(reduced)
                offset += parameter.numel()

        for parameter in participating:
            bytes_needed = parameter.numel() * (8 if parameter.dtype == torch.float64 else 4)
            if bucket and size + bytes_needed > self.gradient_bucket_bytes:
                flush()
                bucket, size = [], 0
            bucket.append(parameter)
            size += bytes_needed
        flush()

    def likelihood_statistics(self, maximum, reduced_maximum, squares, count):
        maxima = torch.tensor([maximum, reduced_maximum], device=self.device, dtype=torch.float32)
        sums = torch.tensor([squares, count], device=self.device, dtype=torch.float32)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=self.gradient_group)
        dist.all_reduce(sums, group=self.gradient_group)
        a, b = maxima.cpu().tolist()
        s, n = sums.cpu().tolist()
        return a, b, s, n


class _MetadataClient:
    """Workers can inspect the agreed contract, but cannot issue physical RPCs."""

    def __init__(self, metadata):
        self.metadata = deepcopy(metadata)

    def call(self, route, payload=None):
        if route != "/health":
            raise RuntimeError("Only the leader may contact simulation or inference")
        return deepcopy(self.metadata)

    def close(self):
        pass


class DistributedDPPOTrainer(DPPOTrainer):
    def __init__(self, config, *, collectives, **kwargs):
        self.collectives = collectives
        self.rank = collectives.rank
        self._replicas_ready = False
        self._initializing_worker = False
        self._distributed_failed = False
        self._publication_failed = False
        if self.rank:
            kwargs.update(inference_client=_MetadataClient({}), simulation_client=_MetadataClient({}))
        super().__init__(config, **kwargs)
        if self.rank:
            self.output = self.output.with_name(self.output.name + "_workers") / f"rank_{self.rank}" / self.run_id
            self.output.mkdir(parents=True, exist_ok=False)

    def _negotiate(self):
        if self.rank:
            raise RuntimeError("Worker initialization is driven by the leader command loop")
        super()._negotiate()
        infer = deepcopy(self.inference_metadata)
        infer['action_expert'].update(policy_version=self.policy_version,
                                     weights_sha256=state_sha256(self.expert.state_dict()))
        self.collectives.broadcast({'command': 'initialize', 'inference': infer,
                                    'simulation': self.simulation_metadata})
        self._assert_replicas()
        self._replicas_ready = True

    def _assert_replicas(self):
        identity = {'expert': state_sha256(self.expert.state_dict()),
                    'value': state_sha256(self.value.state_dict()),
                    'iterations': self.iterations, 'actor_updates': self.actor_updates,
                    'critic_updates': self.critic_updates, 'policy_version': self.policy_version}
        memory = _allocator_memory(next(self.expert.parameters()).device)
        identities = self.collectives.gather({'rank': self.rank, 'device': str(self.config.device),
                                             'state': identity, 'memory': memory})
        if any(item['state'] != identities[0]['state'] for item in identities):
            self._learner_state_consistent = False
            raise RuntimeError('Synchronous learner replica state diverged')
        if self.rank == 0:
            evidence = {'event': 'learner_replicas_verified', 'passed': True,
                'phase': 'before_publication' if self._replicas_ready else 'initialized',
                'backend': dist.get_backend(self.collectives.gradient_group),
                'world_size': self.collectives.world_size,
                'global_minibatch_size': self.config.dppo.minibatch_size,
                'local_minibatch_sizes': [len(x) for x in np.array_split(
                    np.arange(self.config.dppo.minibatch_size), self.collectives.world_size)],
                'canonical_microbatch_size': self.config.action_expert.get('transition_microbatch_size'),
                'gradient_reduction': 'real_row_weighted_sum_before_global_clipping',
                'advantages': 'globally_normalized_on_rank_zero',
                'empty_tail_rank': 'zero_weight_dummy_row',
                'transitions': self.transitions, 'identities': identities}
            self._log('events.jsonl', evidence)
            _atomic_json(self.output / 'distributed.json', evidence)
        return identity

    def serve_worker(self):
        if not self.rank:
            raise RuntimeError("The leader runs the normal DPPO training loop")
        self.started = time.monotonic()
        try:
            while True:
                message = self.collectives.broadcast()
                command = message.get('command')
                if command == 'stop':
                    return
                if command == 'initialize':
                    if self._replicas_ready:
                        raise RuntimeError('Learner worker was initialized twice')
                    self.inference = _MetadataClient(message['inference'])
                    self.simulation = _MetadataClient(message['simulation'])
                    self._initializing_worker = True
                    try:
                        DPPOTrainer._negotiate(self)
                    finally:
                        self._initializing_worker = False
                    self._assert_replicas()
                    self._replicas_ready = True
                elif command == 'update' and self._replicas_ready:
                    coordinates = message['coordinates']
                    for key in ('iterations', 'policy_version', 'updates', 'actor_updates', 'critic_updates'):
                        if getattr(self, key) != coordinates[key]:
                            raise RuntimeError('Worker update coordinates differ from the leader: ' + key)
                    self.transitions = coordinates['transitions']
                    if self._publication_failed:
                        raise RuntimeError('A failed publication requires leader shutdown and a fresh run')
                    try:
                        DPPOTrainer._update(self, message['records'])
                    except _PublicationFailure as error:
                        # Stay alive while rank zero writes its recovery checkpoint.
                        # Exiting here would make torchrun kill that saving process.
                        self._log('events.jsonl', {'event': 'awaiting_leader_recovery',
                            'error': str(error), 'policy_version': self.policy_version})
                else:
                    raise ValueError('Unexpected learner command: ' + str(command))
        except BaseException:
            if not self._publication_failed:
                self._distributed_failed = True
            raise
        finally:
            if self.backend is not None:
                self.backend.close()

    def shutdown_workers(self):
        if not self.rank and not self._distributed_failed:
            self.collectives.broadcast({'command': 'stop'})

    def _update(self, records):
        if self.rank:
            raise RuntimeError('Workers only update through the leader command loop')
        if not records:
            return
        if any(record['policy_version'] != self.policy_version for record in records):
            raise ValueError('Stale samples cannot enter on-policy PPO update')
        self.collectives.broadcast({'command': 'update', 'records': records,
            'coordinates': {key: getattr(self, key) for key in
                            ('transitions', 'iterations', 'policy_version', 'updates', 'actor_updates', 'critic_updates')}})
        try:
            super()._update(records)
        except BaseException:
            if not self._publication_failed:
                self._distributed_failed = True
            raise

    def _minibatch_selection(self, selected):
        return self.collectives.select(selected)

    def _shuffle_update_indices(self, indices):
        if self.rank == 0:
            super()._shuffle_update_indices(indices)
            payload = {'indices': indices, 'rng': self.rng.bit_generator.state}
        else:
            payload = None
        result = self.collectives.broadcast(payload)
        indices[:] = result['indices']
        if self.rank:
            self.rng.bit_generator.state = result['rng']

    def _aggregate_minibatch_metrics(self, metrics, sample_count):
        return self.collectives.metrics(metrics, sample_count)

    def _synchronize_gradients(self, parameters, sample_count):
        self.collectives.gradients(parameters, sample_count)

    def _advantages(self, records):
        if self.rank == 0:
            advantages = super()._advantages(records)
            payload = {'normalized': advantages,
                       'raw': [(row['advantage'], row['return']) for row in records]}
        else:
            payload = None
        result = self.collectives.broadcast(payload)
        if self.rank:
            for row, (advantage, target) in zip(records, result['raw'], strict=True):
                row.update(advantage=advantage, **{'return': target})
        return result['normalized']

    @torch.no_grad()
    def _verify_rollout(self, records):
        prepare = getattr(self.expert, 'prepare_condition_cache', None)
        if callable(prepare):
            prepared = prepare([record['conditioning'] for record in records])
            if prepared is not None:
                self._log('events.jsonl', {'event': 'frozen_prefix_prepared', 'rank': self.rank,
                                         'policy_version': self.policy_version, **prepared})
        cfg = self.config.dppo
        indices = np.arange(len(records) * self.ft_steps)
        maximum = reduced_maximum = square_sum = 0.
        count = 0
        for offset in range(0, len(indices), cfg.minibatch_size):
            local, real_count = self.collectives.select(indices[offset:offset + cfg.minibatch_size])
            *_, old, current = self._pair_batch(records, local)
            if not real_count:
                continue
            difference = current.float() - old.float()
            maximum = max(maximum, float(difference.abs().max()))
            square_sum += float(difference.square().sum())
            count += difference.numel()
            reduced = reduce_logprobs(current, cfg, execution_horizon=self.contract['execution_horizon'])
            reduced -= reduce_logprobs(old, cfg, execution_horizon=self.contract['execution_horizon'])
            reduced_maximum = max(reduced_maximum, float(reduced.abs().max()))
        maximum, reduced_maximum, square_sum, count = self.collectives.likelihood_statistics(
            maximum, reduced_maximum, square_sum, count)
        measured = {'raw_logprob_max_error': maximum, 'raw_logprob_rms_error': math.sqrt(square_sum / count),
                    'reduced_logprob_max_error': reduced_maximum, 'verified_pairs': len(indices)}
        self._log('events.jsonl', {'event': 'rollout_logprob_verified', 'rank': self.rank,
                                 'policy_version': self.policy_version, **measured})
        if maximum > cfg.initial_logprob_atol or reduced_maximum > cfg.initial_reduced_logprob_atol:
            raise ValueError(f'Rollout/trainer log-probability mismatch: raw_max={maximum:.6g}, reduced_max={reduced_maximum:.6g}')
        return measured

    def _publish(self):
        if self._initializing_worker:
            return  # The leader already published a restored checkpoint.
        if not self._replicas_ready:
            return super()._publish()
        self._assert_replicas()
        if self.rank == 0:
            try:
                super()._publish()
            except BaseException as error:
                self._publication_failed = True
                self.collectives.broadcast({'ok': False, 'error': f'{type(error).__name__}: {error}'})
                raise
            result = {'ok': True, 'policy_version': self.policy_version}
        else:
            result = None
        result = self.collectives.broadcast(result)
        if not result['ok']:
            self._publication_failed = True
            raise _PublicationFailure('Leader expert publication failed: ' + result['error'])
        self.policy_version = result['policy_version']
        self._publication_state = 'acknowledged'
