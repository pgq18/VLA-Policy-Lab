#!/usr/bin/env python3
"""Qualify one real Wall BF16 canonical batch before using it for DPPO.

No simulation, task success measurement or production service is involved.
The probe checks every denoising step, row shuffles, singleton/tail requests,
mixed-step gradients, a real FP32-master update and versioned tensor transfer
to a separately loaded model. It also times PPO-sized forward/backward batches.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--device', help='Explicit CUDA/NPU device; overrides model-config device')
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=(8,16,32,64), required=True)
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--encoder-batch-size', type=int, default=16)
    parser.add_argument('--ppo-minibatches', type=int, nargs='+', default=[32,64,128])
    parser.add_argument('--raw-atol', type=float, default=.05)
    parser.add_argument('--reduced-atol', type=float, default=.0005)
    args = parser.parse_args()
    if args.rows < 3 or args.encoder_batch_size < 1 or any(n < 1 for n in args.ppo_minibatches):
        parser.error('rows>=3 and positive batch sizes are required')
    if (not math.isfinite(args.raw_atol) or not 0 < args.raw_atol <= .05
            or not math.isfinite(args.reduced_atol) or not 0 < args.reduced_atol <= .0005):
        parser.error('Likelihood gates may be tightened, never loosened beyond native .05/.0005')
    source = args.source_root.resolve()
    sys.path.insert(0,str(source/'src'))
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report = dict(passed=False, kind='real_wall_bf16_canonical_batch_qualification',
        canonical_batch=args.batch_size, context_length=256, rows=args.rows,
        source_root=str(source), started_unix=time.time(), production_state_modified=False)
    accelerator = None

    def save(stage, **fields):
        report.update(stage=stage, **fields)
        if accelerator is not None and accelerator.is_available():
            report['peak_allocated_gib'] = max(report.get('peak_allocated_gib',0.), accelerator.max_memory_allocated(selected_device)/2**30)
            report['peak_reserved_gib'] = max(report.get('peak_reserved_gib',0.), accelerator.max_memory_reserved(selected_device)/2**30)
        temporary=output/'report.json.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(output/'report.json')
        print(json.dumps(dict(stage=stage, **{k:v for k,v in fields.items() if k!='checks'})),flush=True)

    try:
        import torch
        from PIL import Image
        from pressb.models.walloss import create_backend
        from pressb.runtime.contract import encode_image
        from pressb.runtime.denoising import GaussianChainPolicy
        from pressb.runtime.accelerator import initialize_device, device_module
        from pressb.runtime.dppo import DPPOConfig, reduce_logprobs
        from pressb.runtime.dppo_trainer import MasterAdamW
        from pressb.runtime.expert_sync import ExpertWeights, publish_expert, state_sha256, tree_to_wire
        from pressb.models import flow_experts
        assert Path(flow_experts.__file__).resolve().is_relative_to(source/'src')
        torch.set_num_threads(8)
        cfg=json.loads(args.model_config.read_text());cfg=cfg.get('config',cfg)
        if args.device is not None:cfg={**cfg,'device':args.device}
        if cfg.get('dtype') != 'bfloat16' or cfg.get('flow_prefix_cache',False):
            raise ValueError('Qualification requires native BF16 with prefix caching disabled')
        if cfg.get('num_inference_timesteps',10) != 10:
            raise ValueError('This qualification keeps all native ten flow steps')
        selected_device=initialize_device(cfg.get('device','cuda:0'))
        accelerator=device_module(selected_device)
        if accelerator is None or not accelerator.is_available():
            raise RuntimeError('This probe requires an available CUDA or NPU accelerator')
        report['accelerator_type']=selected_device.type
        report['selected_device']=str(selected_device)
        report['torch_version']=torch.__version__
        save('loading_model')
        backend=create_backend(cfg)
        expert_cfg=dict(transition_std=.03,denoising_steps=10,
            transition_microbatch_size=args.batch_size,transition_context_length=256)
        expert=backend.create_action_expert(expert_cfg);policy=GaussianChainPolicy(expert)
        report['model_identity']={k:backend.describe().get(k) for k in ['model_id','weights_sha256','normalization_sha256']}
        report['device']=accelerator.get_device_name(expert.device)
        report['execution_backend']=backend.describe().get('execution_backend')
        report['sampler']=expert.describe()
        observation=json.loads(args.observation.read_text())
        for name,value in observation['images'].items():
            if isinstance(value,str):
                path=Path(value);path=path if path.is_absolute() else args.observation.parent/path
                observation['images'][name]=encode_image(Image.open(path).convert('RGB'))
        tasks=['Grab the stuff into the box.','Put the red cube into the box.',
            'Carefully pick up the object from the table and place it inside the cardboard box.']
        observations=[dict(observation,task=tasks[i%len(tasks)]) for i in range(args.rows)]
        started=time.perf_counter();encoded=[]
        for first in range(0,args.rows,args.encoder_batch_size):
            encoded.extend(backend.encode_batch(observations[first:first+args.encoder_batch_size],
                           list(range(first,min(first+args.encoder_batch_size,args.rows)))))
        conditions=expert.pack_contexts([row[0] for row in encoded])
        lengths=conditions['attention_mask'].sum(1).tolist()
        frozen={name:(id(value),value._version) for name,value in backend.policy.model.named_parameters()
                if not value.requires_grad}
        expert.assert_frozen_encoder()
        report['context_tensors']={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size()) for k,v in conditions.items()}
        save('encoded',encode_seconds=time.perf_counter()-started,valid_context_lengths=lengths)

        def subset(tree, rows):
            return {key:value[rows.to(value.device)] for key,value in tree.items()}

        def delta_stats(score, expected):
            delta=score.detach()-expected
            reduced=(reduce_logprobs(score.detach(),DPPOConfig(),execution_horizon=expert.horizon)
                     -reduce_logprobs(expected,DPPOConfig(),execution_horizon=expert.horizon))
            result=dict(raw_max=float(delta.abs().max()),raw_rms=float(delta.square().mean().sqrt()),
                row_mean_max=float(delta.flatten(1).mean(1).abs().max()),
                reduced_max=float(reduced.abs().max()))
            if result['raw_max']>args.raw_atol or max(result['row_mean_max'],result['reduced_max'])>args.reduced_atol:
                raise RuntimeError('Native DPPO likelihood tolerance exceeded: '+str(result))
            return result

        # A warm sample is timed separately; the second sample gives steady shape cost.
        accelerator.reset_peak_memory_stats(expert.device)
        started=time.perf_counter();sample=policy.sample(conditions,list(range(1000,1000+args.rows)))
        accelerator.synchronize(expert.device)
        report['cold_sample_seconds']=time.perf_counter()-started
        started=time.perf_counter();sample=policy.sample(conditions,list(range(1000,1000+args.rows)))
        accelerator.synchronize(expert.device)
        duration=time.perf_counter()-started
        row=0
        wire=dict(conditioning=tree_to_wire({k:v[row:row+1] for k,v in conditions.items()}),
            chain=tree_to_wire(sample['chain'][row]),logprobs=tree_to_wire(sample['logprobs'][row]),
            actions=sample['actions'][row].cpu().tolist(),features=encoded[row][1].tolist(),policy_version=0)
        row_bytes=len(json.dumps(wire,separators=(',',':')).encode())
        save('sampled',sample_seconds=duration,sample_transitions_per_second=args.rows/duration,
            sample_peak_allocated_gib=accelerator.max_memory_allocated(expert.device)/2**30,
            sample_wire_row_bytes=row_bytes,estimated_wire_256_rows_mib=row_bytes*256/2**20)
        checks=[]
        order=torch.randperm(args.rows,generator=torch.Generator().manual_seed(713))
        singleton=torch.tensor([0]);tail=torch.tensor([args.rows-1,0,1])
        for step in range(10):
            for label,indices in [('shuffled',order),('singleton',singleton),('tail3',tail)]:
                idx=indices.to(expert.device)
                with torch.no_grad():score=policy.log_prob(subset(conditions,indices),sample['chain'][idx,step],sample['chain'][idx,step+1],step)
                checks.append(dict(step=step,pattern=label,**delta_stats(score,sample['logprobs'][idx,step])))
            save('checking_densities',checked_steps=step+1,checks=checks)
        rows=order[:min(args.rows,args.batch_size)]
        steps=torch.arange(len(rows),device=expert.device)%10;idx=rows.to(expert.device)
        x,y=sample['chain'][idx,steps],sample['chain'][idx,steps+1]
        expected=sample['logprobs'][idx,steps]
        expert.zero_grad(set_to_none=True)
        started=time.perf_counter();score=policy.log_prob(subset(conditions,rows),x,y,steps)
        checks.append(dict(pattern='gradient_mixed_steps',**delta_stats(score,expected)))
        (-score.mean()).backward();accelerator.synchronize(expert.device)
        gradients=[p.grad for p in expert.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Expected finite action-expert gradients')
        grad_norm=float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())
        if grad_norm <= 0:raise RuntimeError('Action expert gradients are zero')
        report['backward_seconds']=time.perf_counter()-started
        report['grad_norm']=grad_norm
        before_hash=state_sha256(expert.state_dict())
        optimizer=MasterAdamW(expert.parameters(),lr=1e-6,weight_decay=0.)
        optimizer.step(1.)
        after_hash=state_sha256(expert.state_dict())
        if after_hash==before_hash:raise RuntimeError('FP32-master step did not change BF16 expert weights')
        if not all(p.dtype==torch.float32 for p in optimizer.masters):raise RuntimeError('Masters are not FP32')
        for name,value in backend.policy.model.named_parameters():
            if name in frozen and (value.requires_grad or value.grad is not None or (id(value),value._version)!=frozen[name]):
                raise RuntimeError('Frozen model parameter changed: '+name)
        save('backward_and_master_step_passed',initial_expert_sha256=before_hash,updated_expert_sha256=after_hash,
            checks=checks,backward_seconds=report['backward_seconds'],grad_norm=grad_norm)
        del score,gradients;optimizer.zero_grad();gc.collect();accelerator.empty_cache()

        # Actual versioned serialization/chunk/commit path, with an independent
        # model receiver. Network latency is not measured by this local dispatch.
        save('loading_independent_receiver')
        receiver=create_backend(cfg);receiver_expert=receiver.create_action_expert(expert_cfg)
        weights=ExpertWeights(receiver_expert)
        class LocalWire:
            def call(self,path,payload):return getattr(weights,path.removeprefix('/weights_'))(payload)
        started=time.perf_counter();receipt=publish_expert(LocalWire(),expert.state_dict(),expected_version=0)
        with torch.no_grad():
            source_mean,source_std=expert.transition(x,steps,subset(conditions,rows))
            receiver_mean,receiver_std=receiver_expert.transition(x,steps,subset(conditions,rows))
        torch.testing.assert_close(source_mean,receiver_mean,rtol=0,atol=0)
        torch.testing.assert_close(source_std,receiver_std,rtol=0,atol=0)
        save('publication_passed',publish_seconds=time.perf_counter()-started,publish_receipt=receipt,
             publication_transport='production chunk/commit protocol, local in-process dispatch')

        timings=[]
        for size in args.ppo_minibatches:
            indices=torch.arange(size)%args.rows;ids=indices.to(expert.device);ks=torch.arange(size,device=expert.device)%10
            optimizer.zero_grad();accelerator.empty_cache();accelerator.reset_peak_memory_stats(expert.device)
            started=time.perf_counter();score=None
            try:
                score=policy.log_prob(subset(conditions,indices),sample['chain'][ids,ks],sample['chain'][ids,ks+1],ks)
                (-score.mean()).backward();accelerator.synchronize(expert.device)
                timings.append(dict(minibatch=size,passed=True,forward_backward_seconds=time.perf_counter()-started,
                    peak_allocated_gib=accelerator.max_memory_allocated(expert.device)/2**30,
                    peak_reserved_gib=accelerator.max_memory_reserved(expert.device)/2**30))
            except torch.OutOfMemoryError:
                timings.append(dict(minibatch=size,passed=False,error=selected_device.type.upper()+' out of memory'))
            finally:
                del score;optimizer.zero_grad();gc.collect();accelerator.empty_cache()
            save('benchmarking_minibatches',minibatches=timings)
        if not any(row['passed'] for row in timings):raise RuntimeError('No PPO minibatch fits')
        expert.assert_frozen_encoder();receiver_expert.assert_frozen_encoder()
        save('passed',passed=True,elapsed_seconds=time.time()-report['started_unix'],checks=checks,
            raw_logprob_max=max(row['raw_max'] for row in checks),
            reduced_logprob_max=max(row['reduced_max'] for row in checks),
            frozen_parameter_count=len(frozen))
    except Exception:
        save('failed',error=traceback.format_exc(),elapsed_seconds=time.time()-report['started_unix'])
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
