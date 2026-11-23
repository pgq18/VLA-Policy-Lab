#!/usr/bin/env python3
"""Compare real BF16 Wall DPPO against its optional frozen-prefix cache.

The full and cached experts are independently loaded copies of the same SFT
model. This is a numerical/performance probe, not an RL success-rate result.
No original checkpoint or production service is modified.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--model-config',type=Path,required=True)
    p.add_argument('--device',help='Explicit CUDA/NPU device; overrides model-config device')
    p.add_argument('--expected-expert-sha256',help='Require this initial SFT expert identity before sampling')
    p.add_argument('--expected-weights-sha256',help='Require this original checkpoint identity before sampling')
    p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32,choices=(8,16,32,64))
    p.add_argument('--rows',type=int,default=32)
    p.add_argument('--minibatches',type=int,nargs='+',default=[32,64,128])
    p.add_argument('--gradient-rows',type=int,
                   help='Rows used for full/cache gradient comparison; default is all sampled rows')
    p.add_argument('--gradient-relative-atol',type=float,default=.001)
    args=p.parse_args()
    if args.rows<3 or any(n<1 for n in args.minibatches):p.error('Positive sizes and rows>=3 required')
    if args.gradient_rows is not None and not 1<=args.gradient_rows<=args.rows:
        p.error('gradient-rows must be between 1 and rows')
    if not math.isfinite(args.gradient_relative_atol) or not 0<args.gradient_relative_atol<=.001:
        p.error('Gradient L2 tolerance may only be tightened from .001')
    root=args.source_root.resolve();sys.path.insert(0,str(root/'src'))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,kind='real_bf16_wall_frozen_prefix_qualification',started_unix=time.time(),
        batch_size=args.batch_size,rows=args.rows,raw_atol=.05,reduced_atol=.0005,
        gradient_relative_atol=args.gradient_relative_atol,production_state_modified=False)
    accelerator=None
    def save(stage,**fields):
        report.update(stage=stage,**fields)
        if accelerator is not None and accelerator.is_available():
            report['peak_allocated_gib']=max(report.get('peak_allocated_gib',0.),accelerator.max_memory_allocated(selected_device)/2**30)
            report['peak_reserved_gib']=max(report.get('peak_reserved_gib',0.),accelerator.max_memory_reserved(selected_device)/2**30)
        tmp=out/'report.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out/'report.json')
        print(json.dumps({'stage':stage,**{k:v for k,v in fields.items() if k not in ['checks','gradient_comparison']}}),flush=True)
    try:
        import torch
        from PIL import Image
        from pressb.models.walloss import create_backend
        from pressb.models import flow_experts
        from pressb.runtime.contract import encode_image
        from pressb.runtime.denoising import GaussianChainPolicy
        from pressb.runtime.accelerator import initialize_device, device_module
        from pressb.runtime.dppo import DPPOConfig,reduce_logprobs,clipped_policy_loss
        from pressb.runtime.dppo_trainer import MasterAdamW
        from pressb.runtime.expert_sync import ExpertWeights,publish_expert,state_sha256,tree_to_wire
        assert Path(flow_experts.__file__).resolve().is_relative_to(root/'src')
        torch.set_num_threads(8)
        cfg=json.loads(args.model_config.read_text());cfg=cfg.get('config',cfg)
        if args.device is not None:cfg={**cfg,'device':args.device}
        if cfg.get('dtype')!='bfloat16' or cfg.get('flow_prefix_cache',False) or cfg.get('num_inference_timesteps',10)!=10:
            raise ValueError('Expected BF16 native K10, normal backend prefix reuse disabled')
        selected_device=initialize_device(cfg.get('device','cuda:0'))
        accelerator=device_module(selected_device)
        if accelerator is None or not accelerator.is_available():
            raise RuntimeError('This probe requires an available CUDA or NPU accelerator')
        report['accelerator_type']=selected_device.type
        report['selected_device']=str(selected_device)
        report['torch_version']=torch.__version__
        report['device']=accelerator.get_device_name(selected_device)
        save('loading_full_model');full_backend=create_backend(cfg)
        settings=dict(transition_std=.03,denoising_steps=10,transition_microbatch_size=args.batch_size,transition_context_length=256)
        full=full_backend.create_action_expert(settings);full_policy=GaussianChainPolicy(full)
        save('loading_cached_model');cached_backend=create_backend(cfg)
        cached=cached_backend.create_action_expert({**settings,'frozen_prefix_cache':True,'prefix_cache_max_entries':1024})
        cached_policy=GaussianChainPolicy(cached)
        before_hash=state_sha256(full.state_dict())
        if state_sha256(cached.state_dict())!=before_hash:raise RuntimeError('Initial model experts differ')
        report['initial_expert_sha256']=before_hash
        report['model_identity']={k:full_backend.describe().get(k) for k in ['model_id','weights_sha256','normalization_sha256']}
        if args.expected_expert_sha256 is not None and before_hash!=args.expected_expert_sha256:
            raise RuntimeError('Original SFT action expert identity differs from the requested checkpoint')
        if args.expected_weights_sha256 is not None and report['model_identity']['weights_sha256']!=args.expected_weights_sha256:
            raise RuntimeError('Original SFT model weights identity differs from the requested checkpoint')
        report['sampler']=cached.describe()
        report['execution_backend']=cached_backend.describe().get('execution_backend')
        observation=json.loads(args.observation.read_text())
        for key,value in observation['images'].items():
            if isinstance(value,str):
                f=Path(value);f=f if f.is_absolute() else args.observation.parent/f
                observation['images'][key]=encode_image(Image.open(f).convert('RGB'))
        prompts=['Grab the stuff into the box.','Put the red cube into the box.',
                 'Carefully pick up the object from the table and place it inside the cardboard box.']
        observations=[]
        for i in range(args.rows):
            state=list(observation['state'])
            for joint in range(6):
                state[joint]+=.005*math.sin((i+1)*(joint+1)*.61803398875*2*math.pi)
            observations.append(dict(observation,task=prompts[i%3],state=state))
        encoded=[]
        for first in range(0,args.rows,16):encoded.extend(full_backend.encode_batch(observations[first:first+16],list(range(first,min(first+16,args.rows)))))
        conditions=full.pack_contexts([x[0] for x in encoded])
        cached_conditions=cached.pack_contexts([x[0] for x in encoded])
        if set(cached_conditions)!=set(conditions) or not all(torch.equal(conditions[k],cached_conditions[k]) for k in conditions):
            raise RuntimeError('Prefix optimization changed the conditioning wire payload')
        # Distinct states avoid falsely cheap prefix construction from just
        # three repeated prompt/image contexts in a nominally large batch.
        unique={hashlib.sha256(conditions['inputs_embeds'][i].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for i in range(args.rows)}
        if len(unique)!=args.rows:raise RuntimeError('Probe needs one distinct encoded context per row')
        tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size()) for k,v in conditions.items()}
        save('contexts_ready',unique_contexts=len(unique),context_tensors=tensors,
            conditioning_wire_bytes_per_row=len(json.dumps(tree_to_wire({k:v[:1] for k,v in conditions.items()}),separators=(',',':'))))
        frozen={name:(id(t),t._version) for name,t in cached_backend.policy.model.named_parameters() if not t.requires_grad}
        params_full=dict(full.named_parameters());params_cached=dict(cached.named_parameters())
        if set(params_full)!=set(params_cached):raise RuntimeError('Cached expert changes trainable parameter set')
        if {id(v) for v in params_full.values()} & {id(v) for v in params_cached.values()}:
            raise RuntimeError('Probe models unexpectedly share trainable parameters')
        dppo=DPPOConfig(clip_epsilon=.01,clip_epsilon_min=.001,logprob_max=2.)
        checks=[]
        def subset(indices):return {k:v[indices.to(v.device)] for k,v in conditions.items()}
        def check(label,actual,expected):
            difference=actual.detach()-expected.detach()
            reduced=reduce_logprobs(actual.detach(),dppo,execution_horizon=32)-reduce_logprobs(expected.detach(),dppo,execution_horizon=32)
            item=dict(label=label,raw_max=float(difference.abs().max()),raw_rms=float(difference.square().mean().sqrt()),reduced_max=float(reduced.abs().max()))
            checks.append(item)
            if item['raw_max']>.05 or item['reduced_max']>.0005:raise RuntimeError('Unchanged native likelihood gate failed: '+str(item))
            return item
        seeds=list(range(900,900+args.rows))
        full_policy.sample(conditions,seeds)
        started=time.perf_counter();sample=full_policy.sample(conditions,seeds);accelerator.synchronize()
        full_seconds=time.perf_counter()-started
        cached.clear_prefix_cache();cached_policy.sample(conditions,seeds)
        cached.clear_prefix_cache();accelerator.reset_peak_memory_stats()
        started=time.perf_counter();cached_sample=cached_policy.sample(conditions,seeds);accelerator.synchronize()
        cached_seconds=time.perf_counter()-started
        sampling_cache=cached.prefix_cache_info()
        check('same_seed_sample_logprob',cached_sample['logprobs'].flatten(0,1),sample['logprobs'].flatten(0,1))
        report['sampling']=dict(full_seconds=full_seconds,cached_cold_seconds=cached_seconds,
            speedup=full_seconds/cached_seconds,cache=sampling_cache,peak_allocated_gib=accelerator.max_memory_allocated()/2**30,
            chain_max_abs=float((cached_sample['chain']-sample['chain']).abs().max()),actions_max_abs=float((cached_sample['actions']-sample['actions']).abs().max()))
        save('sampling_checked',sampling=report['sampling'],checks=checks)
        # The real learner is separate from inference. Start its prefix cache
        # empty, then use the same unique-observation preparation hook that
        # native rollout verification calls before mixed-step minibatches.
        cached.clear_prefix_cache()
        originals=[{key:value[i:i+1] for key,value in conditions.items()} for i in range(args.rows)]
        started=time.perf_counter();cached.prepare_condition_cache(originals);accelerator.synchronize()
        save('learner_prefix_prepared',learner_prefix_prepare_seconds=time.perf_counter()-started,
             learner_prefix_prepared_cache=cached.prefix_cache_info())
        order=torch.randperm(args.rows,generator=torch.Generator().manual_seed(91))
        for step in range(10):
            for label,indices in [('shuffled',order),('singleton',torch.tensor([0])),('tail3',torch.tensor([args.rows-1,0,1]))]:
                ids=indices.to(full.device)
                with torch.no_grad():actual=cached_policy.log_prob(subset(indices),sample['chain'][ids,step],sample['chain'][ids,step+1],step)
                check(f'step{step}/{label}',actual,sample['logprobs'][ids,step])
            save('checking_densities',checked_steps=step+1,checks=checks)
        gradient_rows=args.rows if args.gradient_rows is None else args.gradient_rows
        rows=order[:gradient_rows];ids=rows.to(full.device);steps=torch.arange(gradient_rows,device=full.device)%10
        x=sample['chain'][ids,steps];y=sample['chain'][ids,steps+1];old=sample['logprobs'][ids,steps]
        adv=torch.linspace(-1.,1.,gradient_rows,device=full.device)
        full.zero_grad(set_to_none=True);cached.zero_grad(set_to_none=True)
        started=time.perf_counter();score=full_policy.log_prob(subset(rows),x,y,steps)
        loss,_=clipped_policy_loss(score,old,adv,steps,10,dppo,execution_horizon=32);loss.backward();accelerator.synchronize()
        full_backward_seconds=time.perf_counter()-started
        reference={name:p.grad.detach().float().cpu().clone() for name,p in params_full.items() if p.grad is not None}
        full.zero_grad(set_to_none=True);del score,loss;gc.collect();accelerator.empty_cache()
        started=time.perf_counter();score=cached_policy.log_prob(subset(rows),x,y,steps)
        check('gradient_mode_mixed',score,old)
        loss,_=clipped_policy_loss(score,old,adv,steps,10,dppo,execution_horizon=32);loss.backward();accelerator.synchronize()
        cached_backward_seconds=time.perf_counter()-started
        norm=error=dot=other=0.;maximum=0.
        for name,pv in params_cached.items():
            if (pv.grad is None)!=(name not in reference):raise RuntimeError('Gradient presence differs: '+name)
            if pv.grad is None:continue
            if not torch.isfinite(pv.grad).all():raise RuntimeError('Nonfinite cached gradient')
            a=reference[name];b=pv.grad.detach().float().cpu();delta=b-a
            norm+=float(a.double().square().sum());error+=float(delta.double().square().sum())
            dot+=float((a.double()*b.double()).sum());other+=float(b.double().square().sum())
            maximum=max(maximum,float(delta.abs().max()))
        relative=math.sqrt(error/max(norm,1e-30));cosine=dot/max(math.sqrt(norm*other),1e-30)
        if norm<=0 or other<=0 or relative>args.gradient_relative_atol:raise RuntimeError('Gradient equivalence failed: '+str((relative,cosine)))
        gradient=dict(rows=gradient_rows,relative_l2=relative,cosine=cosine,max_abs=maximum,full_norm=math.sqrt(norm),cached_norm=math.sqrt(other),
            full_seconds=full_backward_seconds,cached_seconds=cached_backward_seconds)
        del reference
        with torch.no_grad():mean_before=cached.transition(x,steps,subset(rows))[0].clone()
        cache_before_update=cached.prefix_cache_info()
        optimizer=MasterAdamW(cached.parameters(),lr=1e-6,weight_decay=0.)
        optimizer.step(1.)
        changed_hash=state_sha256(cached.state_dict())
        if changed_hash==before_hash or not all(t.dtype==torch.float32 for t in optimizer.masters):raise RuntimeError('FP32-master update failed')
        with torch.no_grad():mean_after=cached.transition(x,steps,subset(rows))[0]
        action_mean_change=float((mean_after-mean_before).abs().max())
        if action_mean_change==0:raise RuntimeError('Updated expert produced unchanged probe transitions')
        cache_after_update=cached.prefix_cache_info()
        for name,value in cached_backend.policy.model.named_parameters():
            if name in frozen and (value.requires_grad or value.grad is not None or (id(value),value._version)!=frozen[name]):raise RuntimeError('Frozen parameter mutated: '+name)
        weights=ExpertWeights(full)
        class LocalWire:
            def call(self,path,payload):return getattr(weights,path.removeprefix('/weights_'))(payload)
        receipt=publish_expert(LocalWire(),cached.state_dict(),expected_version=0)
        with torch.no_grad():
            full_after=full_policy.log_prob(subset(rows),x,y,steps)
            cached_after=cached_policy.log_prob(subset(rows),x,y,steps)
        check('updated_cached_prefix_vs_published_full_receiver',cached_after,full_after)
        if state_sha256(full.state_dict())!=changed_hash:raise RuntimeError('Published model identity differs')
        save('gradient_and_update_checked',gradient_comparison=gradient,action_mean_change=action_mean_change,
            cached_prefix_before_update=cache_before_update,cached_prefix_after_update=cache_after_update,
            updated_expert_sha256=changed_hash,publish_receipt=receipt,checks=checks)
        del score,loss;optimizer.zero_grad();gc.collect();accelerator.empty_cache()
        timings=[]
        for size in args.minibatches:
            indices=torch.arange(size)%args.rows;i=indices.to(full.device);ks=torch.arange(size,device=full.device)%10
            # Warm prefix lookups model the PPO phase after mandatory rollout
            # verification has already constructed each observation prefix.
            accelerator.reset_peak_memory_stats();started=time.perf_counter();score=None
            try:
                score=cached_policy.log_prob(subset(indices),sample['chain'][i,ks],sample['chain'][i,ks+1],ks)
                advantages=torch.linspace(-1.,1.,size,device=full.device)
                loss,_=clipped_policy_loss(score,sample['logprobs'][i,ks],advantages,ks,10,dppo,execution_horizon=32)
                loss.backward();accelerator.synchronize()
                timings.append(dict(minibatch=size,passed=True,forward_backward_seconds=time.perf_counter()-started,
                    peak_allocated_gib=accelerator.max_memory_allocated()/2**30,peak_reserved_gib=accelerator.max_memory_reserved()/2**30,
                    cache=cached.prefix_cache_info()))
            except torch.OutOfMemoryError:timings.append(dict(minibatch=size,passed=False,error=selected_device.type.upper()+' out of memory'))
            finally:
                del score;optimizer.zero_grad();gc.collect();accelerator.empty_cache()
            save('benchmarking',minibatches=timings)
        if not any(item['passed'] for item in timings):
            raise RuntimeError('No requested PPO minibatch fits the selected accelerator')
        full.assert_frozen_encoder();cached.assert_frozen_encoder()
        save('passed',passed=True,elapsed_seconds=time.time()-report['started_unix'],checks=checks,
            raw_logprob_max=max(c['raw_max'] for c in checks),reduced_logprob_max=max(c['reduced_max'] for c in checks),
            gradient_comparison=gradient,final_cache=cached.prefix_cache_info())
    except Exception:save('failed',error=traceback.format_exc(),elapsed_seconds=time.time()-report['started_unix'])
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
