"""Native Wall checks for the opt-in DPPO frozen-prefix path."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_full_attention_prefix_forward_gradients_and_lifetime(tmp_path):
    for dependency in ('torch','transformers','torchdiffeq'):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip('Native Wall dependencies unavailable: '+dependency)
    # Reuse the actual tiny native Wall construction from the existing expert
    # test, including the differentiable CPU equivalent of its CUDA permutation.
    tree=ast.parse((ROOT/'tests/test_dppo_flow_experts.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_wall_native_moe_gradients_padding_cached_context_and_weight_transfer')
    source=next(n.value.value for n in function.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))
    setup=source.split("with patch(ops+'.permute'",1)[0]
    body=r'''
full=WallOSSFlowExpert(backend(model),config={'transition_std':.03,'transition_microbatch_size':8,'transition_context_length':8})
cached=WallOSSFlowExpert(backend(copy.deepcopy(model)),config={'transition_std':.03,'transition_microbatch_size':8,
    'transition_context_length':8,'frozen_prefix_cache':True,'prefix_cache_max_entries':8})
assert full.state_dict().keys()==cached.state_dict().keys()
assert 'frozen_prefix_cache' not in full.describe()
assert cached.describe()['frozen_prefix_cache'] is True
with patch(ops+'.permute',side_effect=permute),patch(ops+'.unpermute',side_effect=unpermute):
    pfull,pcache=GaussianChainPolicy(full),GaussianChainPolicy(cached)
    draw=pfull.sample(packed,[11,19],initial_noise=x)
    other=pcache.sample(packed,[11,19],initial_noise=x)
    torch.testing.assert_close(other['chain'],draw['chain'],rtol=0,atol=2e-6)
    torch.testing.assert_close(other['logprobs'],draw['logprobs'],rtol=0,atol=2e-5)
    assert cached.prefix_cache_info()['cached_rows']==2
    before_builds=cached.prefix_cache_info()['builds']
    # Row order, singletons and different denoising steps use the same frozen
    # prefix, but retain an independent and differentiable action stream.
    for k in range(3):
        for indices in (torch.tensor([1,0]),torch.tensor([0])):
            cond={key:value[indices] for key,value in packed.items()}
            a=pcache.log_prob(cond,draw['chain'][indices,k],draw['chain'][indices,k+1],k)
            torch.testing.assert_close(a,draw['logprobs'][indices,k],rtol=0,atol=2e-5)
    assert cached.prefix_cache_info()['builds']==before_builds
    steps=torch.tensor([0,2]);idx=torch.arange(2)
    full.zero_grad();cached.zero_grad()
    lf=pfull.log_prob(packed,draw['chain'][idx,steps],draw['chain'][idx,steps+1],steps).mean()
    lc=pcache.log_prob(packed,draw['chain'][idx,steps],draw['chain'][idx,steps+1],steps).mean()
    lf.backward();lc.backward()
    total=0.
    for (name,a),(other_name,b) in zip(full.named_parameters(),cached.named_parameters()):
        assert name==other_name and (a.grad is None)==(b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=2e-6)
            total+=float(b.grad.abs().sum())
    assert total>0
    assert all(p.grad is None for p in cached._source_model.parameters() if not p.requires_grad)
    initial_mean=cached.transition(x,steps,packed)[0].detach().clone()
    optimizer=torch.optim.AdamW(cached.parameters(),lr=.001)
    optimizer.step();full.load_state_dict(cached.state_dict())
    changed=cached.transition(x,steps,packed)[0]
    assert not torch.equal(initial_mean,changed)
    torch.testing.assert_close(changed,full.transition(x,steps,packed)[0],rtol=1e-5,atol=2e-6)
    assert cached.prefix_cache_info()['builds']==before_builds
    # Prefill runs once per unique-observation canonical batch and survives
    # verify/update collation; the wire keys and state dict remain unchanged.
    cached.prepare_condition_cache(single)
    primed=cached.prefix_cache_info()['builds']
    mixed=cached.collate_conditions([single[1],single[0],single[1]])
    cached.transition(torch.cat([x[1:],x[:1],x[1:]]),torch.tensor([1,2,0]),mixed)
    assert cached.prefix_cache_info()['builds']==primed
    assert set(mixed)==set(packed)
    # Ordinary in-place tensor mutation invalidates the local key memo.
    changed_condition={key:value.clone() for key,value in packed.items()}
    cached.transition(x,steps,changed_condition)
    n=cached.prefix_cache_info()['builds']
    changed_condition['inputs_embeds'][0,2,0]+=.1
    cached.transition(x,steps,changed_condition)
    assert cached.prefix_cache_info()['builds']==n+1
    # Bound row storage even for many distinct observations.
    for j in range(12):
        cond={key:value.clone() for key,value in single[0].items()}
        cond['inputs_embeds'][0,0,0]+=(j+1)*.05
        cached.transition(x[:1],torch.tensor([1]),cond)
    assert cached.prefix_cache_info()['cached_rows']<=8
    cached.clear_prefix_cache();assert cached.prefix_cache_info()['bytes']==0
    # Frozen dependency mutation must reject, not silently reuse activations.
    cached.transition(x,steps,packed)
    with torch.no_grad(): cached._source_model.model.layers[0].self_attn.q_proj.weight.add_(.01)
    try: cached.transition(x,steps,packed)
    except RuntimeError as e: assert 'dependency changed' in str(e)
    else: raise AssertionError('accepted modified shared attention with stale prefix')
print('WALL_PREFIX_CPU_PASSED')
'''
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'src'),str(ROOT/'src/pressb/model_sources/walloss')]),
             CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-c',setup+body],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
