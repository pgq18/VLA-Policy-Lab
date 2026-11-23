"""Validated action-only suffix bypass retains native top-1 MoE math."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_action_suffix_exact_mean_density_gradient_and_validation(tmp_path):
    for dependency in ('torch', 'transformers', 'torchdiffeq'):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip('Native Wall dependencies unavailable: '+dependency)
    tree = ast.parse((ROOT/'tests/test_dppo_flow_experts.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
                    n.name == 'test_wall_native_moe_gradients_padding_cached_context_and_weight_transfer')
    source = next(n.value.value for n in function.body if isinstance(n, ast.Assign) and
                  isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
    setup = source.split("with patch(ops+'.permute'", 1)[0]
    body = r'''
from wall_x.model.flow_prefix import DiffusionPrefix
settings=dict(transition_std=.03,transition_microbatch_size=8,transition_context_length=8,
              frozen_prefix_cache=True,prefix_cache_max_entries=8)
reference=WallOSSFlowExpert(backend(model),config=settings)
candidate=WallOSSFlowExpert(backend(copy.deepcopy(model)),config=settings)
assert reference.state_dict().keys()==candidate.state_dict().keys()
params_before={n:id(p) for n,p in candidate._source_model.named_parameters()}
def direct(self,layer,normalized):
    return self._direct_action_expert(layer,normalized)
with patch(ops+'.permute',side_effect=permute),patch(ops+'.unpermute',side_effect=unpermute):
    old_policy,new_policy=GaussianChainPolicy(reference),GaussianChainPolicy(candidate)
    old=old_policy.sample(packed,[11,19],initial_noise=x)
    with patch.object(DiffusionPrefix,'_action_suffix_moe',direct):
        new=new_policy.sample(packed,[11,19],initial_noise=x)
    for key in ('chain','logprobs','actions'):
        torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)
    steps=torch.tensor([0,2]);idx=torch.arange(2)
    old_score=old_policy.log_prob(packed,old['chain'][idx,steps],old['chain'][idx,steps+1],steps)
    with patch.object(DiffusionPrefix,'_action_suffix_moe',direct):
        new_score=new_policy.log_prob(packed,old['chain'][idx,steps],old['chain'][idx,steps+1],steps)
    torch.testing.assert_close(new_score,old_score,rtol=0,atol=0)
    old_score.mean().backward();new_score.mean().backward()
    for (name,a),(other_name,b) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert name==other_name and (a.grad is None)==(b.grad is None)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert params_before=={n:id(p) for n,p in candidate._source_model.named_parameters()}
    assert all(p.grad is None for p in candidate._source_model.parameters() if not p.requires_grad)
    # The fast suffix must still go through the original constructor checks.
    damaged={k:v.clone() for k,v in packed.items()}
    damaged['input_ids'][1,2],damaged['input_ids'][1,4]=7,4
    damaged['moe_token_types']=damaged['input_ids'].eq(7).long()
    with patch.object(DiffusionPrefix,'_action_suffix_moe',direct):
        try:candidate.transition(x,steps,damaged)
        except ValueError as exc:assert 'contiguous final action' in str(exc)
        else:raise AssertionError('Non-suffix action tokens bypassed initialization validation')
print('WALL_ACTION_SUFFIX_CPU_BIT_EXACT')
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT/'src'), str(ROOT/'src/pressb/model_sources/walloss')]),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', setup+body], cwd=tmp_path, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
