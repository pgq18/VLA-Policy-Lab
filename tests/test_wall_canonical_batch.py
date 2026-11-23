"""Fixed-shape Wall padding stays identical without per-row host reads."""
from types import SimpleNamespace

import pytest
import torch

from pressb.models.flow_experts import WallOSSFlowExpert


def expert(target=8):
    value = WallOSSFlowExpert.__new__(WallOSSFlowExpert)
    torch.nn.Module.__init__(value)
    value.transition_context_length = target
    return value


def conditions(length, counts):
    batch = len(counts)
    valid = torch.arange(length)[None] >= length-torch.tensor(counts)[:, None]
    # Deliberately nonzero padding, including NaN embeddings: row trim must
    # discard it, rather than retain it or multiply NaN by a zero mask.
    embeds = torch.arange(batch*length*3).reshape(batch, length, 3).float()
    embeds[~valid] = float('nan')
    return dict(input_ids=torch.arange(batch*length).reshape(batch, length)+3,
                attention_mask=valid.long()*2, inputs_embeds=embeds,
                position_ids=torch.arange(batch*3*length).reshape(batch, 3, length),
                moe_token_types=torch.ones(batch, length, dtype=torch.long),
                dof_mask=torch.arange(batch*2*4).reshape(batch, 2, 4).float())


@pytest.mark.parametrize('length,counts,target', [
    (6, [1, 3, 6], 8), (8, [1, 8, 5], 8), (12, [3, 8, 7], 8),
    (6, [1], 8), (1, [1, 1, 1], 8), (256, [184, 224, 256], 256),
])
def test_fixed_batch_exactly_matches_original_row_trim_pad(length, counts, target):
    model = expert(target)
    packed = conditions(length, counts)
    expected = [model._canonical_condition_row(packed, i) for i in range(len(counts))]
    actual = model._canonical_condition_batch(packed, len(counts))
    for key, result in actual.items():
        torch.testing.assert_close(result, torch.cat([row[key] for row in expected]), rtol=0, atol=0)
    # No source mutation or accidental input-gradient detachment.
    packed['inputs_embeds'].requires_grad_(True)
    actual = model._canonical_condition_batch(packed, len(counts))['inputs_embeds']
    actual.sum().backward()
    assert torch.equal(packed['inputs_embeds'].grad, packed['attention_mask'].bool().unsqueeze(-1).expand_as(packed['inputs_embeds']).float())


@pytest.mark.parametrize('mask,target,error', [
    ([0, 0, 0, 0], 4, 'contiguous'),
    ([0, 1, 0, 1], 4, 'contiguous'),
    ([1, 1, 1, 0], 4, 'contiguous'),
    ([1, 1, 1, 1], 3, 'exceeding'),
])
def test_fixed_batch_rejects_the_same_invalid_mask_as_original(mask, target, error):
    model = expert(target)
    packed = conditions(len(mask), [len(mask), len(mask)])
    packed['attention_mask'][1] = torch.tensor(mask)
    with pytest.raises(ValueError, match=error):
        model._canonical_condition_row(packed, 1)
    with pytest.raises(ValueError, match=error):
        model._canonical_condition_batch(packed, 2)


@pytest.mark.parametrize('change', ['batch', 'sequence', 'mask_ndim'])
def test_fixed_batch_rejects_misaligned_condition_tensors(change):
    packed = conditions(6, [3, 6])
    if change == 'batch':
        packed['dof_mask'] = packed['dof_mask'][:1]
    elif change == 'sequence':
        packed['inputs_embeds'] = packed['inputs_embeds'][:, :-1]
    else:
        packed['attention_mask'] = packed['attention_mask'][..., None]
    with pytest.raises(ValueError):
        expert()._canonical_condition_batch(packed, 2)


@pytest.mark.parametrize('problem,error', [('horizon','horizon'), ('route','expert 1'), ('padding','padded')])
def test_route_count_shortcut_still_rejects_invalid_routing(problem, error):
    model = expert()
    model.register_parameter('anchor', torch.nn.Parameter(torch.zeros(1)))
    model.horizon, model.action_token_id = 2, 7
    object.__setattr__(model, '_source_model', SimpleNamespace())
    packed = conditions(4, [4, 4])
    packed['input_ids'] = torch.tensor([[2, 3, 7, 7], [2, 3, 7, 7]])
    packed['moe_token_types'] = packed['input_ids'].eq(7).long()
    if problem == 'horizon':
        packed['input_ids'][0, -1] = 3
    elif problem == 'route':
        packed['moe_token_types'][1, 1] = 1
    else:
        packed['attention_mask'][1, -1] = 0
    with pytest.raises(ValueError, match=error):
        model._transition_row(torch.zeros(2, 2, 20), torch.zeros(2, dtype=torch.long), packed)
