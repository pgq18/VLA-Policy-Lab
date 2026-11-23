"""Process-local, bounded reuse of Wall's frozen causal prefix for DPPO.

The original full-sized attention operations are retained. Only invariant
prefix MLP/normalization work is skipped; action-route autograd remains live.
No cached activations enter state_dict, checkpoints, or the node wire payload.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import weakref

import torch


class FrozenPrefixCache:
    def __init__(self, model, trainable_parameters, *, horizon, batch_size, context_length, max_entries):
        self.model, self.backbone = model, model.model
        self.horizon, self.batch_size = horizon, batch_size
        self.context_length, self.max_entries = context_length, max_entries
        self.rows, self.memo = OrderedDict(), OrderedDict()
        selected = {id(p) for p in trainable_parameters}
        self.frozen = tuple((name, p, p._version) for name, p in model.named_parameters() if id(p) not in selected)
        self.builds = self.hits = self.misses = self.evictions = 0

    def guard(self):
        if self.backbone.training or self.backbone.config._attn_implementation != 'eager':
            raise RuntimeError('Frozen prefix reuse requires eval-mode eager attention')
        current = dict(self.model.named_parameters())
        for name, parameter, version in self.frozen:
            if current.get(name) is not parameter or parameter.requires_grad or parameter._version != version:
                self.clear()
                raise RuntimeError('Frozen-prefix dependency changed: '+name)

    @staticmethod
    def _signature(condition):
        result = []
        for name, value in sorted(condition.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError('Wall prefix conditioning must contain tensors')
            try:
                version = value._version
            except RuntimeError:  # Inference tensors lack a version counter.
                return None
            result.append((name, id(value), version, tuple(value.shape), tuple(value.stride()), str(value.device), str(value.dtype)))
        return tuple(result)

    def register_keys(self, condition, keys):
        signature = self._signature(condition)
        if signature is None:
            return
        identity = id(condition['inputs_embeds'])
        self.memo[identity] = (signature, {name: weakref.ref(value) for name,value in condition.items()}, tuple(keys))
        self.memo.move_to_end(identity)
        while len(self.memo) > max(128, self.max_entries*4):
            self.memo.popitem(last=False)

    def keys(self, condition):
        """Hash each immutable source record once; collated GPU batches register keys.

        Versions/references are checked on every memo lookup, so ordinary
        in-place mutations cannot silently reuse an old conditioning prefix.
        Unsafe direct .data/storage mutation is outside the tensor contract.
        """
        signature = self._signature(condition)
        identity = id(condition['inputs_embeds'])
        old = self.memo.get(identity)
        if old and signature == old[0] and all(old[1][name]() is value for name,value in condition.items()):
            self.memo.move_to_end(identity)
            return old[2]
        cpu = {name: value.detach().cpu() for name,value in condition.items()}
        batch = cpu['input_ids'].shape[0]
        result = []
        for i in range(batch):
            mask = cpu['attention_mask'][i].bool()
            count = int(mask.sum())
            start = len(mask)-count
            if count < self.horizon or mask[:start].any() or not mask[start:].all():
                raise ValueError('Prefix keys require contiguous valid left-padded tokens')
            digest = hashlib.sha256()
            for name in sorted(cpu):
                value = cpu[name][i:i+1]
                if name == 'position_ids':
                    value = value[:, :, start:]
                elif name == 'inputs_embeds':
                    # Action placeholders are replaced at every transition.
                    value = value[:, start:-self.horizon]
                elif name != 'dof_mask':
                    value = value[:, start:]
                value = value.contiguous()
                digest.update((name+str(value.dtype)+str(tuple(value.shape))).encode())
                digest.update(value.view(torch.uint8).numpy().tobytes())
            result.append(digest.digest())
        self.register_keys(condition, result)
        return tuple(result)

    def clear(self):
        self.rows.clear()
        self.memo.clear()

    def info(self):
        return dict(cached_rows=len(self.rows), max_entries=self.max_entries,
                    bytes=sum(t.numel()*t.element_size() for row in self.rows.values() for t in row),
                    builds=self.builds, hits=self.hits, misses=self.misses, evictions=self.evictions,
                    storage='local_device', wire_payload='unchanged_inputs_only')

    def get(self, values, keys, action_embeddings, action_mask, starts, ends):
        from wall_x.model.flow_prefix import DiffusionPrefix
        self.guard()
        if len(keys) != self.batch_size or values['input_ids'].shape != (self.batch_size,self.context_length):
            raise ValueError('Prefix cache requires the fixed canonical batch/token shape')
        prefix = DiffusionPrefix(self.backbone,
            input_ids=values['input_ids'], inputs_embeds=values['inputs_embeds'],
            attention_mask=values['attention_mask'], position_ids=values['position_ids'].transpose(0,1),
            moe_token_types=values['moe_token_types'], action_mask=action_mask,
            horizon=self.horizon, mode='full_attention')
        missing = set(keys)-set(self.rows)
        self.hits += sum(key in self.rows for key in keys)
        self.misses += sum(key not in self.rows for key in keys)
        if missing:
            for key in keys:
                if key in self.rows:
                    self.rows.move_to_end(key)
            # Capture under the same full [B,T,D] shapes as the uncached path.
            # Valid prefix queries cannot see the final action tokens; only
            # those action tokens use the trainable expert. Padding outputs
            # may differ, but their keys remain masked from every valid query.
            with torch.no_grad():
                inputs = values['inputs_embeds'].clone()
                inputs[action_mask] = action_embeddings.detach().reshape(-1, inputs.shape[-1])
                hooks = prefix.capture_hooks()
                try:
                    self.backbone(input_ids=None, inputs_embeds=inputs,
                        attention_mask=values['attention_mask'], position_ids=values['position_ids'].transpose(0,1),
                        moe_token_types=values['moe_token_types'], start_indices=starts, end_indices=ends,
                        past_key_values=None, use_cache=False, output_attentions=False,
                        output_hidden_states=False, return_dict=True)
                finally:
                    for hook in hooks:
                        hook.remove()
                prefix.capture(None)
            self.builds += 1
            inserted = set()
            for index,key in enumerate(keys):
                if key not in missing or key in inserted:
                    continue
                while len(self.rows) >= self.max_entries:
                    self.rows.popitem(last=False)
                    self.evictions += 1
                # Clone individual rows: a view would retain a padded full
                # batch even when only one row survives LRU eviction.
                self.rows[key] = tuple(prefix.prefix_normalized[layer][index:index+1].detach().clone()
                                       for layer in range(len(self.backbone.layers)))
                inserted.add(key)
        else:
            prefix.prefix_normalized = {layer:torch.cat([self.rows[key][layer] for key in keys],dim=0)
                                        for layer in range(len(self.backbone.layers))}
            prefix.capture(None)
        for key in keys:
            if key in self.rows:
                self.rows.move_to_end(key)
        return prefix
