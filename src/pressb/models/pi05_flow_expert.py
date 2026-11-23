"""DPPO's Gaussian Euler kernel over the native LeRobot π0.5 action expert."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import uuid
import weakref

import torch
from torch import nn

from .flow_experts import _FlowExpert, _settings


def _remove_owned_files(files):
    while files:
        files.popleft().unlink(missing_ok=True)


class Pi05FlowExpert(_FlowExpert):
    def __init__(self, backend, *, config=None):
        super().__init__()
        config = dict(config or {})
        self.trainable_scope = config.pop("trainable_scope", "full")
        self.transition_microbatch_size = config.pop("transition_microbatch_size", 1)
        shared_dir = config.pop("shared_prefix_dir", None)
        self.shared_prefix_max_entries = config.pop("shared_prefix_max_entries", 512)
        if self.trainable_scope not in ("full", "action_out_proj"):
            raise ValueError("π0.5 trainable_scope must be full or action_out_proj")
        if type(self.transition_microbatch_size) is not int or self.transition_microbatch_size < 1:
            raise ValueError("transition_microbatch_size must be a positive integer")
        if type(self.shared_prefix_max_entries) is not int or self.shared_prefix_max_entries < 1:
            raise ValueError("shared_prefix_max_entries must be a positive integer")
        self.shared_prefix_dir = None
        if shared_dir is not None:
            if not isinstance(shared_dir, str) or not Path(shared_dir).is_absolute():
                raise ValueError("shared_prefix_dir must be an absolute path shared by inference and all learners")
            self.shared_prefix_dir = Path(shared_dir)
            self.shared_prefix_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._files = deque()
        weakref.finalize(self, _remove_owned_files, self._files)
        self._last_packed = None

        model = backend.policy.model
        self.denoising_steps, self.transition_std = _settings(config, model.config.num_inference_steps)
        self.horizon, self.latent_dim = backend.horizon, model.config.max_action_dim
        self.dt = -1.0 / self.denoising_steps
        self.register_buffer("timesteps", torch.tensor(
            [1.0 + index * self.dt for index in range(self.denoising_steps)],
            dtype=torch.float32, device=backend.device), persistent=False)
        model.eval().requires_grad_(False)
        object.__setattr__(self, "_source_model", model)
        self._convert_actions = backend.actions_from_latent
        modules = {"action_out_proj": model.action_out_proj}
        if self.trainable_scope == "full":
            # Native denoise_step calls gemma_expert.model.forward. Its text
            # lm_head is unused and must not enter optimization or publication.
            modules.update(gemma_expert=model.paligemma_with_expert.gemma_expert.model,
                           action_in_proj=model.action_in_proj, time_mlp_in=model.time_mlp_in,
                           time_mlp_out=model.time_mlp_out)
        self.expert_modules = nn.ModuleDict(modules)
        self.expert_modules.requires_grad_(True)
        self.trainable_source_parameters = tuple(name for name, parameter in model.named_parameters()
                                                 if parameter.requires_grad)
        self.train(False)
        self.assert_frozen_encoder()

    def describe(self):
        return dict(super().describe(), transformer_kv_cache=True, trainable_scope=self.trainable_scope,
                    transition_microbatch_size=self.transition_microbatch_size,
                    conditioning="shared_prefix_file" if self.shared_prefix_dir else "inline_prefix_kv")

    def invalidate_cache(self):
        # Publication is a barrier after all learners have consumed the rollout.
        self._last_packed = None
        _remove_owned_files(self._files)

    def _publish_prefix(self, padding, kv):
        key = uuid.uuid4()
        path = self.shared_prefix_dir / (key.hex + ".pt")
        # The filename is only sent after it is closed. Clone the slices so
        # torch.save serializes only this row's storage.
        with path.open("xb") as stream:
            self._files.append(path)
            torch.save({"padding": padding.clone(), "kv": kv.clone()}, stream)
        while len(self._files) > self.shared_prefix_max_entries:
            self._files.popleft().unlink(missing_ok=True)
        return list(key.bytes)

    def pack_contexts(self, contexts):
        if not contexts:
            raise ValueError("Expected nonempty π0.5 prefix contexts")
        padding = torch.stack([ctx.prefix_pad_mask for ctx in contexts])
        kv = torch.stack([torch.stack([torch.stack((key[0], value[0]))
                                       for key, value in ctx.prefix_kv]) for ctx in contexts])
        packed = {"padding": padding, "kv": kv}
        if self.shared_prefix_dir is None:
            return packed
        if len(contexts) > self.shared_prefix_max_entries:
            raise ValueError("shared_prefix_max_entries must hold the sampling microbatch")
        cpu_padding, cpu_kv = padding.detach().cpu(), kv.detach().cpu()
        keys = [self._publish_prefix(cpu_padding[i:i+1], cpu_kv[i:i+1]) for i in range(len(contexts))]
        refs = {"prefix_ref": torch.tensor(keys, dtype=torch.uint8)}
        # All denoising transitions for this sampling request use the original
        # device tensors. Only the learner opens the per-observation files.
        self._last_packed = (refs, packed)
        return refs

    def collate_conditions(self, conditions):
        if not conditions:
            raise ValueError("Expected at least one π0.5 conditioning row")
        if self.shared_prefix_dir is not None:
            if any(set(row) != {"prefix_ref"} for row in conditions):
                raise ValueError("Expected shared π0.5 prefix references")
            refs = torch.cat([row["prefix_ref"] for row in conditions]).detach().cpu()
            if refs.dtype != torch.uint8 or refs.ndim != 2 or refs.shape[1] != 16:
                raise ValueError("Invalid π0.5 shared prefix reference")
            conditions = []
            loaded = {}
            for ref in refs.tolist():
                key = bytes(ref)
                if key not in loaded:
                    path = self.shared_prefix_dir / (key.hex() + ".pt")
                    try:
                        loaded[key] = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
                    except FileNotFoundError as exc:
                        raise FileNotFoundError("Missing π0.5 prefix file. All ranks must share the directory and "
                                                "shared_prefix_max_entries must cover a full rollout per replica.") from exc
                conditions.append(loaded[key])
        return super().collate_conditions(conditions)

    def transition(self, x, step, conditioning):
        from transformers.cache_utils import DynamicCache

        self._check_transition(x, step, conditioning)
        if "prefix_ref" in conditioning:
            if self._last_packed is not None and self._last_packed[0] is conditioning:
                conditioning = self._last_packed[1]
            else:
                conditioning = self.collate_conditions([conditioning])
        padding = conditioning["padding"].detach().to(self.device)
        kv = conditioning["kv"].detach().to(self.device)
        x = x.to(device=self.device, dtype=torch.float32)
        step = step.to(self.device)
        if (padding.dtype != torch.bool or padding.ndim != 2 or kv.ndim != 6 or kv.shape[2] != 2
                or kv.shape[0] != len(x) or padding.shape != (len(x), kv.shape[-2])):
            raise ValueError("π0.5 prefix mask or layer KV geometry differs from the latent batch")
        size = self.transition_microbatch_size
        means = []
        for start in range(0, len(x), size):
            end = min(start + size, len(x))
            count = end - start
            def batch(tensor):
                part = tensor[start:end]
                if count < size:
                    part = torch.cat((part, part[-1:].expand(size - count, *part.shape[1:])))
                return part
            keys_values = batch(kv)
            cache = DynamicCache(tuple((keys_values[:, layer, 0].contiguous(), keys_values[:, layer, 1].contiguous())
                                       for layer in range(kv.shape[1])))
            noisy = batch(x)
            velocity = self._source_model.denoise_step(batch(padding), cache, noisy, self.timesteps[batch(step)])
            means.append((noisy + self.dt * velocity.float())[:count])
        mean = torch.cat(means)
        return mean, torch.full_like(mean, self.transition_std)

    def actions_from_latent(self, x):
        return self._convert_actions(x)
