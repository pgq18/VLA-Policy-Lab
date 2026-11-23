"""Exercise the native π0.5 denoising network with a small CPU Gemma configuration."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pressb.models.pi05 import Pi05Backend, Pi05Context
from pressb.runtime.denoising import GaussianChainPolicy
from pressb.runtime.expert_sync import ExpertWeights, publish_expert, tree_from_wire, tree_to_wire


@pytest.fixture
def backend(monkeypatch):
    native = pytest.importorskip("lerobot.policies.pi05.modeling_pi05")
    paligemma = native.PaliGemmaForConditionalGenerationWithPiGemma
    gemma = native.PiGemmaForCausalLM

    def small_paligemma(config):
        config.text_config.vocab_size = 32
        config.image_token_index = 31
        vision = config.vision_config
        vision.hidden_size = vision.projection_dim = 32
        vision.intermediate_size = 64
        vision.num_attention_heads = 4
        vision.num_hidden_layers = 1
        return paligemma(config)

    def small_gemma(config):
        config.vocab_size = 32
        return gemma(config)

    monkeypatch.setattr(native, "get_gemma_config", lambda _: native.GemmaConfig(32, 2, 64, 4, 1, 8))
    monkeypatch.setattr(native, "PaliGemmaForConditionalGenerationWithPiGemma", small_paligemma)
    monkeypatch.setattr(native, "PiGemmaForCausalLM", small_gemma)
    config = native.PI05Config(chunk_size=4, n_action_steps=4, num_inference_steps=3,
                               max_action_dim=32, dtype="float32", device="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(519)
        model = native.PI05Pytorch(config).eval()
    result = Pi05Backend.__new__(Pi05Backend)
    result.policy = SimpleNamespace(model=model)
    result.device = torch.device("cpu")
    result.horizon, result.dimension = 4, 7
    result.action_scale = np.array([2., 3., 4., 5., 6., 7., .01], dtype=np.float32)
    result.postprocessor = lambda y: y + 0.25
    return result


def inputs(model):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    tokens = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
    with torch.inference_mode():
        embeds, padding, attention = model.embed_prefix([], [], tokens, tokens != 0)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, cache = model.paligemma_with_expert.forward(
            inputs_embeds=[embeds, None], position_ids=padding.cumsum(1) - 1,
            attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(padding, attention)),
            use_cache=True)
        contexts = [Pi05Context(padding[i], tuple((layer.keys[i:i+1], layer.values[i:i+1])
                                                   for layer in cache.layers), 200 + i)
                    for i in range(len(tokens))]
    return tokens, contexts


@pytest.mark.parametrize("scope,shared", [("full", True), ("action_out_proj", False)])
def test_native_euler_and_shuffled_ppo_update(backend, tmp_path, scope, shared):
    local = deepcopy(backend)
    model = backend.policy.model
    tokens, contexts = inputs(model)
    config = dict(trainable_scope=scope, transition_microbatch_size=2, transition_std=.05)
    if shared:
        config["shared_prefix_dir"] = str(tmp_path)
    sampler = backend.create_action_expert(config)
    learner = local.create_action_expert(config)
    policy = GaussianChainPolicy(sampler)
    packed = sampler.pack_contexts(contexts)
    seed = [ctx.seed for ctx in contexts]
    initial = torch.randn((3, 4, 32), generator=torch.Generator().manual_seed(515))
    native = model.sample_actions([], [], tokens, tokens != 0, noise=initial)
    deterministic = policy.sample(packed, seed, deterministic=True, initial_noise=initial)
    torch.testing.assert_close(deterministic["chain"][:, -1], native, atol=2e-6, rtol=2e-6)
    expected_actions = (native[:, :, :7] + .25) * torch.from_numpy(backend.action_scale)
    torch.testing.assert_close(deterministic["actions"], expected_actions, atol=2e-5, rtol=2e-6)
    assert deterministic["logprobs"] is None

    sample = policy.sample(packed, seed, initial_noise=initial)
    row = torch.tensor([2, 0, 1, 2, 0])
    step = torch.tensor([1, 2, 0, 2, 0])
    # Reorder the observations, repeat a prefix, mix denoising steps and leave a tail.
    wire_rows = [tree_from_wire(tree_to_wire({name: value[i:i+1] for name, value in packed.items()}))
                 for i in row.tolist()]
    cond = learner.collate_conditions(wire_rows)
    before, after = sample["chain"][row, step], sample["chain"][row, step + 1]
    actor = GaussianChainPolicy(learner)
    score = actor.log_prob(cond, before, after, step)
    torch.testing.assert_close(score, sample["logprobs"][row, step], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(score).all()

    parameters = dict(local.policy.model.named_parameters())
    trainable = {name for name, param in parameters.items() if param.requires_grad}
    if scope == "action_out_proj":
        assert trainable == {"action_out_proj.weight", "action_out_proj.bias"}
    else:
        for prefix in ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.",
                       "paligemma_with_expert.gemma_expert.model.layers."):
            assert any(name.startswith(prefix) for name in trainable)
        assert not any("paligemma_with_expert.paligemma." in name or "lm_head" in name for name in trainable)
    assert {id(parameters[name]) for name in trainable} == {id(param) for param in learner.parameters()}
    previous = {name: param.detach().clone() for name, param in parameters.items()}
    optimizer = torch.optim.AdamW(learner.parameters(), lr=.002)
    (-score.mean()).backward()
    assert all(torch.isfinite(param.grad).all() for param in learner.parameters() if param.grad is not None)
    assert all(param.grad is None for name, param in parameters.items() if name not in trainable)
    optimizer.step()
    assert not torch.equal(parameters["action_out_proj.weight"], previous["action_out_proj.weight"])
    assert all(torch.equal(param, previous[name]) for name, param in parameters.items() if name not in trainable)

    receiver = ExpertWeights(sampler, on_commit=sampler.invalidate_cache)
    receipt = publish_expert(SimpleNamespace(call=lambda op, request: getattr(
        receiver, op.removeprefix("/weights_"))(request)), learner.state_dict(), expected_version=0)
    assert receipt["policy_version"] == 1
    if shared:
        assert not list(tmp_path.glob("*.pt"))
    received = sampler.transition(before, step, cond)[0]
    torch.testing.assert_close(received, learner.transition(before, step, cond)[0], atol=0, rtol=0)
    assert not torch.equal(actor.log_prob(cond, before, after, step), score)


def test_prefix_files_are_owned_by_replica_and_readable_concurrently(backend, tmp_path):
    _, contexts = inputs(backend.policy.model)
    peer = deepcopy(backend)
    config = {"shared_prefix_dir": str(tmp_path), "shared_prefix_max_entries": 2}
    first, second = backend.create_action_expert(config), peer.create_action_expert(config)
    with ThreadPoolExecutor(2) as pool:
        first_refs, second_refs = list(pool.map(lambda expert: expert.pack_contexts(contexts[:2]),
                                                [first, second]))
        assert len(list(tmp_path.glob("*.pt"))) == 4
        results = list(pool.map(second.collate_conditions, [[first_refs], [first_refs]]))
    assert torch.equal(results[0]["kv"], results[1]["kv"])
    backend.close()
    assert len(list(tmp_path.glob("*.pt"))) == 2
    with pytest.raises(FileNotFoundError, match="prefix"):
        second.collate_conditions([first_refs])
    assert torch.equal(second.collate_conditions([second_refs])["kv"], results[0]["kv"])
    peer.close()
    assert not list(tmp_path.glob("*.pt"))


def test_shared_prefix_rejects_malformed_or_insufficient_storage(backend, tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        backend.create_action_expert({"shared_prefix_dir": "relative"})
    expert = backend.create_action_expert({"shared_prefix_dir": str(tmp_path), "shared_prefix_max_entries": 1})
    _, contexts = inputs(backend.policy.model)
    with pytest.raises(ValueError, match="shared_prefix_max_entries"):
        expert.pack_contexts(contexts[:2])
    with pytest.raises(ValueError, match="reference"):
        expert.collate_conditions([{"prefix_ref": torch.zeros(1, 15, dtype=torch.uint8)}])
    assert not list(tmp_path.iterdir())
