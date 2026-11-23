"""Access the native initial noise of an encoded observation for residual decoding."""
from __future__ import annotations

from operator import attrgetter

from .diffusion_policy import DiffusionPolicyBackend
from .pi05 import Pi05Backend
from .vlajepa import VLAJEPABackend
from .vlajepa_joint import JointVLAJEPABackend
from .walloss import WallOSSBackend


def baseline_noise_reader(backend, shape):
    """Return a (context -> [horizon, latent_dim]) reader, if supported.

    Existing frozen adapters fingerprint their source for checkpoint resume.
    Keep their residual-only bindings here. External adapters can expose the
    same contract through a baseline_noise(context) method.
    """
    reader = getattr(backend, "baseline_noise", None)
    if callable(reader):
        return reader
    if isinstance(backend, VLAJEPABackend):
        return attrgetter("baseline_noise")
    if isinstance(backend, (JointVLAJEPABackend, WallOSSBackend)):
        return lambda ctx: ctx.baseline_noise[0].detach().float().cpu().numpy()
    if isinstance(backend, DiffusionPolicyBackend):
        return lambda ctx: ctx.initial_noise.detach().float().cpu().numpy()
    if isinstance(backend, Pi05Backend):
        import torch

        def seeded_noise(ctx):
            generator = torch.Generator(device=backend.device).manual_seed(ctx.seed)
            return torch.randn(shape, dtype=torch.float32, device=backend.device,
                               generator=generator).cpu().numpy()

        return seeded_noise
    return None
