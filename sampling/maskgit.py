"""maskgit's released sampler, with domino at the positions each step reveals.

everything that maskgit's sampler does is kept: the arc-cosine schedule, classifier-free guidance
and a gumbel temperature both annealed over the steps, and the choice of positions by the noised
confidence of a candidate drawn at every masked position. domino only changes the candidates at
the chosen positions, through the maximal coupling of `Domino.couple`, so with zero potentials this
is maskgit's sampler. with reveal="random" the positions are chosen at random, as in our text
experiments, and the chosen set is drawn from the chain.

note that maskgit's softmax temperature `sm_temp` multiplies the logits, so values above 1 sharpen.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.domino import Domino, arccos_counts


def ramp(i: int, steps: int) -> float:
    return 0.0 if steps <= 1 else i / (steps - 1)


@torch.no_grad()
def sample(model: Domino, n: int, labels: torch.Tensor | None, steps: int, cfg_w: float = 3.0,
           sm_temp: float = 1.0, r_temp: float = 4.5, reveal: str = "confidence",
           product: bool = False, generator: torch.Generator | None = None) -> torch.Tensor:
    """n token grids in `steps` steps, for the given class labels (none for the null class)."""
    den = model.denoiser
    device = den.token_embedding.device
    L, mask_id = den.seq_len, den.mask_id
    z = torch.full((n, L), mask_id, device=device, dtype=torch.long)
    sched = arccos_counts(L, steps)
    for i, count in enumerate(sched):
        masked = z == mask_id
        if not masked.any():
            break
        count = max(1, min(count, int(masked.sum(1).min())))
        w = cfg_w * ramp(i, len(sched))
        den.labels = labels
        logits, h = den(z)
        if w > 0:
            den.labels = None
            logits_u, _ = den(z)
            logits = (1 + w) * logits - w * logits_u
        logits = logits * sm_temp
        logits[..., mask_id] = -float("inf")

        prob = F.softmax(logits, dim=-1)
        cand = torch.multinomial(prob.reshape(-1, prob.shape[-1]), 1, generator=generator).view(n, L)
        temp = r_temp * (1.0 - ramp(i, len(sched)))
        if reveal == "random":
            key = torch.rand(n, L, device=device, generator=generator)
        else:
            conf = prob.gather(-1, cand[..., None]).squeeze(-1)
            g = -torch.log(-torch.log(torch.rand(n, L, device=device, generator=generator).clamp_min(1e-20)))
            key = conf.clamp_min(1e-20).log() + temp * g
        key = key.masked_fill(~masked, -float("inf"))
        S = torch.zeros_like(masked).scatter(1, key.topk(count, dim=-1).indices, True) & masked

        if product:
            z = torch.where(S, cand, z)
        elif reveal == "random":
            z = model.draw(z, logits, h, S)
        else:
            # the threshold a position had to clear is the largest key among those left masked
            tau = key.masked_fill(S, -float("inf")).max(dim=-1).values
            z = model.couple(z, logits, h, S, cand, tau, temp)
    den.labels = None
    return z
