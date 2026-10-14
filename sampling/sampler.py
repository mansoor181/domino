"""the K-step sampler for text and molecules, with the unmasking rules of the paper.

each step queries the denoiser once and reveals a set S of masked positions, drawn jointly from the
chain (or independently from the product kernel). the rules that choose S are
  random      each masked position is revealed with probability 1 / (K - i) at step i, as in the
              forward process
  dilated     a fixed spread-out order (bit reversal), as in the dilated unmasking scheduler
  confidence  the largest top-1 probability first
  margin      the largest gap between the two most probable tokens first
  entropy     the lowest predictive entropy first
  drawn       maskgit's rule: draw a token at every masked position and keep the ones the kernel
              found most probable
  eb<gamma>   eb-sampler: reveal the lowest-entropy positions whose entropies, the largest one
              excluded, sum to at most gamma. the number of steps is adaptive, so K is unused.
every rule but random and eb reveals ceil(remaining / steps left) positions per step.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from models.domino import Domino


def dilated_ranks(L: int, device: torch.device) -> torch.Tensor:
    """the rank of each position in bit-reversal order."""
    bits = max(1, math.ceil(math.log2(L)))
    perm = [r for r in (int(format(i, f"0{bits}b")[::-1], 2) for i in range(2 ** bits)) if r < L]
    ranks = torch.empty(L, dtype=torch.long, device=device)
    ranks[torch.tensor(perm, device=device)] = torch.arange(L, device=device)
    return ranks


@torch.no_grad()
def sample(model: Domino, K: int, n: int, order: str = "random", product: bool = False,
           temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """n rows in K steps. returns the tokens and the step at which each position was revealed.

    the temperature divides the logits and the pairwise potentials alike.
    """
    device = model.denoiser.token_embedding.device
    L, mask_id = model.denoiser.seq_len, model.mask_id
    z = torch.full((n, L), mask_id, device=device, dtype=torch.long)
    step_of = torch.full((n, L), -1, device=device, dtype=torch.long)
    ranks = dilated_ranks(L, device) if order == "dilated" else None
    eb_gamma = float(order[2:]) if order.startswith("eb") else None

    for i in range(L if eb_gamma is not None else K):
        masked = z == mask_id
        if eb_gamma is not None:
            if not masked.any():
                break
            alpha = 1.0 - masked.float().mean(1)
        else:
            alpha = torch.full((n,), i / K, device=device)
        logits, h = model.features(z, 1.0 - alpha)
        if temperature != 1.0:
            logits = logits / temperature

        if eb_gamma is not None:
            prob = F.softmax(logits, dim=-1)
            ent = -(prob * prob.clamp_min(1e-30).log()).sum(-1)
            ent = torch.where(masked, ent, torch.full_like(ent, float("inf")))
            srt, idx = ent.sort(dim=1)
            finite = torch.nan_to_num(srt, posinf=0.0)
            ok = (torch.cumsum(finite, 1) - finite <= eb_gamma) & torch.isfinite(srt)
            ok[:, 0] = torch.isfinite(srt[:, 0])
            S = torch.zeros_like(masked).scatter(1, idx, ok) & masked
            z = model.draw(z, logits, h, S, product, temperature)
            step_of = torch.where(S, torch.full_like(step_of, i), step_of)
            continue

        n_reveal = torch.ceil(masked.sum(1).float() / (K - i)).long()[:, None]
        if order == "drawn":
            if product:
                cand = torch.distributions.Categorical(logits=logits).sample()
                lp = F.log_softmax(logits, dim=-1).gather(-1, cand[..., None]).squeeze(-1)
            else:
                cand, lp = model.draw(z.clone(), logits, h, masked, product, temperature,
                                      return_logp=True)
            key = torch.where(masked, lp, torch.full_like(lp, -float("inf")))
            S = masked & (key.argsort(dim=1, descending=True).argsort(dim=1) < n_reveal)
            z = torch.where(S, cand, z)
            step_of = torch.where(S, torch.full_like(step_of, i), step_of)
            continue

        if order == "random":
            S = masked & (torch.rand(n, L, device=device) < 1.0 / (K - i))
            if i == K - 1:
                S = masked
        elif order == "dilated":
            key = torch.where(masked, ranks[None].expand(n, L).float(),
                              torch.full_like(masked, 10 * L, dtype=torch.float))
            S = masked & (key.argsort(dim=1).argsort(dim=1) < n_reveal)
        else:
            prob = F.softmax(logits, dim=-1)
            if order == "confidence":
                score = prob.max(-1).values
            elif order == "margin":
                top2 = prob.topk(2, dim=-1).values
                score = top2[..., 0] - top2[..., 1]
            elif order == "entropy":
                score = (prob * prob.clamp_min(1e-30).log()).sum(-1)
            else:
                raise ValueError(f"unknown unmasking rule {order!r}")
            key = torch.where(masked, score, torch.full_like(score, -float("inf")))
            S = masked & (key.argsort(dim=1, descending=True).argsort(dim=1) < n_reveal)
        z = model.draw(z, logits, h, S, product, temperature)
        step_of = torch.where(S, torch.full_like(step_of, i), step_of)
    return z, step_of


@torch.no_grad()
def sample_on_schedule(model: Domino, step_of: torch.Tensor, K: int, product: bool = False) -> torch.Tensor:
    """sample along a given reveal schedule, where step_of [n, L] is the step of each position.

    two kernels run on one schedule and one random seed differ only in how each step's tokens are
    drawn, which is how the paired examples are made.
    """
    n, L = step_of.shape
    z = torch.full((n, L), model.mask_id, device=step_of.device, dtype=torch.long)
    for i in range(K):
        logits, h = model.features(z, 1.0 - torch.full((n,), i / K, device=step_of.device))
        z = model.draw(z, logits, h, step_of == i, product)
    return z
