"""domino: a first-order markov chain over the positions revealed in one denoising step.

a masked diffusion model draws the positions it reveals in a step independently from its
per-position posteriors (the product kernel). domino draws them in position order instead. the
first revealed position keeps the denoiser's posterior, and each next position j follows a
transition given the token a of its predecessor i,

    p(z_j = b | z_i = a) = softmax_b( l_j(b) + psi_ij(a, b) ),    psi_ij(a, b) = E_a^T A_ij E_b,
    A_ij = g([u_i; u_j; e(j - i)]),

where l_j are the denoiser's logits, u = W_u h are pair factors read from its final hidden states
h, E is a projection of its input embedding, e embeds the signed gap between the
two positions, and g is a two-layer mlp. the last layer of g starts at zero, so an untrained
module recovers the product kernel. the denoiser stays fixed and only the module is trained.

times follow mdlm: t = 1 is the fully masked state and alpha = 1 - t is the fraction kept.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def chain_parents(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """the chain over each row's revealed set S [B, L] in position order.

    returns parent [B, L], the previous revealed position (-1 for the first one and outside S),
    and rank [B, L], the order within S (-1 outside S).
    """
    B, L = S.shape
    idx = torch.arange(L, device=S.device)[None].expand(B, L)
    rank = torch.cumsum(S.long(), dim=1) - 1
    order = torch.where(S, idx, idx + L).argsort(dim=1)
    prev = torch.gather(order, 1, (rank - 1).clamp_min(0))
    parent = torch.where(S & (rank > 0), prev, torch.full_like(prev, -1))
    return parent, torch.where(S, rank, torch.full_like(rank, -1))


def chain_edges(parent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(row, child, parent) for every edge of the chain."""
    b, j = torch.nonzero(parent >= 0, as_tuple=True)
    return b, j, parent[b, j]


def gumbel_argmax(logits: torch.Tensor) -> torch.Tensor:
    return (logits - torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-20)))).argmax(-1)


class DominoModule(nn.Module):
    """the trained part: pairwise potentials from the denoiser's hidden states.

    rank is the size d_p of the token table and of the potential matrices A_ij.
    """

    def __init__(self, dim: int, embedding: torch.Tensor, rank: int = 128,
                 gap_dim: int = 16, max_gap: int = 32, hidden: int = 256) -> None:
        super().__init__()
        self.rank, self.max_gap = rank, max_gap
        self.W_u = nn.Linear(dim, rank, bias=False)
        self.W_E = nn.Linear(dim, rank, bias=False)
        self.register_buffer("tok_emb", embedding.detach().float().clone(), persistent=False)
        self.gap_emb = nn.Embedding(2 * max_gap + 1, gap_dim)
        self.g = nn.Sequential(nn.Linear(2 * rank + gap_dim, hidden), nn.GELU(),
                               nn.Linear(hidden, rank * rank))
        nn.init.zeros_(self.g[2].weight)
        nn.init.zeros_(self.g[2].bias)

    def token_table(self) -> torch.Tensor:
        """E [V, rank], scaled to unit root-mean-square entries."""
        E = self.W_E(self.tok_emb)
        return E * torch.rsqrt(E.pow(2).mean() + 1e-8)

    def pair_factors(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_u(h)

    def potentials(self, u_i: torch.Tensor, u_j: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        """A_ij [N, rank, rank] for N edges with signed gaps j - i."""
        e = self.gap_emb(gap.clamp(-self.max_gap, self.max_gap) + self.max_gap)
        return self.g(torch.cat([u_i, u_j, e], dim=-1)).view(-1, self.rank, self.rank)

    def transition_logits(self, l_j: torch.Tensor, u_i: torch.Tensor, u_j: torch.Tensor,
                          gap: torch.Tensor, a: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """l_j + psi_ij(a, .) for N edges whose parents hold the tokens a [N]."""
        A = self.potentials(u_i, u_j, gap)
        return l_j + torch.einsum("nd,nde->ne", E[a], A) @ E.t()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Domino(nn.Module):
    """a fixed denoiser with a domino module: chain likelihood, training losses and draws."""

    def __init__(self, denoiser: nn.Module, module: DominoModule) -> None:
        super().__init__()
        self.denoiser = denoiser.eval().requires_grad_(False)
        self.module = module
        self.mask_id = denoiser.mask_id
        self.pad_id = denoiser.pad_id
        self.eps = denoiser.eps

    def train(self, mode: bool = True) -> "Domino":
        super().train(mode)
        self.denoiser.eval()
        return self

    def features(self, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """logits [B, L, V] with the mask token excluded, and hidden states [B, L, D]."""
        with torch.no_grad():
            logits, h = self.denoiser(z, t)
        logits = logits.clone()
        logits[..., self.mask_id] = float("-inf")
        return logits, h

    def log_prob(self, logits: torch.Tensor, h: torch.Tensor, S: torch.Tensor,
                 x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """per-position log-probabilities of the tokens x at the revealed positions S, under the
        chain and under the product kernel (both [B, L], zero outside S)."""
        lp_product = F.log_softmax(logits, dim=-1).gather(-1, x[..., None]).squeeze(-1)
        lp_product = torch.where(S, lp_product, torch.zeros_like(lp_product))
        lp_chain = lp_product.clone()
        b, j, i = chain_edges(chain_parents(S)[0])
        if b.numel() > 0:
            u = self.module.pair_factors(h)
            E = self.module.token_table()
            out = torch.empty(b.numel(), device=logits.device, dtype=logits.dtype)
            chunk = max(1, 2 ** 30 // (4 * logits.shape[-1]))   # about 1 gb of logits at a time
            for c in range(0, b.numel(), chunk):
                s = slice(c, c + chunk)
                cl = self.module.transition_logits(logits[b[s], j[s]], u[b[s], i[s]], u[b[s], j[s]],
                                                   j[s] - i[s], x[b[s], i[s]], E)
                out[s] = F.log_softmax(cl, dim=-1).gather(-1, x[b[s], j[s]][:, None]).squeeze(-1)
            lp_chain = lp_chain.index_put((b, j), out)
        return lp_chain, lp_product

    def _step_loss(self, logits, h, S, x) -> dict[str, torch.Tensor]:
        lp_chain, lp_product = self.log_prob(logits, h, S, x)
        valid = x != self.pad_id if self.pad_id >= 0 else torch.ones_like(S)
        count = (S & valid).float()
        n = count.sum().clamp_min(1.0)
        nll_chain = -(lp_chain * count).sum() / n
        nll_product = -(lp_product * count).sum() / n
        return {"loss": nll_chain, "nll_chain": nll_chain.detach(),
                "nll_product": nll_product.detach(), "gain": (nll_product - nll_chain).detach(),
                "revealed": n.detach() / x.shape[0]}

    def loss(self, x: torch.Tensor, K: int = 4, eta: float = 0.5,
             h_min: float = 1 / 32) -> dict[str, torch.Tensor]:
        """negative log-likelihood of the data's transition from alpha_t to alpha_s under the
        chain, per revealed token.

        with probability 1 - eta the step is one of the K steps of the sampling schedule, and
        otherwise its size h is log-uniform on [h_min, 1] and its start is uniform.
        """
        B, L = x.shape
        dev = x.device
        k = torch.randint(0, K, (B,), device=dev).float()
        log_h = torch.rand(B, device=dev) * (-math.log(h_min)) + math.log(h_min)
        h = log_h.exp()
        a_rand = torch.rand(B, device=dev) * (1.0 - h)
        off_grid = torch.rand(B, device=dev) < eta
        alpha_t = torch.where(off_grid, a_rand, k / K)
        alpha_s = torch.where(off_grid, a_rand + h, (k + 1) / K).clamp(max=1.0)

        # mask with the denoiser's schedule at alpha_t, then reveal each masked position with
        # probability r = (alpha_s - alpha_t) / (1 - alpha_t)
        r = ((alpha_s - alpha_t) / (1.0 - alpha_t).clamp_min(1e-6)).clamp(0.0, 1.0)
        masked = torch.rand(B, L, device=dev) < ((1.0 - self.eps) * (1.0 - alpha_t))[:, None]
        z = torch.where(masked, torch.full_like(x, self.mask_id), x)
        S = masked & (torch.rand(B, L, device=dev) < r[:, None])
        logits, hid = self.features(z, 1.0 - alpha_t)
        return self._step_loss(logits, hid, S, x)

    def loss_confidence(self, x: torch.Tensor, K_choices=(2, 3, 4, 6, 8), r_temp: float = 4.5,
                        sm_temp: float = 1.0) -> dict[str, torch.Tensor]:
        """the chain loss on the reveal sets of maskgit's confidence sampler.

        maskgit reveals the positions whose candidate token it finds most probable after gumbel
        noise that cools over an arc-cosine schedule. with teacher forcing, we rank positions by
        the probability of the true token plus that noise, follow one such trajectory to a step
        drawn uniformly, and score the chain on the set that step reveals.
        """
        B, L = x.shape
        dev = x.device
        K = int(K_choices[int(torch.randint(len(K_choices), (1,)))])
        counts = arccos_counts(L, K)
        j = int(torch.randint(K, (1,)))

        def key(logits, masked, i):
            lq = F.log_softmax(sm_temp * logits, dim=-1).gather(-1, x[..., None]).squeeze(-1)
            temp = r_temp * (1.0 - (i / (K - 1) if K > 1 else 0.0))
            g = -torch.log(-torch.log(torch.rand(B, L, device=dev).clamp_min(1e-20)))
            return (lq + temp * g).masked_fill(~masked, -float("inf"))

        z = torch.full_like(x, self.mask_id)
        for i in range(j):
            logits, _ = self.features(z, 1.0 - torch.full((B,), i / K, device=dev))
            masked = z == self.mask_id
            sel = key(logits, masked, i).topk(min(counts[i], int(masked.sum(1).min())), dim=-1).indices
            z = z.scatter(1, sel, x.gather(1, sel))
        logits, hid = self.features(z, 1.0 - torch.full((B,), j / K, device=dev))
        masked = z == self.mask_id
        if j == K - 1:
            S = masked
        else:
            sel = key(logits, masked, j).topk(min(counts[j], int(masked.sum(1).min())), dim=-1).indices
            S = torch.zeros_like(masked).scatter(1, sel, True) & masked
        return self._step_loss(logits, hid, S, x)

    @torch.no_grad()
    def kstep_bound(self, x: torch.Tensor, K: int, product: bool = False,
                    generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """the negative log-likelihood bound of the K-step sampler on rows x, summed over each
        row's tokens, and the number of tokens per row. each position is revealed at a step
        drawn uniformly, as under the sampler's random reveal rule."""
        B, L = x.shape
        valid = x != self.pad_id if self.pad_id >= 0 else torch.ones_like(x, dtype=torch.bool)
        step = torch.randint(0, K, (B, L), device=x.device, generator=generator)
        total = torch.zeros(B, device=x.device)
        for k in range(K):
            S = step == k
            z = torch.where(step >= k, torch.full_like(x, self.mask_id), x)
            logits, h = self.features(z, 1.0 - torch.full((B,), k / K, device=x.device))
            if product:
                lp = F.log_softmax(logits, dim=-1).gather(-1, x[..., None]).squeeze(-1)
            else:
                lp = self.log_prob(logits, h, S, x)[0]
            total -= (lp * (S & valid).float()).sum(-1)
        return total, valid.sum(-1).float()

    @torch.no_grad()
    def draw(self, z: torch.Tensor, logits: torch.Tensor, h: torch.Tensor, S: torch.Tensor,
             product: bool = False, temperature: float = 1.0, return_logp: bool = False):
        """fill the positions S of z with a draw from the kernel and leave the rest of z unchanged.

        `logits` are expected to be divided by the temperature already, and the pairwise term is
        divided here, so the whole log-joint of the step is tempered. with `return_logp`, also
        return the log-probability of each drawn token.
        """
        B, L = z.shape
        logp = torch.zeros(B, L, device=z.device) if return_logp else None
        if product:
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).view(B, L)
            z = torch.where(S, tok, z)
            if return_logp:
                lp = probs.gather(-1, tok[..., None]).squeeze(-1).clamp_min(1e-20).log()
                logp = torch.where(S, lp, logp)
            return (z, logp) if return_logp else z

        parent, rank = chain_parents(S)
        rb, rj = torch.nonzero(S & (rank == 0), as_tuple=True)
        if rb.numel() > 0:
            first = logits[rb, rj]
            tok = gumbel_argmax(first)
            z = z.index_put((rb, rj), tok)
            if return_logp:
                logp = logp.index_put((rb, rj), F.log_softmax(first, -1).gather(-1, tok[:, None]).squeeze(-1))
        b_all, j_all, i_all = chain_edges(parent)
        if b_all.numel() > 0:
            # the potentials of every edge do not depend on the drawn tokens, so compute them once
            u = self.module.pair_factors(h)
            E = self.module.token_table()
            Et = E.t().contiguous()
            A_all = self.module.potentials(u[b_all, i_all], u[b_all, j_all], j_all - i_all)
            rank_all = rank[b_all, j_all]
            for r in range(1, int(rank_all.max()) + 1):
                sel = torch.nonzero(rank_all == r, as_tuple=True)[0]
                if sel.numel() == 0:
                    continue
                b, j, i = b_all[sel], j_all[sel], i_all[sel]
                v = torch.einsum("nd,nde->ne", E[z[b, i]], A_all[sel])
                cl = logits[b, j] + (1.0 / temperature) * (v @ Et)
                tok = gumbel_argmax(cl)
                z = z.index_put((b, j), tok)
                if return_logp:
                    logp = logp.index_put((b, j), F.log_softmax(cl, -1).gather(-1, tok[:, None]).squeeze(-1))
        return (z, logp) if return_logp else z

    @torch.no_grad()
    def couple(self, z: torch.Tensor, logits: torch.Tensor, h: torch.Tensor, S: torch.Tensor,
               cand: torch.Tensor, tau: torch.Tensor, temp: float) -> torch.Tensor:
        """commit the candidates a product sampler drew at S, corrected toward the chain.

        along the chain over S, each position keeps its candidate c with probability
        min(1, p(c) / q(c)), where q is the product kernel and p the chain's transition given the
        token committed at the predecessor, and otherwise draws from the normalized excess of p over
        q. this is the maximal coupling of speculative decoding: if c is a draw from q, the committed
        token is a draw from p, and with zero potentials every candidate is kept.

        a confidence sampler keeps the candidates that clear its selection threshold, so the kept
        ones are draws from q reweighted by the propensity P(log r(c) + temp * g > tau) of passing,
        with g gumbel noise and tau [B] the threshold of each row. both kernels are reweighted by
        their propensity before they are coupled.
        """
        z = torch.where(S, cand, z)
        parent, rank = chain_parents(S)
        b_all, j_all, i_all = chain_edges(parent)
        if b_all.numel() == 0:
            return z
        u = self.module.pair_factors(h)
        E = self.module.token_table()
        Et = E.t().contiguous()
        A_all = self.module.potentials(u[b_all, i_all], u[b_all, j_all], j_all - i_all)
        rank_all = rank[b_all, j_all]

        def log_propensity(lr, threshold):
            d = ((lr - threshold) / temp).clamp(max=30.0)
            lpi = torch.where(d < -20, d, torch.log(-torch.expm1(-torch.exp(d))).clamp_min(-1e4))
            return torch.where(torch.isfinite(threshold), lpi, torch.zeros_like(lpi))

        for r in range(1, int(rank_all.max()) + 1):
            sel = torch.nonzero(rank_all == r, as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            b, j, i = b_all[sel], j_all[sel], i_all[sel]
            pair = torch.einsum("nd,nde->ne", E[z[b, i]], A_all[sel]) @ Et
            lq = F.log_softmax(logits[b, j].float(), -1)
            lp = F.log_softmax(logits[b, j].float() + pair.float(), -1)
            if temp > 0:
                threshold = tau[b].float()[:, None]
                lq = F.log_softmax(lq + log_propensity(lq, threshold), -1)
                lp = F.log_softmax(lp + log_propensity(lp, threshold), -1)
            c = z[b, j]
            ratio = (lp.gather(-1, c[:, None]) - lq.gather(-1, c[:, None])).squeeze(-1).exp()
            keep = torch.rand_like(ratio) < ratio
            excess = (lp.exp() - lq.exp()).clamp_min(0)
            mass = excess.sum(-1)
            redo = ~keep & (mass > 1e-12)
            tok = c.clone()
            if redo.any():
                tok[redo] = torch.multinomial(excess[redo] / mass[redo, None], 1).squeeze(-1)
            z = z.index_put((b, j), tok)
        return z


def arccos_counts(n_tokens: int, steps: int) -> list[int]:
    """maskgit's arc-cosine schedule as the number of positions each step reveals."""
    r = torch.linspace(0, 1, steps + 1)
    sched = (torch.arccos(r) / (math.pi * 0.5) * n_tokens).round()
    for i in range(len(sched) - 1):
        sched[i] = sched[i] - sched[i + 1]
    sched = sched[:-1]
    sched[sched == 0] = 1
    sched[-1] += n_tokens - sched.sum()
    return [int(v) for v in sched.tolist()]


def build_module(denoiser: nn.Module, cfg: dict) -> DominoModule:
    return DominoModule(denoiser.dim, denoiser.token_embedding, **cfg)


def load_domino(path: str, denoiser: nn.Module, device: torch.device) -> Domino:
    """a trained module on its denoiser, with the module's moving-average weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    module = build_module(denoiser, ckpt["module_cfg"])
    module.load_state_dict(ckpt["ema"]["shadow"] if ckpt.get("ema") else ckpt["module"])
    return Domino(denoiser, module.to(device)).eval()
