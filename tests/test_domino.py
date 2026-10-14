"""checks of the chain on a tiny mdlm, on the cpu.

    python -m pytest tests
"""
import itertools

import pytest
import torch
import torch.nn.functional as F

from models.domino import Domino, DominoModule, arccos_counts, chain_edges, chain_parents
from models.mdlm import MDLM, MDLMConfig
from sampling.sampler import sample

V, L = 6, 8


@pytest.fixture
def model() -> Domino:
    torch.manual_seed(0)
    cfg = MDLMConfig(vocab_size=V + 1, seq_len=L, dim=32, depth=2, heads=2, cond_dim=16,
                     time_embed_dim=16, dropout=0.0, mask_id=V, offset_blocks=1)
    denoiser = MDLM(cfg)
    torch.nn.init.normal_(denoiser.out.weight, std=0.5)
    module = DominoModule(cfg.dim, denoiser.token_embedding, rank=4, gap_dim=4, max_gap=4, hidden=8)
    return Domino(denoiser, module)


def randomize(module: DominoModule) -> None:
    torch.nn.init.normal_(module.g[2].weight, std=0.3)
    torch.nn.init.normal_(module.g[2].bias, std=0.3)


def test_chain_parents():
    S = torch.tensor([[0, 1, 0, 1, 1, 0, 0, 1]], dtype=torch.bool)
    parent, rank = chain_parents(S)
    assert parent.tolist() == [[-1, -1, -1, 1, 3, -1, -1, 4]]
    assert rank.tolist() == [[-1, 0, -1, 1, 2, -1, -1, 3]]
    b, j, i = chain_edges(parent)
    assert list(zip(j.tolist(), i.tolist())) == [(3, 1), (4, 3), (7, 4)]


def test_untrained_module_is_the_product_kernel(model):
    x = torch.randint(0, V, (4, L))
    S = torch.rand(4, L) < 0.6
    z = torch.where(S, torch.full_like(x, V), x)
    logits, h = model.features(z, torch.full((4,), 0.5))
    lp_chain, lp_product = model.log_prob(logits, h, S, x)
    assert torch.allclose(lp_chain, lp_product, atol=1e-6)
    total_chain, _ = model.kstep_bound(x, 3, generator=torch.Generator().manual_seed(1))
    total_product, _ = model.kstep_bound(x, 3, product=True, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(total_chain, total_product, atol=1e-5)


def test_chain_is_normalized(model):
    """the chain's probabilities of every assignment of a reveal set sum to one."""
    randomize(model.module)
    S = torch.zeros(1, L, dtype=torch.bool)
    S[0, [1, 2, 5]] = True
    z = torch.where(S, torch.full((1, L), V), torch.randint(0, V, (1, L)))
    logits, h = model.features(z, torch.full((1,), 0.5))
    total = 0.0
    for tokens in itertools.product(range(V), repeat=3):
        x = z.clone()
        x[0, [1, 2, 5]] = torch.tensor(tokens)
        lp_chain, _ = model.log_prob(logits, h, S, x)
        total += float(lp_chain.sum().exp())
    assert total == pytest.approx(1.0, abs=1e-5)


def test_draw_follows_the_chain(model):
    """the empirical frequencies of a two-position draw match the chain's probabilities."""
    randomize(model.module)
    S = torch.zeros(1, L, dtype=torch.bool)
    S[0, [2, 3]] = True
    z = torch.where(S, torch.full((1, L), V), torch.randint(0, V, (1, L)))
    logits, h = model.features(z, torch.full((1,), 0.5))
    n = 20000
    torch.manual_seed(0)
    draws = model.draw(z.expand(n, L), logits.expand(n, L, -1), h.expand(n, L, -1), S.expand(n, L))
    counts = torch.zeros(V, V)
    counts.index_put_((draws[:, 2], draws[:, 3]), torch.ones(n), accumulate=True)
    probs = torch.zeros(V, V)
    for a, b in itertools.product(range(V), repeat=2):
        x = z.clone()
        x[0, 2], x[0, 3] = a, b
        probs[a, b] = model.log_prob(logits, h, S, x)[0].sum().exp()
    assert (counts / n - probs).abs().max() < 0.02


def test_coupling_keeps_candidates_without_potentials(model):
    x = torch.randint(0, V, (3, L))
    S = torch.rand(3, L) < 0.5
    z = torch.where(S, torch.full_like(x, V), x)
    logits, h = model.features(z, torch.full((3,), 0.5))
    cand = torch.randint(0, V, (3, L))
    out = model.couple(z, logits, h, S, cand, tau=torch.zeros(3), temp=1.0)
    assert torch.equal(out, torch.where(S, cand, z))


def test_loss_trains_only_the_module(model):
    x = torch.randint(0, V, (4, L))
    out = model.loss(x, K=4, eta=0.5, h_min=1 / 8)
    out["loss"].backward()
    assert all(p.grad is None for p in model.denoiser.parameters())
    assert model.module.g[2].weight.grad.abs().sum() > 0
    assert torch.isfinite(out["loss"])


def test_mdlm_loss_and_sampler(model):
    x = torch.randint(0, V, (4, L))
    assert torch.isfinite(model.denoiser.loss(x)["loss"])
    for order in ("random", "dilated", "confidence", "margin", "entropy", "drawn", "eb0.5"):
        z, step_of = sample(model, 4, 3, order)
        assert (z != V).all() and (step_of >= 0).all()


def test_arccos_counts():
    for steps in (2, 3, 4, 8):
        counts = arccos_counts(256, steps)
        assert len(counts) == steps and sum(counts) == 256 and min(counts) >= 1
