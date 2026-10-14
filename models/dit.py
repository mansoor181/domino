"""the diffusion transformer (dit) of mdlm, reimplemented.

each block follows mdlm's ddit block: adaln modulation of a weight-only layer norm, rotary
self-attention and a gelu mlp, with dropout inside the gated residual. a final adaln layer feeds an
untied output projection. the network is conditioned on the noise level sigma of the log-linear
schedule, sigma(t) = -log(1 - (1 - eps) t), where t = 1 is the fully masked state.

two options let one class load every checkpoint we use.
  mdlm_exact     reproduce the numerics of mdlm's released code: one silu on the conditioning,
                 mdlm's frequency grid for the time embedding, and float32 for the embedding, the
                 conditioning, the normalized activations and the rotary angles. our own models
                 were trained with this switched off, which applies silu twice, divides the
                 frequency exponent by half - 1 instead of half, and computes those parts inside
                 the bfloat16 region.
  offset_blocks  the last `offset_blocks` blocks and the output layer add a learned vector to the
                 conditioning. our text and molecule models use 4, and 0 is mdlm's layout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    vocab_size: int
    seq_len: int
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_mult: int = 4
    cond_dim: int = 128
    time_embed_dim: int = 256
    dropout: float = 0.1
    eps: float = 1e-3
    time_conditioning: bool = True
    mdlm_exact: bool = False
    offset_blocks: int = 0


def timestep_embedding(sigma: torch.Tensor, dim: int, mdlm_grid: bool) -> torch.Tensor:
    half = dim // 2
    denom = half if mdlm_grid else half - 1
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=sigma.device) / denom)
    args = sigma.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LayerNorm(nn.Module):
    """mdlm's layer norm: computed in float32 with a weight and no bias."""

    def __init__(self, dim: int, keep_fp32: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = (dim,)
        self.keep_fp32 = keep_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            y = F.layer_norm(x.float(), self.dim)
        out = y * self.weight
        return out if self.keep_fp32 else out.to(x.dtype)


class Rotary(nn.Module):
    def __init__(self, dim: int, fp32: bool, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.fp32 = fp32

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device).float()
        if self.fp32:
            with torch.cuda.amp.autocast(enabled=False):
                freqs = torch.einsum("i,j->ij", pos, self.inv_freq.float())
        else:
            freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, L, D], cos and sin: [L, D/2]."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, rotary_fp32: bool) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.rot = Rotary(self.head_dim, rotary_fp32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self.rot(L, x.device)
        q, k = apply_rotary(qkv[0], cos, sin), apply_rotary(qkv[1], cos, sin)
        out = F.scaled_dot_product_attention(q, k, qkv[2], is_causal=False)
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mult * dim)
        self.fc2 = nn.Linear(mult * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class AdaLN(nn.Module):
    """per-sample scales, shifts and gates from the conditioning vector, zero at initialization."""

    def __init__(self, dim: int, cond_dim: int, n: int) -> None:
        super().__init__()
        self.map = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n * dim))
        nn.init.zeros_(self.map[1].weight)
        nn.init.zeros_(self.map[1].bias)
        self.n = n

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.map(c).chunk(self.n, dim=-1)


class Block(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.norm1 = LayerNorm(cfg.dim, keep_fp32=cfg.mdlm_exact)
        self.norm2 = LayerNorm(cfg.dim, keep_fp32=cfg.mdlm_exact)
        self.attn = Attention(cfg.dim, cfg.heads, rotary_fp32=cfg.mdlm_exact)
        self.mlp = MLP(cfg.dim, cfg.mlp_mult)
        self.ada = AdaLN(cfg.dim, cfg.cond_dim, 6)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale1, shift1, gate1, scale2, shift2, gate2 = self.ada(c)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None] * self.drop1(self.attn(h))
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.drop2(self.mlp(h))


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        nn.init.kaiming_uniform_(self.embed.weight, a=math.sqrt(5))
        self.sigma_map = nn.Sequential(nn.Linear(cfg.time_embed_dim, cfg.cond_dim), nn.SiLU(),
                                       nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.final_norm = LayerNorm(cfg.dim, keep_fp32=cfg.mdlm_exact)
        self.final_ada = AdaLN(cfg.dim, cfg.cond_dim, 2)
        self.out = nn.Linear(cfg.dim, cfg.vocab_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if cfg.offset_blocks:
            # the offset only reaches the output through the adaln maps of these blocks, so they
            # start small instead of at zero
            self.cond_offset = nn.Parameter(0.02 * torch.randn(cfg.cond_dim))
            for block in self.blocks[cfg.depth - cfg.offset_blocks:]:
                nn.init.normal_(block.ada.map[1].weight, std=0.02)

    def condition(self, t: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.time_conditioning:
            sigma = -torch.log1p(-(1.0 - cfg.eps) * t.clamp(min=cfg.eps, max=1.0))
        else:
            sigma = torch.zeros_like(t)
        c = self.sigma_map(timestep_embedding(sigma, cfg.time_embed_dim, mdlm_grid=cfg.mdlm_exact))
        return c if cfg.mdlm_exact else F.silu(c)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: [B, L] token ids, t: [B] diffusion times. returns float32 logits [B, L, V] and the
        final hidden states [B, L, D] that the output projection reads."""
        cfg = self.cfg
        if cfg.mdlm_exact:
            with torch.cuda.amp.autocast(enabled=False):
                x, c = self.embed(z), self.condition(t.float())
        with torch.cuda.amp.autocast(enabled=z.is_cuda, dtype=torch.bfloat16):
            if not cfg.mdlm_exact:
                x, c = self.embed(z), self.condition(t)
            c_late = c + self.cond_offset if cfg.offset_blocks else c
            first_late = cfg.depth - cfg.offset_blocks
            for i, block in enumerate(self.blocks):
                x = block(x, c if i < first_late else c_late)
            scale, shift = self.final_ada(c_late)
            h = self.final_norm(x) * (1 + scale[:, None]) + shift[:, None]
            logits = self.out(h)
        return logits.float(), h.float()
