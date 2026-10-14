"""masked diffusion language model (mdlm) on the dit of `models/dit.py`.

the loss is mdlm's substitution parameterization under the log-linear schedule: a position is
masked at time t with probability (1 - eps) t, the denoiser does not predict the mask token, and the
cross entropy at the masked positions is weighted by dsigma / expm1(sigma), which is 1 / t.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.dit import DiT, DiTConfig


@dataclass
class MDLMConfig(DiTConfig):
    mask_id: int = 0
    pad_id: int = -1   # rows padded with this id leave the padding out of the loss


class MDLM(DiT):
    def __init__(self, cfg: MDLMConfig) -> None:
        super().__init__(cfg)
        self.mask_id = cfg.mask_id
        self.pad_id = cfg.pad_id
        self.eps = cfg.eps
        self.seq_len = cfg.seq_len
        self.vocab_size = cfg.vocab_size
        self.dim = cfg.dim

    @property
    def token_embedding(self) -> torch.Tensor:
        return self.embed.weight

    def loss(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """negative evidence lower bound per token on a batch of clean rows x [B, L]."""
        B, L = x.shape
        eps = self.eps
        # antithetic times, as mdlm samples them
        u = torch.rand(B, device=x.device)
        u = (u / B + torch.arange(B, device=x.device) / B) % 1
        t = eps + (1.0 - eps) * u
        move = (1.0 - eps) * t
        sigma = -torch.log1p(-move)
        weight = ((1.0 - eps) / (1.0 - move)) / torch.expm1(sigma)

        z = torch.where(torch.rand_like(x, dtype=torch.float) < move[:, None],
                        torch.full_like(x, self.mask_id), x)
        logits, _ = self(z, t)
        logits[..., self.mask_id] = float("-inf")
        logp = F.log_softmax(logits, dim=-1).gather(-1, x[..., None]).squeeze(-1)

        masked = (z == self.mask_id).float()
        if self.pad_id >= 0:
            valid = (x != self.pad_id).float()
            masked = masked * valid
            norm = valid.sum().clamp_min(1.0)
        else:
            norm = torch.tensor(float(B * L), device=x.device)
        nll = -(weight * (logp * masked).sum(-1)).sum() / norm
        return {"loss": nll, "nll": nll.detach()}


def load_mdlm(path: str, device: torch.device) -> MDLM:
    """build an mdlm from one of our checkpoints, with its moving-average weights if it has them."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MDLM(MDLMConfig(**ckpt["cfg"]))
    state = ckpt["model"]
    if ckpt.get("ema") is not None:
        state = {**state, **ckpt["ema"]["shadow"]}
    model.load_state_dict(state)
    return model.to(device).eval()
