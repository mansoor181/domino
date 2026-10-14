"""absorbing sedd through the denoiser interface.

for the absorbing graph, the concrete score of a clean token at a masked position is the posterior
of that token times a factor shared by all tokens, so a softmax over the scores is the clean-token
posterior. we read the output of sedd's final layer and the state that enters it through hooks,
before sedd scales the scores by the noise level and zeroes the current token. sedd is conditioned
on sigma, which we set from the fraction of masked positions in each row, since a position is masked
with probability 1 - exp(-sigma) under its log-linear schedule. moving-average weights are applied.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from paths import SEDD_DIR

sys.path.append(str(SEDD_DIR))
from model.ema import ExponentialMovingAverage
from model.transformer import SEDD


class SEDDDenoiser(nn.Module):
    def __init__(self, net: SEDD) -> None:
        super().__init__()
        cfg = net.config
        self.net = net
        self.noise_eps = float(cfg.noise.get("eps", 1e-3))
        self.mask_id = int(cfg.tokens)
        self.vocab_size = self.mask_id + 1
        self.pad_id = -1
        self.eps = 0.0   # sigma follows the masked fraction, so the masking schedule needs no offset
        self.seq_len = int(cfg.model.length)
        self.dim = int(cfg.model.hidden_size)
        self.register_buffer("token_embedding", net.vocab_embed.embedding.detach().float().clone(),
                             persistent=False)
        self._io: dict[str, torch.Tensor] = {}
        net.output_layer.register_forward_hook(
            lambda module, inputs, output: self._io.update(hidden=inputs[0], logits=output))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frac = (z == self.mask_id).float().mean(1).clamp(max=1.0 - self.noise_eps)
        self.net(z, -torch.log1p(-frac))
        return self._io["logits"].float(), self._io["hidden"].float()


def load_sedd(run: str, device: torch.device) -> SEDDDenoiser:
    """a sedd training run directory, or a released model on the hub such as louaaron/sedd-small."""
    if (Path(run) / ".hydra").is_dir():
        cfg = OmegaConf.load(Path(run) / ".hydra" / "config.yaml")
        net = SEDD(cfg)
        ckpt = torch.load(Path(run) / "checkpoints-meta" / "checkpoint.pth", map_location="cpu")
        net.load_state_dict(ckpt["model"])
        ema = ExponentialMovingAverage(net.parameters(), decay=cfg.training.ema)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(net.parameters())
    else:
        net = SEDD.from_pretrained(run)   # the hub weights are already the moving average
    return SEDDDenoiser(net.to(device).eval().requires_grad_(False)).to(device)
