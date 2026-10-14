"""the released maskgit imagenet-256 model through the denoiser interface.

we wrap the maskgit-pytorch reimplementation rather than convert it, since its learned absolute
positions and per-position output bias have no counterpart in our dit. the model takes the 16 x 16
lattice, appends a class token after the image positions (the hidden state of that row is dropped),
and has no noise level, so the time argument is ignored. setting `labels` reads the model with
those classes, and leaving it at none reads it with the null class.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from paths import MASKGIT_DIR

sys.path.append(str(MASKGIT_DIR))
from Network.transformer import Attention, MaskTransformer


def _attention(self, x: torch.Tensor):
    # the attention weights go unused, and asking for them disables the fused kernel
    return self.mha(x, x, x, need_weights=False)


Attention.forward = _attention


class MaskGITDenoiser(nn.Module):
    def __init__(self, net: MaskTransformer, grid: int, codebook: int, n_classes: int) -> None:
        super().__init__()
        self.net = net
        self.grid = grid
        self.n_classes = n_classes
        self.mask_id = codebook
        self.vocab_size = codebook + 1
        self.pad_id = -1
        self.eps = 0.0
        self.seq_len = grid * grid
        self.dim = net.tok_emb.weight.shape[1]
        self.register_buffer("token_embedding",
                             net.tok_emb.weight[:self.vocab_size].detach().clone(), persistent=False)
        self.labels: torch.Tensor | None = None
        self._hidden: dict[str, torch.Tensor] = {}
        net.last_layer.register_forward_hook(
            lambda module, inputs, output: self._hidden.update(h=output))

    def forward(self, z: torch.Tensor, t: torch.Tensor | None = None):
        B, L = z.shape
        if self.labels is None:
            # maskgit shifts labels past the codebook, so class index n_classes is its null class
            y = torch.full((B,), self.n_classes, device=z.device, dtype=torch.long)
        else:
            y = self.labels.to(z.device).expand(B)
        drop = torch.zeros(B, dtype=torch.bool, device=z.device)
        logits = self.net(z.view(B, self.grid, self.grid), y=y, drop_label=drop)
        return logits.float()[:, :L, :self.vocab_size], self._hidden["h"][:, :L].float()


def load_maskgit(path: str, device: torch.device, grid: int = 16, codebook: int = 1024,
                 n_classes: int = 1000) -> MaskGITDenoiser:
    """the released checkpoint. it does not record the number of attention heads, which leaves the
    shapes of its weights unchanged, so we use the value of maskgit-pytorch, 16."""
    net = MaskTransformer(img_size=256, hidden_dim=768, codebook_size=codebook, depth=24, heads=16,
                          mlp_dim=3072, dropout=0.0, nclass=n_classes)
    state = torch.load(path, map_location="cpu")
    state = state.get("model_state_dict", state)
    net.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return MaskGITDenoiser(net.to(device).eval().requires_grad_(False), grid, codebook,
                           n_classes).to(device)
