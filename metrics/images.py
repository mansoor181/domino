"""image metrics for imagenet-256 tokens.

token grids are decoded to pixels with maskgit's vqgan and scored with the inception metrics of
the maskgit-pytorch repository (fid, inception score, precision and recall) against cached
inception features of the 50,000 validation images, which `scripts/build_image_refs.py` writes.
"""
from __future__ import annotations

import sys
from typing import Iterator

import numpy as np
import torch
from omegaconf import OmegaConf

from paths import IMAGE_ROOT, MASKGIT_DIR

sys.path.append(str(MASKGIT_DIR))
from Metrics.inception_metrics import MultiInceptionMetrics
from Network.Taming.models.vqgan import VQModel

VQGAN_DIR = IMAGE_ROOT / "maskgit"
REF_DIR = IMAGE_ROOT / "ref"


def build_vqgan(device: torch.device) -> VQModel:
    cfg = OmegaConf.load(VQGAN_DIR / "model.yaml")
    vq = VQModel(**cfg.model.params)
    # the checkpoint also holds the discriminator and the perceptual loss, which VQModel lacks
    vq.load_state_dict(torch.load(VQGAN_DIR / "last.ckpt", map_location="cpu")["state_dict"],
                       strict=False)
    return vq.eval().to(device).requires_grad_(False)


@torch.no_grad()
def tokens_to_uint8(tokens: np.ndarray | torch.Tensor, vq: VQModel, device: torch.device,
                    chunk: int = 64, grid: int = 16) -> Iterator[torch.Tensor]:
    """uint8 images [chunk, 3, 256, 256] from token rows [N, 256]."""
    t = torch.as_tensor(np.asarray(tokens)).view(-1, grid, grid)
    for i in range(0, t.shape[0], chunk):
        img = vq.decode_code(t[i:i + chunk].to(device).long())
        img = img.float().clamp(-1.0, 1.0) * 0.5 + 0.5
        yield (img * 255).round().clamp(0, 255).to(torch.uint8)


def build_metric(device: torch.device) -> MultiInceptionMetrics:
    return MultiInceptionMetrics(reset_real_features=False, compute_unconditional_metrics=True,
                                 compute_conditional_metrics=False,
                                 compute_conditional_metrics_per_class=False,
                                 manifold_k=3).to(device)


@torch.no_grad()
def inception_features(metric: MultiInceptionMetrics, images: torch.Tensor) -> torch.Tensor:
    feats, _ = metric.inception(images)
    return feats.view(feats.size(0), -1).double()


def load_reference(name: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.load(REF_DIR / f"{name}.npz")["features"]).to(device).double()


def score(metric: MultiInceptionMetrics, images: Iterator[torch.Tensor],
          reference: torch.Tensor) -> dict[str, float]:
    """score a stream of uint8 batches against cached reference features."""
    metric.reset()
    metric.real_features = [reference]
    for batch in images:
        metric.update(batch, image_type="unconditional")
    return {k.replace("_unconditional", ""): float(v) for k, v in metric.compute().items()}
