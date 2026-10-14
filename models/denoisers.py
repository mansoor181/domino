"""load any of the denoisers domino is integrated with."""
from __future__ import annotations

import torch

from models.maskgit import load_maskgit
from models.mdlm import load_mdlm
from models.sedd import load_sedd

LOADERS = {"mdlm": load_mdlm, "sedd": load_sedd, "maskgit": load_maskgit}


def load_denoiser(kind: str, path: str, device: torch.device) -> torch.nn.Module:
    """kind is mdlm (our checkpoints, including sdtt students), sedd or maskgit."""
    return LOADERS[kind](path, device)
