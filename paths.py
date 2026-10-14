"""filesystem roots, read from the environment or from a `.env` file next to this module.

  DOMINO_TEXT_ROOT    text caches (wikitext103, lm1b, lm1b_gpt2, openwebtext)
  DOMINO_MOL_ROOT     qm9 and zinc-250k caches built by pairflow
  DOMINO_IMAGE_ROOT   imagenet-256 tokens, the maskgit checkpoints and the fid references
  DOMINO_EXTERNAL     checkouts of the base-model repositories (mdlm, sedd, maskgit)
  DOMINO_RESULTS      checkpoints and evaluation outputs

config files may refer to these roots as `${DOMINO_RESULTS}` and so on.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent

env_file = REPO_DIR / ".env"
if env_file.is_file():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

for key, default in (("DOMINO_TEXT_ROOT", "storage/text"), ("DOMINO_MOL_ROOT", "storage/mols"),
                     ("DOMINO_IMAGE_ROOT", "storage/images"), ("DOMINO_EXTERNAL", "external"),
                     ("DOMINO_RESULTS", "results")):
    os.environ.setdefault(key, str(REPO_DIR / default))

TEXT_ROOT = Path(os.environ["DOMINO_TEXT_ROOT"])
MOL_ROOT = Path(os.environ["DOMINO_MOL_ROOT"])
IMAGE_ROOT = Path(os.environ["DOMINO_IMAGE_ROOT"])
EXTERNAL = Path(os.environ["DOMINO_EXTERNAL"])
RESULTS = Path(os.environ["DOMINO_RESULTS"])

MDLM_DIR = EXTERNAL / "mdlm"
SEDD_DIR = EXTERNAL / "sedd"
MASKGIT_DIR = EXTERNAL / "maskgit"
