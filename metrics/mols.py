"""molecule metrics, following pairflow's evaluation.

a sample is valid when it is not empty and rdkit parses it. uniqueness and novelty count distinct
canonical smiles among the valid samples, novelty against the canonical smiles of the training set,
and the frechet chemnet distance (fcd) compares the valid samples with 10,000 training molecules.
"""
from __future__ import annotations

from functools import lru_cache

import datasets as hfds
import numpy as np
from fcd_torch import FCD
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REFERENCE = {"qm9": "yairschiff/qm9", "zinc250k": "yairschiff/zinc250k"}


@lru_cache(maxsize=2)
def reference_smiles(name: str) -> frozenset[str]:
    """the canonical smiles of the training split."""
    return frozenset(hfds.load_dataset(REFERENCE[name], split="train")["canonical_smiles"])


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except RuntimeError:   # a few rings parse but cannot be ranked
        return None


def score(smiles: list[str], name: str) -> dict:
    valid = [s for s in smiles if s and Chem.MolFromSmiles(s) is not None]
    canon = {c for c in map(canonical, valid) if c is not None}
    n_valid = max(len(valid), 1)
    return {"validity": len(valid) / len(smiles), "uniqueness": len(canon) / n_valid,
            "novelty": len(canon - reference_smiles(name)) / n_valid, "valid": valid}


def fcd(valid: list[str], name: str, n_ref: int = 10000, device: str = "cuda") -> float:
    ref = sorted(reference_smiles(name))
    ref = [ref[i] for i in np.random.default_rng(0).choice(len(ref), n_ref, replace=False)]
    # fcd_torch parses every string again, and a few canonical strings fail to kekulize
    gen = [c for c in map(canonical, valid) if c and Chem.MolFromSmiles(c) is not None]
    return float(FCD(device=device, n_jobs=1)(ref=ref, gen=gen))
