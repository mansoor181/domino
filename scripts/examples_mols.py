"""paired molecule samples from the denoiser alone and with domino, and a figure of them.

    python -m scripts.examples_mols --ckpt <mdlm.pt> --module <module.pt> --dataset qm9 \
        --out examples_qm9

each pair reveals every position at a step drawn uniformly and independently, which is the
evaluator's random rule, and both kernels start from one random seed, so a pair differs only in
how the tokens of a step are drawn. writes the smiles of every pair (`.json`) and a figure with the
first valid molecules of each kernel above a size cutoff (`.png`), since tiny fragments such as a
single atom also count as valid.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

import data
from models.domino import load_domino
from models.mdlm import load_mdlm
from sampling.sampler import sample_on_schedule

RDLogger.DisableLog("rdApp.*")
MIN_HEAVY_ATOMS = {"qm9": 5, "zinc250k": 10}


def substantial(smiles: str, name: str) -> bool:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return mol is not None and mol.GetNumHeavyAtoms() >= MIN_HEAVY_ATOMS[name]


def render(smiles: str) -> Image.Image:
    canvas = Draw.MolDraw2DCairo(300, 240)
    opts = canvas.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.padding = 0.08
    opts.minFontSize = 16
    canvas.DrawMolecule(Chem.MolFromSmiles(smiles))
    canvas.FinishDrawing()
    return Image.open(io.BytesIO(canvas.GetDrawingText()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True, choices=["qm9", "zinc250k"])
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_domino(args.module, load_mdlm(args.ckpt, device), device)
    g = torch.Generator().manual_seed(args.seed)
    step_of = torch.randint(0, args.K, (args.n, model.denoiser.seq_len), generator=g).to(device)
    smiles = {}
    for name, product in (("MDLM", True), ("MDLM + Domino", False)):
        torch.manual_seed(args.seed)
        smiles[name] = data.decode(args.dataset, sample_on_schedule(model, step_of, args.K, product).cpu())
    record = {"dataset": args.dataset, "K": args.K, **smiles}
    Path(f"{args.out}.json").write_text(json.dumps(record, indent=1))

    fig, axes = plt.subplots(2, args.show, figsize=(1.1 * args.show, 2.4))
    for row, (name, mols) in enumerate(smiles.items()):
        shown = [s for s in mols if substantial(s, args.dataset)][:args.show]
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col < len(shown):
                ax.imshow(render(shown[col]))
        axes[row][0].set_title(name, loc="left", fontsize=8)
    fig.savefig(f"{args.out}.png", dpi=300, bbox_inches="tight", facecolor="white")
    # an alpha channel renders as a grey box once the figure is placed in a pdf
    Image.open(f"{args.out}.png").convert("RGB").save(f"{args.out}.png")


if __name__ == "__main__":
    main()
