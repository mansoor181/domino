"""uncurated imagenet-256 samples from maskgit and from maskgit with domino, side by side.

    python -m scripts.examples_images --module <module.pt> --out samples.png

both samplers run at maskgit's released settings (guidance 3, softmax temperature 1) at 3 and 4
steps, with the same class and seed in each column. the classes are those of maskgit's demo, and
the first row shows a held-out image of each class as the vqgan reconstructs it.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

import data
from metrics import images
from models.domino import load_domino
from models.maskgit import load_maskgit
from paths import IMAGE_ROOT
from sampling.maskgit import sample

CLASSES = [1, 7, 282, 604, 724, 179, 751, 404]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maskgit", default=str(IMAGE_ROOT / "maskgit" / "MaskGIT_ImageNet_256.pth"))
    ap.add_argument("--module", required=True)
    ap.add_argument("--steps", default="3,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_domino(args.module, load_maskgit(args.maskgit, device), device)
    vq = images.build_vqgan(device)
    labels = torch.tensor(CLASSES, device=device)

    def pixels(tokens: torch.Tensor) -> list:
        return [img.permute(1, 2, 0).cpu().numpy() for batch in images.tokens_to_uint8(tokens, vq, device)
                for img in batch]

    val, _ = data.load("imagenet256", "val", labels=True)
    g = torch.Generator().manual_seed(args.seed)
    picks = [torch.nonzero(val.y == c, as_tuple=True)[0] for c in CLASSES]
    rows = [("held-out",
             pixels(torch.stack([val.x[p[torch.randint(len(p), (1,), generator=g)]][0] for p in picks])))]
    for steps in [int(k) for k in args.steps.split(",")]:
        for name, product in (("MaskGIT", True), ("+ Domino", False)):
            torch.manual_seed(args.seed)
            gen = torch.Generator(device=device).manual_seed(args.seed)
            z = sample(model, len(CLASSES), labels, steps, product=product, generator=gen)
            rows.append((f"{name}, {steps} steps", pixels(z.cpu())))

    fig, axes = plt.subplots(len(rows), len(CLASSES), figsize=(1.2 * len(CLASSES), 1.25 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        for c, ax in enumerate(axes[r]):
            ax.imshow(imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
    fig.savefig(args.out, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
