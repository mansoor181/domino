"""sample imagenet-256 with maskgit's sampler, with or without domino, and score fid and is.

    python -m eval.evaluate_images --module <module.pt> --out results/eval/imagenet256_domino.json

the defaults are the released sampler: guidance 3, softmax temperature 1, a confidence reveal with
gumbel temperature 4.5, and 10,000 samples of random classes. `--reveal random --uncond --cfg_w 0`
reveals random positions with the null class instead, as in our text experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from metrics import images
from models.domino import Domino, load_domino
from models.maskgit import load_maskgit
from paths import IMAGE_ROOT
from sampling.maskgit import sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maskgit", default=str(IMAGE_ROOT / "maskgit" / "MaskGIT_ImageNet_256.pth"))
    ap.add_argument("--module", default=None, help="a trained domino module (omit for maskgit)")
    ap.add_argument("--steps", default="2,3,4,5,6,8")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--cfg_w", type=float, default=3.0)
    ap.add_argument("--sm_temp", type=float, default=1.0)
    ap.add_argument("--r_temp", type=float, default=4.5)
    ap.add_argument("--reveal", default="confidence", choices=["confidence", "random"])
    ap.add_argument("--uncond", action="store_true", help="sample with the null class")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    denoiser = load_maskgit(args.maskgit, device)
    model = load_domino(args.module, denoiser, device) if args.module else Domino(denoiser, None)
    vq = images.build_vqgan(device)
    metric = images.build_metric(device)
    reference = images.load_reference("val_real", device)

    result = {"meta": vars(args), "by_steps": {}}
    for steps in [int(k) for k in args.steps.split(",")]:
        gen = torch.Generator(device=device).manual_seed(args.seed + steps)
        torch.manual_seed(args.seed + 1000 + steps)
        label_gen = torch.Generator().manual_seed(args.seed + steps)

        def batches():
            for start in range(0, args.samples, args.batch):
                n = min(args.batch, args.samples - start)
                labels = torch.randint(0, denoiser.n_classes, (n,), generator=label_gen).to(device)
                z = sample(model, n, None if args.uncond else labels, steps, args.cfg_w, args.sm_temp,
                           args.r_temp, args.reveal, args.module is None, gen)
                yield from images.tokens_to_uint8(z.cpu(), vq, device)

        result["by_steps"][steps] = images.score(metric, batches(), reference)
        print(steps, json.dumps(result["by_steps"][steps]), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
