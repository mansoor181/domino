"""sampling time per sequence, for the cost table.

    python -m scripts.time_sampler --denoiser mdlm --ckpt <mdlm.pt> [--module <module.pt>]

the median of five timed runs after one warm-up run, at batch size 32, on an otherwise idle gpu.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from models.denoisers import load_denoiser
from models.domino import Domino, load_domino
from sampling.sampler import sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoiser", required=True, choices=["mdlm", "sedd"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--module", default=None)
    ap.add_argument("--nfe", default="4,32")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda")
    denoiser = load_denoiser(args.denoiser, args.ckpt, device)
    model = load_domino(args.module, denoiser, device) if args.module else Domino(denoiser, None)
    result = {}
    for nfe in [int(k) for k in args.nfe.split(",")]:
        times = []
        for _ in range(args.reps + 1):
            torch.cuda.synchronize()
            t0 = time.time()
            sample(model, nfe, args.batch, product=args.module is None)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        result[f"ms_per_seq_nfe{nfe}"] = 1000 * float(np.median(times[1:])) / args.batch
    print(json.dumps(result))


if __name__ == "__main__":
    main()
