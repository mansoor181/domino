"""sample a text or molecule model across numbers of steps and score the samples.

    python -m eval.evaluate --denoiser mdlm --ckpt <mdlm.pt> --module <module.pt> \
        --dataset wikitext103 --out results/eval/wikitext103_mdlm_domino.json

without --module the denoiser samples with its own product kernel. text is scored by generative
perplexity under gpt-2 large and the per-position entropy, and molecules by validity, uniqueness
and novelty per draw, plus uniqueness, novelty and fcd on the first `--pooled` samples of the draws
at the largest number of steps.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import data
from metrics import mols, text
from models.denoisers import load_denoiser
from models.domino import Domino, load_domino
from sampling.sampler import sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoiser", required=True, choices=["mdlm", "sedd"])
    ap.add_argument("--ckpt", required=True, help="our checkpoint, a sedd run directory or a hub id")
    ap.add_argument("--module", default=None, help="a trained domino module")
    ap.add_argument("--dataset", required=True, choices=list(data.DATASETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", default="1,2,4,8,16,32")
    ap.add_argument("--orders", default="random", help="comma-separated unmasking rules")
    ap.add_argument("--samples", type=int, default=512, help="samples per draw")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pooled", type=int, default=10000, help="molecules: samples for fcd")
    ap.add_argument("--save_texts", action="store_true", help="keep the samples, for mauve")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    denoiser = load_denoiser(args.denoiser, args.ckpt, device)
    model = load_domino(args.module, denoiser, device) if args.module else Domino(denoiser, None)
    product = args.module is None
    domain = data.DATASETS[args.dataset]["domain"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nfes = [int(k) for k in args.nfe.split(",")]
    result = {"meta": vars(args), "gen": []}
    if domain == "text":
        scorer = text.GPT2Scorer(device)
        result["reference_entropy"] = text.reference_entropy(args.dataset, args.samples)

    for order in args.orders.split(","):
        pooled = []
        for nfe in nfes:
            for draw in range(args.draws):
                torch.manual_seed(args.seed * 1000 + draw * 100 + nfe)
                rows, steps = [], []
                t0 = time.time()
                for start in range(0, args.samples, args.batch):
                    x, step_of = sample(model, nfe, min(args.batch, args.samples - start), order,
                                        product, args.temperature)
                    rows.append(x.cpu())
                    steps.append(step_of.cpu())
                torch.cuda.synchronize()
                samples = torch.cat(rows).numpy()
                rec = {"order": order, "nfe": nfe, "draw": draw,
                       "sec_per_seq": (time.time() - t0) / len(samples),
                       "steps": float((torch.cat(steps).max(1).values + 1).float().mean()),
                       "entropy": text.position_entropy(samples)}
                decoded = data.decode(args.dataset, samples)
                if domain == "text":
                    texts = [t for t in decoded if t.strip()]
                    if args.save_texts:
                        texts_dir = out_path.with_suffix("")
                        texts_dir.mkdir(exist_ok=True)
                        (texts_dir / f"{order}_nfe{nfe}_draw{draw}.json").write_text(json.dumps(texts))
                    nll, count = scorer.nll(texts, max_length=1024 if samples.shape[1] >= 512 else 512)
                    rec["gen_ppl"] = math.exp(nll / count)
                else:
                    rec.update({k: v for k, v in mols.score(decoded, args.dataset).items() if k != "valid"})
                    if nfe == nfes[-1]:
                        pooled += decoded
                result["gen"].append(rec)
                print(json.dumps(rec), flush=True)
                out_path.write_text(json.dumps(result, indent=1))
        if domain == "mol":
            scored = mols.score(pooled[:args.pooled], args.dataset)
            result.setdefault("pooled", {})[order] = {
                "nfe": nfes[-1], "samples": len(pooled[:args.pooled]),
                "uniqueness": scored["uniqueness"], "novelty": scored["novelty"],
                "fcd": mols.fcd(scored["valid"], args.dataset)}
            print(json.dumps(result["pooled"][order]), flush=True)
            out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
