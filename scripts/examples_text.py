"""paired text samples from the denoiser alone and with domino, on one shared reveal schedule.

    python -m scripts.examples_text --ckpt <mdlm.pt> --module <module.pt> --dataset wikitext103 \
        --out examples_wikitext103.json

each pair reveals the same positions at each step, drawn in advance as a random permutation cut
into K groups, and both kernels start from one random seed, so a pair differs only in how the
tokens of a step are drawn. every token is annotated with its surprise under gpt-2 large, the
largest negative log-probability of the gpt-2 tokens that overlap it, and the threshold for a
surprising token is the 90th percentile of that surprise on held-out text.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import data
from models.domino import load_domino
from models.mdlm import load_mdlm
from sampling.sampler import sample_on_schedule


def pieces(name: str, ids: list[int], tok) -> list[str]:
    """the surface string of each token, with the spacing it adds to the decoded text."""
    if data.DATASETS[name]["tokenizer"] == "gpt2":
        return [tok.decode([i]) for i in ids]
    out = []
    for j, i in enumerate(ids):
        w = tok.convert_ids_to_tokens(i)
        if w == "[PAD]":
            out.append("")
        elif w in ("[CLS]", "[SEP]"):
            out.append(" ¶")
        else:
            out.append(w[2:] if w.startswith("##") else (" " + w if j else w))
    return out


@torch.no_grad()
def surprise(gpt, gtok, text: str, device: torch.device) -> tuple[np.ndarray, list[float]]:
    """surprise per character of text, and the per-token negative log-likelihoods."""
    enc = gtok(text, return_offsets_mapping=True, truncation=True, max_length=1024)
    ids = torch.tensor([enc["input_ids"]], device=device)
    nll = [0.0] + (-F.log_softmax(gpt(ids).logits[0, :-1].float(), -1)
                   .gather(-1, ids[0, 1:, None]).squeeze(-1)).tolist()
    per_char = np.zeros(len(text))
    for (s, e), v in zip(enc["offset_mapping"], nll):
        per_char[s:e] = np.maximum(per_char[s:e], v)
    return per_char, nll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_domino(args.module, load_mdlm(args.ckpt, device), device)
    L = model.denoiser.seq_len
    g = torch.Generator().manual_seed(args.seed)
    step_of = torch.stack([torch.randperm(L, generator=g) * args.K // L for _ in range(args.n)]).to(device)
    samples = {}
    for name, product in (("MDLM", True), ("Domino", False)):
        torch.manual_seed(args.seed)
        samples[name] = sample_on_schedule(model, step_of, args.K, product).tolist()

    gtok = AutoTokenizer.from_pretrained("gpt2-large")
    gpt = AutoModelForCausalLM.from_pretrained("gpt2-large").to(device).eval()
    tok = data.tokenizer(args.dataset)
    held_out = []
    for row in data.first_rows(args.dataset, "val", 32):
        held_out += surprise(gpt, gtok, "".join(pieces(args.dataset, row.tolist(), tok)), device)[1][1:]
    pairs = []
    for b in range(args.n):
        pair = {"steps": step_of[b].tolist()}
        for name, rows in samples.items():
            parts = pieces(args.dataset, rows[b], tok)
            per_char, nll = surprise(gpt, gtok, "".join(parts), device)
            tokens, pos = [], 0
            for p in parts:
                value = float(per_char[pos:pos + len(p)].max()) if p.strip() else 0.0
                tokens.append({"text": p, "surprise": value})
                pos += len(p)
            pair[name] = {"tokens": tokens, "gen_ppl": math.exp(float(np.mean(nll[1:])))}
        pairs.append(pair)
    Path(args.out).write_text(json.dumps({"dataset": args.dataset, "K": args.K,
                                          "threshold": float(np.percentile(held_out, 90)),
                                          "pairs": pairs}, indent=1))


if __name__ == "__main__":
    main()
