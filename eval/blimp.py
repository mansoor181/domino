"""subject-verb agreement on blimp, scored in closed form from one denoiser call.

each blimp item is a minimal pair that differs in one word, either the verb (subsets ending in _1
and the two distractor subsets) or the subject noun (subsets ending in _2). we embed the
grammatical sentence in a held-out window with every other token visible, mask the subject noun and
the finite verb, and measure how much of the kernel's joint mass on the two blimp forms falls on the
agreeing pair when both positions are revealed in one step,

    verb varies:  A = P(n, v_good) / (P(n, v_good) + P(n, v_bad)),
    noun varies:  A = P(n_good, v) / (P(n_good, v) + P(n_bad, v)).

the product kernel factorizes P(n, v), so A reduces to the marginal preference at the varying
position. domino's chain factorizes it as P(n) P(v | n). the two-call reference reveals the noun
first and queries the denoiser again for the verb. all three are exact, so no sampling is involved.

    python -m eval.blimp --ckpt <mdlm.pt> --module <module.pt> --dataset wikitext103 \
        --blimp_dir blimp/data --out blimp.json

the subject noun and verb are located with nltk's part-of-speech tagger.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nltk import pos_tag

import data
from models.domino import Domino, load_domino
from models.mdlm import load_mdlm

SUBSETS = ["regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2",
           "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
           "distractor_agreement_relational_noun", "distractor_agreement_relative_clause"]
VERBS = {"is", "was", "has", "does", "isn't", "wasn't", "hasn't", "doesn't",
         "are", "were", "have", "do", "aren't", "weren't", "haven't", "don't"}
MODES = ("product", "one_step", "two_step")


def locate(item: dict) -> tuple[list[str], list[str], int, int] | None:
    """the words of both sentences and the positions of the subject noun and the verb."""
    good = item["sentence_good"].strip().rstrip(".").split(" ")
    bad = item["sentence_bad"].strip().rstrip(".").split(" ")
    if len(good) != len(bad):
        return None
    diff = [i for i, (g, b) in enumerate(zip(good, bad)) if g != b]
    if len(diff) != 1:
        return None
    d = diff[0]
    tags = [tag for _, tag in pos_tag(good)]
    if item["UID"].endswith("_2"):
        noun = d
        verb = next((k for k in range(d + 1, len(good))
                     if tags[k] in ("VBZ", "VBP", "VBD") or good[k].lower() in VERBS), None)
    else:
        verb = d
        noun = next((k for k in range(d) if tags[k] in ("NN", "NNS", "NNP", "NNPS")), None)
    if noun is None or verb is None or noun >= verb:
        return None
    return good, bad, noun, verb


@torch.no_grad()
def joint_log_probs(model: Domino, z: torch.Tensor, alpha: float, pn: int, pv: int,
                    nouns: list[int], verbs: list[int]) -> dict[str, dict]:
    """log P(n, v) for the candidate tokens at the noun position pn and the verb position pv."""
    device = model.denoiser.token_embedding.device
    Z = z[None].to(device)
    a = torch.full((1,), alpha, device=device)
    logits, h = model.features(Z, 1.0 - a)
    lp_n = F.log_softmax(logits[0, pn], -1)
    lp_v = F.log_softmax(logits[0, pv], -1)
    out = {m: {} for m in MODES}
    u = model.module.pair_factors(h)
    E = model.module.token_table()
    for n in nouns:
        cl = model.module.transition_logits(logits[0:1, pv], u[0:1, pn], u[0:1, pv],
                                            torch.tensor([pv - pn], device=device),
                                            torch.tensor([n], device=device), E)
        lp_c = F.log_softmax(cl[0], -1)
        Z2 = Z.clone()
        Z2[0, pn] = n
        lp_v2 = F.log_softmax(model.features(Z2, 1.0 - (a + 1.0 / Z.shape[1]))[0][0, pv], -1)
        for v in verbs:
            out["product"][(n, v)] = float(lp_n[n] + lp_v[v])
            out["one_step"][(n, v)] = float(lp_n[n] + lp_c[v])
            out["two_step"][(n, v)] = float(lp_n[n] + lp_v2[v])
    return out


def agreement(lp: dict, good: tuple[int, int], bad: tuple[int, int]) -> float:
    g, b = lp[good], lp[bad]
    m = max(g, b)
    return math.exp(g - m) / (math.exp(g - m) + math.exp(b - m))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--dataset", default="wikitext103")
    ap.add_argument("--blimp_dir", required=True, help="the data directory of the blimp repository")
    ap.add_argument("--max_items", type=int, default=1000, help="per subset")
    ap.add_argument("--offset", type=int, default=32, help="where the sentence starts in its window")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_domino(args.module, load_mdlm(args.ckpt, device), device)
    L, mask_id = model.denoiser.seq_len, model.mask_id
    tok = data.tokenizer(args.dataset)
    gpt2_spaces = data.DATASETS[args.dataset]["tokenizer"] == "gpt2"
    windows = torch.from_numpy(data.first_rows(args.dataset, "val", 1024))

    def encode(words: list[str]) -> list[list[int]]:
        return [tok.encode((" " + w) if gpt2_spaces and i else w, add_special_tokens=False)
                for i, w in enumerate(words)]

    scores = {m: [] for m in MODES}
    per_subset = {}
    for subset in SUBSETS:
        items = [json.loads(line) for line in (Path(args.blimp_dir) / f"{subset}.jsonl").open()]
        values = {m: [] for m in MODES}
        used = 0
        for item in items:
            if used >= args.max_items:
                break
            found = locate(item)
            if found is None:
                continue
            good, bad, ni, vi = found
            enc_good, enc_bad = encode(good), encode(bad)
            d = ni if subset.endswith("_2") else vi
            if len(enc_good[ni]) != 1 or len(enc_good[vi]) != 1 or len(enc_bad[d]) != 1:
                continue
            ids = [t for e in enc_good for t in e]
            off = args.offset if args.offset + len(ids) + 1 <= L else 0
            z = windows[used % len(windows)].clone()
            z[off:off + len(ids)] = torch.tensor(ids)
            pos = np.cumsum([0] + [len(e) for e in enc_good])
            pn, pv = int(pos[ni]) + off, int(pos[vi]) + off
            z[pn] = mask_id
            z[pv] = mask_id
            n_good, v_good = enc_good[ni][0], enc_good[vi][0]
            if subset.endswith("_2"):
                nouns, verbs = [n_good, enc_bad[ni][0]], [v_good]
                good_pair, bad_pair = (n_good, v_good), (enc_bad[ni][0], v_good)
            else:
                nouns, verbs = [n_good], [v_good, enc_bad[vi][0]]
                good_pair, bad_pair = (n_good, v_good), (n_good, enc_bad[vi][0])
            lp = joint_log_probs(model, z, (L - 2) / L, pn, pv, nouns, verbs)
            for m in MODES:
                values[m].append(agreement(lp[m], good_pair, bad_pair))
            used += 1
        per_subset[subset] = {"items": used, **{m: float(np.mean(values[m])) for m in MODES}}
        for m in MODES:
            scores[m] += values[m]
        print(subset, json.dumps(per_subset[subset]), flush=True)

    total = {m: {"items": len(scores[m]), "agreement": float(np.mean(scores[m])),
                 "accuracy": float(np.mean(np.array(scores[m]) > 0.5))} for m in MODES}
    print("total", json.dumps(total), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"total": total, "per_subset": per_subset}, indent=1))


if __name__ == "__main__":
    main()
