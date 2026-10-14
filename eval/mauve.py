"""mauve, per-sample entropy and distinct bigrams of saved text samples.

    python -m eval.mauve --dataset wikitext103 --out mauve.json \
        --set "mdlm 4=results/eval/wt103_mdlm/random_nfe4_draw*.json" --set ...

each set is NAME=GLOB over the files that `eval.evaluate --save_texts` writes. mauve runs at its
default settings (gpt-2 large features, scaling factor 5) against held-out text of the same corpus,
decoded like the samples. the entropy is that of each sample's gpt-2 token histogram, averaged over
samples, and distinct-2 is the share of distinct bigrams among all bigrams of the pooled samples. a
last row scores a second draw of held-out text, which is the ceiling at this sample count.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import mauve
import mauve.utils as mu
import numpy as np

import data


def references(name: str, n: int, seed: int) -> list[str]:
    """n held-out windows decoded like the samples.

    a corpus padded to one sentence per row is packed into full-length windows first, so that mauve
    compares content rather than length.
    """
    ds, meta = data.load(name, "val")
    rows = data.first_rows(name, "val", len(ds) // 64 * 64)
    if meta["pad_id"] < 0:
        idx = np.random.default_rng(seed).permutation(len(rows))[:n]
        return [t for t in data.decode(name, rows[idx]) if t.strip()]
    tok = data.tokenizer(name)
    specials = set(tok.all_special_ids)
    L = meta["seq_len"]
    stream: list[int] = []
    for start in range(0, len(rows), 256):
        stream += [int(c) for row in rows[start:start + 256] for c in row if int(c) not in specials]
        if len(stream) >= 4 * n * L:
            break
    windows = [stream[i:i + L] for i in range(0, len(stream) - L, L)]
    idx = np.random.default_rng(seed).permutation(len(windows))[:n]
    return [tok.decode(windows[i], skip_special_tokens=True).strip() for i in idx]


def entropy(ids: list[int]) -> float:
    p = np.array(list(Counter(ids).values()), dtype=np.float64)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def distinct(ids: list[list[int]], n: int = 2) -> float:
    grams = [tuple(t[i:i + n]) for t in ids for i in range(len(t) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--set", action="append", required=True, help="NAME=GLOB, repeatable")
    ap.add_argument("--n_ref", type=int, default=1536)
    ap.add_argument("--max_len", type=int, default=256, help="tokens per text that mauve reads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = mu.get_tokenizer("gpt2-large")
    model = mu.get_model("gpt2-large", tok, 0)

    def features(texts: list[str], name: str) -> np.ndarray:
        toks = [tok.encode(t, return_tensors="pt", truncation=True, max_length=args.max_len)
                for t in texts]
        return mu.featurize_tokens_from_model(model, toks, 32, name).cpu().numpy()

    def row(name: str, texts: list[str]) -> dict:
        score = mauve.compute_mauve(p_features=ref_features, q_features=features(texts, name),
                                    verbose=False).mauve
        ids = [tok.encode(t) for t in texts]
        out = {"set": name, "n": len(texts), "mauve": float(score),
               "entropy": float(np.mean([entropy(t) for t in ids if t])), "distinct_2": distinct(ids)}
        print(json.dumps(out), flush=True)
        return out

    ref_features = features(references(args.dataset, args.n_ref, args.seed), "reference")
    rows = []
    for spec in args.set:
        name, pattern = spec.split("=", 1)
        texts = [t for f in sorted(glob.glob(pattern)) for t in json.loads(Path(f).read_text()) if t.strip()]
        rows.append(row(name, texts))
    rows.append(row("held-out text", references(args.dataset, args.n_ref, args.seed + 1)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": args.dataset, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
