"""build the text caches.

    python -m scripts.prep_text --corpus wikitext103   # and lm1b, lm1b_gpt2, openwebtext

wikitext103 and lm1b go through mdlm's own data pipeline (`dataloader.get_dataset` in the mdlm
checkout, with patches/mdlm.patch applied), so the caches match the ones mdlm trains on.

lm1b_gpt2 is lm1b in the tokenization of our sedd model, which follows sedd's pipeline: sedd's lm1b
detokenizer, gpt-2 tokens, an end-of-text token after each sentence, and the stream cut into rows
of 128 with no leading token. the training split keeps every fourth sentence, which is plenty for a
domino module, and the test split serves as the validation set.

openwebtext is a bounded subset for the released openwebtext models, wrapped as mdlm wraps it:
each document gets a trailing end-of-text token, the documents are concatenated, and the stream is
cut into pieces of 1022 tokens framed by a leading and a trailing end-of-text token. the validation
rows come from the last parquet shard of a local snapshot, so they do not overlap the training rows.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import datasets
import numpy as np
from omegaconf import OmegaConf
from transformers import GPT2TokenizerFast

from paths import MDLM_DIR, TEXT_ROOT

sys.path.append(str(MDLM_DIR))
import dataloader as mdlm_data


def lm1b_detokenizer(x: str) -> str:
    """sedd's lm1b detokenizer (data.py in the sedd repository)."""
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def prep_mdlm(name: str, tokenizer: str, wrap: bool, modes: tuple[str, ...]) -> None:
    tok = mdlm_data.get_tokenizer(OmegaConf.create({"data": {"tokenizer_name_or_path": tokenizer}}))
    for mode in modes:
        mdlm_data.get_dataset(name, tok, wrap=wrap, mode=mode, cache_dir=str(TEXT_ROOT / name),
                              block_size=128)


def prep_lm1b_gpt2(block: int = 128, val_rows: int = 4096) -> None:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(batch: dict) -> dict:
        ids = tok([lm1b_detokenizer(t) for t in batch["text"]])["input_ids"]
        return {"input_ids": [x + [tok.eos_token_id] for x in ids]}

    raw = datasets.load_dataset("jdeschena/lm1b")
    out_dir = TEXT_ROOT / "lm1b_gpt2"
    for split, stride, max_rows in (("test", 1, val_rows), ("train", 4, None)):
        ds = raw[split].select(range(0, len(raw[split]), stride))
        ds = ds.map(encode, batched=True, num_proc=8, remove_columns=ds.column_names)
        flat = np.fromiter((t for row in ds["input_ids"] for t in row), dtype=np.int32)
        n = len(flat) // block if max_rows is None else min(len(flat) // block, max_rows)
        rows = flat[: n * block].reshape(n, block)
        datasets.Dataset.from_dict({"input_ids": rows.tolist()}).save_to_disk(
            str(out_dir / f"lm1b_gpt2_{split}_bs{block}_wrapped.dat"))
        print(f"lm1b_gpt2 {split}: {n} rows")


def prep_openwebtext(snapshot: str, train_rows: int, val_rows: int, block: int = 1024) -> None:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    eos = bos = tok.eos_token_id
    shards = sorted(str(p) for p in Path(snapshot).glob("*.parquet"))
    out_dir = TEXT_ROOT / "openwebtext"
    for split, files, n_rows in (("valid", shards[-1:], val_rows), ("train", shards[:-1], train_rows)):
        stream = datasets.load_dataset("parquet", data_files=files, split="train", streaming=True)
        rows, carry = [], []
        for doc in stream:
            if not doc["text"]:
                continue
            carry += tok(doc["text"], add_special_tokens=False)["input_ids"] + [eos]
            while len(carry) >= block - 2 and len(rows) < n_rows:
                rows.append([bos] + carry[:block - 2] + [eos])
                carry = carry[block - 2:]
            if len(rows) >= n_rows:
                break
        datasets.Dataset.from_dict({"input_ids": rows}).save_to_disk(
            str(out_dir / f"openwebtext-train_{split}_bs{block}_wrapped.dat"))
        print(f"openwebtext {split}: {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["wikitext103", "lm1b", "lm1b_gpt2", "openwebtext"])
    ap.add_argument("--snapshot", help="openwebtext: a local directory of the parquet shards")
    ap.add_argument("--train_rows", type=int, default=200000)
    ap.add_argument("--val_rows", type=int, default=2048)
    args = ap.parse_args()
    if args.corpus == "wikitext103":
        prep_mdlm("wikitext103", "gpt2", wrap=True, modes=("train", "validation"))
    elif args.corpus == "lm1b":
        prep_mdlm("lm1b", "bert-base-uncased", wrap=False, modes=("train", "test"))
    elif args.corpus == "lm1b_gpt2":
        prep_lm1b_gpt2()
    else:
        prep_openwebtext(args.snapshot, args.train_rows, args.val_rows)


if __name__ == "__main__":
    main()
