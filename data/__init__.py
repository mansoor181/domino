"""corpus loaders.

every corpus is a set of `[L]` int64 token rows plus a `meta` dict. the text caches are the
pre-tokenized dumps of mdlm's data pipeline (built by `scripts/prep_text.py`), the molecule caches
are pairflow's, and the imagenet-256 tokens are written by `scripts/prep_imagenet256.py`.

text corpora append the mask token past the end of the vocabulary, while the molecule and image
corpora reserve one inside it. `meta["mask_id"]` and `meta["model_vocab"]` cover both cases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import datasets as hfds
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from paths import IMAGE_ROOT, MOL_ROOT, TEXT_ROOT

DATASETS = {
    "wikitext103": dict(domain="text", root=TEXT_ROOT, seq_len=128, vocab=50257, tokenizer="gpt2",
                        file="wikitext103/wikitext103_{split}_bs128_wrapped.dat",
                        splits={"train": "train", "val": "validation"}),
    # one sentence per row, padded with [PAD] = 0, which the mdlm loss leaves out
    "lm1b": dict(domain="text", root=TEXT_ROOT, seq_len=128, vocab=30522, pad_id=0,
                 tokenizer="bert-base-uncased", file="lm1b/lm1b_{split}_bs128_unwrapped.dat",
                 splits={"train": "train", "val": "test"}),
    # lm1b in the gpt-2 tokenization that our sedd model was trained on
    "lm1b_gpt2": dict(domain="text", root=TEXT_ROOT, seq_len=128, vocab=50257, tokenizer="gpt2",
                      file="lm1b_gpt2/lm1b_gpt2_{split}_bs128_wrapped.dat",
                      splits={"train": "train", "val": "test"}),
    "openwebtext": dict(domain="text", root=TEXT_ROOT, seq_len=1024, vocab=50257, tokenizer="gpt2",
                        file="openwebtext/openwebtext-train_{split}_bs1024_wrapped.dat",
                        splits={"train": "train", "val": "valid"}),
    "qm9": dict(domain="mol", root=MOL_ROOT, seq_len=32, vocab=40, mask_id=2,
                tokenizer="yairschiff/qm9-tokenizer", file="qm9/parsed/{split}.pt",
                splits={"train": "train", "val": "valid"}),
    "zinc250k": dict(domain="mol", root=MOL_ROOT, seq_len=74, vocab=72, mask_id=2,
                     tokenizer="yairschiff/zinc250k-tokenizer", file="zinc-250k/parsed/{split}.pt",
                     splits={"train": "train", "val": "valid"}),
    "imagenet256": dict(domain="image", root=IMAGE_ROOT, seq_len=256, vocab=1025, mask_id=1024,
                        grid=16, n_classes=1000, tokenizer=None, file="imagenet256/{split}.pt",
                        splits={"train": "train", "val": "val"}),
}


class ArrowRows(Dataset):
    def __init__(self, path: Path) -> None:
        self.ds = hfds.load_from_disk(str(path)).with_format("numpy", columns=["input_ids"])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.ds[i]["input_ids"].astype(np.int64))


class TensorRows(Dataset):
    """a `.pt` dump with `x_clean` [N, L], and class labels `y` [N] for images."""

    def __init__(self, path: Path, labels: bool = False) -> None:
        blob = torch.load(str(path), map_location="cpu")
        self.x = blob["x_clean"]
        self.y = blob["y"] if labels else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int):
        x = self.x[i].to(torch.int64)
        return x if self.y is None else (x, int(self.y[i]))


def load(name: str, split: str = "train", labels: bool = False) -> tuple[Dataset, dict]:
    spec = DATASETS[name]
    path = Path(spec["root"]) / spec["file"].format(split=spec["splits"][split])
    if not path.exists():
        raise FileNotFoundError(f"missing {name} cache at {path}; see the readme for how to build it")
    ds = ArrowRows(path) if spec["domain"] == "text" else TensorRows(path, labels)
    mask_id = spec.get("mask_id")
    meta = {
        "domain": spec["domain"],
        "seq_len": spec["seq_len"],
        "vocab_size": spec["vocab"],
        "mask_id": spec["vocab"] if mask_id is None else mask_id,
        "model_vocab": spec["vocab"] + 1 if mask_id is None else spec["vocab"],
        "pad_id": spec.get("pad_id", -1),
        "tokenizer": spec["tokenizer"],
    }
    return ds, meta


def make_loader(name: str, split: str, batch_size: int, shuffle: bool = True,
                num_workers: int = 2, labels: bool = False) -> tuple[DataLoader, dict]:
    ds, meta = load(name, split, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                        drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)
    return loader, meta


def infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def first_rows(name: str, split: str, n: int) -> np.ndarray:
    """the first n rows of a split as an int64 array."""
    ds, _ = load(name, split)
    if isinstance(ds, ArrowRows):
        return np.asarray(ds.ds[:n]["input_ids"], dtype=np.int64)
    return ds.x[:n].numpy().astype(np.int64)


_tokenizers: dict[str, object] = {}


def tokenizer(name: str):
    tid = DATASETS[name]["tokenizer"]
    if tid not in _tokenizers:
        _tokenizers[tid] = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
    return _tokenizers[tid]


def decode(name: str, ids: np.ndarray) -> list[str]:
    """token ids to text, or to smiles strings for the molecule corpora."""
    ids = np.atleast_2d(np.asarray(ids))
    tok = tokenizer(name)
    if DATASETS[name]["domain"] == "mol":
        # pairflow cuts at the first end marker before stripping the other markers, so a sample
        # that starts with one decodes to the empty string and counts as invalid
        out = []
        for seq in tok.batch_decode(torch.as_tensor(ids)):
            seq = (seq + "<eos>").split("<eos>")[0]
            for marker in ("<bos>", "<eos>", "<pad>"):
                seq = seq.replace(marker, "")
            out.append(seq)
        return out
    specials = set(tok.all_special_ids)
    return [tok.decode([int(c) for c in row if int(c) not in specials],
                       skip_special_tokens=True).strip() for row in ids]
