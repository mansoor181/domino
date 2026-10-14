"""download the released maskgit and its vqgan, and build the imagenet-256 token cache.

    python -m scripts.prep_imagenet256

the tokens come from the `Jiayi-Pan/imagenet-maskgit-256-vqgan` dump, where each image is a 16 x 16
grid of maskgit's vqgan codes (codebook 1024) with its class. we flatten each grid in row-major
order, as maskgit does, and write `{train,val}.pt` with the codes `x_clean` [N, 256] and the
classes `y` [N]. the mask token is 1024, just past the codebook, as in maskgit.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from paths import IMAGE_ROOT

MASKGIT_FILES = ("pretrained_maskgit/VQGAN/last.ckpt", "pretrained_maskgit/VQGAN/model.yaml",
                 "pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth")


def fetch_maskgit(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in MASKGIT_FILES:
        if not (dest / Path(name).name).exists():
            path = hf_hub_download("llvictorll/Maskgit-pytorch", name, local_dir=dest / "_hub")
            shutil.move(path, dest / Path(name).name)


def read_split(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=["token", "label"])
            # list<list<int16>> flattened twice is a contiguous buffer of 256 codes per row
            codes = np.asarray(table["token"].combine_chunks().flatten().flatten())
            xs.append(codes.reshape(len(table), 256).astype(np.int16, copy=False))
            ys.append(np.asarray(table["label"]).astype(np.int64, copy=False))
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    fetch_maskgit(IMAGE_ROOT / "maskgit")
    out = IMAGE_ROOT / "imagenet256"
    dump = out / "_hub"
    snapshot_download("Jiayi-Pan/imagenet-maskgit-256-vqgan", repo_type="dataset", local_dir=dump)
    for split in ("train", "val"):
        x, y = read_split(sorted(str(p) for p in dump.rglob(f"{split}-*.parquet")))
        torch.save({"x_clean": torch.from_numpy(x), "y": torch.from_numpy(y)}, out / f"{split}.pt")
        print(f"{split}: {x.shape[0]} images")


if __name__ == "__main__":
    main()
