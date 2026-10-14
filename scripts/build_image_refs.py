"""inception features of the 50,000 imagenet validation images, the reference of every fid.

    python -m scripts.build_image_refs

the images are streamed from `evanarlian/imagenet_1k_resized_256`, resized so that the short side is
256 and center-cropped, and only their features are kept (`val_real.npz`). the script then scores
the vqgan reconstructions of the first 10,000 validation token grids against them, which is the fid
of the tokenizer itself and bounds what any model of these tokens can reach.
"""
from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

import data
from metrics import images

SIZE = 256


def center_crop(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) != SIZE:
        scale = SIZE / min(w, h)
        img = img.resize((max(SIZE, round(w * scale)), max(SIZE, round(h * scale))), Image.BICUBIC)
        w, h = img.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    return np.asarray(img.crop((left, top, left + SIZE, top + SIZE)), dtype=np.uint8)


def main() -> None:
    device = torch.device("cuda")
    metric = images.build_metric(device)
    images.REF_DIR.mkdir(parents=True, exist_ok=True)

    def features(batch: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).contiguous().to(device)
        return images.inception_features(metric, x).float().cpu().numpy()

    feats, batch = [], []
    for rec in load_dataset("evanarlian/imagenet_1k_resized_256", split="val", streaming=True):
        batch.append(center_crop(rec["image"]))
        if len(batch) == 64:
            feats.append(features(batch))
            batch = []
    if batch:
        feats.append(features(batch))
    np.savez(images.REF_DIR / "val_real.npz", features=np.concatenate(feats))

    tokens = data.first_rows("imagenet256", "val", 10000)
    vq = images.build_vqgan(device)
    reference = images.load_reference("val_real", device)
    print("tokenizer reconstructions:", images.score(metric, images.tokens_to_uint8(tokens, vq, device),
                                                      reference))


if __name__ == "__main__":
    main()
