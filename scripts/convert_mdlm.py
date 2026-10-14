"""convert an mdlm model released on the hub into one of our mdlm checkpoints.

    python -m scripts.convert_mdlm --repo kuleshov-group/mdlm-owt --out $DOMINO_RESULTS/mdlm/openwebtext.pt
    python -m scripts.convert_mdlm --repo jdeschena/sdtt --revision baselines_kld_step_70000 \
        --out $DOMINO_RESULTS/sdtt/openwebtext.pt

mdlm's dit and ours agree block for block once two layout differences are undone: mdlm packs each
adaln projection as (shift, scale, gate) where ours is (scale, shift, gate), and it builds the
sinusoidal time embedding as cosine then sine where ours is sine then cosine. the converted model
uses mdlm's exact numerics. both released openwebtext models (mdlm, and sdtt's kl round 7) were
trained without time conditioning, which is why that is the default here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from models.mdlm import MDLM, MDLMConfig


def permute(w: torch.Tensor, order: list[int]) -> torch.Tensor:
    chunks = w.chunk(len(order), dim=0)
    return torch.cat([chunks[i] for i in order], dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--time_conditioning", action="store_true")
    args = ap.parse_args()

    raw = load_file(hf_hub_download(args.repo, "model.safetensors", revision=args.revision))
    src = {k.removeprefix("backbone."): v.float() for k, v in raw.items() if k.startswith("backbone.")}
    vocab, dim = src["vocab_embed.embedding"].shape
    depth = len({k.split(".")[1] for k in src if k.startswith("blocks.")})
    cfg = MDLMConfig(vocab_size=vocab, seq_len=args.seq_len, dim=dim, depth=depth, heads=args.heads,
                     cond_dim=src["sigma_map.mlp.2.weight"].shape[0],
                     time_embed_dim=src["sigma_map.mlp.0.weight"].shape[1],
                     time_conditioning=args.time_conditioning, mdlm_exact=True, mask_id=vocab - 1)
    half = cfg.time_embed_dim // 2
    w0 = src["sigma_map.mlp.0.weight"]
    state = {
        "embed.weight": src["vocab_embed.embedding"],
        "sigma_map.0.weight": torch.cat([w0[:, half:], w0[:, :half]], dim=1),
        "sigma_map.0.bias": src["sigma_map.mlp.0.bias"],
        "sigma_map.2.weight": src["sigma_map.mlp.2.weight"],
        "sigma_map.2.bias": src["sigma_map.mlp.2.bias"],
        "final_norm.weight": src["output_layer.norm_final.weight"],
        "final_ada.map.1.weight": permute(src["output_layer.adaLN_modulation.weight"], [1, 0]),
        "final_ada.map.1.bias": permute(src["output_layer.adaLN_modulation.bias"], [1, 0]),
        "out.weight": src["output_layer.linear.weight"],
        "out.bias": src["output_layer.linear.bias"],
    }
    for i in range(depth):
        b = f"blocks.{i}"
        state.update({
            f"{b}.norm1.weight": src[f"{b}.norm1.weight"],
            f"{b}.norm2.weight": src[f"{b}.norm2.weight"],
            f"{b}.attn.qkv.weight": src[f"{b}.attn_qkv.weight"],
            f"{b}.attn.o.weight": src[f"{b}.attn_out.weight"],
            f"{b}.mlp.fc1.weight": src[f"{b}.mlp.0.weight"],
            f"{b}.mlp.fc1.bias": src[f"{b}.mlp.0.bias"],
            f"{b}.mlp.fc2.weight": src[f"{b}.mlp.2.weight"],
            f"{b}.mlp.fc2.bias": src[f"{b}.mlp.2.bias"],
            f"{b}.ada.map.1.weight": permute(src[f"{b}.adaLN_modulation.weight"], [1, 0, 2, 4, 3, 5]),
            f"{b}.ada.map.1.bias": permute(src[f"{b}.adaLN_modulation.bias"], [1, 0, 2, 4, 3, 5]),
        })
    MDLM(cfg).load_state_dict(state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(cfg), "model": state, "ema": None, "dataset": "openwebtext",
                "step": -1, "source": args.repo}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
