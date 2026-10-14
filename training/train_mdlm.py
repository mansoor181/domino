"""train an mdlm from scratch.

    python -m training.train_mdlm --config configs/mdlm/wikitext103.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import data
from models.mdlm import MDLM, MDLMConfig
from training.utils import EMA, load_config, prune, save_atomic


@torch.no_grad()
def validate(model: MDLM, loader, device: torch.device, max_rows: int = 8192) -> float:
    model.eval()
    total, rows = 0.0, 0
    for x in loader:
        total += float(model.loss(x.to(device))["nll"]) * x.shape[0]
        rows += x.shape[0]
        if rows >= max_rows:
            break
    model.train()
    return total / rows


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    micro = cfg["micro_batch_size"]
    accum = cfg["batch_size"] // micro
    loader, meta = data.make_loader(cfg["dataset"], "train", micro, num_workers=cfg["num_workers"])
    val_loader, _ = data.make_loader(cfg["dataset"], "val", micro, shuffle=False,
                                     num_workers=cfg["num_workers"])
    model = MDLM(MDLMConfig(vocab_size=meta["model_vocab"], seq_len=meta["seq_len"],
                            mask_id=meta["mask_id"], pad_id=meta["pad_id"], **cfg["model"])).to(device)
    print(f"[mdlm] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, "
          f"batch {micro} x {accum}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"],
                            betas=tuple(cfg["betas"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / cfg["warmup_steps"]))
    ema = EMA(model, cfg["ema"])

    start = 0
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    if ckpts:
        state = torch.load(ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        ema.load_state_dict(state["ema"], device)
        start = state["step"]
        print(f"[resume] {ckpts[-1].name} at step {start}")

    batches = data.infinite(loader)
    log = (out_dir / "log.jsonl").open("a")

    def emit(row: dict) -> None:
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    t0 = time.time()
    for step in range(start + 1, cfg["steps"] + 1):
        opt.zero_grad(set_to_none=True)
        nll = 0.0
        for _ in range(accum):
            out = model.loss(next(batches).to(device, non_blocking=True))
            (out["loss"] / accum).backward()
            nll += float(out["nll"]) / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(model)

        if step % cfg["log_every"] == 0 or step == 1:
            emit({"step": step, "nll": round(nll, 4), "lr": opt.param_groups[0]["lr"],
                  "elapsed": round(time.time() - t0, 1)})
        if step % cfg["val_every"] == 0:
            emit({"step": step, "val_nll": round(validate(model, val_loader, device), 4)})
        if step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
            save_atomic({"step": step, "cfg": asdict(model.cfg), "dataset": cfg["dataset"],
                         "model": model.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "ema": ema.state_dict()},
                        out_dir / f"ckpt_{step:07d}.pt")
            prune(out_dir, "ckpt_*.pt", cfg["keep_last_ckpts"])
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    train(load_config(args.config, args.override))
