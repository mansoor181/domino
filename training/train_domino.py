"""train a domino module on a fixed pre-trained denoiser.

    python -m training.train_domino --config configs/domino/wikitext103_mdlm.yaml

only the module is optimized. the log records the chain's loss next to the product kernel's on
the same revealed positions (their difference is the gain per revealed token), and every
`val_every` steps the K-step likelihood bound of both kernels on held-out rows.
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
from models.denoisers import load_denoiser
from models.domino import Domino, build_module
from training.utils import EMA, load_config, prune, save_atomic


@torch.no_grad()
def validate(model: Domino, batches, device: torch.device, rows: int, K: int) -> dict:
    """K-step bounds of the product kernel and the chain, in nats per token. both kernels are scored
    on the same reveal steps, so their difference is not masked by sampling noise."""
    model.eval()
    nll = {"product": 0.0, "chain": 0.0}
    tokens = {"product": 0.0, "chain": 0.0}
    seen = batch = 0
    while seen < rows:
        b = next(batches)
        if isinstance(b, (list, tuple)):
            b, y = b
            model.denoiser.labels = y.to(device)
        x = b.to(device)
        for kernel in nll:
            gen = torch.Generator(device=device).manual_seed(1234 + batch)
            total, count = model.kstep_bound(x, K, product=kernel == "product", generator=gen)
            nll[kernel] += float(total.sum())
            tokens[kernel] += float(count.sum())
        seen += x.shape[0]
        batch += 1
    model.train()
    out = {f"val_bound_{k}": nll[k] / tokens[k] for k in nll}
    out["val_gain"] = out["val_bound_product"] - out["val_bound_chain"]
    return out


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = cfg["denoiser"]
    denoiser = load_denoiser(spec["kind"], spec["path"], device)
    class_cond = cfg.get("class_cond", False)
    micro = cfg["micro_batch_size"]
    accum = cfg["batch_size"] // micro
    loader, _ = data.make_loader(cfg["dataset"], "train", micro, num_workers=cfg["num_workers"],
                                 labels=class_cond)
    val_loader, _ = data.make_loader(cfg["dataset"], "val", micro, shuffle=False, num_workers=2,
                                     labels=class_cond)

    # the module's initialization depends on the seed alone, whatever the denoiser consumed
    torch.manual_seed(cfg["seed"])
    module = build_module(denoiser, cfg["module"]).to(device)
    if cfg.get("init_module"):
        init = torch.load(cfg["init_module"], map_location=device, weights_only=False)
        module.load_state_dict(init["ema"]["shadow"] if init.get("ema") else init["module"])
    model = Domino(denoiser, module)
    print(f"[domino] module {module.n_params() / 1e6:.2f}M parameters on a "
          f"{sum(p.numel() for p in denoiser.parameters()) / 1e6:.1f}M {spec['kind']} denoiser, "
          f"batch {micro} x {accum}", flush=True)

    opt = torch.optim.AdamW(module.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"],
                            betas=tuple(cfg["betas"]))
    warmup, steps = cfg["warmup_steps"], cfg["steps"]

    def lr_lambda(s: int) -> float:
        if s < warmup:
            return s / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, steps - warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(module, cfg["ema"])

    start = 0
    ckpts = sorted(out_dir.glob("module_*.pt"))
    if ckpts:
        state = torch.load(ckpts[-1], map_location=device, weights_only=False)
        module.load_state_dict(state["module"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        ema.load_state_dict(state["ema"], device)
        start = state["step"]
        print(f"[resume] {ckpts[-1].name} at step {start}", flush=True)

    batches, val_batches = data.infinite(loader), data.infinite(val_loader)
    log = (out_dir / "log.jsonl").open("a")

    def emit(row: dict) -> None:
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    model.train()
    t0 = time.time()
    totals: dict[str, float] = {}
    n_totals = 0
    for step in range(start + 1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            b = next(batches)
            if class_cond:
                # read the denoiser as maskgit was trained, with the true class and a share of the
                # labels replaced by the null class, so one module serves both passes of guidance
                x, y = b[0].to(device, non_blocking=True), b[1].to(device, non_blocking=True)
                drop = torch.rand(y.shape[0], device=device) < cfg["label_drop"]
                denoiser.labels = torch.where(drop, torch.full_like(y, denoiser.n_classes), y)
            else:
                x = b.to(device, non_blocking=True)
            if cfg.get("reveal") == "confidence":
                out = model.loss_confidence(x, tuple(cfg["reveal_K"]), cfg["reveal_r_temp"],
                                            cfg["reveal_sm_temp"])
            else:
                out = model.loss(x, cfg["K"], cfg["eta"], cfg["h_min"])
            (out["loss"] / accum).backward()
            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n_totals += 1
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(module)

        if step % cfg["log_every"] == 0 or step == 1:
            emit({"step": step, **{k: round(v / n_totals, 4) for k, v in totals.items()},
                  "lr": opt.param_groups[0]["lr"], "elapsed": round(time.time() - t0, 1)})
            totals, n_totals = {}, 0
        if step % cfg["val_every"] == 0 or step == steps:
            emit({"step": step, **{k: round(v, 4) for k, v in
                                   validate(model, val_batches, device, cfg["val_rows"], cfg["K"]).items()}})
        if step % cfg["ckpt_every"] == 0 or step == steps:
            save_atomic({"step": step, "module_cfg": cfg["module"], "denoiser": spec,
                         "dataset": cfg["dataset"], "module": module.state_dict(),
                         "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()},
                        out_dir / f"module_{step:07d}.pt")
            prune(out_dir, "module_*.pt", cfg["keep_last_ckpts"])
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    train(load_config(args.config, args.override))
