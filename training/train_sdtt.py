"""distill an sdtt student from one of our mdlm models.

this transcribes the released objective of sdtt (`sdtt/core/distill/multi_round_sdtt.py`) onto our
mdlm, so that the student is a checkpoint domino can be integrated with. in each round, a student
learns to match in one ancestral step of size dt what its teacher produces in two steps of size
dt / 2. we mask the data at a noise time t and roll the teacher over two ancestral steps. a position
the rollout reveals takes as its target the teacher's clean-token posterior from the step that
revealed it, and a position still masked at the end takes the posterior of the last step. the loss
is kl(student || teacher) on the masked positions (sdtt's kl-bwd). every `grow_dt_every` steps the
student, without its moving average, becomes the teacher and dt doubles, from 1/1024 of the time
axis until the student takes 8 steps.

    python -m training.train_sdtt --config configs/sdtt/wikitext103.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

import data
from models.mdlm import MDLM, load_mdlm
from training.utils import load_config

NEG = -1.0e6   # sdtt's finite stand-in for minus infinity, which keeps the kl finite


def log_x0(model: MDLM, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """log p(x0 | z_t) under the substitution parameterization: the posterior over clean tokens at
    masked positions, and a point mass on the token at unmasked ones."""
    logits, _ = model(z, t)
    logits[..., model.mask_id] += NEG
    lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked = (z != model.mask_id)[..., None]
    lp = torch.where(unmasked, torch.full_like(lp, NEG), lp)
    return lp.scatter(-1, z[..., None], torch.where(unmasked, 0.0, lp.gather(-1, z[..., None])))


@torch.no_grad()
def teacher_targets(teacher: MDLM, z: torch.Tensor, t: torch.Tensor, dt: float, n_sub: int,
                    eps: float) -> torch.Tensor:
    """sdtt's `_teacher_logprobs_on_mask` with the ancestral sampler."""
    B, L = z.shape
    space = torch.linspace(1, 0, n_sub, device=z.device).double()[:, None]
    t_end = t.double() - dt * n_sub
    ts = (t.double()[None, :] * space + (1 - t.double()[None, :]) * t_end[None, :]).clamp_min(eps).float()
    preds = torch.zeros(B, L, teacher.vocab_size, device=z.device)
    revealed = torch.zeros(B, L, dtype=torch.bool, device=z.device)
    cur = z
    for k in range(n_sub):
        lp = log_x0(teacher, cur, ts[k])
        s = (ts[k] - dt).clamp_min(0.0)
        # a masked position stays masked with probability move(s) / move(t), and otherwise takes
        # a token from the posterior
        stay = (((1 - eps) * s) / ((1 - eps) * ts[k]))[:, None]
        reveal = (cur == teacher.mask_id) & (torch.rand(B, L, device=z.device) >= stay)
        new = cur.clone()
        rows = lp[reveal]
        if rows.numel():
            g = -torch.log(-torch.log(torch.rand_like(rows).clamp_min(1e-20)))
            new[reveal] = (rows + g).argmax(-1)
        updated = new != cur
        preds[updated] = lp[updated]
        revealed |= updated
        cur = new
    last = (cur == teacher.mask_id) & ~revealed
    return torch.where(last[..., None], lp, preds)


def sample_t(B: int, eps: float, device: torch.device) -> torch.Tensor:
    """antithetic uniform times on [eps, 1], as sdtt's `_sample_t`."""
    u = (torch.rand(1, device=device) + torch.linspace(0, 1, B + 1, device=device)[:-1]) % 1
    return (1 - eps) * u + eps


def train(cfg: dict) -> None:
    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher = load_mdlm(cfg["teacher"], device).requires_grad_(False)
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    mask_id = student.mask_id

    eps = cfg["sampling_eps"]
    n_sub = cfg["num_distill_steps"]
    dt = (1 - eps) / cfg["orig_num_sampling_steps"]
    bs, micro = cfg["batch_size"], cfg["micro_batch_size"]
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup_steps"]))
    ema = {n: p.detach().clone() for n, p in student.named_parameters()}
    loader, _ = data.make_loader(cfg["dataset"], "train", micro, num_workers=cfg["num_workers"])
    batches = data.infinite(loader)

    def save(name: str, state: dict, extra: dict) -> None:
        torch.save({"model": state, "cfg": asdict(student.cfg), "dataset": cfg["dataset"],
                    "step": step, **extra}, out_dir / name)

    step, rnd, t0 = 0, 0, time.time()
    log = (out_dir / "log.jsonl").open("a")
    while True:
        if step > 0 and step % cfg["grow_dt_every"] == 0:
            rnd = step // cfg["grow_dt_every"]
            dt = (1 - eps) / cfg["orig_num_sampling_steps"] * n_sub ** rnd
            teacher.load_state_dict(student.state_dict())
            save(f"student_round{rnd}.pt", student.state_dict(),
                 {"ema": {"shadow": {n: v.clone() for n, v in ema.items()}}})
            for old in sorted(out_dir.glob("student_round*.pt"))[:-1]:
                old.unlink()
            print(f"[sdtt] step {step}: round {rnd} done, the student now takes {round(1 / dt)} steps",
                  flush=True)
            if round(1 / dt) < cfg["min_num_sampling_steps"]:
                break
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(bs // micro):
            x = next(batches).to(device)
            t = sample_t(x.shape[0], eps, device)
            z = torch.where(torch.rand_like(x, dtype=torch.float) < ((1 - eps) * t)[:, None],
                            torch.full_like(x, mask_id), x)
            target = teacher_targets(teacher, z, t, dt, n_sub, eps)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = log_x0(student, z, t)
            m = z == mask_id
            loss = F.kl_div(target[m].float(), pred[m].float(), log_target=True, reduction="batchmean")
            (loss * micro / bs).backward()
            total += float(loss) * micro / bs
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        with torch.no_grad():
            for n, p in student.named_parameters():
                ema[n].mul_(cfg["ema"]).add_(p.detach(), alpha=1 - cfg["ema"])
        if step % 100 == 0:
            row = {"step": step, "round": rnd, "loss": round(total, 5), "steps": round(1 / dt),
                   "lr": sched.get_last_lr()[0], "elapsed": round(time.time() - t0, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
    # the final student is the last round's, with its moving average, which is how sdtt samples
    save("student_final.pt", {**student.state_dict(), **ema}, {"ema": None})
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    train(load_config(args.config, args.override))
