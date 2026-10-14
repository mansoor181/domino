"""helpers shared by the trainers."""
from __future__ import annotations

import os
from pathlib import Path

import torch
import yaml

import paths  # sets the default roots that ${...} in a config refers to


def expand(node):
    """environment variables in every string of a config."""
    if isinstance(node, dict):
        return {k: expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [expand(v) for v in node]
    return os.path.expandvars(node) if isinstance(node, str) else node


def merge(into: dict, update: dict) -> dict:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict):
            merge(into[k], v)
        else:
            into[k] = v
    return into


def load_config(path: str, overrides: list[str] = ()) -> dict:
    """a yaml config with environment variables expanded and `key=value` overrides applied.

    a config can name another one under `base` (relative to its own directory) and then lists only
    the settings it adds or changes.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:
        cfg = merge(load_config(Path(path).parent / cfg.pop("base")), cfg)
    cfg = expand(cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


class EMA:
    """parameter average with mdlm's warmup of the decay, d_n = min(decay, (1 + n) / (10 + n))."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay, self.updates = decay, 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].sub_((1.0 - d) * (self.shadow[n] - p))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.decay, self.updates = state["decay"], state["updates"]
        self.shadow = {n: v.to(device) for n, v in state["shadow"].items()}


def save_atomic(obj: dict, path: Path) -> None:
    """write through a temporary file, so an interrupted save cannot leave a truncated checkpoint."""
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def prune(out_dir: Path, pattern: str, keep: int) -> None:
    for old in sorted(out_dir.glob(pattern))[:-keep]:
        old.unlink()
