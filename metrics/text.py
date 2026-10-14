"""text metrics: generative perplexity under gpt-2 large and the per-position entropy of samples."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import data


class GPT2Scorer:
    def __init__(self, device: torch.device, name: str = "gpt2-large") -> None:
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def nll(self, texts: list[str], max_length: int, batch: int = 16) -> tuple[float, int]:
        """summed negative log-likelihood of the texts and the number of scored tokens."""
        total, count = 0.0, 0
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True,
                           max_length=max_length).to(self.device)
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
            mask = enc["attention_mask"][:, 1:].float()
            nll = -logp.gather(-1, enc["input_ids"][:, 1:, None]).squeeze(-1)
            total += float((nll * mask).sum())
            count += int(mask.sum())
        return total, count


def position_entropy(samples: np.ndarray) -> float:
    """mean over positions of the entropy of the token histogram across samples, in nats."""
    ents = []
    for column in samples.T:
        counts = np.bincount(column).astype(np.float64)
        p = counts[counts > 0] / len(column)
        ents.append(float(-(p * np.log(p)).sum()))
    return float(np.mean(ents))


def reference_entropy(name: str, n: int, subsets: int = 8) -> float:
    """the same entropy on held-out rows, averaged over subsets of n rows.

    a corpus padded to one sentence per row (lm1b) is packed first, since samples fill every
    position and the padding would otherwise dominate the histogram.
    """
    ds, meta = data.load(name, "val")
    if meta["pad_id"] >= 0:
        tokens = data.first_rows(name, "val", 200000)
        tokens = tokens[tokens != meta["pad_id"]]
        rows = tokens[: len(tokens) // meta["seq_len"] * meta["seq_len"]].reshape(-1, meta["seq_len"])
    else:
        rows = data.first_rows(name, "val", len(ds))
    rows = rows[np.random.default_rng(0).permutation(len(rows))]
    k = max(1, min(subsets, len(rows) // n))
    return float(np.mean([position_entropy(rows[i * n:(i + 1) * n]) for i in range(k)]))
