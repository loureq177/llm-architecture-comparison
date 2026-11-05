import functools
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from config import cfg


@functools.cache
def get_tokenizer():
    return AutoTokenizer.from_pretrained(cfg.tokenizer_name)


class _Split:
    def __init__(self, split):
        L = cfg.max_seq_len
        names = [f"train_{i:04d}" for i in range(cfg.train_shards)] if split == "train" else [split]
        self.parts = [
            np.memmap(os.path.join(cfg.data_dir, f"{n}.bin"), dtype=np.uint16, mode="r")
            for n in names
        ]
        self.ends = np.cumsum([len(p) // L - 1 for p in self.parts])
        self.n = int(self.ends[-1])

    def get(self, idx, shift=0):
        idx, L = np.asarray(idx), cfg.max_seq_len
        part = np.searchsorted(self.ends, idx, side="right")
        first = (idx - np.concatenate([[0], self.ends])[part]) * L + shift
        tokens = np.stack([self.parts[p][s : s + L] for p, s in zip(part, first)])
        return torch.from_numpy(tokens.astype(np.int64)).to(cfg.device)


@functools.cache
def _load_split(split, data_dir, max_seq_len, train_shards):
    return _Split(split)


def load_split(split: str) -> _Split:
    return _load_split(split, cfg.data_dir, cfg.max_seq_len, cfg.train_shards)


def train_batches(start_step: int = 0, batch_size: int | None = None, rank=0, world=1):
    data = load_split("train")
    bs = batch_size or cfg.batch_size
    per_step = cfg.batch_size * cfg.grad_accum_steps
    local = per_step // world
    assert local % bs == 0, f"micro-batch {bs} must divide {local} sequences per rank"
    n = data.n // per_step * per_step
    assert n > 0, f"training split too small: {data.n} sequences < {per_step} per step"
    step, current = start_step, None
    while True:
        for j in range(0, local, bs):
            epoch, offset = divmod(step * per_step + rank * local + j, n)
            if epoch != current:
                g = np.random.default_rng([cfg.seed, epoch])
                perm, shift, current = g.permutation(data.n)[:n], int(g.integers(cfg.max_seq_len)), epoch
            yield data.get(perm[offset : offset + bs], shift)
        step += 1


def eval_batches(split: str = "validation", n_batches: int | None = None, batch_size=None):
    data = load_split(split)
    bs = batch_size or cfg.batch_size
    total = data.n // bs
    if n_batches:
        total = min(total, n_batches)
    for b in range(total):
        yield data.get(np.arange(b * bs, (b + 1) * bs))
