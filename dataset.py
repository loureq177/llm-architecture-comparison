import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from config import cfg

_tokenizer = None
_cache = {}


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    return _tokenizer


def load_meta() -> dict:
    path = os.path.join(cfg.data_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run `python prepare_data.py` first."
        )
    with open(path) as f:
        return json.load(f)


def prompt_len() -> int:
    """Tokens a model is conditioned on at generation / evaluation time: a single
    start token (EOS), i.e. generation is unconditional."""
    return 1


def gen_len() -> int:
    """Tokens generated after the prompt (fills the context window)."""
    return cfg.max_seq_len - prompt_len()


class _Split:
    """A split seen as `n` sequences of cfg.max_seq_len tokens: consecutive
    windows of the token stream(s); `shift` moves all windows by a few tokens,
    so that every epoch cuts new ones."""

    def __init__(self, split):
        L = cfg.max_seq_len
        names = [f"train_{i:04d}" for i in range(cfg.train_shards)] if split == "train" else [split]
        self.parts = [
            np.memmap(os.path.join(cfg.data_dir, f"{n}.bin"), dtype=np.uint16, mode="r")
            for n in names
        ]
        # whole windows per piece, minus one to leave room for the shift
        self.ends = np.cumsum([len(p) // L - 1 for p in self.parts])
        self.n = int(self.ends[-1])

    def get(self, idx, shift=0):
        """(tokens [len(idx), L] int64, length [len(idx)]) as tensors on cfg.device;
        every window is full, length = L (kept for the padding-aware losses)."""
        idx, L = np.asarray(idx), cfg.max_seq_len
        part = np.searchsorted(self.ends, idx, side="right")
        first = (idx - np.concatenate([[0], self.ends])[part]) * L + shift
        tokens = np.stack([self.parts[p][s : s + L] for p, s in zip(part, first)])
        length = np.full(len(idx), L, dtype=np.int32)
        x = torch.from_numpy(tokens.astype(np.int64)).to(cfg.device)
        return x, torch.from_numpy(length).to(cfg.device)


def load_split(split: str) -> _Split:
    """Splits: train/validation/test."""
    key = (cfg.data_dir, split, cfg.max_seq_len, cfg.train_shards)
    if key not in _cache:
        _cache[key] = _Split(split)
    return _cache[key]


def train_batches(start_step: int = 0, batch_size: int | None = None):
    """Infinite stream of micro-batches (x, length), beginning at optimizer step `start_step`.

    Sequence number i of the stream depends only on (cfg.seed, i): every epoch
    is a fresh permutation and a fresh window shift. All models thus
    see the same data in the same order, whatever their micro-batch size, and a
    resumed run continues exactly where it stopped.
    """
    data = load_split("train")
    bs = batch_size or cfg.batch_size
    per_step = cfg.batch_size * cfg.grad_accum_steps
    n = data.n // per_step * per_step  # an epoch is a whole number of optimizer steps
    assert n > 0, f"training split too small: {data.n} sequences < {per_step} per step"
    i, current = start_step * per_step, None
    while True:
        epoch, offset = divmod(i, n)
        if epoch != current:
            g = np.random.default_rng([cfg.seed, epoch])
            perm, shift, current = g.permutation(data.n)[:n], int(g.integers(cfg.max_seq_len)), epoch
        yield data.get(perm[offset : offset + bs], shift)
        i += bs


def eval_batches(split: str = "validation", n_batches: int | None = None, batch_size=None):
    """Deterministic pass over the first sequences of a split."""
    data = load_split(split)
    bs = batch_size or cfg.batch_size
    total = data.n // bs
    if n_batches:
        total = min(total, n_batches)
    for b in range(total):
        yield data.get(np.arange(b * bs, (b + 1) * bs))
