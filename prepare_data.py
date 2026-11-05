import argparse
import functools
import multiprocessing
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

from config import PRESETS, cfg

PIECE_GROUPS = 100
HELDOUT_PER_GROUP = 100


@functools.cache
def subset_files(subset):
    files = list_repo_files(cfg.lm_dataset_name, repo_type="dataset")
    return sorted(f for f in files if f.startswith(subset + "/") and f.endswith(".parquet"))


def train_files():
    files = subset_files(cfg.lm_subset)
    return files[:-1] if cfg.lm_subset == cfg.lm_heldout_subset else files


@functools.cache
def shard(path):
    return pq.ParquetFile(hf_hub_download(cfg.lm_dataset_name, path, repo_type="dataset"))


@functools.cache
def row_groups(index, path):
    n = shard(path).metadata.num_row_groups
    return [(path, int(g)) for g in np.random.default_rng(index).permutation(n)]


def heldout_groups(i):
    files = subset_files(cfg.lm_heldout_subset)
    n = cfg.lm_heldout_docs // HELDOUT_PER_GROUP
    return row_groups(len(files) - 1, files[-1])[i * n : (i + 1) * n]


def docs(groups, per_group=None, exclude=frozenset()):
    for path, g in groups:
        t = shard(path).read_row_group(g, columns=["id", "text"])
        ids, texts = t.column("id").to_pylist()[:per_group], t.column("text").to_pylist()[:per_group]
        yield [x for i, x in zip(ids, texts) if i not in exclude]


def heldout_ids():
    return frozenset(
        i for path, g in heldout_groups(0) + heldout_groups(1)
        for i in shard(path).read_row_group(g, columns=["id"]).column("id").to_pylist()[:HELDOUT_PER_GROUP]
    )


def train_groups(piece):
    first, groups = piece * PIECE_GROUPS, []
    for index, path in enumerate(train_files()):
        order = row_groups(index, path)
        if first >= len(order):
            first -= len(order)
            continue
        groups += order[first : first + PIECE_GROUPS - len(groups)]
        first = 0
        if len(groups) == PIECE_GROUPS:
            return groups
    raise ValueError(f"piece {piece}: training documents exhausted")


def tokenizer():
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tok.model_max_length = 10**9
    return tok


def write(name, tok, text_batches):
    path, n = os.path.join(cfg.data_dir, f"{name}.bin"), 0
    with open(path + ".tmp", "wb") as f:
        for texts in text_batches:
            ids = tok(texts, add_special_tokens=False)["input_ids"]
            tokens = np.concatenate([np.asarray(i + [tok.eos_token_id], dtype=np.uint16) for i in ids])
            tokens.tofile(f)
            n += len(tokens)
    os.replace(path + ".tmp", path)
    print(f"[{name}] {n / 1e6:.1f}M tokens")


def _init_worker(config, exclude):
    global _tok, _exclude
    vars(cfg).update(config)
    _tok, _exclude = tokenizer(), exclude


def _write_piece(piece):
    write(f"train_{piece:04d}", _tok, docs(train_groups(piece), exclude=_exclude))


def files():
    return ["validation", "test"] + [f"train_{i:04d}" for i in range(cfg.train_shards)]


def exists(name):
    return os.path.exists(os.path.join(cfg.data_dir, f"{name}.bin"))


def prepare(workers=1):
    os.makedirs(cfg.data_dir, exist_ok=True)
    tok = tokenizer()
    for i, split in enumerate(["validation", "test"]):
        if not exists(split):
            write(split, tok, docs(heldout_groups(i), HELDOUT_PER_GROUP))
    missing = [i for i in range(cfg.train_shards) if not exists(f"train_{i:04d}")]
    if missing:
        exclude = frozenset() if cfg.lm_subset == cfg.lm_heldout_subset else heldout_ids()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(min(workers, len(missing)), _init_worker, (vars(cfg), exclude)) as pool:
            for _ in pool.imap_unordered(_write_piece, missing):
                pass


def ensure_data():
    if not all(exists(n) for n in files()):
        prepare()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    if args.preset:
        cfg.apply_preset(args.preset)
    prepare(args.workers)
