"""Builds the tokenized dataset: FineWeb-Edu sample-10BT, plain language modelling.

Documents are tokenized, terminated with EOS and concatenated into flat uint16
token streams in cfg.data_dir:
    train_0000.bin ...         training pieces of 100k documents (cfg.train_shards of them)
    validation.bin, test.bin   cfg.lm_heldout_docs documents each, taken from the
                               last shard (the sample has no official split)
    meta.json                  eos id, vocabulary size, token counts
Training later cuts windows of cfg.max_seq_len tokens out of these streams.
"""

import argparse
import functools
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from config import PRESETS, cfg

LM_SHARDS = 14  # sample/10BT: 000_00000.parquet ... 013_00000.parquet
LM_HELDOUT_SHARD = LM_SHARDS - 1
# A shard is stored as row groups of 1000 documents, sorted by crawl (2013 ->
# 2024): the data is therefore read in row groups, in a fixed random order, so
# that every piece and the held-out splits cover all crawls.
LM_PIECE_GROUPS = 100  # row groups per training piece (100k documents, ~103M GPT-2 tokens)
LM_HELDOUT_PER_GROUP = 100  # held-out documents taken from each row group


@functools.cache
def lm_shard(shard):
    path = hf_hub_download(
        cfg.lm_dataset_name, f"{cfg.lm_subset}/{shard:03d}_00000.parquet", repo_type="dataset"
    )
    return pq.ParquetFile(path)


@functools.cache
def lm_row_groups(shard):
    """Row groups of a shard as (shard, group) pairs, in a fixed random order."""
    n = lm_shard(shard).metadata.num_row_groups
    return [(shard, int(g)) for g in np.random.default_rng(shard).permutation(n)]


def lm_docs(groups, per_group=None):
    """Yields the raw texts of each row group (its first `per_group` documents)."""
    for shard, g in groups:
        yield lm_shard(shard).read_row_group(g, columns=["text"]).column("text").to_pylist()[:per_group]


def lm_train_groups(piece):
    """Row groups of training piece `piece`: the next LM_PIECE_GROUPS of the
    training shards' row groups (each shard in its random order), so a larger
    preset extends the data of a smaller one."""
    first, groups = piece * LM_PIECE_GROUPS, []
    for shard in range(LM_HELDOUT_SHARD):
        order = lm_row_groups(shard)
        if first >= len(order):
            first -= len(order)
            continue
        groups += order[first : first + LM_PIECE_GROUPS - len(groups)]
        first = 0
        if len(groups) == LM_PIECE_GROUPS:
            return groups
    raise ValueError(f"piece {piece}: training documents exhausted")


def lm_encode(tok, texts):
    """Token stream of the documents, each followed by EOS."""
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    return np.concatenate([np.asarray(i + [tok.eos_token_id], dtype=np.uint16) for i in ids])


def lm_write(name, tok, text_batches):
    """Tokenizes batch by batch straight into the file (RAM use independent of size)."""
    path, n = os.path.join(cfg.data_dir, f"{name}.bin"), 0
    with open(path + ".tmp", "wb") as f:
        for texts in text_batches:
            tokens = lm_encode(tok, texts)
            tokens.tofile(f)
            n += len(tokens)
    os.replace(path + ".tmp", path)  # a .bin file is either complete or absent
    print(f"[{name}] {n / 1e6:.1f}M tokens")
    return n


def lm_files():
    return ["validation", "test"] + [f"train_{i:04d}" for i in range(cfg.train_shards)]


def prepare():
    """Download + tokenize the pieces that are still missing in cfg.data_dir."""
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tok.model_max_length = 10**9  # whole documents are tokenized; no length warning
    os.makedirs(cfg.data_dir, exist_ok=True)
    meta_path = os.path.join(cfg.data_dir, "meta.json")
    meta = {"tokens": {}}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta.update(
        dataset=f"{cfg.lm_dataset_name}/{cfg.lm_subset}", eos_id=tok.eos_token_id,
        vocab_size=tok.vocab_size, piece_row_groups=LM_PIECE_GROUPS,
    )

    def exists(name):
        return os.path.exists(os.path.join(cfg.data_dir, f"{name}.bin"))

    def save_meta():
        with open(meta_path + ".tmp", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(meta_path + ".tmp", meta_path)

    # validation / test: LM_HELDOUT_PER_GROUP documents from each of n row groups
    # of the held-out shard (validation the first n of its random order, test the next n)
    n = cfg.lm_heldout_docs // LM_HELDOUT_PER_GROUP
    for i, split in enumerate(["validation", "test"]):
        if not exists(split):
            groups = lm_row_groups(LM_HELDOUT_SHARD)[i * n : (i + 1) * n]
            meta["tokens"][split] = lm_write(split, tok, lm_docs(groups, LM_HELDOUT_PER_GROUP))
            save_meta()
    for piece in range(cfg.train_shards):
        name = f"train_{piece:04d}"
        if not exists(name):
            meta["tokens"][name] = lm_write(name, tok, lm_docs(lm_train_groups(piece)))
            save_meta()
    save_meta()
    total = sum(meta["tokens"][f"train_{i:04d}"] for i in range(cfg.train_shards))
    print(f"Training stream: {total / 1e6:.0f}M tokens in {cfg.train_shards} pieces")


def ensure_data():
    """Prepare the dataset only if it is not there yet."""
    if not all(os.path.exists(os.path.join(cfg.data_dir, f"{n}.bin")) for n in lm_files()):
        print(f"Dataset not complete in {cfg.data_dir} - preparing it...")
        prepare()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default=None)
    args = p.parse_args()
    if args.preset:
        cfg.apply_preset(args.preset)
    prepare()


if __name__ == "__main__":
    main()
