import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch

from config import PRESETS, cfg
from dataset import eval_batches, get_tokenizer
from decoding import release_graph
from losses import token_nll

ARCH_FIELDS = ["dim_model", "n_head", "n_layers", "max_seq_len", "attn_every", "block_size"]


def load_model():
    from models.deltablock.train import build_model

    ck = torch.load(os.path.join(cfg.checkpoints_dir, "deltablock.pt"), map_location=cfg.device)
    for k in ARCH_FIELDS:
        setattr(cfg, k, ck["cfg"][k])
    model = build_model(get_tokenizer().vocab_size)
    model.load_state_dict(ck["state_dict"])
    model = model.to(cfg.device).to(torch.bfloat16).eval()
    for name, p in model.named_parameters():
        if name.endswith(("A_log", "dt_bias")):
            p.data = p.data.float()
    return model


def sync():
    torch.cuda.synchronize()


def timed(fn):
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    sec = time.perf_counter() - t0
    if sec <= 10:
        repeats = 3 if sec <= 2 else 1
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn()
        sync()
        sec = (time.perf_counter() - t0) / repeats
    return sec, torch.cuda.max_memory_allocated() / 2**20


def eos_prompts(split, n, batch_size):
    eos = get_tokenizer().eos_token_id
    for x in eval_batches(split, max(1, n // batch_size), batch_size=batch_size):
        yield torch.full_like(x[:, :1], eos)


def latency_sweep(model, gen_fn, prompt, batch_sizes):
    latency, memory = {}, {}
    for bs in batch_sizes:
        p = prompt.repeat(math.ceil(bs / prompt.size(0)), 1)[:bs]
        try:
            latency[bs], memory[bs] = timed(lambda: gen_fn(p))
        except torch.OutOfMemoryError:
            release_graph(model)
            torch.cuda.empty_cache()
            break
        release_graph(model)
        print(f"batch {bs}: {latency[bs]:.2f} s, {memory[bs]:.0f} MiB", flush=True)
    return latency, memory


def mean_sem(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))


def unigram_entropy(tokens):
    ent = []
    for row in tokens:
        p = row.unique(return_counts=True)[1].double()
        p = p / p.sum()
        ent.append(float(-(p * p.log()).sum()))
    return mean_sem(ent)


@torch.no_grad()
def generative_perplexity(tokens, judge_name):
    from transformers import AutoModelForCausalLM

    eos = get_tokenizer().eos_token_id
    judge = AutoModelForCausalLM.from_pretrained(judge_name, dtype=torch.float16).to(cfg.device).eval()
    head = judge.get_output_embeddings().weight
    per_seq = []
    for row in tokens:
        ids = torch.cat([row.new_tensor([eos]), row]).to(cfg.device)[None]
        hidden = judge.base_model(ids).last_hidden_state[0, :-1]
        per_seq.append(token_nll(hidden, head, ids[0, 1:]).double().mean().item())
    nll, sem = mean_sem(per_seq)
    return math.exp(nll), math.exp(nll) * sem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="4gb")
    p.add_argument("--split", default="test")
    p.add_argument("--n-quality", type=int, default=64)
    p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge", default="gpt2-large")
    args = p.parse_args()
    cfg.apply_preset(args.preset)
    torch.manual_seed(cfg.seed)

    G = cfg.max_seq_len - 1
    model = load_model()
    prompt = next(eos_prompts(args.split, args.gen_batch, args.gen_batch))
    g = {"steps": None, "nfe": G}
    g["ttft_s"] = timed(lambda: model.generate(prompt[:1], 1))[0]
    g["latency_s"], g["peak_mem_mib"] = latency_sweep(
        model, lambda x: model.generate(x, G, 0.0, cuda_graph=True), prompt, args.batch_sizes)
    g["latency_eager_s"], g["peak_mem_eager_mib"] = latency_sweep(
        model, lambda x: model.generate(x, G, 0.0), prompt, args.batch_sizes)

    samples = torch.cat([model.generate(x, G, args.temperature, cuda_graph=True).cpu()
                         for x in eos_prompts(args.split, args.n_quality, args.gen_batch)])
    release_graph(model)
    g["entropy"], g["entropy_sem"] = unigram_entropy(samples)
    model = None
    torch.cuda.empty_cache()
    g["gen_ppl"], g["gen_ppl_sem"] = generative_perplexity(samples, args.judge)
    print(g)

    path = os.path.join(cfg.results_dir, "deltablock.json")
    res = {}
    if os.path.isfile(path):
        with open(path) as f:
            res = json.load(f)
    os.makedirs(cfg.results_dir, exist_ok=True)
    res["generation"] = {"ancestral": g}
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    torch.save(samples, os.path.join(cfg.results_dir, "samples_deltablock.pt"))
    tok = get_tokenizer()
    with open(os.path.join(cfg.results_dir, "samples.txt"), "a") as f:
        for row in samples[:2]:
            f.write(f"=== deltablock/ancestral ===\n{tok.decode(row[:200].tolist())}\n\n")
    print("saved", path)


if __name__ == "__main__":
    main()
