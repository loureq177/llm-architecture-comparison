"""Final comparison of the trained models on the test split -> results/<run>/metrics.json.

Per model: test perplexity (MDLM and DeltaBlock's diffusion mode: ELBO bound),
training throughput from the log, and per decoding setting the latency / peak
memory of generating a full sequence over a sweep of batch sizes (greedy, bf16,
AR decoding as CUDA graphs plus the eager loop), and sample quality at
--temperature: generative perplexity under --judge and unigram entropy.

    python evaluate.py --preset 16gb [--models transformer mamba]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import tracking
from config import PRESETS, cfg
from dataset import eval_batches, get_tokenizer
from decoding import release_graph
from losses import token_nll
from metrics import validation_metrics

KINDS = ["transformer", "diffusion", "mamba", "deltablock"]
ARCH_FIELDS = ["dim_model", "n_head", "n_layers", "max_seq_len", "tie_embeddings", "mamba_layers_per_layer",
               "mamba_d_state", "mamba_expand", "mamba_d_conv", "attn_every", "block_size"]
FP32_PARAMS = ("A_log", "dt_bias", ".D")


def load_model(kind):
    import importlib

    ck = torch.load(os.path.join(cfg.checkpoints_dir, f"{kind}.pt"), map_location=cfg.device)
    for k in ARCH_FIELDS:
        if k in ck["cfg"]:
            setattr(cfg, k, ck["cfg"][k])
    model = importlib.import_module(f"models.{kind}.train").build_model(get_tokenizer().vocab_size)
    model.load_state_dict(ck["state_dict"])
    return model.to(cfg.device).eval(), ck["step"] + 1


def to_inference(model):
    model.to(torch.bfloat16)
    for name, p in model.named_parameters():
        if name.endswith(FP32_PARAMS):
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
        yield torch.full_like(x[:, :1], eos), x


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
        print(f"  batch {bs}: {latency[bs]:.2f} s, {memory[bs]:.0f} MiB", flush=True)
    return latency, memory


def decoders(kind, model, G, diffusion_steps):
    """(label, steps, generate(prompt, temperature), eager variant or None) per decoding setting."""
    if kind == "diffusion":
        return [(f"MDLM {s} steps", s, lambda p, t, s=s: model.sample(p, G, s, t), None)
                for s in diffusion_steps]
    return [("ancestral", None, lambda p, t: model.generate(p, G, t, cuda_graph=True),
             lambda p, t: model.generate(p, G, t))]


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
def generative_perplexity(samples, judge_name):
    """{label: (perplexity, standard error)} of token sequences {label: [N, G]} under the judge."""
    from transformers import AutoModelForCausalLM

    eos = get_tokenizer().eos_token_id
    judge = AutoModelForCausalLM.from_pretrained(judge_name, dtype=torch.float16).to(cfg.device).eval()
    head = judge.get_output_embeddings().weight
    out = {}
    for label, tokens in samples.items():
        per_seq = []
        for row in tokens[:, : judge.config.n_positions - 1]:  # GPT-2 sees at most 1024 tokens
            ids = torch.cat([row.new_tensor([eos]), row]).to(cfg.device)[None]
            hidden = judge.base_model(ids).last_hidden_state[0, :-1]
            per_seq.append(token_nll(hidden, head, ids[0, 1:]).double().mean().item())
        nll, sem = mean_sem(per_seq)
        out[label] = (math.exp(nll), math.exp(nll) * sem)
    return out


def training_stats(kind):
    with open(os.path.join(cfg.logs_dir, f"{kind}.jsonl")) as f:
        train = [r for r in map(json.loads, f) if "loss" in r]
    return {"train_tokens_per_s": float(np.median([r["tokens_per_s"] for r in train[1:]])),
            "train_hours": max(r["elapsed_s"] for r in train) / 3600}


def test_perplexity(kind, model, split, n_batches, n_mc):
    with torch.autocast("cuda", torch.bfloat16):
        r = validation_metrics("diffusion" if kind == "diffusion" else "ar", model, split, n_batches, n_mc)
        r = {k.replace("val_", "test_"): v for k, v in r.items()}
        if kind == "deltablock":
            d = validation_metrics("diffusion", model, split, n_batches, n_mc)
            r.update(test_nll_diff=d["val_nll"], test_ppl_diff=d["val_ppl"])
    print(f"[{kind}] {r}", flush=True)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="4gb")
    p.add_argument("--models", nargs="+", choices=KINDS, default=KINDS)
    p.add_argument("--split", default="test")
    p.add_argument("--ppl-batches", type=int, default=100)
    p.add_argument("--n-mc", type=int, default=8, help="Monte-Carlo samples of the diffusion ELBO")
    p.add_argument("--n-quality", type=int, default=64, help="generated sequences per decoding setting")
    p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--diffusion-max-batch", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge", default="gpt2-large")
    args = p.parse_args()
    cfg.apply_preset(args.preset)

    G = cfg.max_seq_len - 1
    out = os.path.join(cfg.results_dir, "metrics.json")
    results = {"models": {}}
    if os.path.isfile(out):  # re-running a subset of the models keeps the others
        with open(out) as f:
            results = json.load(f)
    results.update(run=cfg.run_name, split=args.split, prompt_len=1, gen_len=G,
                   temperature=args.temperature, n_quality=args.n_quality, judge=args.judge)
    os.makedirs(cfg.results_dir, exist_ok=True)

    def save():
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    samples = {}
    for kind in args.models:
        torch.manual_seed(cfg.seed)
        model, steps = load_model(kind)
        r = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6, "trained_steps": steps,
             "train_tokens_M": steps * cfg.batch_size * cfg.grad_accum_steps * cfg.max_seq_len / 1e6,
             **training_stats(kind), **test_perplexity(kind, model, args.split, args.ppl_batches, args.n_mc)}
        results["models"][kind] = r
        save()
        model = to_inference(model)
        torch.cuda.empty_cache()

        batch_sizes = [b for b in args.batch_sizes if kind != "diffusion" or b <= args.diffusion_max_batch]
        prompt = next(eos_prompts(args.split, args.gen_batch, args.gen_batch))[0]
        r["generation"] = {}
        for label, n_steps, gen, eager in decoders(kind, model, G, args.diffusion_steps):
            print(f"[{kind}] {label}", flush=True)
            g = {"steps": n_steps, "nfe": n_steps or G}
            if eager is not None:
                g["ttft_s"] = timed(lambda: model.generate(prompt[:1], 1))[0]
            g["latency_s"], g["peak_mem_mib"] = latency_sweep(model, lambda x: gen(x, 0.0), prompt, batch_sizes)
            if eager is not None:
                g["latency_eager_s"], g["peak_mem_eager_mib"] = latency_sweep(
                    model, lambda x: eager(x, 0.0), prompt, batch_sizes)
            tokens = torch.cat([gen(x, args.temperature).cpu()
                                for x, _ in eos_prompts(args.split, args.n_quality, args.gen_batch)])
            release_graph(model)
            samples[f"{kind}/{label}"] = tokens
            g["entropy"], g["entropy_sem"] = unigram_entropy(tokens)
            r["generation"][label] = g
        results["models"][kind] = r
        save()
        model = None
        torch.cuda.empty_cache()

    real = torch.cat([x[:, 1:] for _, x in eos_prompts(args.split, args.n_quality, args.gen_batch)])
    samples["data"] = real
    torch.save(samples, os.path.join(cfg.results_dir, "samples.pt"))
    tok = get_tokenizer()
    with open(os.path.join(cfg.results_dir, "samples.txt"), "w") as f:
        for label, tokens in samples.items():
            for row in tokens[:2]:
                f.write(f"=== {label} ===\n{tok.decode(row[:200].tolist())}\n\n")

    gen_ppl = generative_perplexity(samples, args.judge)
    ref, ref_sem = gen_ppl.pop("data")
    ent, ent_sem = unigram_entropy(real)
    results["data"] = {"gen_ppl": ref, "gen_ppl_sem": ref_sem, "entropy": ent, "entropy_sem": ent_sem}
    for key, (value, sem) in gen_ppl.items():
        kind, label = key.split("/", 1)
        results["models"][kind]["generation"][label].update(gen_ppl=value, gen_ppl_sem=sem)
    save()
    if tracking.enabled():  # results as the last point of every model's training curves
        for kind in args.models:
            r = results["models"][kind]
            tracking.start(kind)
            tracking.log({"step": r["trained_steps"], **r, "data": results["data"]}, prefix="final/")
            tracking.finish()
    for kind in args.models:
        r = results["models"][kind]
        for label, g in r["generation"].items():
            print(f"{kind:12s} {label:16s} test ppl {r['test_ppl']:7.2f} | gen ppl {g['gen_ppl']:7.1f} "
                  f"| entropy {g['entropy']:.2f} | latency b=1 {g['latency_s'][1]:.2f} s")
    print("saved", out)


if __name__ == "__main__":
    main()
