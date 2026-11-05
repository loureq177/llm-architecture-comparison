"""KDA vs GDN-2 in the same 3:1 hybrid (34M params, equal parameter counts).

    python -m experiments.mixer_ablation check   # read-only trick is exact (both kernels)
    python -m experiments.mixer_ablation speed   # training throughput
    python -m experiments.mixer_ablation mqar    # associative recall (Arora et al. 2023)
    python -m experiments.mixer_ablation lm --steps 500 --seeds 0 1   # FineWeb-Edu, val PPL

Results: results/ablation_mixer/*.json.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from config import cfg

OUT = "ablation_mixer"
MIXERS = ("kda", "gdn2")


def save(name, obj):
    d = os.path.join(cfg.root_dir, "results", OUT)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"-> {path}")


# ---------------------------------------------------------------- check
def check():
    from fla.ops.gdn2 import chunk_gdn2
    from fla.ops.kda import chunk_kda

    B, nblk, L, H, K = 2, 16, 16, 4, 64
    T, dev = nblk * L, "cuda"

    def call(kind, q, k, v, g, gate, **kw):
        if kind == "kda":
            return chunk_kda(q, k, v, g, gate[..., 0], **kw)  # scalar beta per head
        return chunk_gdn2(q, k, v, g, gate, gate.flip(-1), **kw)  # erase b, write w

    def interleave(n, c):
        sh = n.shape
        return torch.stack([n.reshape(B, nblk, L, *sh[2:]), c.reshape(B, nblk, L, *sh[2:])],
                           2).reshape(B, 2 * T, *sh[2:])

    def deinterleave(x):
        x = x.reshape(B, nblk, 2, L, *x.shape[2:])
        return x[:, :, 0].flatten(1, 2), x[:, :, 1].flatten(1, 2)

    results = {}
    for kind in MIXERS:
        for dt in (torch.bfloat16, torch.float32):
            torch.manual_seed(0)
            unit = lambda: F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1)  # noqa: E731
            qc, kc, qn, kn = unit(), unit(), unit(), unit()
            vc, vn = torch.randn(B, T, H, K, device=dev), torch.randn(B, T, H, K, device=dev)
            gc = -F.softplus(torch.randn(B, T, H, K, device=dev)) * 0.1
            bc = torch.rand(B, T, H, K, device=dev)  # gates in [0, 1]
            zero = torch.zeros_like
            args = [interleave(qn, qc), interleave(kn, kc), interleave(vn, vc),
                    interleave(zero(gc), gc), interleave(zero(bc), bc)]
            o, _ = call(kind, *(a.to(dt) for a in args))
            o_noisy, o_clean = deinterleave(o.float())
            ref_clean, _ = call(kind, *(a.to(dt) for a in (qc, kc, vc, gc, bc)))
            err_clean = (o_clean - ref_clean.float()).abs().max().item()
            err_noisy, S = 0.0, torch.zeros(B, H, K, K, device=dev)
            for b in range(nblk):
                sl = slice(b * L, (b + 1) * L)
                ref = K**-0.5 * torch.einsum("bthk,bhkv->bthv", qn[:, sl], S)
                err_noisy = max(err_noisy, (o_noisy[:, sl] - ref).abs().max().item())
                _, S = call(kind, *(a[:, sl].to(dt) for a in (qc, kc, vc, gc, bc)),
                            initial_state=S, output_final_state=True)
                S = S.float()
            scale = o.float().abs().max().item()
            key = f"{kind}_{str(dt).split('.')[-1]}"
            results[key] = {"max_err_clean": err_clean, "max_err_noisy": err_noisy,
                            "max_abs_output": scale}
            print(f"{key:14s} clean err {err_clean:.2e} | noisy err {err_noisy:.2e} | "
                  f"max |o| {scale:.2e} | relative {max(err_clean, err_noisy) / scale:.1%}")

    # layer level, two-stream mode: clean outputs must not depend on noisy tokens
    from models.deltablock.mixers import DeltaMixer, TwoStream
    ts = TwoStream(T, L, dev)
    for kind in MIXERS:
        torch.manual_seed(0)
        layer = DeltaMixer(256, 4, kind).to(dev)
        x = torch.randn(1, 2 * T, 256, device=dev)
        x2 = x.clone()
        x2[:, ts.noisy] = torch.randn_like(x2[:, ts.noisy])
        with torch.no_grad():
            leak = (layer(x, ts) - layer(x2, ts))[:, ~ts.noisy].abs().max().item()
        print(f"{kind:5s} layer leak noisy -> clean: {leak:.1e}")
        results[f"{kind}_layer_leak"] = leak
    save("check", results)


# ---------------------------------------------------------------- speed
def build(kind, vocab, seed=0, **kw):
    from models.deltablock.model import HybridLM

    torch.manual_seed(1000 + seed)  # initialisation seed only; data order is cfg.seed
    if kind == "transformer":
        from models.transformer.model import LanguageModel
        return LanguageModel(vocab, cfg.dim_model, cfg.n_head, cfg.max_seq_len, cfg.n_layers)
    return HybridLM(vocab, kw.pop("dim", cfg.dim_model), kw.pop("heads", cfg.n_head),
                    kw.pop("layers", cfg.n_layers), mixer=kind, **kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def speed():
    from dataset import get_tokenizer, train_batches
    from train_utils import next_token_loss

    cfg.apply_preset("4gb")
    vocab = get_tokenizer().vocab_size
    batches = train_batches(0, cfg.batch_size)
    x, length = next(batches)
    res = {}
    for kind in ("transformer",) + MIXERS:
        model = build(kind, vocab).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        torch.cuda.reset_peak_memory_stats()
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast("cuda", torch.bfloat16):
                loss = next_token_loss(model, x, length)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tps = 20 * x.numel() / (time.time() - t0)
        N = n_params(model)
        res[kind] = {"params": N, "tokens_per_s": tps, "achieved_tflops_6N": 6 * N * tps / 1e12,
                     "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20,
                     "layers": getattr(model, "layer_kinds", ["attn"] * cfg.n_layers)}
        print(f"{kind:12s} {N / 1e6:6.2f}M params | {tps / 1e3:5.1f}k tok/s | "
              f"6N*tok/s = {6 * N * tps / 1e12:4.1f} TFLOP/s | "
              f"peak {res[kind]['peak_mem_mib']:.0f} MiB")
        del model, opt
        torch.cuda.empty_cache()
    save("speed", res)


# ---------------------------------------------------------------- mqar
def mqar_batch(n, seq_len, n_kv, vocab, gen):
    """Keys from [1, vocab/2), values from [vocab/2, vocab). Tokens: k1 v1 ... kN vN,
    then each key once at a random position of the rest (0 = filler). Target at
    the position of a repeated key: its value (-100 elsewhere)."""
    half, dev, rest = vocab // 2, gen.device, seq_len - 2 * n_kv
    # distinct keys / query positions per row: first n_kv of a random permutation
    keys = torch.rand(n, half - 1, device=dev, generator=gen).argsort(-1)[:, :n_kv] + 1
    vals = torch.randint(half, vocab, (n, n_kv), device=dev, generator=gen)
    pos = torch.rand(n, rest, device=dev, generator=gen).argsort(-1)[:, :n_kv] + 2 * n_kv
    x = torch.zeros(n, seq_len, dtype=torch.long, device=dev)
    y = torch.full((n, seq_len), -100, dtype=torch.long, device=dev)
    x[:, 0 : 2 * n_kv : 2], x[:, 1 : 2 * n_kv : 2] = keys, vals
    x.scatter_(1, pos, keys)
    y.scatter_(1, pos, vals)
    return x, y


def mqar(steps=3000, batch=64, seq_len=256, dim=128, heads=2, layers=2,
         kvs=(16, 32, 64), lrs=(1e-3, 3e-3), vocab=8192):
    res = []
    for n_kv in kvs:
        for kind in MIXERS:
            best = None
            for lr in lrs:
                model = build(kind, vocab, dim=dim, heads=heads, layers=layers,
                              attn_every=10**9).cuda()  # no attention layers
                opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
                gen = torch.Generator("cuda").manual_seed(0)  # same training data for every run
                for step in range(steps):
                    for g in opt.param_groups:  # warmup 10 %, cosine decay
                        w = min(1.0, (step + 1) / (steps // 10))
                        g["lr"] = lr * w * 0.5 * (1 + math.cos(math.pi * step / steps))
                    x, y = mqar_batch(batch, seq_len, n_kv, vocab, gen)
                    m = y != -100  # logits only at the query positions
                    with torch.autocast("cuda", torch.bfloat16):
                        logits = (model.features(x)[m] @ model.head_weight.T).float()
                    loss = F.cross_entropy(logits, y[m])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                gen_eval = torch.Generator("cuda").manual_seed(12345)
                correct = total = 0
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    for _ in range(8):
                        x, y = mqar_batch(128, seq_len, n_kv, vocab, gen_eval)
                        m = y != -100
                        pred = (model.features(x)[m] @ model.head_weight.T).argmax(-1)
                        correct += (pred == y[m]).sum().item()
                        total += m.sum().item()
                acc = correct / total
                print(f"mqar kv={n_kv:3d} {kind:5s} lr={lr:.0e} | acc {acc:.3f} | "
                      f"params {n_params(model) / 1e6:.2f}M", flush=True)
                if best is None or acc > best["acc"]:
                    best = {"n_kv": n_kv, "mixer": kind, "lr": lr, "acc": acc,
                            "params": n_params(model)}
                del model, opt
            res.append(best)
    save("mqar", {"setup": dict(steps=steps, batch=batch, seq_len=seq_len, dim=dim,
                                heads=heads, layers=layers, lrs=lrs, vocab=vocab),
                  "best_over_lr": res})


# ---------------------------------------------------------------- lm
def lm(steps, seeds):
    from dataset import get_tokenizer
    from metrics import validation_metrics
    from train_utils import next_token_loss, run_loop

    cfg.apply_preset("4gb")
    cfg.run_name = OUT
    cfg.max_steps, cfg.warmup_steps = steps, max(1, steps // 10)
    cfg.eval_interval, cfg.eval_batches = max(1, steps // 6), 32
    cfg.save_interval, cfg.sample_interval = 10**9, 10**9
    vocab = get_tokenizer().vocab_size
    res = {}
    for seed in seeds:
        for kind in MIXERS:
            name = f"hybrid_{kind}_s{seed}"
            model = build(kind, vocab, seed)
            path = os.path.join(cfg.checkpoints_dir, f"{name}.pt")
            run_loop(model, lambda x, length, m=model: next_token_loss(m, x, length), name,
                     path, eval_fn=lambda m: validation_metrics("ar", m), resume=True)
            with open(os.path.join(cfg.logs_dir, f"{name}.jsonl")) as f:
                rows = [json.loads(line) for line in f]
            res[name] = {"params": n_params(model),
                         "curve": [(r["tokens"], r["eval"]["val_ppl"]) for r in rows if "eval" in r],
                         "tokens_per_s": [r["tokens_per_s"] for r in rows if "tokens_per_s" in r][-1]}
            del model
            torch.cuda.empty_cache()
    save(f"lm_{steps}steps", res)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=["check", "speed", "mqar", "lm"])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = p.parse_args()
    if a.part == "lm":
        lm(a.steps, a.seeds)
    else:
        globals()[a.part]()
