"""DeltaBlock: test metrics and self-speculative decoding.

The noisy stream drafts the rest of the current block in one parallel step and
the clean (autoregressive) stream verifies the draft, so the output is that of
plain AR decoding: exact match for greedy decoding, speculative sampling
(Leviathan et al. 2023) at temperature > 0. Reported: tokens per draft-verify
round, i.e. tokens per forward pass when drafting and verification share one
pass (TiDAR, Liu et al. 2025).

    python -m experiments.self_speculation --preset 4gb

Results: results/<run>/deltablock.json.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import PRESETS, cfg
from dataset import eval_batches, get_tokenizer
from metrics import validation_metrics

ARCH_FIELDS = ["dim_model", "n_head", "n_layers", "max_seq_len", "deltablock_mixer",
               "attn_every", "block_size", "intra_block"]


def load_model():
    from models.deltablock.train import build_model

    ck = torch.load(os.path.join(cfg.checkpoints_dir, "deltablock.pt"), map_location=cfg.device)
    for k in ARCH_FIELDS:
        if k in ck["cfg"]:
            setattr(cfg, k, ck["cfg"][k])
    model = build_model(get_tokenizer().vocab_size)
    model.load_state_dict(ck["state_dict"])
    return model.to(cfg.device).eval(), ck["step"]


def draft_logits(model, s):
    """Logits [k, V] of the noisy stream for the k = L - n % L positions left in
    the block of position n = len(s), given the clean blocks before it."""
    L, n = model.block_size, s.size(1)
    b = n // L
    clean = F.pad(s[:, : b * L], (0, L))  # clean block b is invisible to noisy block b
    noisy = torch.full_like(clean, model.mask_token_id)
    noisy[:, b * L : n] = s[:, b * L :]
    h, _ = model.two_stream(noisy, clean)
    return F.linear(h[0, n : (b + 1) * L], model.head_weight)


def pick(logits, temperature):
    if temperature <= 0:
        return logits.argmax(-1)
    return torch.multinomial(F.softmax(logits / temperature, -1), 1)[:, 0]


@torch.no_grad()
def speculate(model, prompt, new_tokens, temperature, draft_temperature):
    """[1, P] -> ([1, P + new_tokens], tokens accepted per round). The draft is
    sampled at draft_temperature (0: argmax, a point-mass proposal); the output
    distribution is that of AR sampling at temperature either way."""
    s, per_round = prompt, []
    while s.size(1) < prompt.size(1) + new_tokens:
        q = draft_logits(model, s)
        d = pick(q, draft_temperature)
        p = model(torch.cat([s, d[None]], 1))[0, s.size(1) - 1 :]  # [k + 1, V]
        if temperature <= 0:
            ok = (d == p[:-1].argmax(-1)).int()
            a = int(ok.cumprod(0).sum())
            nxt = p[a].argmax()
        else:
            pp = F.softmax(p / temperature, -1)
            qq = (F.one_hot(d, q.size(-1)).float() if draft_temperature <= 0
                  else F.softmax(q / draft_temperature, -1))
            idx = torch.arange(d.numel(), device=d.device)
            ratio = (pp[idx, d] / qq[idx, d]).clamp(max=1)
            ok = (torch.rand_like(ratio) < ratio).int()
            a = int(ok.cumprod(0).sum())
            res = pp[a] if a == d.numel() else (pp[a] - qq[a]).clamp(min=0)
            nxt = torch.multinomial(res / res.sum(), 1)[0]
        s = torch.cat([s, d[None, :a], nxt.view(1, 1)], 1)
        per_round.append(a + 1)
    return s[:, : prompt.size(1) + new_tokens], per_round


@torch.no_grad()
def greedy(model, prompt, new_tokens):
    s = prompt
    for _ in range(new_tokens):
        s = torch.cat([s, model(s)[:, -1:].argmax(-1)], 1)
    return s


def prompts(n, length):
    rows = torch.cat([x for x, _ in eval_batches("test", n)])[:n]
    return rows[:, :length]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--ppl-batches", type=int, default=100)
    p.add_argument("--n-mc", type=int, default=8)
    p.add_argument("--n", type=int, default=64, help="prompts")
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--new-tokens", type=int, default=192)
    p.add_argument("--settings", nargs="+", default=["0:0", "1:1", "1:0"],
                   help="temperature:draft_temperature")
    args = p.parse_args()
    if args.preset:
        cfg.apply_preset(args.preset)
    torch.manual_seed(cfg.seed)
    model, step = load_model()
    r = {"trained_steps": step + 1, "params_M": sum(q.numel() for q in model.parameters()) / 1e6}

    with torch.autocast(cfg.device, torch.bfloat16, enabled=cfg.use_bf16):
        ar = validation_metrics("ar", model, "test", args.ppl_batches)
        diff = validation_metrics("diffusion", model, "test", args.ppl_batches, args.n_mc)
    r.update(test_nll=ar["val_nll"], test_ppl=ar["val_ppl"],
             test_nll_diff=diff["val_nll"], test_ppl_diff=diff["val_ppl"])
    print({k: round(v, 3) for k, v in r.items()}, flush=True)

    prompt_rows = prompts(args.n, args.prompt_len).to(cfg.device)
    r["speculation"] = {"prompt_len": args.prompt_len, "new_tokens": args.new_tokens,
                        "n": args.n, "block_size": model.block_size}
    for setting in args.settings:
        temp, draft_temp = map(float, setting.split(":"))
        rounds, outs, t0 = [], [], time.perf_counter()
        for row in prompt_rows:
            out, per_round = speculate(model, row[None], args.new_tokens, temp, draft_temp)
            rounds += per_round
            outs.append(out)
        g = {"tokens_per_round": args.new_tokens * args.n / len(rounds),
             "rounds_hist": np.bincount(rounds, minlength=model.block_size + 2)[1:].tolist(),
             "seconds": time.perf_counter() - t0}
        if temp <= 0:  # lossless: identical to token-by-token greedy decoding
            same = [(o == greedy(model, row[None], args.new_tokens)).all().item()
                    for o, row in zip(outs, prompt_rows)]
            g["identical_to_greedy"] = sum(same) / len(same)
        r["speculation"][f"T={temp:g}, draft T={draft_temp:g}"] = g
        print(f"T={temp:g}, draft T={draft_temp:g}: {g}", flush=True)

    tok = get_tokenizer()
    out_dir = cfg.results_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "deltablock.json"), "w") as f:
        json.dump(r, f, indent=2)
    print(tok.decode(outs[0][0].tolist()))
    print(f"-> {os.path.join(out_dir, 'deltablock.json')}")


if __name__ == "__main__":
    main()
