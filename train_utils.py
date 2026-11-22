import json
import math
import os
import time

import torch

import distributed
import tracking
from config import cfg
from dataset import train_batches
from losses import weighted_ce


def make_optimizer(model):
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=cfg.device == "cuda"
    )


def next_token_loss(model, x):
    hidden = model.features(x[:, :-1]).reshape(-1, model.head_weight.size(1))
    weight = hidden.new_full((hidden.size(0),), 1.0 / hidden.size(0), dtype=torch.float32)
    return weighted_ce(hidden, model.head_weight, x[:, 1:].reshape(-1), weight, cfg.loss_chunk)


def save_checkpoint(path, **state):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def decay_start() -> int:
    return int(cfg.max_steps * (1 - cfg.decay_fraction))


def lr_at(step: int) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if cfg.lr_schedule == "wsd":
        decay = (step - decay_start()) / max(1, cfg.max_steps - decay_start())
        return cfg.learning_rate * (1 - (1 - cfg.min_lr_ratio) * min(1.0, max(0.0, decay)))
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def run_loop(model, loss_fn, name, save_path, eval_fn, resume=False, step_hook=None):
    key = name.lower()
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    latest_path = os.path.join(cfg.checkpoints_dir, f"{key}_latest.pt")
    log_path = os.path.join(cfg.logs_dir, f"{key}.jsonl")

    rank, world, main = distributed.rank(), distributed.world(), distributed.is_main()
    per_step = cfg.batch_size * cfg.grad_accum_steps
    assert per_step % world == 0, f"{per_step} sequences per step do not split over {world} GPUs"
    local = per_step // world
    micro = min(cfg.micro_batch.get(key, cfg.batch_size), local)
    accum = local // micro
    tokens_per_step = per_step * cfg.max_seq_len

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.device} x {world} | {n_params / 1e6:.1f}M params | micro-batch {micro} x {accum}")
    model.to(cfg.device).train()
    if world > 1:
        torch.manual_seed(cfg.seed + 1 + rank)

    if resume and os.path.exists(save_path):
        ck = torch.load(save_path, map_location=cfg.device)
        if ck["step"] >= ck["cfg"]["max_steps"] - 1:
            print(f"[{name}] already trained ({save_path})")
            model.load_state_dict(ck["state_dict"])
            return model

    optimizer = make_optimizer(model)
    start, elapsed = 0, 0.0
    if resume and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=cfg.device)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        start, elapsed = ck["step"] + 1, ck.get("elapsed_s", 0.0)
        for k, v in ck.get("plan", {}).items():
            setattr(cfg, k, v)
        print(f"[{name}] resumed from step {ck['step']}")
        del ck
    elif os.path.exists(log_path) and main:
        os.remove(log_path)
    if main:
        tracking.start(key)

    def log(record):
        if main:
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            tracking.log(record)

    def sync():
        if cfg.device == "cuda":
            torch.cuda.synchronize()

    use_amp = cfg.use_bf16 and cfg.device == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    batches = train_batches(start, micro, rank, world)
    session_start, t0 = elapsed, time.time()

    def clock():
        return elapsed + time.time() - t0

    def off_the_clock(fn, step):
        nonlocal elapsed, t0
        sync()
        elapsed = clock()
        fn(step)
        sync()
        t0 = time.time()

    def run_eval(step):
        if not main:
            return
        model.eval()
        with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=use_amp):
            metrics = eval_fn(model)
        model.train()
        print(f"[{name}] eval @ {step} | " + " | ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        log({"step": step, "eval": metrics, "tokens": step * tokens_per_step, "elapsed_s": elapsed})

    def save_latest(step):
        if main:
            save_checkpoint(latest_path, step=step, state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(), elapsed_s=elapsed,
                            plan={"max_steps": cfg.max_steps, "joint_start": cfg.joint_start})

    def accumulate_gradients():
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            with torch.autocast(cfg.device, torch.bfloat16, enabled=use_amp):
                loss = loss_fn(next(batches)) / accum
            loss.backward()
            total = total + loss.detach()
        return total

    step, done = start - 1, start - 1
    try:
        while step + 1 < cfg.max_steps:
            step += 1
            if step_hook is not None:
                step_hook(step)
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            while True:
                try:
                    total = accumulate_gradients()
                    break
                except torch.OutOfMemoryError:
                    if micro == 1:
                        raise
                micro = max(m for m in range(1, micro // 2 + 1) if local % m == 0)
                accum = local // micro
                batches = train_batches(step, micro, rank, world)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[{name}] out of memory - micro-batch {micro}")
            distributed.average_gradients(model.parameters())
            total = distributed.mean(total)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            done = step

            if step % cfg.log_interval == 0:
                total = float(total)
                if not math.isfinite(total):
                    raise RuntimeError(f"[{name}] loss is {total} at step {step}")
                now = clock()
                tps = (step - start + 1) * tokens_per_step / max(now - session_start, 1e-9)
                eta = (cfg.max_steps - step - 1) * tokens_per_step / tps
                print(f"[{name}] step {step} | loss {total:.4f} | {tps / 1e3:.1f}k tok/s | ETA {eta / 3600:.1f} h")
                log({"step": step, "loss": total, "lr": lr_at(step),
                     "grad_norm": float(gnorm), "tokens_per_s": tps, "elapsed_s": now})

            if step > 0 and step % cfg.eval_interval == 0:
                off_the_clock(run_eval, step)
            if step > 0 and step % cfg.save_interval == 0:
                off_the_clock(save_latest, step)
        off_the_clock(run_eval, cfg.max_steps)
    except BaseException:
        if done >= start:
            elapsed = clock()
            save_latest(done)
        raise

    if main:
        save_checkpoint(save_path, state_dict=model.state_dict(), cfg=vars(cfg), step=step, elapsed_s=elapsed)
        tracking.finish()
    print(f"[{name}] saved {save_path}")
    return model
