import json
import math
import os
import time

import torch

from config import cfg
from dataset import get_tokenizer, load_meta, train_batches
from losses import weighted_ce


def count_params(model) -> int:
    """Unique parameters (tied weights counted once)."""
    return sum(p.numel() for p in model.parameters())


def make_optimizer(model):
    """AdamW; weight decay only on matrices (not norms, biases, A_log, D)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=cfg.device == "cuda"
    )


def next_token_loss(model, x, length):
    """Mean next-token cross-entropy over the real (non-padding) targets."""
    hidden = model.features(x[:, :-1])
    pos = torch.arange(1, x.size(1), device=x.device)[None]
    weight = (pos < length[:, None]).float()
    weight = (weight / weight.sum()).reshape(-1)
    return weighted_ce(
        hidden.reshape(-1, hidden.size(-1)), model.head_weight, x[:, 1:].reshape(-1),
        weight, cfg.loss_chunk,
    )


SAMPLE_TOKENS = 128  # length of the progress samples printed during training


def progress_prompt():
    """(prompt [1, 1], temperature) for the unconditional samples printed during
    training: EOS marks the start of a document."""
    return torch.full((1, 1), load_meta()["eos_id"], device=cfg.device), 1.0


def print_sample(name, step, tokens):
    print(f"\n--- [{name}] Sample (iter {step}) ---")
    print(get_tokenizer().decode(tokens[:SAMPLE_TOKENS].tolist(), skip_special_tokens=True))
    print("----------------------------------------\n")


def save_checkpoint(path, **state):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)  # never leaves a half-written checkpoint behind


def lr_at(step: int) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * learning_rate."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def run_loop(model, loss_fn, name, save_path=None, eval_fn=None, on_sample=None, resume=False,
             step_hook=None):
    """Shared training loop.

    loss_fn(x, length) -> scalar loss for one micro-batch.
    eval_fn(model)     -> dict of validation metrics (logged every eval_interval).
    step_hook(step)    -> called before every optimizer step (e.g. a training stage switch).
    Logs "[Name] Iteration N | Loss: x" to stdout and JSON lines to logs/<name>.jsonl.

    Two checkpoints are written: <name>_latest.pt (with the optimizer, every
    save_interval steps and when interrupted) and the final <name>.pt. With
    resume=True the run continues from the former, or is skipped altogether if
    the latter is already complete.
    """
    key = name.lower()
    if save_path is None:
        save_path = os.path.join(cfg.checkpoints_dir, f"{key}.pt")
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)
    latest_path = os.path.join(cfg.checkpoints_dir, f"{key}_latest.pt")
    log_path = os.path.join(cfg.logs_dir, f"{key}.jsonl")

    micro = cfg.micro_batch.get(key, cfg.batch_size)
    per_step = cfg.batch_size * cfg.grad_accum_steps  # sequences per optimizer step
    assert per_step % micro == 0, f"micro-batch {micro} must divide {per_step}"
    accum = per_step // micro
    tokens_per_step = per_step * cfg.max_seq_len

    print(f"Using device: {cfg.device} | params: {count_params(model) / 1e6:.1f}M | "
          f"micro-batch {micro} x {accum} accumulation steps")
    model.to(cfg.device).train()

    if resume and os.path.exists(save_path):
        ck = torch.load(save_path, map_location=cfg.device)
        if ck.get("step", -1) >= cfg.max_steps - 1:
            print(f"[{name}] Already trained for {cfg.max_steps} steps - skipping ({save_path})")
            model.load_state_dict(ck["state_dict"])
            return model
        del ck

    optimizer = make_optimizer(model)
    start, elapsed = 0, 0.0  # elapsed = pure training time in seconds, across restarts
    if resume and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=cfg.device)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        start, elapsed = ck["step"] + 1, ck.get("elapsed_s", 0.0)
        print(f"[{name}] Resumed from step {ck['step']}")
        del ck
    elif os.path.exists(log_path):
        os.remove(log_path)

    def log(record):
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def sync():
        if cfg.device == "cuda":
            torch.cuda.synchronize()

    use_amp = cfg.use_bf16 and cfg.device == "cuda"
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    batches = train_batches(start, micro)  # same data order for every model
    session_start = elapsed
    t0 = time.time()

    def clock():
        return elapsed + time.time() - t0

    def off_the_clock(fn, step):
        """Runs evaluation / sampling / checkpointing without counting its time."""
        nonlocal elapsed, t0
        sync()
        elapsed = clock()
        fn(step)
        sync()
        t0 = time.time()

    def run_eval(step):
        if eval_fn is None:
            return
        model.eval()
        with torch.no_grad(), torch.autocast(cfg.device, torch.bfloat16, enabled=use_amp):
            metrics = eval_fn(model)
        model.train()
        print(f"[{name}] Eval @ {step} | " + " | ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        log({"step": step, "eval": metrics, "tokens": step * tokens_per_step, "elapsed_s": elapsed})

    def save_latest(step):
        save_checkpoint(latest_path, step=step, state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(), elapsed_s=elapsed)
        print(f"[{name}] Checkpoint saved: {latest_path}")

    def accumulate_gradients():
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            x, length = next(batches)
            with torch.autocast(cfg.device, torch.bfloat16, enabled=use_amp):
                loss = loss_fn(x, length) / accum
            loss.backward()
            total = total + loss.detach()  # no .item(): keeps the GPU queue full
        return total

    step, done = start - 1, start - 1  # done = last step whose update was applied
    try:
        for step in range(start, cfg.max_steps):
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
                # Something else took GPU memory: redo the step with half the
                # micro-batch (same sequences, twice as many accumulation steps).
                micro = max(m for m in range(1, micro // 2 + 1) if per_step % m == 0)
                accum = per_step // micro
                batches = train_batches(step, micro)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[{name}] Out of GPU memory - continuing with micro-batch {micro}")
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
                print(f"[{name}] Iteration {step} | Loss: {total:.4f} | "
                      f"{tps / 1e3:.1f}k tok/s | ETA {eta / 3600:.1f} h")
                log({"step": step, "loss": total, "lr": lr_at(step),
                     "grad_norm": float(gnorm), "tokens_per_s": tps, "elapsed_s": now})

            if step > 0 and step % cfg.eval_interval == 0:
                off_the_clock(run_eval, step)
            if on_sample is not None and step > 0 and step % cfg.sample_interval == 0:
                off_the_clock(on_sample, step)
            if step > 0 and step % cfg.save_interval == 0:
                off_the_clock(save_latest, step)
        off_the_clock(run_eval, cfg.max_steps)
    except BaseException:
        if done >= start:  # keep the progress of an interrupted or crashed run
            elapsed = clock()
            save_latest(done)
        raise

    save_checkpoint(save_path, state_dict=model.state_dict(), cfg=vars(cfg), step=step,
                    elapsed_s=elapsed)
    print(f"[{name}] Final model saved to: {save_path}")
    return model
