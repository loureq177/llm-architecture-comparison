import argparse
import gc
import os
import sys
import time
from datetime import datetime

import torch

from config import PRESETS, cfg
from dataset import load_split
from models.deltablock.train import train as train_deltablock
from models.diffusion.train import train as train_diffusion
from models.mamba.train import train as train_mamba
from models.transformer.train import train as train_transformer
from prepare_data import ensure_data

STAGES = {
    "transformer": ("Causal Transformer", train_transformer),
    "diffusion": ("Masked Diffusion LM (MDLM)", train_diffusion),
    "mamba": ("Mamba SSM", train_mamba),
    "deltablock": ("DeltaBlock", train_deltablock),
}


MAX_ATTEMPTS = 3  # a long unattended run should survive a transient failure


def format_duration(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}h {mins:02d}m {secs:02d}s"


def existing_checkpoints(models):
    paths = [os.path.join(cfg.checkpoints_dir, f"{k}{suffix}.pt") for k in models for suffix in ("", "_latest")]
    return [p for p in paths if os.path.exists(p)]


def check_single_epoch(allow_repeats=False):
    """Every training sequence is seen at most once: the validation loss then
    follows the training loss and the models cannot memorise their data."""
    available = load_split("train").n
    needed = cfg.max_steps * cfg.batch_size * cfg.grad_accum_steps
    print(f"Training windows: {needed} of {available} ({needed / available:.2f} epochs)")
    if needed > available and not allow_repeats:
        sys.exit(f"The budget needs {needed / available:.2f} passes over the training data. "
                 f"Increase train_shards or pass --allow-repeats.")


def run_training_suite(models, resume=False, allow_repeats=False):
    ensure_data()
    check_single_epoch(allow_repeats)
    start_total = time.time()
    print("=" * 70)
    print(f"BENCHMARK TRAINING SUITE STARTED AT {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Device: {cfg.device} | steps per model: {cfg.max_steps}")
    print(
        f"Tokens per model: {cfg.max_steps * cfg.batch_size * cfg.grad_accum_steps * cfg.max_seq_len / 1e6:.0f}M"
    )
    print(f"Checkpoints directory: {os.path.abspath(cfg.checkpoints_dir)}")
    print("=" * 70)

    results = []
    for idx, key in enumerate(models, 1):
        name, train_fn = STAGES[key]
        target_path = os.path.join(cfg.checkpoints_dir, f"{key}.pt")
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(models)}] TRAINING {name.upper()} -> {target_path}")
        print("=" * 70)
        t0 = time.time()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                # a retry continues from the checkpoint the failed attempt left behind
                train_fn(save_path=target_path, resume=resume or attempt > 1)
                status = "SUCCESS"
                break
            except Exception as e:
                print(f"\n[ERROR] {name}, attempt {attempt}/{MAX_ATTEMPTS}: {e}", file=sys.stderr)
                status = f"FAILED: {e}"
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(30)
        results.append((name, status, format_duration(time.time() - t0)))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70 + "\nALL RUNS FINISHED - SUMMARY\n" + "=" * 70)
    for name, status, dur in results:
        print(f"- {name:<28} | {status:<10} | {dur}")
    print(f"Total time elapsed: {format_duration(time.time() - start_total)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # live progress even when piped to a file
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--models", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="continue interrupted runs and skip the models that are already trained")
    p.add_argument("--overwrite", action="store_true", help="discard existing checkpoints")
    p.add_argument("--allow-repeats", action="store_true",
                   help="allow a budget of more than one pass over the training data")
    args = p.parse_args()
    if args.preset:
        cfg.apply_preset(args.preset)
    if args.max_steps:
        cfg.max_steps = args.max_steps
    found = existing_checkpoints(args.models)
    if found and not (args.resume or args.overwrite):
        sys.exit("Checkpoints already exist:\n  " + "\n  ".join(found)
                 + "\nPass --resume to continue them or --overwrite to train from scratch.")
    if args.overwrite and not args.resume:
        for path in found:
            os.remove(path)
    run_training_suite(args.models, args.resume, args.allow_repeats)
