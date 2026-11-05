import argparse
import gc
import os
import sys
import time

import torch

import distributed
from config import PRESETS, cfg
from dataset import load_split
from models.deltablock.train import train as train_deltablock
from models.diffusion.train import train as train_diffusion
from models.mamba.train import train as train_mamba
from models.transformer.train import train as train_transformer
from prepare_data import ensure_data

STAGES = {
    "transformer": train_transformer,
    "diffusion": train_diffusion,
    "mamba": train_mamba,
    "deltablock": train_deltablock,
}
MAX_ATTEMPTS = 3


def check_single_epoch(allow_repeats=False):
    available = load_split("train").n
    needed = cfg.max_steps * cfg.batch_size * cfg.grad_accum_steps
    print(f"training windows: {needed} of {available} ({needed / available:.2f} epochs)")
    if needed > available and not allow_repeats:
        sys.exit(f"{needed / available:.2f} epochs needed: increase train_shards or pass --allow-repeats")


def run(models, resume=False, allow_repeats=False):
    if distributed.world() == 1:
        ensure_data()
    check_single_epoch(allow_repeats)
    for key in models:
        path = os.path.join(cfg.checkpoints_dir, f"{key}.pt")
        t0 = time.time()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                STAGES[key](save_path=path, resume=resume or attempt > 1)
                break
            except Exception as e:
                if distributed.world() > 1 or attempt == MAX_ATTEMPTS:
                    raise
                print(f"[{key}] attempt {attempt} failed: {e}", file=sys.stderr)
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(30)
        print(f"[{key}] done in {(time.time() - t0) / 3600:.1f} h")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    distributed.init()
    if not distributed.is_main():
        sys.stdout = open(os.devnull, "w")
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default=None)
    p.add_argument("--models", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--deadline", type=float, default=None, help="unix time at which training ends")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-repeats", action="store_true")
    args = p.parse_args()
    if args.preset:
        cfg.apply_preset(args.preset)
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.deadline:
        assert cfg.lr_schedule == "wsd", "a deadline needs the WSD schedule"
        cfg.deadline = args.deadline
    found = [os.path.join(cfg.checkpoints_dir, f"{k}{s}.pt") for k in args.models for s in ("", "_latest")]
    found = [f for f in found if os.path.exists(f)]
    if found and not (args.resume or args.overwrite):
        sys.exit("checkpoints exist (pass --resume or --overwrite):\n  " + "\n  ".join(found))
    if args.overwrite and not args.resume and distributed.is_main():
        for f in found:
            os.remove(f)
    run(args.models, args.resume, args.allow_repeats)
    distributed.close()
