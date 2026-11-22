"""Weights & Biases logging, on when WANDB_API_KEY is set (off otherwise).

One W&B run per training run (logs/<run>/<model>.wandb_id), so a resubmitted
job continues the same curves and evaluate.py adds its results to it.
"""

import os
import uuid

from config import cfg

_run = None


def _id_path(key):
    return os.path.join(cfg.logs_dir, f"{key}.wandb_id")


def enabled():
    return bool(os.environ.get("WANDB_API_KEY"))


def start(key, fresh=False):
    global _run
    if not enabled():
        return
    import wandb

    path = _id_path(key)
    if fresh and os.path.exists(path):
        os.remove(path)
    if os.path.exists(path):
        with open(path) as f:
            run_id = f.read().strip()
    else:
        run_id = uuid.uuid4().hex[:12]
        os.makedirs(cfg.logs_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(run_id)
    _run = wandb.init(project=os.environ.get("WANDB_PROJECT", "llm-architecture-comparison"),
                      name=f"{cfg.run_name}/{key}", id=run_id, resume="allow",
                      config={k: v for k, v in vars(cfg).items() if not k.startswith("_")})


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = v
    return out


def log(record):
    if _run is not None:
        r = dict(record)
        step = r.pop("step")
        _run.log(_flat(r), step=step)


def summary(d, prefix=""):
    if _run is not None:
        _run.summary.update(_flat(d, prefix))


def finish():
    global _run
    if _run is not None:
        _run.finish()
        _run = None
