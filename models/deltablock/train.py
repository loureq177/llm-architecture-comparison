import time

import torch

import distributed
from config import cfg
from dataset import get_tokenizer, load_split
from metrics import validation_metrics
from models.deltablock.model import HybridLM
from train_utils import decay_start, next_token_loss, run_loop

JOINT_COST = 2.0


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)
    return HybridLM(vocab_size, cfg.dim_model, cfg.n_head, cfg.n_layers, cfg.attn_every, cfg.block_size)


def joint_from() -> int:
    return cfg.joint_start if cfg.joint_start is not None else int(cfg.ar_fraction * cfg.max_steps)


class Deadline:
    def __init__(self):
        self.max_by_data = load_split("train").n // (cfg.batch_size * cfg.grad_accum_steps)
        self.marks = {}

    def __call__(self, step):
        if step % cfg.log_interval:
            return
        now, joint = time.time(), step >= joint_from()
        marks = self.marks.setdefault(joint, [])
        if len(marks) < 2:
            marks.append((step, now))
        sec = {k: (now - m[1][1]) / (step - m[1][0]) for k, m in self.marks.items()
               if len(m) == 2 and step > m[1][0]}
        left = cfg.deadline - now
        if left <= 0:
            n = step + 1
        elif joint in sec and step < decay_start():
            if joint:
                n = step + left / sec[True]
            else:
                t_ar, a = sec[False], cfg.ar_fraction
                n = (left + step * t_ar) / (a * t_ar + (1 - a) * sec.get(True, JOINT_COST * t_ar))
        else:
            n = cfg.max_steps
        n = int(distributed.broadcast(min(n, self.max_by_data)))
        if abs(n - cfg.max_steps) > 0.005 * cfg.max_steps:
            print(f"[DeltaBlock] deadline: {n} steps")
        cfg.max_steps = max(n, step + 1)


def train(save_path, resume=False):
    model = build_model(get_tokenizer().vocab_size)
    plan = Deadline() if cfg.deadline else None

    def hook(step):
        if cfg.joint_start is None and step >= joint_from():
            cfg.joint_start = step
        if plan is not None:
            plan(step)

    def loss_fn(x):
        if cfg.joint_start is None:
            return next_token_loss(model, x)
        return model.joint_loss(x, cfg.diff_weight, cfg.diff_eps, cfg.loss_chunk)

    def eval_fn(m):
        out = validation_metrics("ar", m)
        if cfg.joint_start is not None:
            d = validation_metrics("diffusion", m)
            out.update(val_nll_diff=d["val_nll"], val_ppl_diff=d["val_ppl"])
        return out

    return run_loop(model, loss_fn, "DeltaBlock", save_path, eval_fn, resume, hook)
