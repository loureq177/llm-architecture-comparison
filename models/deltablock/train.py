import os

import torch

from config import cfg
from dataset import get_tokenizer
from metrics import validation_metrics
from models.deltablock.model import HybridLM
from train_utils import next_token_loss, run_loop


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)  # identical seed for every architecture
    return HybridLM(
        vocab_size, cfg.dim_model, cfg.n_head, cfg.n_layers, mixer=cfg.deltablock_mixer,
        attn_every=cfg.attn_every, block_size=cfg.block_size, intra_block=cfg.intra_block,
    )


def joint_from() -> int:
    """First step of the joint stage; before it only the next-token loss."""
    return int(cfg.ar_fraction * cfg.max_steps)


def train(save_path: str = None, resume: bool = False):
    if save_path is None:
        save_path = os.path.join(cfg.checkpoints_dir, "deltablock.pt")
    model = build_model(get_tokenizer().vocab_size)
    state = {"step": 0}

    def loss_fn(x, length):
        if state["step"] < joint_from():
            return next_token_loss(model, x, length)
        return model.joint_loss(x, cfg.diff_weight, cfg.diff_eps, cfg.loss_chunk)

    def eval_fn(m):
        out = validation_metrics("ar", m)
        if state["step"] >= joint_from():
            d = validation_metrics("diffusion", m)
            out.update(val_nll_diff=d["val_nll"], val_ppl_diff=d["val_ppl"])
        return out

    print(f"Starting DeltaBlock training (joint stage from step {joint_from()})...")
    return run_loop(
        model, loss_fn, "DeltaBlock", save_path, resume=resume, eval_fn=eval_fn,
        step_hook=lambda s: state.update(step=s),
    )


if __name__ == "__main__":
    train()
