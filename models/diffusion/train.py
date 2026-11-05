import os

import torch

from config import cfg
from dataset import gen_len, get_tokenizer
from metrics import validation_metrics
from models.diffusion.model import DiffusionLanguageModel
from train_utils import print_sample, progress_prompt, run_loop


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)  # identical seed for every architecture
    return DiffusionLanguageModel(
        vocab_size=vocab_size,
        dim_model=cfg.dim_model,
        n_head=cfg.n_head,
        max_sequence_len=cfg.max_seq_len,
        n_layers=cfg.n_layers,
        tie_embeddings=cfg.tie_embeddings,
    )


def train(save_path: str = None, resume: bool = False):
    if save_path is None:
        save_path = os.path.join(cfg.checkpoints_dir, "diffusion.pt")
    vocab_size = get_tokenizer().vocab_size
    model = build_model(vocab_size)

    def loss_fn(x, _length):
        return model.training_loss(x, cfg.diff_eps, cfg.loss_chunk)

    def on_sample(step):
        prompt, temperature = progress_prompt()
        out = model.sample(prompt, gen_len(), steps=64, temperature=temperature)
        print_sample("Diffusion", step, out[0])

    print("Starting Diffusion (MDLM) training...")
    return run_loop(
        model, loss_fn, "Diffusion", save_path, resume=resume,
        eval_fn=lambda m: validation_metrics("diffusion", m), on_sample=on_sample,
    )


if __name__ == "__main__":
    train()
