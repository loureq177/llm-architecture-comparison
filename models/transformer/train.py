import os

import torch

from config import cfg
from dataset import gen_len, get_tokenizer
from metrics import validation_metrics
from models.transformer.model import LanguageModel
from train_utils import (
    SAMPLE_TOKENS,
    next_token_loss,
    print_sample,
    progress_prompt,
    run_loop,
)


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)  # identical seed for every architecture
    return LanguageModel(
        vocab_size=vocab_size,
        dim_model=cfg.dim_model,
        n_head=cfg.n_head,
        max_sequence_len=cfg.max_seq_len,
        n_layers=cfg.n_layers,
        tie_embeddings=cfg.tie_embeddings,
    )


def train(save_path: str = None, resume: bool = False):
    if save_path is None:
        save_path = os.path.join(cfg.checkpoints_dir, "transformer.pt")
    vocab_size = get_tokenizer().vocab_size
    model = build_model(vocab_size)

    def on_sample(step):
        prompt, temperature = progress_prompt()
        out = model.generate(prompt, min(gen_len(), SAMPLE_TOKENS), temperature)
        print_sample("Transformer", step, out[0])

    print("Starting Transformer training...")
    return run_loop(
        model, lambda x, length: next_token_loss(model, x, length), "Transformer", save_path,
        resume=resume, eval_fn=lambda m: validation_metrics("ar", m), on_sample=on_sample,
    )


if __name__ == "__main__":
    train()
