import os

import torch

from config import cfg
from dataset import gen_len, get_tokenizer
from metrics import validation_metrics
from models.mamba.model import HAS_KERNEL, MambaLanguageModel
from train_utils import (
    SAMPLE_TOKENS,
    next_token_loss,
    print_sample,
    progress_prompt,
    run_loop,
)


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)  # identical seed for every architecture
    return MambaLanguageModel(
        vocab_size=vocab_size,
        dim_model=cfg.dim_model,
        n_layers=cfg.mamba_n_layers,
        max_sequence_len=cfg.max_seq_len,
        d_state=cfg.mamba_d_state,
        expand=cfg.mamba_expand,
        d_conv=cfg.mamba_d_conv,
        tie_embeddings=cfg.tie_embeddings,
        use_kernel=cfg.use_mamba_kernel,
    )


def train(save_path: str = None, resume: bool = False):
    if save_path is None:
        save_path = os.path.join(cfg.checkpoints_dir, "mamba.pt")
    vocab_size = get_tokenizer().vocab_size
    if cfg.device == "cuda" and cfg.use_mamba_kernel and not HAS_KERNEL:
        print("WARNING: mamba-ssm not installed - using the slow, memory-hungry PyTorch scan.")
    model = build_model(vocab_size)

    def on_sample(step):
        prompt, temperature = progress_prompt()
        out = model.generate(prompt, min(gen_len(), SAMPLE_TOKENS), temperature)
        print_sample("Mamba", step, out[0])

    print("Starting Mamba training...")
    return run_loop(
        model, lambda x, length: next_token_loss(model, x, length), "Mamba", save_path,
        resume=resume, eval_fn=lambda m: validation_metrics("ar", m), on_sample=on_sample,
    )


if __name__ == "__main__":
    train()
