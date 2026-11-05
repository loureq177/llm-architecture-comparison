import torch

from config import cfg
from dataset import get_tokenizer
from metrics import validation_metrics
from models.mamba.model import MambaLanguageModel
from train_utils import next_token_loss, run_loop


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)
    return MambaLanguageModel(vocab_size, cfg.dim_model, cfg.mamba_n_layers, cfg.mamba_d_state,
                              cfg.mamba_expand, cfg.mamba_d_conv, cfg.tie_embeddings, cfg.use_mamba_kernel)


def train(save_path, resume=False):
    model = build_model(get_tokenizer().vocab_size)
    return run_loop(model, lambda x: next_token_loss(model, x), "Mamba", save_path,
                    lambda m: validation_metrics("ar", m), resume)
