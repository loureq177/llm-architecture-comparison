import torch

from config import cfg
from dataset import get_tokenizer
from metrics import validation_metrics
from models.diffusion.model import DiffusionLanguageModel
from train_utils import run_loop


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)
    return DiffusionLanguageModel(vocab_size, cfg.dim_model, cfg.n_head, cfg.max_seq_len, cfg.n_layers)


def train(save_path, resume=False):
    model = build_model(get_tokenizer().vocab_size)
    return run_loop(model, lambda x: model.training_loss(x, cfg.diff_eps, cfg.loss_chunk), "Diffusion",
                    save_path, lambda m: validation_metrics("diffusion", m), resume)
