import torch

from config import cfg
from dataset import get_tokenizer
from metrics import validation_metrics
from models.transformer.model import LanguageModel
from train_utils import next_token_loss, run_loop


def build_model(vocab_size):
    torch.manual_seed(cfg.seed)
    return LanguageModel(vocab_size, cfg.dim_model, cfg.n_head, cfg.max_seq_len, cfg.n_layers, cfg.tie_embeddings)


def train(save_path, resume=False):
    model = build_model(get_tokenizer().vocab_size)
    return run_loop(model, lambda x: next_token_loss(model, x), "Transformer", save_path,
                    lambda m: validation_metrics("ar", m), resume)
