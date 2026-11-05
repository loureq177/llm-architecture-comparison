import math

import torch

from config import cfg
from dataset import eval_batches
from losses import token_nll


@torch.no_grad()
def _ar_nll(model, x):
    hidden = model.features(x[:, :-1])
    nll = token_nll(hidden.reshape(-1, hidden.size(-1)), model.head_weight, x[:, 1:].reshape(-1))
    return nll.sum().double(), nll.numel()


@torch.no_grad()
def validation_metrics(kind, model, split="validation", n_batches=None, n_mc=4):
    nll, n = 0.0, 0
    for x in eval_batches(split, n_batches or cfg.eval_batches):
        s, c = model.elbo_nll(x, start=1, n_mc=n_mc) if kind == "diffusion" else _ar_nll(model, x)
        nll, n = nll + float(s), n + int(c)
    return {"val_nll": nll / n, "val_ppl": math.exp(nll / n)}
