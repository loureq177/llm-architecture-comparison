"""Validation perplexity for all three architectures.

Autoregressive models: exact perplexity exp(mean NLL per token).
Diffusion (MDLM):      perplexity *upper bound* from the ELBO (>= true ppl).
Reported over the same tokens for every architecture: all tokens after the
first one, i.e. p(x[1:] | x[0]) (a next-token model cannot score the first
token of a window).
"""

import math

import torch

from config import cfg
from dataset import eval_batches, prompt_len
from losses import token_nll


@torch.no_grad()
def _ar_nll(model, x, length):
    """(NLL sum, token count) of a next-token model over the real target positions."""
    hidden = model.features(x[:, :-1])
    nll = token_nll(
        hidden.reshape(-1, hidden.size(-1)), model.head_weight, x[:, 1:].reshape(-1)
    ).view(x.size(0), -1)
    pos = torch.arange(1, x.size(1), device=x.device)[None]  # target positions
    valid = pos < length[:, None]
    return (nll * valid).sum().double(), valid.sum()


@torch.no_grad()
def validation_metrics(kind, model, split="validation", n_batches=None, n_mc=4):
    """kind: "diffusion" or "ar". n_mc: Monte-Carlo samples of the diffusion ELBO."""
    n_batches = n_batches or cfg.eval_batches
    acc = torch.zeros(2, dtype=torch.float64, device=cfg.device)
    for x, length in eval_batches(split, n_batches):
        if kind == "diffusion":
            vals = model.elbo_nll(x, length, prompt_len(), n_mc)
        else:
            vals = _ar_nll(model, x, length)
        acc += torch.stack([v.double() for v in vals])
    nll, n = acc.tolist()
    return {"val_nll": nll / n, "val_ppl": math.exp(nll / n)}
