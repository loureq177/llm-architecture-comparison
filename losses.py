"""Cross-entropy over a large vocabulary without materialising all logits.

With a 50k vocabulary the [tokens, vocab] logits and their gradients need far
more GPU memory than the network itself. Both helpers apply the LM head to
slices of `chunk` tokens, so the peak memory of the loss no longer grows with
the batch and the micro-batch can be as large as the network body allows.
"""

import torch
import torch.nn.functional as F


def _compute_dtype(hidden):
    """dtype of the matmuls: the autocast dtype if mixed precision is on."""
    kind = hidden.device.type
    return torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else hidden.dtype


def _log_probs(hidden, weight):
    logits = hidden @ weight.T
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()  # the softmax needs full precision
    return F.log_softmax(logits, dim=-1)


class _WeightedCE(torch.autograd.Function):
    """sum_i w_i * CE(hidden_i @ weight^T, target_i).

    The gradients w.r.t. `hidden` and `weight` are computed in closed form
    (softmax - one_hot) slice by slice during the forward pass, so the logits of
    a slice are freed before the next slice is evaluated.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, token_w, chunk):
        dtype = _compute_dtype(hidden)
        w_cast = weight.to(dtype)  # cast once, not in every matmul
        grad_h = torch.empty_like(hidden)
        grad_w = torch.zeros_like(weight)  # accumulated in full precision
        loss = 0.0
        for s in range(0, hidden.size(0), chunk):
            h, t = hidden[s : s + chunk].to(dtype), targets[s : s + chunk, None]
            logp = _log_probs(h, w_cast)
            w = token_w[s : s + chunk, None].to(logp.dtype)
            loss = loss - (logp.gather(1, t) * w).sum()
            d = logp.exp_()  # softmax ...
            d.scatter_add_(1, t, torch.full_like(w, -1.0))  # ... minus one-hot
            d = d.mul_(w).to(dtype)  # d loss / d logits
            grad_h[s : s + chunk] = d @ w_cast
            grad_w += d.T @ h
        ctx.save_for_backward(grad_h, grad_w)
        return loss if torch.is_tensor(loss) else hidden.new_zeros((), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        return grad_h * grad_out.to(grad_h.dtype), grad_w * grad_out.to(grad_w.dtype), None, None, None


def weighted_ce(hidden, weight, targets, token_w, chunk=512):
    """Scalar sum_i token_w[i] * CE_i that back-propagates into `hidden` and `weight`.

    hidden [N, D], weight [V, D] (LM head), targets [N], token_w [N]. Put the
    normalisation into `token_w` (e.g. 1/N for a mean, 0 for ignored tokens).
    """
    return _WeightedCE.apply(hidden, weight, targets, token_w, chunk)


@torch.no_grad()
def token_nll(hidden, weight, targets, chunk=512):
    """Per-token negative log-likelihood [N] in nats (evaluation, no gradient)."""
    dtype = _compute_dtype(hidden)
    w_cast = weight.to(dtype)
    out = [
        -_log_probs(hidden[s : s + chunk].to(dtype), w_cast).gather(1, targets[s : s + chunk, None])[:, 0]
        for s in range(0, hidden.size(0), chunk)
    ]
    return torch.cat(out) if out else hidden.new_zeros(0, dtype=torch.float32)
