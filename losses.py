import torch
import torch.nn.functional as F


def _compute_dtype(hidden):
    kind = hidden.device.type
    return torch.get_autocast_dtype(kind) if torch.is_autocast_enabled(kind) else hidden.dtype


def _log_probs(hidden, weight):
    logits = hidden @ weight.T
    if logits.dtype in (torch.bfloat16, torch.float16):
        logits = logits.float()
    return F.log_softmax(logits, dim=-1)


class _WeightedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, token_w, chunk):
        dtype = _compute_dtype(hidden)
        w_cast = weight.to(dtype)
        grad_h = torch.empty_like(hidden)
        grad_w = torch.zeros_like(weight)
        loss = 0.0
        for s in range(0, hidden.size(0), chunk):
            h, t = hidden[s : s + chunk].to(dtype), targets[s : s + chunk, None]
            logp = _log_probs(h, w_cast)
            w = token_w[s : s + chunk, None].to(logp.dtype)
            loss = loss - (logp.gather(1, t) * w).sum()
            d = logp.exp_()
            d.scatter_add_(1, t, torch.full_like(w, -1.0))
            d = d.mul_(w).to(dtype)
            grad_h[s : s + chunk] = d @ w_cast
            grad_w += d.T @ h
        ctx.save_for_backward(grad_h, grad_w)
        return loss if torch.is_tensor(loss) else hidden.new_zeros((), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        return grad_h * grad_out.to(grad_h.dtype), grad_w * grad_out.to(grad_w.dtype), None, None, None


def weighted_ce(hidden, weight, targets, token_w, chunk=512):
    return _WeightedCE.apply(hidden, weight, targets, token_w, chunk)


@torch.no_grad()
def token_nll(hidden, weight, targets, chunk=512):
    dtype = _compute_dtype(hidden)
    w_cast = weight.to(dtype)
    out = [
        -_log_probs(hidden[s : s + chunk].to(dtype), w_cast).gather(1, targets[s : s + chunk, None])[:, 0]
        for s in range(0, hidden.size(0), chunk)
    ]
    return torch.cat(out) if out else hidden.new_zeros(0, dtype=torch.float32)
