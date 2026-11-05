"""DeltaBlock: block diffusion on a 3:1 hybrid of delta-rule linear attention
and softmax attention (pre-norm RMSNorm blocks, SwiGLU, tied embeddings).

Autoregressive mode: `features(x)` is a causal LM over the clean tokens.
Two-stream mode: the clean stream (next-token loss) and the noisy stream
(masked diffusion inside blocks, conditioned on the clean blocks before)
share one forward pass; the clean stream is identical to the AR mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import token_nll, weighted_ce
from models.deltablock.mixers import Attention, DeltaMixer, TwoStream


class SwiGLU(nn.Module):
    def __init__(self, dim_model, hidden):
        super().__init__()
        self.gate_up = nn.Linear(dim_model, 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, dim_model, n_head, mixer, mlp_hidden, intra_block):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim_model)
        self.mixer = (Attention(dim_model, n_head) if mixer == "attn"
                      else DeltaMixer(dim_model, n_head, kind=mixer, intra_block=intra_block))
        self.norm2 = nn.RMSNorm(dim_model)
        self.mlp = SwiGLU(dim_model, mlp_hidden)

    def forward(self, x, ts=None):
        x = x + self.mixer(self.norm1(x), ts)
        return x + self.mlp(self.norm2(x))


def mlp_hidden_for(dim_model, mixer, base_hidden):
    """GDN-2 has two extra d x d gates (erase, write): a narrower MLP keeps the
    parameter count of a layer independent of the mixer."""
    if mixer != "gdn2":
        return base_hidden
    return base_hidden - round(2 * dim_model / 3 / 64) * 64


class HybridLM(nn.Module):
    def __init__(self, vocab_size, dim_model, n_head, n_layers, mixer="kda", attn_every=4,
                 mlp_hidden=None, block_size=16, intra_block=True):
        super().__init__()
        self.vocab_size, self.block_size = vocab_size, block_size
        self.mask_token_id = vocab_size  # extra embedding row, never predicted
        base = mlp_hidden or 64 * round(8 * dim_model / 3 / 64)  # ~ 8 d^2 per SwiGLU
        kinds = ["attn" if (i + 1) % attn_every == 0 else mixer for i in range(n_layers)]
        self.layer_kinds = kinds
        self.token_embedding = nn.Embedding(vocab_size + 1, dim_model)
        self.blocks = nn.ModuleList(
            [Block(dim_model, n_head, k, mlp_hidden_for(dim_model, k, base), intra_block)
             for k in kinds]
        )
        self.norm_f = nn.RMSNorm(dim_model)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2: residual outputs scaled by 1/sqrt(2L)
            if name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    @property
    def head_weight(self):  # tied LM head over the real vocabulary
        return self.token_embedding.weight[: self.vocab_size]

    def features(self, idx):
        """Autoregressive mode: hidden states [B, T, D] of a causal LM."""
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)

    def forward(self, idx):
        return F.linear(self.features(idx), self.head_weight)

    def two_stream(self, noisy, clean):
        """Hidden states (noisy, clean), each [B, T, D], of one joint pass."""
        ts = TwoStream(clean.size(1), self.block_size, clean.device)
        x = self.token_embedding(ts.interleave(noisy, clean))
        for block in self.blocks:
            x = block(x, ts)
        return ts.split(self.norm_f(x))

    # ------------------------------------------------------------------ diffusion
    def noise(self, x0, t, start=1):
        """Mask every position >= start with probability t of its block.
        t [B, nblk] -> (x_t, mask [B, T], t per token [B, T])."""
        t_tok = t.repeat_interleave(self.block_size, dim=1)
        pos = torch.arange(x0.size(1), device=x0.device)[None]
        mask = (torch.rand(x0.shape, device=x0.device) < t_tok) & (pos >= start)
        return torch.where(mask, self.mask_token_id, x0), mask, t_tok

    def joint_loss(self, x0, diff_weight=0.3, eps=1e-3, chunk=512):
        """L_AR + diff_weight * L_diff, both per token (Nemotron-Labs-Diffusion).
        L_diff is the block-diffusion ELBO: masked CE weighted by 1/t, with t
        drawn per block, stratified over all blocks of the batch."""
        B, T = x0.shape
        n = B * (T // self.block_size)
        u = (torch.randperm(n, device=x0.device) + torch.rand(n, device=x0.device)) / n
        t = (eps + (1 - eps) * u).view(B, -1)
        x_t, mask, t_tok = self.noise(x0, t)
        h_noisy, h_clean = self.two_stream(x_t, x0)
        w_diff = (1.0 / (t_tok * B * T))[mask]
        l_diff = weighted_ce(h_noisy[mask], self.head_weight, x0[mask], w_diff, chunk)
        h = h_clean[:, :-1].reshape(-1, h_clean.size(-1))
        w_ar = h.new_full((h.size(0),), 1.0 / h.size(0), dtype=torch.float32)
        l_ar = weighted_ce(h, self.head_weight, x0[:, 1:].reshape(-1), w_ar, chunk)
        return l_ar + diff_weight * l_diff

    @torch.no_grad()
    def elbo_nll(self, x0, length, start=1, n_mc=4):
        """Block-diffusion ELBO bound on -log p(x[start:] | x[:start]), summed
        over blocks, with a deterministic stratified grid of t (as MDLM's
        elbo_nll). Returns (nll_sum, n_tokens). Assumes full windows."""
        B, T = x0.shape
        n = B * (T // self.block_size)
        total = x0.new_zeros((), dtype=torch.float64)
        for k in range(n_mc):  # fixed permutation: blocks differ in context, so the
            # grid point of a block must not depend on its position
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(k)).to(x0.device)
            t = ((perm + (k + 0.5) / n_mc) / n).clamp(min=1e-3)
            x_t, mask, t_tok = self.noise(x0, t.view(B, -1), start)
            h_noisy, _ = self.two_stream(x_t, x0)
            nll = token_nll(h_noisy[mask], self.head_weight, x0[mask])
            total += (nll / t_tok[mask]).sum().double()
        pos = torch.arange(T, device=x0.device)[None]
        return total / n_mc, ((pos >= start) & (pos < length[:, None])).sum()
