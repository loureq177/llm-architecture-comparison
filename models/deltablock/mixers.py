"""Sequence mixers of DeltaBlock: delta-rule linear attention (KDA, GDN-2) and
softmax attention with RoPE.

Two-stream mode works on the interleaved sequence [N_0 C_0 N_1 C_1 ...] of
noisy blocks N_b and clean blocks C_b (see TwoStream). Clean tokens are causal
and never see noisy ones; noisy block b sees the clean blocks < b and itself.
"""

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops.gdn2 import chunk_gdn2
from fla.ops.kda import chunk_kda


class TwoStream:
    """Layout of T clean + T noisy tokens interleaved block by block."""

    def __init__(self, T, block, device):
        assert T % block == 0
        self.T, self.L, self.nblk = T, block, T // block
        idx = torch.arange(2 * T, device=device)
        self.blk = idx // (2 * block)
        self.noisy = idx % (2 * block) < block
        self.pos = self.blk * block + idx % block

    def interleave(self, noisy, clean):  # 2 x [B, T, ...] -> [B, 2T, ...]
        B, rest = noisy.size(0), noisy.shape[2:]
        n = noisy.reshape(B, self.nblk, 1, self.L, *rest)
        c = clean.reshape(B, self.nblk, 1, self.L, *rest)
        return torch.cat([n, c], 2).reshape(B, 2 * self.T, *rest)

    def split(self, x):  # [B, 2T, ...] -> (noisy, clean), each [B, T, ...]
        x = x.reshape(x.size(0), self.nblk, 2, self.L, *x.shape[2:])
        return x[:, :, 0].flatten(1, 2), x[:, :, 1].flatten(1, 2)

    @property
    def attn_mask(self):  # [2T, 2T], True = may attend
        return _attn_mask(self.T, self.L, self.noisy.device)


@lru_cache(maxsize=8)
def _attn_mask(T, L, device):
    ts = TwoStream(T, L, device)
    nq, nk = ts.noisy[:, None], ts.noisy[None, :]
    bq, bk = ts.blk[:, None], ts.blk[None, :]
    clean = ~nq & ~nk & (ts.pos[None, :] <= ts.pos[:, None])
    noisy = nq & ((nk & (bk == bq)) | (~nk & (bk < bq)))
    return clean | noisy


def rope(x, pos, base=10000.0):
    """Rotary position embedding of x [B, H, T, d] at positions pos [T]."""
    d = x.size(-1)
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.float()[:, None] * inv
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class ShortConv(nn.Module):
    """Causal depthwise convolution over time + SiLU (kernel 4, no bias)."""

    def __init__(self, channels, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels,
                              padding=kernel_size - 1, bias=False)

    def forward(self, x):  # [B, T, C]
        T = x.size(1)
        return F.silu(self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2))

    def two_stream(self, x, ts):
        """Clean stream: causal over the whole clean sequence. Noisy stream:
        causal inside each block, so no information crosses the streams."""
        n, c = ts.split(x)
        B = x.size(0)
        n = self(n.reshape(B * ts.nblk, ts.L, -1)).reshape(B, ts.T, -1)
        return ts.interleave(n, self(c))


class DeltaMixer(nn.Module):
    """KDA or GDN-2 layer (H heads, key/value head dim K), as in
    fla/layers/{kda,gdn2}.py, with gates computed here so that noisy tokens can
    be made read-only: beta / erase / write = 0 (no write), log-decay = 0."""

    def __init__(self, dim_model, n_head, kind="kda", conv_size=4, intra_block=True):
        super().__init__()
        assert kind in ("kda", "gdn2")
        assert dim_model % n_head == 0
        self.kind, self.H, self.intra_block = kind, n_head, intra_block
        self.K = dim_model // n_head
        hd = n_head * self.K
        self.q_proj = nn.Linear(dim_model, hd, bias=False)
        self.k_proj = nn.Linear(dim_model, hd, bias=False)
        self.v_proj = nn.Linear(dim_model, hd, bias=False)
        self.q_conv, self.k_conv, self.v_conv = (ShortConv(hd, conv_size) for _ in range(3))
        self.f_proj = nn.Sequential(nn.Linear(dim_model, self.K, bias=False),
                                    nn.Linear(self.K, hd, bias=False))
        if kind == "kda":
            self.b_proj = nn.Linear(dim_model, n_head, bias=False)
        else:
            self.b_proj = nn.Linear(dim_model, hd, bias=False)
            self.w_proj = nn.Linear(dim_model, hd, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(n_head).uniform_(1, 16)))
        dt = torch.exp(torch.rand(hd) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(dim_model, self.K, bias=False),
                                    nn.Linear(self.K, hd, bias=True))
        self.o_norm = nn.RMSNorm(self.K, eps=1e-5)
        self.out_proj = nn.Linear(hd, dim_model, bias=False)

    def _heads(self, x):
        return x.unflatten(-1, (self.H, self.K))

    def forward(self, x, ts=None):
        """x [B, T, D], or the interleaved [B, 2T, D] when ts (TwoStream) is given."""
        conv = (lambda m, t: m(t)) if ts is None else (lambda m, t: m.two_stream(t, ts))
        q = self._heads(conv(self.q_conv, self.q_proj(x)))
        k = self._heads(conv(self.k_conv, self.k_proj(x)))
        v = self._heads(conv(self.v_conv, self.v_proj(x)))
        g = F.softplus(self.f_proj(x).float() + self.dt_bias)
        g = -self.A_log.float().exp()[:, None] * self._heads(g)  # [B, T, H, K], <= 0
        if self.kind == "kda":
            gates = [self.b_proj(x).float().sigmoid()]  # beta [B, T, H]
        else:
            gates = [self._heads(self.b_proj(x).float().sigmoid()),
                     self._heads(self.w_proj(x).float().sigmoid())]
        if ts is not None:  # noisy tokens: read the state, never write or decay it
            keep = (~ts.noisy).float()[None, :, None]
            g = g * keep[..., None]
            gates = [gt * (keep if gt.dim() == 3 else keep[..., None]) for gt in gates]
        if self.kind == "kda":
            o, _ = chunk_kda(q, k, v, g, gates[0].to(q.dtype), use_qk_l2norm_in_kernel=True)
        else:
            o, _ = chunk_gdn2(q, k, v, g, *(gt.to(q.dtype) for gt in gates),
                              use_qk_l2norm_in_kernel=True)
        if ts is not None and self.intra_block:
            o = o + self._intra_block(q, k, v, ts)
        o = self.o_norm(o) * self._heads(self.g_proj(x)).sigmoid()
        return self.out_proj(o.flatten(-2))

    def _intra_block(self, q, k, v, ts):
        """Bidirectional softmax attention among the noisy tokens of each block
        (RoPE at in-block positions); zero for clean tokens."""
        B = q.size(0)
        qn, kn, vn = (ts.split(t)[0].reshape(B * ts.nblk, ts.L, self.H, self.K).transpose(1, 2)
                      for t in (q, k, v))
        pos = torch.arange(ts.L, device=q.device)
        o = F.scaled_dot_product_attention(rope(qn, pos), rope(kn, pos), vn)
        o = o.transpose(1, 2).reshape(B, ts.T, self.H, self.K)
        return ts.interleave(o, torch.zeros_like(o))


class Attention(nn.Module):
    """Causal multi-head softmax attention with RoPE; block mask in two-stream mode."""

    def __init__(self, dim_model, n_head):
        super().__init__()
        self.H = n_head
        self.qkv_proj = nn.Linear(dim_model, 3 * dim_model, bias=False)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=False)

    def forward(self, x, ts=None):
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.H, -1).transpose(1, 2) for t in self.qkv_proj(x).split(D, -1))
        if ts is None:
            pos = torch.arange(T, device=x.device)
            out = F.scaled_dot_product_attention(rope(q, pos), rope(k, pos), v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(rope(q, ts.pos), rope(k, ts.pos), v,
                                                 attn_mask=ts.attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))
