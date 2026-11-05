import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.ops.kda import chunk_kda, fused_recurrent_kda

from decoding import generate_eager, generate_graphed
from losses import token_nll, weighted_ce


class TwoStream:
    def __init__(self, T, block, device):
        self.T, self.L, self.nblk = T, block, T // block
        idx = torch.arange(2 * T, device=device)
        self.blk = idx // (2 * block)
        self.noisy = idx % (2 * block) < block
        self.pos = self.blk * block + idx % block

    def interleave(self, noisy, clean):
        B, rest = noisy.size(0), noisy.shape[2:]
        n = noisy.reshape(B, self.nblk, 1, self.L, *rest)
        c = clean.reshape(B, self.nblk, 1, self.L, *rest)
        return torch.cat([n, c], 2).reshape(B, 2 * self.T, *rest)

    def split(self, x):
        x = x.reshape(x.size(0), self.nblk, 2, self.L, *x.shape[2:])
        return x[:, :, 0].flatten(1, 2), x[:, :, 1].flatten(1, 2)

    @property
    def attn_mask(self):
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
    d = x.size(-1)
    inv = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.float()[:, None] * inv
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class ShortConv(nn.Module):
    def __init__(self, channels, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=kernel_size - 1, bias=False)

    def forward(self, x):
        return F.silu(self.conv(x.transpose(1, 2))[..., : x.size(1)].transpose(1, 2))

    def two_stream(self, x, ts):
        n, c = ts.split(x)
        n = self(n.reshape(x.size(0) * ts.nblk, ts.L, -1)).reshape(x.size(0), ts.T, -1)
        return ts.interleave(n, self(c))


class KDA(nn.Module):
    def __init__(self, dim_model, n_head, conv_size=4):
        super().__init__()
        self.H, self.K = n_head, dim_model // n_head
        self.q_proj = nn.Linear(dim_model, dim_model, bias=False)
        self.k_proj = nn.Linear(dim_model, dim_model, bias=False)
        self.v_proj = nn.Linear(dim_model, dim_model, bias=False)
        self.q_conv, self.k_conv, self.v_conv = (ShortConv(dim_model, conv_size) for _ in range(3))
        self.f_proj = nn.Sequential(nn.Linear(dim_model, self.K, bias=False), nn.Linear(self.K, dim_model, bias=False))
        self.b_proj = nn.Linear(dim_model, n_head, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(n_head).uniform_(1, 16)))
        dt = torch.exp(torch.rand(dim_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(dim_model, self.K, bias=False), nn.Linear(self.K, dim_model, bias=True))
        self.o_norm = nn.RMSNorm(self.K, eps=1e-5)
        self.out_proj = nn.Linear(dim_model, dim_model, bias=False)

    def _heads(self, x):
        return x.unflatten(-1, (self.H, self.K))

    def _gates(self, x):
        g = -self.A_log.float().exp()[:, None] * self._heads(F.softplus(self.f_proj(x).float() + self.dt_bias))
        return g, self.b_proj(x).float().sigmoid()

    def _out(self, o, x):
        o = self.o_norm(o) * self._heads(self.g_proj(x)).sigmoid()
        return self.out_proj(o.flatten(-2))

    def forward(self, x, ts=None):
        conv = (lambda m, t: m(t)) if ts is None else (lambda m, t: m.two_stream(t, ts))
        q = self._heads(conv(self.q_conv, self.q_proj(x)))
        k = self._heads(conv(self.k_conv, self.k_proj(x)))
        v = self._heads(conv(self.v_conv, self.v_proj(x)))
        g, beta = self._gates(x)
        if ts is not None:
            keep = (~ts.noisy).float()[None, :, None]
            g, beta = g * keep[..., None], beta * keep
        o, _ = chunk_kda(q, k, v, g, beta.to(q.dtype), use_qk_l2norm_in_kernel=True)
        if ts is not None:
            o = o + self._intra_block(q, k, v, ts)
        return self._out(o, x)

    def init_state(self, B, max_len, device, dtype):
        w = self.q_conv.conv.kernel_size[0] - 1
        return {"conv": torch.zeros(3, B, self.H * self.K, w, device=device, dtype=dtype),
                "S": torch.zeros(B, self.H, self.K, self.K, device=device, dtype=torch.float32)}

    def _projs(self):
        return zip((self.q_proj, self.k_proj, self.v_proj), (self.q_conv, self.k_conv, self.v_conv))

    def prefill(self, x, st):
        w, T = st["conv"].size(-1), x.size(1)
        qkv = []
        for i, (proj, conv) in enumerate(self._projs()):
            t = proj(x)
            st["conv"][i].zero_()
            st["conv"][i][..., max(w - T, 0):] = t[:, -w:].transpose(1, 2)
            qkv.append(self._heads(conv(t)))
        g, beta = self._gates(x)
        o, S = chunk_kda(*qkv, g, beta.to(x.dtype), output_final_state=True, use_qk_l2norm_in_kernel=True)
        st["S"].copy_(S)
        return self._out(o, x)

    def step(self, x, pos, st):
        qkv = []
        for i, (proj, conv) in enumerate(self._projs()):
            win = torch.cat([st["conv"][i], proj(x).transpose(1, 2)], -1)
            st["conv"][i].copy_(win[..., 1:])
            h = F.silu((win.float() * conv.conv.weight[:, 0].float()).sum(-1)).to(x.dtype)
            qkv.append(self._heads(h)[:, None])
        g, beta = self._gates(x)
        o, S = fused_recurrent_kda(*qkv, g, beta.to(x.dtype), initial_state=st["S"], output_final_state=True,
                                   use_qk_l2norm_in_kernel=True)
        st["S"].copy_(S)
        return self._out(o, x)

    def _intra_block(self, q, k, v, ts):
        B = q.size(0)
        qn, kn, vn = (ts.split(t)[0].reshape(B * ts.nblk, ts.L, self.H, self.K).transpose(1, 2) for t in (q, k, v))
        pos = torch.arange(ts.L, device=q.device)
        o = F.scaled_dot_product_attention(rope(qn, pos), rope(kn, pos), vn)
        o = o.transpose(1, 2).reshape(B, ts.T, self.H, self.K)
        return ts.interleave(o, torch.zeros_like(o))


class Attention(nn.Module):
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
            out = F.scaled_dot_product_attention(rope(q, ts.pos), rope(k, ts.pos), v, attn_mask=ts.attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))

    def init_state(self, B, max_len, device, dtype):
        shape = (B, self.H, max_len, self.qkv_proj.in_features // self.H)
        return {"k": torch.zeros(shape, device=device, dtype=dtype), "v": torch.zeros(shape, device=device, dtype=dtype)}

    def prefill(self, x, st):
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.H, -1).transpose(1, 2) for t in self.qkv_proj(x).split(D, -1))
        pos = torch.arange(T, device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        st["k"][:, :, :T], st["v"][:, :, :T] = k, v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))

    def step(self, x, pos, st):
        B, _, D = x.shape
        q, k, v = (t.view(B, 1, self.H, -1).transpose(1, 2) for t in self.qkv_proj(x).split(D, -1))
        p = pos.view(1)
        st["k"].index_copy_(2, p, rope(k, p))
        st["v"].index_copy_(2, p, v)
        mask = torch.arange(st["k"].size(2), device=x.device) <= pos
        out = F.scaled_dot_product_attention(rope(q, p), st["k"], st["v"], attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, 1, D))


class SwiGLU(nn.Module):
    def __init__(self, dim_model, hidden):
        super().__init__()
        self.gate_up = nn.Linear(dim_model, 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, dim_model, n_head, attn, mlp_hidden):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim_model)
        self.mixer = Attention(dim_model, n_head) if attn else KDA(dim_model, n_head)
        self.norm2 = nn.RMSNorm(dim_model)
        self.mlp = SwiGLU(dim_model, mlp_hidden)

    def forward(self, x, ts=None):
        x = x + self.mixer(self.norm1(x), ts)
        return x + self.mlp(self.norm2(x))

    def cached(self, x, st, pos=None):
        h = self.norm1(x)
        x = x + (self.mixer.prefill(h, st) if pos is None else self.mixer.step(h, pos, st))
        return x + self.mlp(self.norm2(x))


class HybridLM(nn.Module):
    def __init__(self, vocab_size, dim_model, n_head, n_layers, attn_every=4, block_size=16):
        super().__init__()
        self.vocab_size, self.block_size = vocab_size, block_size
        self.mask_token_id = vocab_size
        hidden = 64 * round(8 * dim_model / 3 / 64)
        self.token_embedding = nn.Embedding(vocab_size + 1, dim_model)
        self.blocks = nn.ModuleList(
            [Block(dim_model, n_head, (i + 1) % attn_every == 0, hidden) for i in range(n_layers)]
        )
        self.norm_f = nn.RMSNorm(dim_model)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    @property
    def head_weight(self):
        return self.token_embedding.weight[: self.vocab_size]

    def features(self, idx):
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x)
        return self.norm_f(x)

    def forward(self, idx):
        return F.linear(self.features(idx), self.head_weight)

    def alloc_decode_state(self, batch, max_len):
        w = self.head_weight
        return [b.mixer.init_state(batch, max_len, w.device, w.dtype) for b in self.blocks]

    def _cached(self, idx, state, pos=None):
        x = self.token_embedding(idx)
        for block, st in zip(self.blocks, state):
            x = block.cached(x, st, pos)
        return F.linear(self.norm_f(x[:, -1]), self.head_weight)

    def prefill_into(self, state, prompt):
        return self._cached(prompt, state)

    def decode_step(self, tok, pos, state):
        return self._cached(tok[:, None], state, pos)

    @torch.no_grad()
    def generate(self, prompt, new_tokens, temperature=0.0, cuda_graph=False):
        decode = generate_graphed if cuda_graph else generate_eager
        return decode(self, prompt, new_tokens, temperature)

    def two_stream(self, noisy, clean):
        ts = TwoStream(clean.size(1), self.block_size, clean.device)
        x = self.token_embedding(ts.interleave(noisy, clean))
        for block in self.blocks:
            x = block(x, ts)
        return ts.split(self.norm_f(x))

    def noise(self, x0, t, start=1):
        t_tok = t.repeat_interleave(self.block_size, dim=1)
        pos = torch.arange(x0.size(1), device=x0.device)[None]
        mask = (torch.rand(x0.shape, device=x0.device) < t_tok) & (pos >= start)
        return torch.where(mask, self.mask_token_id, x0), mask, t_tok

    def joint_loss(self, x0, diff_weight=0.3, eps=1e-3, chunk=512):
        B, T = x0.shape
        n = B * (T // self.block_size)
        u = (torch.randperm(n, device=x0.device) + torch.rand(n, device=x0.device)) / n
        x_t, mask, t_tok = self.noise(x0, (eps + (1 - eps) * u).view(B, -1))
        h_noisy, h_clean = self.two_stream(x_t, x0)
        w_diff = (1.0 / (t_tok * B * T))[mask]
        l_diff = weighted_ce(h_noisy[mask], self.head_weight, x0[mask], w_diff, chunk)
        h = h_clean[:, :-1].reshape(-1, h_clean.size(-1))
        w_ar = h.new_full((h.size(0),), 1.0 / h.size(0), dtype=torch.float32)
        l_ar = weighted_ce(h, self.head_weight, x0[:, 1:].reshape(-1), w_ar, chunk)
        return l_ar + diff_weight * l_diff

    @torch.no_grad()
    def elbo_nll(self, x0, start=1, n_mc=4):
        B, T = x0.shape
        n = B * (T // self.block_size)
        total = x0.new_zeros((), dtype=torch.float64)
        for k in range(n_mc):
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(k)).to(x0.device)
            t = ((perm + (k + 0.5) / n_mc) / n).clamp(min=1e-3)
            x_t, mask, t_tok = self.noise(x0, t.view(B, -1), start)
            h_noisy, _ = self.two_stream(x_t, x0)
            nll = token_nll(h_noisy[mask], self.head_weight, x0[mask])
            total += (nll / t_tok[mask]).sum().double()
        return total / n_mc, B * (T - start)
