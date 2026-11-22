import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    HAS_KERNEL = True
except Exception:
    HAS_KERNEL = False
try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except Exception:
    selective_state_update = None

from decoding import generate_eager, generate_graphed


class S6(nn.Module):
    def __init__(self, d_model, d_state=16, use_kernel=True):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.dt_rank = math.ceil(d_model / 32)
        self.use_kernel = use_kernel
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        self.reset_dt()

    def reset_dt(self, dt_min=1e-3, dt_max=0.1, dt_floor=1e-4):
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, h_out=None):
        A = -torch.exp(self.A_log.float())
        dt_r, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        if self.use_kernel and HAS_KERNEL and x.is_cuda:
            y = selective_scan_fn(
                x.transpose(1, 2).contiguous(),
                F.linear(dt_r, self.dt_proj.weight).transpose(1, 2).contiguous(),
                A,
                B.transpose(1, 2).contiguous(),
                C.transpose(1, 2).contiguous(),
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=h_out is not None,
            )
            if h_out is not None:
                y, last = y
                h_out.copy_(last)
            return y.transpose(1, 2)
        xf = x.float()
        delta = F.softplus(self.dt_proj(dt_r).float())
        dA = torch.exp(delta.unsqueeze(-1) * A)
        Bx = delta.unsqueeze(-1) * B.float().unsqueeze(-2) * xf.unsqueeze(-1)
        H = _parallel_scan(dA, Bx)
        if h_out is not None:
            h_out.copy_(H[:, -1])
        return (torch.einsum("btdn,btn->btd", H, C.float()) + xf * self.D).to(x.dtype)

    def step(self, x, h):
        A = -torch.exp(self.A_log.float())
        dt_r, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        if self.use_kernel and selective_state_update is not None and x.is_cuda:
            return selective_state_update(
                h, x, F.linear(dt_r, self.dt_proj.weight), A, B, C, self.D.float(),
                z=None, dt_bias=self.dt_proj.bias.float(), dt_softplus=True,
            )
        delta = F.softplus(self.dt_proj(dt_r).float())
        h.copy_(torch.exp(delta.unsqueeze(-1) * A) * h
                + delta.unsqueeze(-1) * B.float().unsqueeze(1) * x.float().unsqueeze(-1))
        return ((h * C.float().unsqueeze(1)).sum(-1) + x.float() * self.D).to(x.dtype)


def _parallel_scan(a, b):
    A, H, stride = a, b, 1
    while stride < a.shape[1]:
        A_prev, H_prev = A, H
        A = torch.cat([A_prev[:, :stride], A_prev[:, stride:] * A_prev[:, :-stride]], 1)
        H = torch.cat([H_prev[:, :stride], A_prev[:, stride:] * H_prev[:, :-stride] + H_prev[:, stride:]], 1)
        stride *= 2
    return H


class MambaBlock(nn.Module):
    def __init__(self, dim_model, d_state=16, expand=2, d_conv=4, use_kernel=True):
        super().__init__()
        d_inner = expand * dim_model
        self.in_proj = nn.Linear(dim_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1)
        self.s6 = S6(d_inner, d_state=d_state, use_kernel=use_kernel)
        self.out_proj = nn.Linear(d_inner, dim_model, bias=False)

    def forward(self, x):
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = F.silu(self.conv1d(x_in.transpose(1, 2))[:, :, : x.size(1)]).transpose(1, 2)
        return self.out_proj(self.s6(x_conv) * F.silu(z))

    def init_state(self, B, device, dtype):
        d_inner, k = self.conv1d.weight.size(0), self.conv1d.weight.size(-1)
        return (torch.zeros(B, d_inner, k - 1, device=device, dtype=dtype),
                torch.zeros(B, d_inner, self.s6.d_state, device=device, dtype=torch.float32))

    def prefill(self, x, st):
        conv, h = st
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        x_t = x_in.transpose(1, 2)
        conv.copy_(F.pad(x_t, (conv.size(-1), 0))[:, :, -conv.size(-1):])
        x_conv = F.silu(self.conv1d(x_t)[:, :, : x.size(1)]).transpose(1, 2)
        return self.out_proj(self.s6(x_conv, h_out=h) * F.silu(z))

    def step(self, x, st):
        conv, h = st
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        window = torch.cat([conv, x_in.unsqueeze(-1)], dim=-1)
        conv.copy_(window[:, :, 1:])
        x_conv = F.silu((window * self.conv1d.weight[:, 0]).sum(-1) + self.conv1d.bias)
        return self.out_proj(self.s6.step(x_conv, h) * F.silu(z))


class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim_model, n_layers, d_state=16, expand=2, d_conv=4,
                 tie_embeddings=True, use_kernel=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.norms = nn.ModuleList([nn.RMSNorm(dim_model) for _ in range(n_layers)])
        self.blocks = nn.ModuleList(
            [MambaBlock(dim_model, d_state, expand, d_conv, use_kernel) for _ in range(n_layers)]
        )
        self.ln_f = nn.RMSNorm(dim_model)
        self.lm_head = nn.Linear(dim_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / n_layers**0.5)
        for m in self.modules():
            if isinstance(m, S6):
                m.reset_dt()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    @property
    def head_weight(self):
        return self.lm_head.weight

    def features(self, idx):
        x = self.token_embedding(idx)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        return self.ln_f(x)

    def forward(self, idx):
        return self.lm_head(self.features(idx))

    def alloc_decode_state(self, batch, max_len):
        w = self.head_weight
        return [b.init_state(batch, w.device, w.dtype) for b in self.blocks]

    def prefill_into(self, state, prompt):
        x = self.token_embedding(prompt)
        for norm, block, st in zip(self.norms, self.blocks, state):
            x = x + block.prefill(norm(x), st)
        return self.lm_head(self.ln_f(x[:, -1]))

    def decode_step(self, tok, pos, state):
        x = self.token_embedding(tok)
        for norm, block, st in zip(self.norms, self.blocks, state):
            x = x + block.step(norm(x), st)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, prompt, new_tokens, temperature=0.0, cuda_graph=False):
        decode = generate_graphed if cuda_graph else generate_eager
        return decode(self, prompt, new_tokens, temperature)
