import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from decoding import generate_graphed, sample_next

try:  # optional CUDA kernel: `uv sync --extra mamba` (needs nvcc + a GPU)
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    HAS_KERNEL = True
except Exception:
    selective_scan_fn = None
    HAS_KERNEL = False

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except Exception:
    selective_state_update = None


class S6(nn.Module):
    """Selective state space core (Mamba, Gu & Dao 2023).

    Two interchangeable training/prefill paths with identical maths:
      * mamba-ssm fused CUDA kernel (fast, O(T) memory) when available,
      * pure PyTorch parallel scan (reference; materialises [B,T,D,N]).
    plus a single-token recurrent `step` for generation.
    """

    def __init__(self, d_model, d_state=16, dt_rank=None, use_kernel=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 32)
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
        """Paper init: dt ~ logUniform[dt_min, dt_max], stored via inverse softplus."""
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.d_model) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _kernel_ok(self, x):
        return self.use_kernel and HAS_KERNEL and x.is_cuda

    def forward(self, x, return_state=False):
        """x: [B, T, D] -> y: [B, T, D] (and last SSM state [B, D, N])."""
        A = -torch.exp(self.A_log.float())
        dt_r, B, C = torch.split(
            self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        if self._kernel_ok(x):
            delta = F.linear(dt_r, self.dt_proj.weight)  # bias/softplus fused in kernel
            res = selective_scan_fn(
                x.transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                A,
                B.transpose(1, 2).contiguous(),
                C.transpose(1, 2).contiguous(),
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=return_state,
            )
            y, last = res if return_state else (res, None)
            y = y.transpose(1, 2)
            return (y, last) if return_state else y

        xf, Bf, Cf = x.float(), B.float(), C.float()
        delta = F.softplus(self.dt_proj(dt_r).float())
        dA = torch.exp(delta.unsqueeze(-1) * A)  # [B,T,D,N]
        Bx = delta.unsqueeze(-1) * Bf.unsqueeze(-2) * xf.unsqueeze(-1)
        H = _parallel_scan(dA, Bx)  # h_t = dA_t * h_{t-1} + Bx_t
        y = torch.einsum("btdn,btn->btd", H, Cf) + xf * self.D
        y = y.to(x.dtype)
        return (y, H[:, -1]) if return_state else y

    def step(self, x, h):
        """One recurrent step. x: [B, D], h: [B, D, N] -> (y [B, D], h)."""
        A = -torch.exp(self.A_log.float())
        dt_r, B, C = torch.split(
            self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        if self._kernel_ok(x) and selective_state_update is not None:
            h = h.contiguous()  # updated in place by the kernel
            y = selective_state_update(
                h, x, F.linear(dt_r, self.dt_proj.weight), A, B, C,
                self.D.float(), z=None, dt_bias=self.dt_proj.bias.float(),
                dt_softplus=True,
            )
            return y, h
        delta = F.softplus(self.dt_proj(dt_r).float())
        h = torch.exp(delta.unsqueeze(-1) * A) * h + (
            delta.unsqueeze(-1) * B.float().unsqueeze(1) * x.float().unsqueeze(-1)
        )
        y = (h * C.float().unsqueeze(1)).sum(-1) + x.float() * self.D
        return y.to(x.dtype), h


def _parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inclusive scan of h_t = a_t * h_{t-1} + b_t, a/b: [B, T, ...] (Hillis-Steele)."""
    A, H = a, b
    stride = 1
    while stride < a.shape[1]:
        A_prev, H_prev = A, H
        A = torch.cat([A_prev[:, :stride], A_prev[:, stride:] * A_prev[:, :-stride]], 1)
        H = torch.cat(
            [
                H_prev[:, :stride],
                A_prev[:, stride:] * H_prev[:, :-stride] + H_prev[:, stride:],
            ],
            1,
        )
        stride *= 2
    return H


class MambaBlock(nn.Module):
    def __init__(self, dim_model, d_state=16, expand=2, d_conv=4, use_kernel=True):
        super().__init__()
        d_inner = expand * dim_model
        self.d_inner = d_inner
        self.d_conv = d_conv

        self.in_proj = nn.Linear(dim_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1
        )
        self.s6 = S6(d_inner, d_state=d_state, use_kernel=use_kernel)
        self.out_proj = nn.Linear(d_inner, dim_model, bias=False)

    def forward(self, x, return_state=False):
        seq_len = x.size(1)
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        x_t = x_in.transpose(1, 2)  # [B, D, T]
        x_conv = F.silu(self.conv1d(x_t)[:, :, :seq_len]).transpose(1, 2)

        if return_state:
            y, ssm_state = self.s6(x_conv, return_state=True)
            conv_state = F.pad(x_t, (self.d_conv - 1, 0))[:, :, -(self.d_conv - 1) :]
            out = self.out_proj(y * F.silu(z))
            return out, (conv_state, ssm_state)
        return self.out_proj(self.s6(x_conv) * F.silu(z))

    def step(self, x, state):
        """x: [B, dim_model] single token; state = (conv_state, ssm_state)."""
        conv_state, h = state
        x_in, z = self.in_proj(x).chunk(2, dim=-1)
        window = torch.cat([conv_state, x_in.unsqueeze(-1)], dim=-1)  # [B, D, d_conv]
        x_conv = F.silu((window * self.conv1d.weight[:, 0]).sum(-1) + self.conv1d.bias)
        y, h = self.s6.step(x_conv, h)
        return self.out_proj(y * F.silu(z)), (window[:, :, 1:], h)


class MambaLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim_model,
        n_layers,
        max_sequence_len,
        d_state=16,
        expand=2,
        d_conv=4,
        tie_embeddings=True,
        use_kernel=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.max_sequence_len = max_sequence_len

        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.norms = nn.ModuleList([nn.RMSNorm(dim_model) for _ in range(n_layers)])
        self.blocks = nn.ModuleList(
            [
                MambaBlock(dim_model, d_state, expand, d_conv, use_kernel)
                for _ in range(n_layers)
            ]
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
                m.reset_dt()  # must run after the generic init above

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    @property
    def head_weight(self):
        return self.lm_head.weight

    def features(self, idx, return_states=False):
        """Final hidden states [B, T, D] (and the recurrent states of every block)."""
        x = self.token_embedding(idx)
        states = []
        for norm, block in zip(self.norms, self.blocks):
            if return_states:
                out, st = block(norm(x), return_state=True)
                states.append(st)
            else:
                out = block(norm(x))
            x = x + out
        x = self.ln_f(x)
        return (x, states) if return_states else x

    def forward(self, idx, return_states=False):
        if return_states:
            x, states = self.features(idx, return_states=True)
            return self.lm_head(x), states
        return self.lm_head(self.features(idx))

    def step(self, idx, states):
        """idx: [B] one token per sequence -> (logits [B, V], new states)."""
        x = self.token_embedding(idx)
        new = []
        for norm, block, st in zip(self.norms, self.blocks, states):
            out, st = block.step(norm(x), st)
            x = x + out
            new.append(st)
        return self.lm_head(self.ln_f(x)), new

    # ---- fixed-size state: decoding step recordable as a CUDA graph (decoding.py)
    def alloc_decode_state(self, batch, max_len):
        w = self.token_embedding.weight
        return [
            (w.new_zeros(batch, b.d_inner, b.d_conv - 1),
             w.new_zeros(batch, b.d_inner, b.s6.d_state, dtype=torch.float32))
            for b in self.blocks
        ]

    def prefill_into(self, state, prompt):
        x, states = self.features(prompt, return_states=True)
        for (conv, h), (conv_new, h_new) in zip(state, states):
            conv.copy_(conv_new)
            h.copy_(h_new)
        return self.lm_head(x[:, -1])

    def decode_step(self, tok, pos, state):
        logits, states = self.step(tok, state)
        for (conv, h), (conv_new, h_new) in zip(state, states):
            conv.copy_(conv_new)
            if h_new is not h:  # the mamba-ssm kernel updates h in place
                h.copy_(h_new)
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=0.0, cuda_graph=False):
        """Recurrent decoding: O(1) state per token, no KV cache. [B,P] -> [B,new].

        cuda_graph=True: every decoding step replayed as one CUDA graph (no
        kernel-launch overhead; see decoding.py)."""
        was_training = self.training
        self.eval()
        if cuda_graph:
            out = generate_graphed(self, prompt, max_new_tokens, temperature)
            self.train(was_training)
            return out
        x, states = self.features(prompt, return_states=True)
        last = self.lm_head(x[:, -1])  # the LM head is only needed for the last position
        out = []
        for _ in range(max_new_tokens):
            nxt = sample_next(last, temperature)
            out.append(nxt)
            if len(out) < max_new_tokens:
                last, states = self.step(nxt, states)
        self.train(was_training)
        return torch.stack(out, dim=1)
