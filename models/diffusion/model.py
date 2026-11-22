import torch
from torch import nn
from torch.nn import functional as F

from decoding import sample_next
from losses import token_nll, weighted_ce


class BidirectionalMultiHeadAttention(nn.Module):
    def __init__(self, dim_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv_proj = nn.Linear(dim_model, 3 * dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in self.qkv_proj(x).split(C, -1))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class FeedForward(nn.Sequential):
    def __init__(self, dim_model):
        super().__init__(nn.Linear(dim_model, 4 * dim_model), nn.GELU(), nn.Linear(4 * dim_model, dim_model))


class TransformerBlock(nn.Module):
    def __init__(self, dim_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model)
        self.attn = BidirectionalMultiHeadAttention(dim_model, n_head)
        self.ln_2 = nn.LayerNorm(dim_model)
        self.ffwd = FeedForward(dim_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.ffwd(self.ln_2(x))


class DiffusionLanguageModel(nn.Module):
    def __init__(self, vocab_size, dim_model, n_head, max_sequence_len, n_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size
        self.token_embedding = nn.Embedding(vocab_size + 1, dim_model)
        self.position_embedding = nn.Embedding(max_sequence_len, dim_model)
        self.blocks = nn.ModuleList([TransformerBlock(dim_model, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim_model)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ffwd.2.weight"):
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
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(idx.size(1), device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx):
        return F.linear(self.features(idx), self.head_weight)

    def noise(self, x0, t, region):
        mask = (torch.rand(x0.shape, device=x0.device) < t[:, None]) & region
        return torch.where(mask, self.mask_token_id, x0), mask

    def training_loss(self, x0, eps=1e-3, chunk=512):
        B, T = x0.shape
        t = (torch.randperm(B, device=x0.device) + torch.rand(B, device=x0.device)) / B
        t = eps + (1 - eps) * t
        x_t, mask = self.noise(x0, t, torch.ones_like(x0, dtype=torch.bool))
        weight = (1.0 / (t * T * B))[:, None].expand(B, T)[mask]
        return weighted_ce(self.features(x_t)[mask], self.head_weight, x0[mask], weight, chunk)

    @torch.no_grad()
    def elbo_nll(self, x0, start=1, n_mc=4):
        B, T = x0.shape
        region = (torch.arange(T, device=x0.device) >= start)[None].expand(B, T)
        total = x0.new_zeros((), dtype=torch.float64)
        for k in range(n_mc):
            t = ((torch.arange(B, device=x0.device) + (k + 0.5) / n_mc) / B).clamp(min=1e-3)
            x_t, mask = self.noise(x0, t, region)
            nll = token_nll(self.features(x_t)[mask], self.head_weight, x0[mask])
            total += (nll / t[:, None].expand(B, T)[mask]).sum().double()
        return total / n_mc, region.sum()

    @torch.no_grad()
    def sample(self, prompt, new_tokens, steps, temperature=0.0, chunk=128):
        B, P = prompt.shape
        x = torch.cat([prompt, prompt.new_full((B, new_tokens), self.mask_token_id)], 1)
        for i in range(steps):
            unmask = (x == self.mask_token_id) & (torch.rand(x.shape, device=x.device) < 1.0 / (steps - i))
            h = self.features(x)[unmask]
            x[unmask] = torch.cat([sample_next(F.linear(h[s : s + chunk], self.head_weight), temperature)
                                   for s in range(0, h.size(0), chunk)] or [x.new_zeros(0)])
        return x[:, P:]
