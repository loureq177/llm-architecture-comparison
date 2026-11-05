import torch
from torch import nn
from torch.nn import functional as F


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, dim_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv_proj = nn.Linear(dim_model, 3 * dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = (t.view(B, T, self.n_head, -1).transpose(1, 2) for t in self.qkv_proj(x).split(C, -1))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.l1 = nn.Linear(dim_model, 4 * dim_model)
        self.l2 = nn.Linear(4 * dim_model, dim_model)

    def forward(self, x):
        return self.l2(F.gelu(self.l1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model)
        self.attn = CausalMultiHeadAttention(dim_model, n_head)
        self.ln_2 = nn.LayerNorm(dim_model)
        self.ffwd = FeedForward(dim_model)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.ffwd(self.ln_2(x))


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, dim_model, n_head, max_sequence_len, n_layers, tie_embeddings=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.position_embedding = nn.Embedding(max_sequence_len, dim_model)
        self.blocks = nn.ModuleList([TransformerBlock(dim_model, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim_model)
        self.lm_head = nn.Linear(dim_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("l2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

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
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(idx.size(1), device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx):
        return self.lm_head(self.features(idx))
