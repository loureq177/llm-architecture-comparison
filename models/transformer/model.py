import torch
from torch import nn
from torch.nn import functional as F

from decoding import generate_graphed, sample_next


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, dim_model, n_head):
        super().__init__()
        assert dim_model % n_head == 0, (
            "Model dimensions must be divisible by the number of attention heads."
        )
        self.n_head = n_head
        self.d_k = dim_model // n_head
        self.qkv_proj = nn.Linear(dim_model, 3 * dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

    def forward(self, x, cache=None):
        """cache: optional (k, v) of past tokens, shape [B, H, T_past, d_k]."""
        batch, time_step, channels = x.size()
        q, k, v = self.qkv_proj(x).split(channels, dim=-1)
        q, k, v = (
            t.view(batch, time_step, self.n_head, self.d_k).transpose(1, 2)
            for t in (q, k, v)
        )
        if cache is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Incremental decoding: new queries may attend to every cached key.
            # (used with time_step == 1, so no causal mask inside the new chunk)
            assert time_step == 1, "cached attention supports one new token at a time"
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(batch, time_step, channels)
        return self.out_proj(out), (k, v)

    def decode_static(self, x, k_cache, v_cache, pos, valid):
        """One token x [B, 1, C] at position `pos` (0-dim tensor), with caches
        preallocated for the whole sequence [B, H, T_max, d_k]: writes its key
        and value at `pos` and attends to the positions marked in `valid` [T_max].
        Shapes and addresses never change, so it can be recorded in a CUDA graph."""
        batch, _, channels = x.size()
        q, k, v = (
            t.view(batch, 1, self.n_head, self.d_k).transpose(1, 2)
            for t in self.qkv_proj(x).split(channels, dim=-1)
        )
        k_cache.index_copy_(2, pos.view(1), k)
        v_cache.index_copy_(2, pos.view(1), v)
        out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=valid)
        return self.out_proj(out.transpose(1, 2).reshape(batch, 1, channels))


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

    def forward(self, x, cache=None):
        a, kv = self.attn(self.ln_1(x), cache)
        x = x + a
        x = x + self.ffwd(self.ln_2(x))
        return x, kv

    def decode_static(self, x, k_cache, v_cache, pos, valid):
        x = x + self.attn.decode_static(self.ln_1(x), k_cache, v_cache, pos, valid)
        return x + self.ffwd(self.ln_2(x))


class LanguageModel(nn.Module):
    def __init__(
        self, vocab_size, dim_model, n_head, max_sequence_len, n_layers, tie_embeddings=True
    ):
        super().__init__()
        self.max_sequence_len = max_sequence_len
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.position_embedding = nn.Embedding(max_sequence_len, dim_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim_model, n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(dim_model)
        self.lm_head = nn.Linear(dim_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        # GPT-2 style: scale residual-branch output projections by 1/sqrt(2N)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("l2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layers) ** 0.5)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @property
    def head_weight(self):
        return self.lm_head.weight

    def features(self, idx, caches=None, pos_offset=0):
        """Final hidden states [B, T, D], or (hidden, caches) when `caches` is not None.

        caches=[] with a full prompt does the prefill and returns fresh caches;
        passing the returned caches with one new token does a decode step.
        """
        _, time_step = idx.size()
        positions = torch.arange(
            pos_offset, pos_offset + time_step, device=idx.device
        )
        x = self.token_embedding(idx) + self.position_embedding(positions)

        new_caches = []
        for i, block in enumerate(self.blocks):
            past = caches[i] if caches else None
            x, kv = block(x, past)
            new_caches.append(kv)
        x = self.ln_f(x)
        return x if caches is None else (x, new_caches)

    def forward(self, idx, caches=None, pos_offset=0):
        """Returns logits, or (logits, caches) when `caches` is not None."""
        if caches is None:
            return self.lm_head(self.features(idx))
        x, caches = self.features(idx, caches, pos_offset)
        return self.lm_head(x), caches

    # ---- static KV cache: decoding step recordable as a CUDA graph (decoding.py)
    def alloc_decode_state(self, batch, max_len):
        attn = self.blocks[0].attn
        shape = (batch, attn.n_head, max_len, attn.d_k)
        w = self.token_embedding.weight
        return [(w.new_zeros(shape), w.new_zeros(shape)) for _ in self.blocks]

    def prefill_into(self, state, prompt):
        x, caches = self.features(prompt, caches=[])
        P = prompt.size(1)
        for (k_cache, v_cache), (k, v) in zip(state, caches):
            k_cache[:, :, :P].copy_(k)
            v_cache[:, :, :P].copy_(v)
        return self.lm_head(x[:, -1])

    def decode_step(self, tok, pos, state):
        x = self.token_embedding(tok[:, None]) + self.position_embedding(pos.view(1))
        valid = torch.arange(state[0][0].size(2), device=tok.device) <= pos
        for block, (k_cache, v_cache) in zip(self.blocks, state):
            x = block.decode_static(x, k_cache, v_cache, pos, valid)
        return self.lm_head(self.ln_f(x))[:, 0]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=0.0, cuda_graph=False):
        """Autoregressive decoding with a KV cache. prompt: [B, P] -> [B, new].

        cuda_graph=True: static KV cache, every decoding step replayed as one
        CUDA graph (no kernel-launch overhead; see decoding.py)."""
        was_training = self.training
        self.eval()
        if cuda_graph:
            out = generate_graphed(self, prompt, max_new_tokens, temperature)
            self.train(was_training)
            return out
        x, caches = self.features(prompt, caches=[])
        x = x[:, -1:]  # the LM head is only needed for the last prompt position
        out = []
        offset = prompt.size(1)
        for _ in range(max_new_tokens):
            nxt = sample_next(self.lm_head(x)[:, -1], temperature)[:, None]
            out.append(nxt)
            if len(out) < max_new_tokens:
                x, caches = self.features(nxt, caches, offset)
                offset += 1
        self.train(was_training)
        return torch.cat(out, dim=1)
