"""Masked discrete diffusion LM (MDLM, Sahoo et al. 2024) with absorbing [MASK] state.

Forward process (linear schedule alpha_t = 1 - t): every token is replaced by
[MASK] independently with probability t. The ELBO reduces to a time-weighted
masked cross-entropy:

    NLL <= E_t E_{x_t|x_0} [ (1/t) * sum_{i masked} -log p_theta(x_0^i | x_t) ]

so, unlike plain "BERT-style" masking, it yields a proper likelihood bound and
a perplexity comparable with autoregressive models. The network is
time-agnostic (the mask rate can be inferred from the input), as in MDLM/RADD.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from decoding import sample_next
from losses import token_nll, weighted_ce


class BidirectionalMultiHeadAttention(nn.Module):
    def __init__(self, dim_model: int, n_head: int):
        super().__init__()
        assert dim_model % n_head == 0, (
            "Model dimension must be divisible by number of heads."
        )
        self.n_head = n_head
        self.d_k = dim_model // n_head
        self.qkv_proj = nn.Linear(dim_model, 3 * dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_step, channels = x.size()
        q, k, v = self.qkv_proj(x).split(channels, dim=-1)
        q, k, v = (
            t.view(batch, time_step, self.n_head, self.d_k).transpose(1, 2)
            for t in (q, k, v)
        )
        out = F.scaled_dot_product_attention(q, k, v)  # no causal mask
        out = out.transpose(1, 2).contiguous().view(batch, time_step, channels)
        return self.out_proj(out)


class FeedForward(nn.Sequential):
    def __init__(self, dim_model: int):
        super().__init__(
            nn.Linear(dim_model, 4 * dim_model),
            nn.GELU(),
            nn.Linear(4 * dim_model, dim_model),
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim_model: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model)
        self.attn = BidirectionalMultiHeadAttention(dim_model, n_head)
        self.ln_2 = nn.LayerNorm(dim_model)
        self.ffwd = FeedForward(dim_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.ffwd(self.ln_2(x))


class DiffusionLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_model: int,
        n_head: int,
        max_sequence_len: int,
        n_layers: int,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_len = max_sequence_len
        self.mask_token_id = vocab_size  # one extra embedding row for [MASK]
        self.tie_embeddings = tie_embeddings

        self.token_embedding = nn.Embedding(vocab_size + 1, dim_model)
        self.position_embedding = nn.Embedding(max_sequence_len, dim_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim_model, n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(dim_model)
        # Output covers real vocabulary only: [MASK] can never be predicted.
        self.lm_head = None if tie_embeddings else nn.Linear(dim_model, vocab_size, bias=False)

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
        if self.tie_embeddings:
            return self.token_embedding.weight[: self.vocab_size]
        return self.lm_head.weight

    def features(self, idx: torch.Tensor) -> torch.Tensor:
        """Final hidden states [B, T, D] (before the LM head)."""
        positions = torch.arange(idx.size(1), device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.linear(self.features(idx), self.head_weight)

    # ------------------------------------------------------------------ loss
    def noise(self, x0: torch.Tensor, t: torch.Tensor, region: torch.Tensor):
        """Mask `region` positions of x0 independently with prob t: (x_t, mask [B,T])."""
        mask = (torch.rand(x0.shape, device=x0.device) < t[:, None]) & region
        return torch.where(mask, self.mask_token_id, x0), mask

    def training_loss(self, x0, eps=1e-3, chunk=512):
        """ELBO loss per token: every position is masked with prob t.

        Only masked positions contribute to the loss, so the LM head is applied
        to them alone - on average half of the sequence."""
        B, T = x0.shape
        # stratified t over the batch lowers the variance of the 1/t weighting
        t = (torch.randperm(B, device=x0.device) + torch.rand(B, device=x0.device)) / B
        t = eps + (1 - eps) * t
        region = torch.ones_like(x0, dtype=torch.bool)
        x_t, mask = self.noise(x0, t, region)
        weight = (1.0 / (t * T * B))[:, None].expand(B, T)[mask]
        return weighted_ce(self.features(x_t)[mask], self.head_weight, x0[mask], weight, chunk)

    @torch.no_grad()
    def elbo_nll(self, x0, length, start: int = 0, n_mc: int = 4):
        """Monte-Carlo ELBO estimate of the NLL (nats) of the real tokens at
        positions >= `start`; everything before `start` stays clean, i.e. the
        bound is on -log p(x[start:] | x[:start]).

        Returns (nll_sum, n_tokens); perplexity = exp(nll_sum / n_tokens).
        """
        B, T = x0.shape
        pos = torch.arange(T, device=x0.device)[None]
        region = (pos >= start) & (pos < length[:, None])
        total = x0.new_zeros((), dtype=torch.float64)
        for k in range(n_mc):  # stratified, deterministic grid of t
            t = (torch.arange(B, device=x0.device) + (k + 0.5) / n_mc) / B
            t = t.clamp(min=1e-3)
            x_t, mask = self.noise(x0, t, region)
            nll = token_nll(self.features(x_t)[mask], self.head_weight, x0[mask])
            total += (nll / t[:, None].expand(B, T)[mask]).sum().double()
        return total / n_mc, region.sum()

    # -------------------------------------------------------------- sampling
    def decode(self, hidden, temperature: float = 0.0, chunk: int = 128):
        """Tokens [N] for hidden states [N, D]: argmax, or a sample at `temperature`
        (the exact sampler shared with the autoregressive models, decoding.py)."""
        # chunks bound the [N, vocab] logits, as a batch of 128 does in the
        # autoregressive samplers
        out = [sample_next(F.linear(hidden[s : s + chunk], self.head_weight), temperature)
               for s in range(0, hidden.size(0), chunk)]
        return torch.cat(out) if out else hidden.new_zeros(0, dtype=torch.long)

    @torch.no_grad()
    def sample(
        self,
        prompt: torch.Tensor,
        gen_len: int,
        steps: int = 32,
        temperature: float = 0.0,
        remask: str = "random",
    ) -> torch.Tensor:
        """Generate `gen_len` tokens after `prompt` ([B, P]) in `steps` denoising steps.

        remask="random":     MDLM ancestral sampler - each masked token is
                             unmasked with prob (t-s)/t = 1/(steps-i). The LM
                             head is evaluated only for the tokens revealed in
                             the current step, i.e. once per generated token.
        remask="confidence": LLaDA-style - unmask the most confident
                             predictions, keeping ceil(s*gen_len) masked (needs
                             the logits of every generated position per step).
        Returns only the generated part [B, gen_len].
        """
        was_training = self.training
        self.eval()
        B, P = prompt.shape
        x = torch.cat(
            [prompt, torch.full((B, gen_len), self.mask_token_id, device=prompt.device, dtype=prompt.dtype)],
            dim=1,
        )
        for i in range(steps):
            s = 1.0 - (i + 1) / steps
            hidden = self.features(x)
            is_masked = x == self.mask_token_id

            if remask == "random":
                unmask = is_masked & (torch.rand(x.shape, device=x.device) < 1.0 / (steps - i))
                x[unmask] = self.decode(hidden[unmask], temperature)
            elif remask == "confidence":
                is_masked = is_masked[:, P:]
                logits = F.linear(hidden[:, P:], self.head_weight).float()
                if temperature > 0:
                    sampled = torch.distributions.Categorical(logits=logits / temperature).sample()
                else:
                    sampled = logits.argmax(-1)
                conf = F.softmax(logits, dim=-1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                conf = torch.where(is_masked, conf, torch.full_like(conf, -1.0))
                k = int(is_masked[0].sum()) - math.ceil(s * gen_len)
                if k > 0:
                    unmask = torch.zeros_like(is_masked)
                    unmask.scatter_(1, conf.topk(k, dim=-1).indices, True)
                    x[:, P:] = torch.where(unmask & is_masked, sampled, x[:, P:])
            else:
                raise ValueError(f"unknown remask strategy: {remask}")

        self.train(was_training)
        return x[:, P:]
