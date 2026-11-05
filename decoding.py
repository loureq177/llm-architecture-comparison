"""Autoregressive decoding with every per-token step replayed as a CUDA graph.

Generating one token with a small model takes a few hundred tiny GPU kernels.
Launched one by one from Python, the GPU mostly waits for the CPU (launch
overhead), so the measured speed says more about the interpreter than about the
architecture. A CUDA graph records the kernels of one decoding step once and
replays them with a single launch - the standard way LLMs are served (vLLM,
TensorRT-LLM, `generate(cg=True)` of mamba-ssm). The recorded step must keep its
shapes and memory addresses, so it works on preallocated state: a KV cache of the
full length for the Transformer, the fixed-size recurrent state for Mamba.

A model provides
    alloc_decode_state(batch, max_len) -> state (tensors of fixed shape)
    prefill_into(state, prompt)        -> logits [B, V] of the last prompt token
    decode_step(token, pos, state)     -> logits [B, V] of the token after `token`
                                          ([B], at position `pos`), whose state
                                          it writes in place
The graph runs the model in its own dtype: convert the weights (e.g. to bf16)
instead of using autocast, whose weight-cast cache cannot live inside a graph.
"""

import torch
from torch.nn import functional as F


def sample_next(logits, temperature, block=256):
    """Next tokens [B] for logits [B, V]: argmax, or an exact sample at `temperature`.

    Two-level inverse-CDF sampling: pick a block of `block` tokens by its total
    probability, then the token inside that block. Float32 random numbers would
    bias it - torch.rand lies on a 2^-24 grid, so u = 0 happens and hands the
    draw to an arbitrary token (~0.3% of draws over 50k tokens, +0.016 nats of
    surprisal in our test) - and float64 over the full vocabulary is slow on
    consumer GPUs (1/64 rate). So the probabilities stay float32 and only the
    cumulative sums over [B, V/block] and [B, block] use float64. No host
    synchronisation: safe inside a CUDA graph.
    """
    if temperature <= 0:
        return logits.argmax(-1)
    B, V = logits.shape
    lf = logits.float()
    w = torch.exp((lf - lf.amax(-1, keepdim=True)) / temperature)  # unnormalised
    blocks = F.pad(w, (0, (-V) % block)).view(B, -1, block)
    mass = blocks.sum(-1).double()
    cum = mass.cumsum(-1)
    u = torch.rand(B, 1, device=logits.device, dtype=torch.float64) * cum[:, -1:]
    blk = (cum < u).sum(-1).clamp_(max=cum.size(1) - 1)  # first block with cum >= u
    before = torch.where(blk > 0, cum.gather(1, (blk - 1).clamp(min=0)[:, None])[:, 0], 0.0)
    frac = ((u[:, 0] - before) / mass.gather(1, blk[:, None])[:, 0]).clamp_(max=1 - 2**-52)
    inner = blocks[torch.arange(B, device=logits.device), blk].double().cumsum(-1)
    return blk * block + (inner < frac[:, None] * inner[:, -1:]).sum(-1)


class GraphedDecoder:
    """Generation loop for one (batch, prompt length, new tokens, temperature):
    eager prefill, then one graph replay per generated token."""

    def __init__(self, model, batch, prompt_len, new_tokens, temperature):
        device = model.head_weight.device
        self.model, self.key = model, (batch, prompt_len, new_tokens, temperature)
        self.state = model.alloc_decode_state(batch, prompt_len + new_tokens)
        self.tok = torch.zeros(batch, dtype=torch.long, device=device)
        self.pos = torch.full((), prompt_len, dtype=torch.long, device=device)
        self.out = torch.zeros(batch, new_tokens, dtype=torch.long, device=device)

        def step():  # token at `pos` -> next token, appended to `out`
            logits = model.decode_step(self.tok, self.pos, self.state)
            self.tok.copy_(sample_next(logits, temperature))
            self.pos += 1
            self.out.index_copy_(1, (self.pos - prompt_len).view(1), self.tok[:, None])

        # warm-up on a side stream (lazy initialisation, Triton autotuning), then
        # record; the garbage state left behind is overwritten by the prefill
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.pos.fill_(prompt_len)
                step()
        torch.cuda.current_stream().wait_stream(side)
        self.pos.fill_(prompt_len)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            step()

    def __call__(self, prompt):
        _, prompt_len, new_tokens, temperature = self.key
        logits = self.model.prefill_into(self.state, prompt)
        self.tok.copy_(sample_next(logits, temperature))
        self.pos.fill_(prompt_len)
        self.out[:, 0] = self.tok
        for _ in range(new_tokens - 1):
            self.graph.replay()
        return self.out.clone()


@torch.no_grad()
def generate_graphed(model, prompt, new_tokens, temperature=0.0):
    """[B, P] -> [B, new_tokens]. The graph of the last configuration is kept on
    the model, so repeated calls (benchmarks, sampling in batches) record it once."""
    assert prompt.is_cuda, "CUDA graphs need a GPU"
    assert not torch.is_autocast_enabled("cuda"), (
        "run the graphed model in its own dtype (e.g. model.to(torch.bfloat16)), not under autocast"
    )
    key = (prompt.size(0), prompt.size(1), new_tokens, temperature)
    runner = getattr(model, "_graphed", None)
    if runner is None or runner.key != key:
        release_graph(model)
        runner = model._graphed = GraphedDecoder(model, *key)
    return runner(prompt)


def release_graph(model):
    """Frees the recorded graph and its preallocated state."""
    if getattr(model, "_graphed", None) is not None:
        model._graphed = None
        torch.cuda.empty_cache()
