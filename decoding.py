import torch
from torch.nn import functional as F


def sample_next(logits, temperature, block=256):
    if temperature <= 0:
        return logits.argmax(-1)
    B, V = logits.shape
    lf = logits.float()
    w = torch.exp((lf - lf.amax(-1, keepdim=True)) / temperature)
    blocks = F.pad(w, (0, (-V) % block)).view(B, -1, block)
    mass = blocks.sum(-1).double()
    cum = mass.cumsum(-1)
    u = torch.rand(B, 1, device=logits.device, dtype=torch.float64) * cum[:, -1:]
    blk = (cum < u).sum(-1).clamp_(max=cum.size(1) - 1)
    before = torch.where(blk > 0, cum.gather(1, (blk - 1).clamp(min=0)[:, None])[:, 0], 0.0)
    frac = ((u[:, 0] - before) / mass.gather(1, blk[:, None])[:, 0]).clamp_(max=1 - 2**-52)
    inner = blocks[torch.arange(B, device=logits.device), blk].double().cumsum(-1)
    return blk * block + (inner < frac[:, None] * inner[:, -1:]).sum(-1)


@torch.no_grad()
def generate_eager(model, prompt, new_tokens, temperature=0.0):
    B, P = prompt.shape
    state = model.alloc_decode_state(B, P + new_tokens)
    tok = sample_next(model.prefill_into(state, prompt), temperature)
    out = [tok]
    for i in range(new_tokens - 1):
        pos = torch.tensor(P + i, device=prompt.device)
        tok = sample_next(model.decode_step(tok, pos, state), temperature)
        out.append(tok)
    return torch.stack(out, 1)


class GraphedDecoder:
    def __init__(self, model, batch, prompt_len, new_tokens, temperature):
        device = model.head_weight.device
        self.model, self.key = model, (batch, prompt_len, new_tokens, temperature)
        self.state = model.alloc_decode_state(batch, prompt_len + new_tokens)
        self.tok = torch.zeros(batch, dtype=torch.long, device=device)
        self.pos = torch.full((), prompt_len, dtype=torch.long, device=device)
        self.out = torch.zeros(batch, new_tokens, dtype=torch.long, device=device)

        def step():
            logits = model.decode_step(self.tok, self.pos, self.state)
            self.tok.copy_(sample_next(logits, temperature))
            self.pos += 1
            self.out.index_copy_(1, (self.pos - prompt_len).view(1), self.tok[:, None])

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
    key = (prompt.size(0), prompt.size(1), new_tokens, temperature)
    runner = getattr(model, "_graphed", None)
    if runner is None or runner.key != key:
        release_graph(model)
        runner = model._graphed = GraphedDecoder(model, *key)
    return runner(prompt)


def release_graph(model):
    if getattr(model, "_graphed", None) is not None:
        model._graphed = None
        torch.cuda.empty_cache()
