"""Correctness tests for the model implementations (CPU, tiny sizes)."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import token_nll, weighted_ce  # noqa: E402
from models.diffusion.model import DiffusionLanguageModel  # noqa: E402
from models.mamba.model import MambaLanguageModel, _parallel_scan  # noqa: E402
from models.transformer.model import LanguageModel  # noqa: E402

V, D, T = 100, 32, 24


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_parallel_scan_matches_sequential():
    a, b = torch.rand(2, 13, 5, 3), torch.randn(2, 13, 5, 3)
    h, ref = torch.zeros(2, 5, 3), []
    for t in range(13):
        h = a[:, t] * h + b[:, t]
        ref.append(h)
    assert torch.allclose(_parallel_scan(a, b), torch.stack(ref, 1), atol=1e-5)


def make_mamba():
    return MambaLanguageModel(V, D, 2, T, use_kernel=False).eval()


def test_mamba_is_causal():
    m = make_mamba()
    x = torch.randint(0, V, (2, T))
    y = x.clone()
    y[:, -1] = (y[:, -1] + 1) % V
    assert torch.allclose(m(x)[:, :-1], m(y)[:, :-1], atol=1e-5)


def test_mamba_recurrent_step_matches_parallel():
    m = make_mamba()
    x = torch.randint(0, V, (2, T))
    full = m(x)
    _, states = m(x[:, :10], return_states=True)
    for t in range(10, T):
        step_logits, states = m.step(x[:, t], states)
        assert torch.allclose(step_logits, full[:, t], atol=1e-4), t


def test_transformer_kv_cache_matches_full_forward():
    m = LanguageModel(V, D, 4, T, 2).eval()
    x = torch.randint(0, V, (2, T))
    full = m(x)
    logits, caches = m(x[:, :10], caches=[])
    for t in range(10, T):
        logits, caches = m(x[:, t : t + 1], caches=caches, pos_offset=t)
        assert torch.allclose(logits[:, 0], full[:, t], atol=1e-4), t


def test_generation_is_consistent_with_argmax():
    for m in (make_mamba(), LanguageModel(V, D, 4, T, 2).eval()):
        prompt = torch.randint(0, V, (2, 8))
        out = m.generate(prompt, 5)
        full = torch.cat([prompt, out], 1)
        assert torch.equal(m(full[:, :-1]).argmax(-1)[:, 7:], out)


def test_weight_tying_counts_params_once():
    tied = LanguageModel(V, D, 4, T, 2, tie_embeddings=True)
    untied = LanguageModel(V, D, 4, T, 2, tie_embeddings=False)
    n = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert n(untied) - n(tied) == V * D


def test_mamba_dt_init_in_paper_range():
    m = make_mamba()
    dt = torch.nn.functional.softplus(m.blocks[0].s6.dt_proj.bias)
    assert 1e-4 <= dt.min() and dt.max() <= 0.11


def test_chunked_ce_matches_autograd():
    # the closed-form, slice-by-slice loss must agree with F.cross_entropy + autograd
    h = torch.randn(23, D, dtype=torch.float64, requires_grad=True)
    w = torch.randn(V, D, dtype=torch.float64, requires_grad=True)
    y, tw = torch.randint(0, V, (23,)), torch.rand(23, dtype=torch.float64)
    ref = (torch.nn.functional.cross_entropy(h @ w.T, y, reduction="none") * tw).sum()
    ref_h, ref_w = torch.autograd.grad(3 * ref, (h, w))
    loss = weighted_ce(h, w, y, tw, chunk=5)
    got_h, got_w = torch.autograd.grad(3 * loss, (h, w))
    assert torch.allclose(loss, ref) and torch.allclose(got_h, ref_h) and torch.allclose(got_w, ref_w)
    assert torch.allclose((token_nll(h, w, y, chunk=5) * tw).sum(), ref)


def test_next_token_loss_ignores_padding():
    from train_utils import next_token_loss

    for m in (make_mamba(), LanguageModel(V, D, 4, T, 2).eval()):
        x, length = torch.randint(0, V, (3, T)), torch.tensor([T, 10, 17])
        pos = torch.arange(1, T)[None]
        targets = x[:, 1:].masked_fill(pos >= length[:, None], -100)
        ref = torch.nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, V), targets.reshape(-1), ignore_index=-100)
        assert torch.allclose(next_token_loss(m, x, length), ref, atol=1e-5)


def make_diffusion():
    return DiffusionLanguageModel(V, D, 4, T, 2)


def test_diffusion_loss_matches_dense_reference():
    # head on masked positions only == masked, 1/t-weighted cross-entropy over all positions
    m = make_diffusion().eval()
    x = torch.randint(0, V, (8, T))
    torch.manual_seed(1)
    loss = m.training_loss(x)
    torch.manual_seed(1)
    t = (torch.randperm(8) + torch.rand(8)) / 8
    t = 1e-3 + (1 - 1e-3) * t
    x_t, mask = m.noise(x, t, torch.ones_like(x, dtype=torch.bool))
    ce = torch.nn.functional.cross_entropy(m(x_t).transpose(1, 2), x, reduction="none")
    ref = ((ce * mask).sum(1) / t / T).mean()
    assert torch.allclose(loss, ref, atol=1e-5)


def test_diffusion_sampler_matches_dense_reference():
    # LM head on revealed tokens only == argmax over the full logits at every step
    m = make_diffusion().eval()
    prompt, steps, G = torch.randint(0, V, (3, 6)), 5, T - 6
    torch.manual_seed(2)
    out = m.sample(prompt, G, steps=steps)
    torch.manual_seed(2)
    x = torch.cat([prompt, torch.full((3, G), m.mask_token_id)], 1)
    for i in range(steps):
        sampled = m(x).argmax(-1)
        unmask = (x == m.mask_token_id) & (torch.rand(x.shape) < 1.0 / (steps - i))
        x = torch.where(unmask, sampled, x)
    assert torch.equal(out, x[:, 6:])


def test_diffusion_loss_at_init_is_log_vocab():
    # random model => CE ~ log V, ELBO weighting makes E[(1/t) * masked_frac * CE] ~ CE
    m = make_diffusion().eval()
    x = torch.randint(0, V, (64, T))
    loss = m.training_loss(x)
    assert abs(loss.item() - torch.log(torch.tensor(float(V))).item()) < 0.5


def test_diffusion_elbo_matches_training_objective():
    m = make_diffusion().eval()
    x = torch.randint(0, V, (32, T))
    length = torch.full((32,), T)
    nll, n = m.elbo_nll(x, length, n_mc=8)
    assert abs(nll.item() / n.item() - torch.log(torch.tensor(float(V))).item()) < 0.5


def test_diffusion_sampler_fills_everything_and_keeps_prompt():
    m = make_diffusion().eval()
    prompt = torch.randint(0, V, (2, 10))
    for remask in ("random", "confidence"):
        for steps in (1, 4, 8):
            out = m.sample(prompt, 8, steps=steps, remask=remask)
            assert out.shape == (2, 8)
            assert (out < V).all(), "[MASK] left in output"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and __import__("models.mamba.model", fromlist=["x"]).HAS_KERNEL),
    reason="needs a GPU and the mamba-ssm package",
)
def test_mamba_kernel_matches_reference():
    """Run on the training server: `uv sync --extra mamba && pytest tests`."""
    ref = MambaLanguageModel(V, 64, 2, 64, use_kernel=False).cuda().eval()
    ker = MambaLanguageModel(V, 64, 2, 64, use_kernel=True).cuda().eval()
    ker.load_state_dict(ref.state_dict())
    x = torch.randint(0, V, (2, 64), device="cuda")
    assert torch.allclose(ref(x), ker(x), atol=1e-3, rtol=1e-3)
    out_ref, out_ker = ref.generate(x[:, :32], 16), ker.generate(x[:, :32], 16)
    assert torch.equal(out_ref, out_ker)


@pytest.fixture
def lm_stream(tmp_path):
    """A small random token stream laid out like the prepared FineWeb-Edu data."""
    from config import cfg

    saved = dict(vars(cfg))
    cfg.root_dir, cfg.device = str(tmp_path), "cpu"
    cfg.max_seq_len, cfg.train_shards, cfg.batch_size, cfg.grad_accum_steps = 16, 2, 4, 2
    os.makedirs(cfg.data_dir)
    rng = np.random.default_rng(0)
    for name, n in [("train_0000", 1000), ("train_0001", 700), ("validation", 300)]:
        rng.integers(0, V, n).astype(np.uint16).tofile(os.path.join(cfg.data_dir, f"{name}.bin"))
    yield cfg
    vars(cfg).update(saved)


def test_lm_stream_is_deterministic_and_resumable(lm_stream):
    from dataset import eval_batches, load_split, train_batches

    def take(batches, k):
        return torch.cat([next(batches)[0] for _ in range(k)])

    assert load_split("train").n == (1000 // 16 - 1) + (700 // 16 - 1)
    a = take(train_batches(0, 4), 2 * 30)  # 30 optimizer steps of 8 sequences: > 2 epochs
    assert a.shape == (240, 16) and a.max() < V
    assert torch.equal(a, take(train_batches(0, 2), 4 * 30))  # micro-batch does not matter
    assert torch.equal(a[17 * 8 : 18 * 8], take(train_batches(17, 4), 2))  # resume at step 17
    epoch = {tuple(row.tolist()) for row in a[:96]}
    assert len(epoch) == 96  # no window twice within an epoch
    assert not epoch & {tuple(row.tolist()) for row in a[96:192]}  # next epoch: shifted windows
    x, length = next(eval_batches("validation", 1, batch_size=3))
    stream = np.fromfile(os.path.join(lm_stream.data_dir, "validation.bin"), dtype=np.uint16)
    assert torch.equal(x.flatten(), torch.from_numpy(stream[:48].astype(np.int64)))
    assert length.tolist() == [16, 16, 16]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_decoding_matches_eager():
    torch.backends.cuda.matmul.allow_tf32 = False
    models = [LanguageModel(V, 64, 4, T, 2)]
    if __import__("models.mamba.model", fromlist=["x"]).HAS_KERNEL:
        models.append(MambaLanguageModel(V, 64, 2, T))
    for m in models:
        m = m.cuda().eval()
        prompt = torch.randint(0, V, (3, 4), device="cuda")
        eager = m.generate(prompt, T - 4)
        for _ in range(2):  # the second call replays the recorded graph
            assert torch.equal(m.generate(prompt, T - 4, cuda_graph=True), eager)
        sampled = m.generate(prompt[:2], 8, temperature=1.0, cuda_graph=True)
        assert sampled.shape == (2, 8) and (sampled < V).all()


def test_sampler_is_exact():
    # two-level sampler: frequencies match softmax(logits / T), padding never drawn
    from decoding import sample_next

    torch.manual_seed(3)
    vocab, n = 1000, 400_000  # 1000 is not a multiple of the 256-token block
    logits = torch.randn(vocab) * 2
    for temperature in (1.0, 0.7):
        p = torch.softmax(logits.double() / temperature, -1)
        tokens = sample_next(logits.expand(n, -1), temperature)
        assert tokens.max() < vocab
        freq = torch.bincount(tokens, minlength=vocab).double() / n
        assert 0.5 * (freq - p).abs().sum() < 0.02  # total variation (noise ~0.01)
        assert abs(-(freq * p.log()).sum() + (p * p.log()).sum()) < 0.01  # mean surprisal
