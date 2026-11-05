"""DeltaBlock: no information leaks between the streams (GPU, fla kernels)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fla kernels need CUDA")

V, D, H, LAYERS, T, L, B = 300, 128, 2, 4, 128, 16, 2


@pytest.fixture(params=["kda", "gdn2"])
def model(request):
    from models.deltablock.model import HybridLM

    torch.manual_seed(0)
    return HybridLM(V, D, H, LAYERS, mixer=request.param, block_size=L).cuda().eval()


def tokens():
    return torch.randint(0, V, (B, T), device="cuda")


def masked(x, p=0.5):
    return torch.where(torch.rand(x.shape, device=x.device) < p, V, x)


@torch.no_grad()
def test_clean_stream_ignores_noisy_stream(model):
    x = tokens()
    _, c1 = model.two_stream(masked(x), x)
    _, c2 = model.two_stream(tokens(), x)  # completely different noisy stream
    assert (c1 - c2).abs().max() < 1e-5


@torch.no_grad()
def test_clean_stream_equals_autoregressive_mode(model):
    x = tokens()
    _, c = model.two_stream(masked(x), x)
    ar = model.features(x)
    assert (c - ar).abs().max() / ar.abs().max() < 1e-2  # kernel chunking differs


@torch.no_grad()
@pytest.mark.parametrize("b", [0, 3, 7])
def test_noisy_block_sees_only_its_block_and_the_clean_past(model, b):
    x, xn = tokens(), masked(tokens())
    n1, _ = model.two_stream(xn, x)
    x2, xn2 = x.clone(), xn.clone()
    x2[:, b * L:] = torch.randint(0, V, x2[:, b * L:].shape, device="cuda")  # clean blocks >= b
    other = torch.ones(T, dtype=torch.bool, device="cuda")
    other[b * L:(b + 1) * L] = False
    xn2[:, other] = masked(tokens())[:, other]  # noisy tokens of every other block
    n2, _ = model.two_stream(xn2, x2)
    blk = slice(b * L, (b + 1) * L)
    assert (n1[:, blk] - n2[:, blk]).abs().max() < 1e-5


def test_joint_loss_and_elbo(model):
    model.train()
    x = tokens()
    loss = model.joint_loss(x)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    nll, n = model.elbo_nll(x, torch.full((B,), T, device="cuda"))
    assert n == B * (T - 1) and torch.isfinite(nll) and nll > 0
