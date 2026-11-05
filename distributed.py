import datetime
import os

import torch
import torch.distributed as dist

BUCKET = 2**25


def init():
    if int(os.environ.get("WORLD_SIZE", 1)) == 1:
        return
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count())
    dist.init_process_group(os.environ.get("DIST_BACKEND", "nccl"), timeout=datetime.timedelta(hours=1))


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return rank() == 0


def average_gradients(params):
    if world() == 1:
        return
    buckets, size = [[]], 0
    for g in (p.grad for p in params if p.grad is not None):
        if buckets[-1] and size + g.numel() > BUCKET:
            buckets.append([])
            size = 0
        buckets[-1].append(g)
        size += g.numel()
    for bucket in buckets:
        flat = torch.cat([g.reshape(-1) for g in bucket])
        dist.all_reduce(flat)
        flat /= world()
        for g, f in zip(bucket, flat.split([g.numel() for g in bucket])):
            g.copy_(f.view_as(g))


def mean(x: torch.Tensor) -> torch.Tensor:
    if world() == 1:
        return x
    x = x.detach().clone().float()
    dist.all_reduce(x)
    return x / world()


def broadcast(value: float) -> float:
    if world() == 1:
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.broadcast(t, 0)
    return float(t)


def close():
    if dist.is_initialized():
        dist.destroy_process_group()
