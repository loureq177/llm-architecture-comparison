import os
from dataclasses import dataclass, field

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PRESETS = {
    "4gb": dict(
        run_name="4gb",
        dim_model=384, n_head=6, n_layers=8, max_seq_len=1024, train_shards=8,
        batch_size=4, grad_accum_steps=8, micro_batch={"mamba": 4},
        max_steps=20700, warmup_steps=1000,
        eval_interval=1000, eval_batches=32, save_interval=250,
    ),
    # RTX 5070 Ti: same recipe as 4gb (20 tokens/param, 32k tokens/step, cosine), width x2;
    # ~48 h for all four models, estimated from the 4gb logs - check the ETA lines before committing.
    "16gb": dict(
        run_name="16gb",
        dim_model=768, n_head=12, n_layers=8, max_seq_len=1024, train_shards=20,
        batch_size=16, grad_accum_steps=2, micro_batch={"deltablock": 8},
        max_steps=58700, warmup_steps=1000,
        eval_interval=1000, eval_batches=8, save_interval=500,
    ),
    "helios": dict(
        run_name="helios", lm_subset="sample/100BT", train_shards=485,
        dim_model=1280, n_head=10, n_layers=24, max_seq_len=2048,
        batch_size=8, grad_accum_steps=64, learning_rate=4e-4, lr_schedule="wsd",
        max_steps=40000, warmup_steps=1000,
        eval_interval=1000, eval_batches=16, save_interval=250,
    ),
}


def _default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    dim_model: int = 512
    n_head: int = 8
    n_layers: int = 8
    mamba_layers_per_layer: int = 2
    max_seq_len: int = 1024
    tie_embeddings: bool = True

    run_name: str = ""
    lm_dataset_name: str = "HuggingFaceFW/fineweb-edu"
    lm_subset: str = "sample/10BT"
    lm_heldout_subset: str = "sample/10BT"
    lm_heldout_docs: int = 4000
    train_shards: int = 2
    tokenizer_name: str = "gpt2"
    data_dir_name: str = "data"

    batch_size: int = 16
    grad_accum_steps: int = 2
    micro_batch: dict = field(default_factory=dict)
    loss_chunk: int = 512
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"
    decay_fraction: float = 0.2
    deadline: float | None = None
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_steps: int = 30000
    seed: int = 1337
    use_bf16: bool = True

    diff_eps: float = 1e-3

    attn_every: int = 4
    block_size: int = 16
    diff_weight: float = 0.3
    ar_fraction: float = 0.7
    joint_start: int | None = None

    mamba_d_state: int = 16
    mamba_expand: int = 2
    mamba_d_conv: int = 4
    use_mamba_kernel: bool = True

    log_interval: int = 50
    eval_interval: int = 1000
    eval_batches: int = 50
    save_interval: int = 5000

    root_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    device: str = field(default_factory=_default_device)

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.root_dir, "checkpoints", self.run_name)

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.root_dir, "logs", self.run_name)

    @property
    def results_dir(self) -> str:
        return os.path.join(self.root_dir, "results", self.run_name)

    @property
    def data_dir(self) -> str:
        sample = "" if self.lm_subset == "sample/10BT" else "-" + self.lm_subset.split("/")[-1]
        return os.path.join(self.root_dir, self.data_dir_name, f"fineweb-edu{sample}_{self.tokenizer_name}")

    @property
    def mamba_n_layers(self) -> int:
        return self.n_layers * self.mamba_layers_per_layer

    def apply_preset(self, name: str):
        for k, v in PRESETS[name].items():
            setattr(self, k, v)


cfg = Config()
