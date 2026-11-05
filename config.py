import os
from dataclasses import dataclass, field

# Batches of varying size (masked diffusion) fragment the default CUDA allocator;
# must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Presets: language modelling on FineWeb-Edu, one per hardware tier, sized so
# that all three models train in ~30 h (about 10 h per model) on that hardware.
# Everything not listed keeps the Config default.
#
# Training budget = 20 tokens per parameter (compute-optimal, Hoffmann et al.
# 2022), so no model is undertrained for its size, and always less than one
# pass over the training pieces, so no sequence is seen twice (no overfitting;
# train_all.py refuses a budget that would need a second epoch).
# A piece is 100k documents = ~103M GPT-2 tokens; 32 sequences of 1024 tokens
# per optimizer step on 4 GB, 256 on 48 GB.
PRESETS = {
    # 4 GB laptop GPU (RTX 3050 Ti): ~34M params, 678M tokens per model
    # (0.82 epoch of 8 pieces). Measured: transformer 8.5 h, MDLM 6.3 h, Mamba 14 h.
    "4gb": dict(
        run_name="4gb",
        dim_model=384, n_head=6, n_layers=8, max_seq_len=1024, train_shards=8,
        batch_size=4, grad_accum_steps=8, micro_batch={"mamba": 4},
        max_steps=20700, warmup_steps=1000,
        eval_interval=1000, eval_batches=32, save_interval=250, sample_interval=4000,
    ),
    # 48 GB GPU (A6000 / A40 / RTX 6000 Ada): GPT-2 small shape, ~124M params,
    # 2.49B tokens per model (0.92 epoch of 26 pieces), ~30-45 h on an A6000.
    "48gb": dict(
        run_name="48gb",
        dim_model=768, n_head=12, n_layers=12, max_seq_len=1024, train_shards=26,
        batch_size=32, grad_accum_steps=8, micro_batch={"mamba": 16},
        max_steps=9500, warmup_steps=500,
        eval_interval=500, eval_batches=32, save_interval=250, sample_interval=2000,
    ),
}


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _default_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # Architecture. n_layers is the Transformer/Diffusion depth; Mamba uses
    # mamba_layers_per_layer * n_layers blocks so that parameter counts match
    # (one Transformer layer ~ 12 d^2, one Mamba block ~ 6 d^2).
    dim_model: int = 512
    n_head: int = 8
    n_layers: int = 8
    mamba_layers_per_layer: int = 2
    max_seq_len: int = 1024
    tie_embeddings: bool = True

    # Data: plain language modelling on a stream of documents
    run_name: str = ""  # sub-directory of checkpoints/ logs/ results/ ("" = top level)
    # FineWeb-Edu (Penedo et al. 2024, ODC-By 1.0), its 10B-token random sample:
    # 14 parquet shards of ~690k documents; the last one is held out for evaluation.
    lm_dataset_name: str = "HuggingFaceFW/fineweb-edu"
    lm_subset: str = "sample/10BT"
    lm_heldout_docs: int = 4000  # validation and test documents (each), from the last shard
    train_shards: int = 2  # training pieces of 100k documents (~103M tokens) each
    tokenizer_name: str = "gpt2"
    data_dir_name: str = "data"

    # Optimisation
    batch_size: int = 16  # micro-batch
    grad_accum_steps: int = 2
    # Per-model micro-batch (e.g. {"mamba": 2}) for models that need more memory;
    # the number of sequences per optimizer step stays batch_size * grad_accum_steps.
    micro_batch: dict = field(default_factory=dict)
    loss_chunk: int = 512  # tokens per slice of the LM head in the loss (see losses.py)
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_steps: int = 30000
    seed: int = 1337
    use_bf16: bool = True

    # Diffusion (MDLM)
    diff_eps: float = 1e-3  # min masking time t

    # DeltaBlock
    deltablock_mixer: str = "kda"
    attn_every: int = 4  # every 4th layer is softmax attention
    block_size: int = 16
    intra_block: bool = True  # noisy tokens also attend to each other in KDA layers
    diff_weight: float = 0.3  # L = L_AR + diff_weight * L_diff
    ar_fraction: float = 0.7  # first 70 % of the steps: next-token loss only

    # Mamba
    mamba_d_state: int = 16
    mamba_expand: int = 2
    mamba_d_conv: int = 4
    use_mamba_kernel: bool = True  # use mamba-ssm CUDA kernel when available

    # Logging, evaluation, checkpointing
    log_interval: int = 50
    eval_interval: int = 1000
    eval_batches: int = 50
    sample_interval: int = 5000
    save_interval: int = 5000

    root_dir: str = field(default_factory=_default_root)
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
        # token stream, independent of the context length; shared by all presets
        name = f"fineweb-edu_{self.tokenizer_name}"
        return os.path.join(self.root_dir, self.data_dir_name, name)

    @property
    def mamba_n_layers(self) -> int:
        return self.n_layers * self.mamba_layers_per_layer

    def apply_preset(self, name: str):
        for k, v in PRESETS[name].items():
            setattr(self, k, v)


cfg = Config()
