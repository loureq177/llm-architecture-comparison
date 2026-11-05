# llm-architecture-comparison

Autoregressive Transformer, masked diffusion (MDLM), Mamba and DeltaBlock
(block diffusion on a 3:1 hybrid of KDA linear attention and softmax attention)
trained on FineWeb-Edu.

## Helios

GPU nodes are aarch64 and the login node is x86_64, so build the environment
on a GPU node:

```bash
srun -p plgrid-gpu-gh200 -A <grant>-gpu-gh200 --gres=gpu:1 -t 01:00:00 --pty bash
cd $SCRATCH
git clone https://github.com/loureq177/llm-architecture-comparison && cd llm-architecture-comparison
curl -LsSf https://astral.sh/uv/install.sh | sh
export HF_HOME=$SCRATCH/hf
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
uv run pytest -q tests
uv run python prepare_data.py --preset 4gb
uv run python -m experiments.mixer_ablation check
```
