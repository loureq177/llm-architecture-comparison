#!/bin/bash
#SBATCH --job-name=deltablock
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN before sbatch (README)}"
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=${HF_HOME:-$SCRATCH/hf} UV_CACHE_DIR=$SCRATCH/uv TRITON_CACHE_DIR=$SCRATCH/triton
export UV_PROJECT_ENVIRONMENT=.venv-$(uname -m) UV_FROZEN=1
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$HOME/.local/bin:$PATH
fi

uv sync
uv run python prepare_data.py --preset helios --workers $(( $(nproc) < 128 ? $(nproc) : 128 ))
end=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)
for attempt in 1 2 3; do
    uv run python -m torch.distributed.run --standalone --nproc_per_node=4 \
        train_all.py --preset helios --models deltablock --resume --deadline $(( end - 2700 )) && break
done
if [ -f checkpoints/helios/deltablock.pt ]; then
    uv run python tracking.py upload --preset helios --models deltablock || true   # the model first
    uv run python evaluate.py --preset helios --models deltablock
    uv run python tracking.py upload --preset helios --models deltablock            # + results
fi
