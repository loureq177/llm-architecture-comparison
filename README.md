# llm-architecture-comparison

Transformer, MDLM, Mamba and DeltaBlock (block diffusion on a KDA/attention hybrid)
trained on FineWeb-Edu. Needs a Hugging Face "Write" token
(huggingface.co/settings/tokens); models, logs and results go to the private repo
`<user>/llm-architecture-comparison`, live curves to the private Space
`<user>/llm-architecture-comparison-live`.

```bash
read -rs HF_TOKEN && export HF_TOKEN   # paste the token (not shown, not kept in the history)
```

## Helios (DeltaBlock 0.55B, 4x GH200, 48 h)

```bash
ssh <plgrid login>@helios.cyfronet.pl
cd $SCRATCH
git clone https://github.com/loureq177/llm-architecture-comparison && cd llm-architecture-comparison
read -rs HF_TOKEN && export HF_TOKEN
sbatch -A <grant>-gpu-gh200 helios.sbatch
```

That is all: the job prepares the data, trains, uploads the model, evaluates it
and uploads the results. Progress: `squeue -u $USER`, `tail -f deltablock-<job id>.log`.
If the job stops early, run the last two commands again in the same directory: it resumes.

## Single GPU (16 GB, all four models)

```bash
MAX_JOBS=4 uv sync --frozen --extra mamba
uv run --no-sync python prepare_data.py --preset 16gb --workers 4
nohup uv run --no-sync python train_all.py --preset 16gb --resume > train-16gb.log 2>&1 &
# when done:
uv run --no-sync python evaluate.py --preset 16gb
uv run --no-sync python tracking.py upload --preset 16gb
```

## Download

```bash
uv run --no-sync python tracking.py download --preset helios --models deltablock
uv run --no-sync python tracking.py download --preset 16gb
```
