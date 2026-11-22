# llm-architecture-comparison

Autoregressive Transformer, masked diffusion (MDLM), Mamba and DeltaBlock
(block diffusion on a 3:1 hybrid of KDA linear attention and softmax attention)
trained on FineWeb-Edu.

Live training curves, results and the trained models go to your Hugging Face
account (both private): the dashboard Space `<user>/llm-architecture-comparison-live`
and the model repository `<user>/llm-architecture-comparison`. Every command below
that talks to Hugging Face needs a token with the "Write" role from
https://huggingface.co/settings/tokens in the shell:

```bash
export HF_TOKEN=<token>
```

## Helios (DeltaBlock 0.55B, 4x GH200, 48 h)

```bash
ssh <plgrid login>@helios.cyfronet.pl
cd $SCRATCH
git clone https://github.com/loureq177/llm-architecture-comparison && cd llm-architecture-comparison
export HF_TOKEN=<token>
sbatch -A <grant>-gpu-gh200 helios.sbatch
```

Nothing else to run. The job builds the environment, tokenizes ~50B tokens of
FineWeb-Edu sample-100BT, trains on the 4 GPUs until 45 minutes before the end,
uploads the model to Hugging Face, evaluates it (test perplexity, generation speed
and quality) and uploads the results. It refuses to start without `HF_TOKEN`.

While it runs:

```bash
squeue -u $USER                 # job state (PD = waiting, R = running)
tail -f deltablock-<job id>.log # progress
```

Live curves: huggingface.co/spaces/<user>/llm-architecture-comparison-live, run
`helios/deltablock` (the first open of the Space takes a minute to start).

If the job is cancelled or fails, submit it again from the same directory (with
`HF_TOKEN` exported): it resumes from the last checkpoint.

## Single GPU (RTX 5070 Ti, 16 GB; all four models, ~48 h)

```bash
git clone https://github.com/loureq177/llm-architecture-comparison && cd llm-architecture-comparison
MAX_JOBS=4 uv sync --frozen --extra mamba
uv run --no-sync python prepare_data.py --preset 16gb --workers 4
nohup uv run --no-sync python train_all.py --preset 16gb --resume > train-16gb.log 2>&1 &
```

Running the last command again resumes. Then evaluate (~1-2 h) and, once
`evaluate_16gb.out` ends with `saved .../metrics.json`, upload:

```bash
nohup uv run --no-sync python evaluate.py --preset 16gb > evaluate_16gb.out 2>&1 &
export HF_TOKEN=<token>
uv run --no-sync python tracking.py upload --preset 16gb
```

## Getting the models and results

On any machine with this repository and `uv sync` done:

```bash
export HF_TOKEN=<token>
uv run --no-sync python tracking.py download --preset helios --models deltablock
uv run --no-sync python tracking.py download --preset 16gb
```

Files land where they were on the training machine: `checkpoints/<preset>/`,
`logs/<preset>/`, `results/<preset>/`.
