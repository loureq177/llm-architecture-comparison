# llm-architecture-comparison

Autoregressive Transformer, masked diffusion (MDLM), Mamba and DeltaBlock
(block diffusion on a 3:1 hybrid of KDA linear attention and softmax attention)
trained on FineWeb-Edu.

## Helios

On the login node, in `$SCRATCH`:

```bash
git clone https://github.com/loureq177/llm-architecture-comparison && cd llm-architecture-comparison
sbatch -A <grant>-gpu-gh200 helios.sbatch
```

The job (one node, 4x GH200, 48 h) builds the environment, tokenizes ~50B
tokens of FineWeb-Edu sample-100BT and trains DeltaBlock 0.55B on the 4 GPUs
until 30 minutes before the end of the job. Progress: `deltablock-<job id>.log`,
metrics: `logs/helios/deltablock.jsonl`, model: `checkpoints/helios/deltablock.pt`.
Submitting it again resumes. Set `HF_TOKEN` to avoid Hugging Face download limits.
