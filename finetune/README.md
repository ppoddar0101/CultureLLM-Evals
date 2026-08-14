# Fine-Tuning

LoRA fine-tuning of Llama-3.1-8B-Instruct on country-specific WVS/EVS data.

## Script

`llama_finetune.py` handles training, evaluation, and inference. The `--country` flag sets the wandb project name and run name automatically.

## Usage

```bash
python llama_finetune.py train \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --new_model "./models/culturellm-india-8b-morality" \
    --data_files "data/India/Finetune/WVQ_India_v2_llama_minus_morality.jsonl" \
    --country "India"
```

## Hyperparameters

| Parameter | US/English | All Others |
|-----------|-----------|------------|
| Learning rate | 2e-5 | 1e-5 |
| Epochs | 6 | 12 |
| LoRA rank | 64 | 64 |
| LoRA alpha | 16 | 16 |
| Batch size | 4 | 4 |
| Train/test split | 80/20 (seed=42) | 80/20 (seed=42) |

## SLURM

See `slurm/` for per-country job scripts. Each requests 1 GPU and 48 hours. Tested on NVIDIA A100 80GB and A6000 GPUs.

## Output

Adapters are saved to the `--new_model` path (e.g., `models/culturellm-india-8b-morality/`). These are LoRA adapters, not full model weights — they're small (~50MB) and require the base model for inference.
