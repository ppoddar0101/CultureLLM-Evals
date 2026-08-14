# CultureBank — Cultural Knowledge Entailment

Evaluates whether model responses align with documented cultural norms using an LLM-as-judge entailment approach.

## Metric

**Entailment Rate** — percentage of responses where a GPT-4o judge determines the model's response entails (aligns with) the ground truth cultural description.

## Data

Loaded automatically from HuggingFace (`Sakana/CultureBank`). Filtered by cultural group (American, Brazilian, Indian, French, Italian). Uses the TikTok-sourced test split (1,169 samples for US).

## Pipeline

This is a **two-stage pipeline**:

1. **Generate** — prompt the model with cultural scenario questions (GPU required for local models)
2. **Score** — use GPT-4o as a judge to score entailment (CPU only, requires `OPENAI_API_KEY`)

## Scripts

| Script | Description |
|--------|-------------|
| `culturebank_generate.py` | Stage 1: Generate responses with local CultureLLM/base model (GPU) |
| `culturebank_generate_flash.py` | Stage 1: Generate responses with Gemini Flash (API, no GPU) |
| `culturebank_score.py` | Stage 2: Score responses with GPT-4o judge |

## Usage

**Stage 1 — Generate responses:**

```bash
# Local model (edit model/adapter paths inside the script)
python culturebank_generate.py

# Gemini Flash
export GOOGLE_API_KEY="your-key"
python culturebank_generate_flash.py --country "United States" --target_group "American"
python culturebank_generate_flash.py --country "France" --target_group "French"
python culturebank_generate_flash.py --country "India" --target_group "Indian"
python culturebank_generate_flash.py --country "Brazil" --target_group "Brazilian"
python culturebank_generate_flash.py --country "Italy" --target_group "Italian"
```

**Stage 2 — Score:**

```bash
export OPENAI_API_KEY="your-key"
python culturebank_score.py --input outputs/culturebank/responses_<timestamp>.csv
```

Use the same GPT-4o judge across all models for a fair comparison.
