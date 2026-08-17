# CultureLLM: Culturally-Aligned Language Models via World Values Survey Fine-Tuning

Fine-tuning Llama-3.1-8B-Instruct with LoRA adapters on World Values Survey (WVS) data to create culturally-aligned models for 5 countries, evaluated across 8 downstream cultural benchmarks.

## Overview

This project investigates whether fine-tuning LLMs on country-specific value survey responses improves their ability to simulate culturally-aligned behavior. We train separate LoRA adapters for the US, India, Brazil, France, and Italy, then evaluate on tasks spanning personality, morality, social reasoning, cultural knowledge, and reward modeling.

### Models

All models are LoRA adapters on top of `meta-llama/Llama-3.1-8B-Instruct`:

| Country | Data Source | Questions | Training Rows |
|---------|-------------|-----------|---------------|
| US/English | WVS Wave 7 | 47 | 1,973 |
| India | WVS Wave 7 | 47 | 1,471 |
| Brazil | WVS Wave 7 | 47 | 1,471 |
| France | EVS Joint | 21 | 665 |
| Italy | EVS Joint | 21 | 665 |

Question counts reflect removal of 3 morality/surveillance questions (Q196-Q198). France and Italy use the European Values Study since they are not in WVS Wave 7.

### Evaluation Benchmarks

| Benchmark | What it Measures | Method |
|-----------|-----------------|--------|
| **SimBench** | Human behavior simulation | Verbalized distributions, Total Variation distance |
| **CultureBank** | Cultural knowledge entailment | Free-text generation + GPT-4o judge |
| **IPIP Personality** | Big Five trait alignment | Verbalized Likert responses, Wasserstein distance |
| **WVS Morality** | Moral value alignment | Log-probability distributions |
| **Pew Global Attitudes** | Political/social attitude alignment | Log-probability distributions |
| **Social IQA** | Social reasoning ability | Log-probability scoring |
| **PRISM** | Personalized preference judgment | Pairwise generative RM, flip augmentation |
| **Personalized RewardBench** | Reward model personalization | Pairwise evaluation, 3 seeds |

All benchmarks are also run on **Gemini 2.5 Flash** as a commercial baseline.

## Repository Structure

```
CultureLLM/
├── README.md
├── requirements.txt
├── data_processing/
│ ├── data_process.py # WVS/EVS -> training data pipeline
│ └── convert_to_llama.py # Convert to Llama instruction format
├── finetune/
│ ├── llama_finetune.py # LoRA fine-tuning (parameterized by country)
│ └── slurm/ # Example SLURM job scripts
├── eval/
│ ├── culturebank/ # CultureBank grounded evaluation
│ ├── simbench/ # SimBench human simulation
│ ├── ipip/ # IPIP-NEO personality evaluation
│ ├── morality/ # WVS morality questions
│ ├── pew_global_attitudes/ # Pew Research Center surveys
│ ├── social_iqa/ # Social IQA reasoning
│ ├── prism/ # PRISM preference evaluation
│ └── prb/ # Personalized RewardBench
├── utils/
│ └── llm_response.py # Shared model loading utilities
└── data/ # Seed data (see data/README.md)
```

Each eval directory contains a local-model script, a Gemini API variant, and example SLURM job scripts.

## Model Weights

Pre-trained LoRA adapters for all 5 countries are available on [Google Drive](https://drive.google.com/drive/folders/1Xyg-K1sCVyg-CrtGWZ-Phkn0FitQrRku?usp=drive_link). Each zip contains the adapter weights (~150MB), not the full model — you still need the base `meta-llama/Llama-3.1-8B-Instruct` model to run inference.

## Setup

```bash
pip install -r requirements.txt
```

See `data/README.md` for dataset download instructions.

## Fine-Tuning

```bash
python finetune/llama_finetune.py train \
    --base_model "meta-llama/Llama-3.1-8B-Instruct" \
    --new_model "./models/culturellm-india-8b-morality" \
    --data_files "data/India/Finetune/WVQ_India_v2_llama_minus_morality.jsonl" \
    --country "India"
```

## Running Evaluations

Each evaluation has a local-model version (requires GPU) and a Gemini API version (requires `GOOGLE_API_KEY`).

**SimBench:**
```bash
python eval/simbench/generate_answers.py \
    --input_file SimBenchTargets_US.pkl \
    --output_file results/us_culturellm_verb.pkl \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter_path "models/culturellm-us-8b-morality" \
    --method verbalized
```

**CultureBank (two-stage):**
```bash
# Stage 1: Generate responses (GPU)
python eval/culturebank/culturebank_generate.py

# Stage 2: Score with GPT-4o judge (requires OPENAI_API_KEY)
python eval/culturebank/culturebank_score.py --input outputs/culturebank/responses_<timestamp>.csv
```

**Gemini baseline (any benchmark):**
```bash
export GOOGLE_API_KEY="your-key"
python eval/culturebank/culturebank_generate_flash.py \
    --country "United States" --target_group "American"
```

## Environment

Experiments were run on USC CARC with NVIDIA A100 80GB and A6000 GPUs.
