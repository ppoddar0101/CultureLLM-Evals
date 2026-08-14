# Social IQA — Social Reasoning Evaluation

Evaluates social commonsense reasoning ability using the Social IQA benchmark.

## Metric

**Accuracy** — percentage of questions where the model assigns the highest probability to the correct answer (A, B, or C).

## Data

Download Social IQA from [HuggingFace](https://huggingface.co/datasets/allenai/social_i_qa) or place the `.jsonl` files in a `social_iqa_data/` directory.

## Scripts

| Script | Description |
|--------|-------------|
| `eval_social_iqa.py` | Evaluate local CultureLLM/base model using log-probabilities (GPU) |
| `eval_social_iqa_gemini.py` | Evaluate Gemini Flash via chat-based scoring (API, no GPU) |

## Usage

```bash
# Base model
python eval_social_iqa.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --split validation \
    --output results_siqa_val_base.json

# CultureLLM-US
python eval_social_iqa.py \
    --model /path/to/culturellm-us-adapter \
    --split validation \
    --output results_siqa_val_us.json

# Gemini Flash
export GOOGLE_API_KEY="your-key"
python eval_social_iqa_gemini.py \
    --model gemini-2.5-flash \
    --split validation \
    --output results_siqa_val_gemini.json
```

Run on both `validation` and `test` splits for complete results.
