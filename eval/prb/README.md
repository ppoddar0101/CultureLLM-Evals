# Personalized RewardBench (PRB) — Reward Model Personalization

Evaluates models as personalized reward models using the Personalized RewardBench dataset.

## Metric

**Pairwise Accuracy** — how often the model correctly identifies the user-preferred response. Tested across 3 seeds (42, 123, 456) with randomized choice position to measure position bias. Results broken down by subset: Art & Entertainment, Lifestyle & Personal Development, Society & Culture.

## Data

Loaded automatically from HuggingFace (`QiyaoMa/Personalized-RewardBench`).

## Scripts

| Script | Description |
|--------|-------------|
| `pairwise.py` | Evaluate local CultureLLM/base model (GPU) |
| `pairwise_gemini.py` | Evaluate Gemini Flash (API, no GPU) |
| `aggregate_results.py` | Aggregate results across seeds and subsets into summary table |
| `category_breakdown.py` | Per-category accuracy analysis |

## Usage

**Step 1 — Run evaluation:**

```bash
# CultureLLM-US (run for each subset and seed)
python pairwise.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter_path "/path/to/culturellm-us-adapter" \
    --subset "Art_and_Entertainment" \
    --seed 42 \
    --output_dir prb_results

# Base model (no adapter)
python pairwise.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --subset "Art_and_Entertainment" \
    --seed 42 \
    --output_dir prb_results

# Gemini Flash
export GOOGLE_API_KEY="your-key"
python pairwise_gemini.py \
    --model_name gemini-2.5-flash \
    --subset "Art_and_Entertainment" \
    --seed 42 \
    --output_dir prb_results
```

**Step 2 — Aggregate:**

```bash
python aggregate_results.py --results_dir prb_results
```

**SLURM (recommended):**

`run_prb_eval.slurm` runs all 18 jobs (3 subsets × 2 models × 3 seeds) as an array job. `run_prb_eval_gemini.slurm` loops through all combinations for Gemini.
