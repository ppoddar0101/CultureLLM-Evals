# PRISM — Personalized Preference Evaluation

Evaluates models as personalized preference judges using the PRISM dataset. Tests whether cultural fine-tuning improves a model's ability to judge which response better matches a user's preferences.

## Two Evaluations

This directory contains two related but different evaluations:

### 1. PRISM Personalized (`prism_pairwise.py`)

Uses the pre-made PRISM Personalized dataset (`MichaelR207/prism_personalized_0125`) with existing preference pairs. Evaluates all 5 CultureLLM models + base model as judges. Simpler setup — pairs are pre-constructed.

### 2. PRISM Original (`prism_original_pairwise.py`)

Uses the original PRISM utterances (raw scored responses) and constructs preference pairs from the scores. **This is the more informative evaluation** because it breaks down accuracy by user country, which revealed that PRISM's user population is 85%+ Anglophone (0 Brazil/India users, 7-8 France/Italy users), making it unsuitable for cross-cultural comparison.

## Metric

**Pairwise Accuracy** with flip augmentation — each pair is tested in both orderings (chosen-first and chosen-second) to measure and correct for position bias. Results are averaged across orderings and reported with 95% confidence intervals.

## Scripts

| Script | What it does | When to use |
|--------|-------------|-------------|
| `prism_original_pairwise.py` | Local model eval on original PRISM with country breakdowns | **Start here** for local models |
| `prism_gemini_resumable.py` | Gemini eval on original PRISM (supports sharding and resume) | For Gemini baseline |
| `prism_pairwise.py` | Local model eval on PRISM Personalized (simpler, no country breakdown) | Alternative simpler eval |
| `aggregate_prism_original.py` | Aggregate results from `prism_original_pairwise.py` | After running original eval |
| `aggregate_prism.py` | Aggregate results from `prism_pairwise.py` | After running personalized eval |
| `prism_demographic_analysis.py` | Analyze user demographics in PRISM | To see the Anglophone skew finding |

## Usage

**Recommended workflow (Original PRISM):**

```bash
# Step 1 — Run local models
python prism_original_pairwise.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --adapter_path /path/to/culturellm-us-adapter \
    --output_dir prism_original_results

# Step 2 — Run Gemini
export GOOGLE_API_KEY="your-key"
python prism_gemini_resumable.py \
    --output_dir prism_original_results \
    --model gemini-2.5-flash \
    --concurrency 50

# Step 3 — Aggregate
python aggregate_prism_original.py

# Step 4 — Demographic analysis
python prism_demographic_analysis.py
```

**SLURM (array job across all models):**

See `run_prism_eval.slurm` for a 6-model array job, or `run_prism_original_eval.slurm` for the sharded Gemini evaluation.
