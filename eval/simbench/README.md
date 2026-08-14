# SimBench — Human Behavior Simulation

Evaluates how well models replicate group-level human response distributions on behavioral/social science survey questions.

## Metric

**SimBench Score** (higher is better, 0 = uniform random, 100 = perfect match):

SimBench_Score = 100 × (1 − TV(model, human) / TV(uniform, human))


Where TV is Total Variation distance.

## Data

Download `SimBenchPop.pkl` and `SimBenchGrouped.pkl` from [HuggingFace](https://huggingface.co/datasets/pitehu/SimBench). The evaluation filters for country-specific subsets (US, Brazil, India, France, Italy).

## Scripts

| Script | Description |
|--------|-------------|
| `generate_answers.py` | Main script — handles base model, CultureLLM adapters, and Gemini via CLI args |
| `calculate_simbench_score.py` | Computes SimBench scores from generated answer files |
| `generate_answers_culturellm.py` | Standalone CultureLLM variant (alternative to using `generate_answers.py` with `--adapter_path`) |
| `generate_answers_gemini.py` | Standalone Gemini variant (alternative to using `generate_answers.py` with `--model_name gemini-2.5-flash`) |

## Usage

**Step 1 — Generate answers (pick one):**

```bash
# CultureLLM model
python generate_answers.py \
    --input_file SimBenchTargets_US.pkl \
    --output_file results/us_culturellm_verb.pkl \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter_path "/path/to/culturellm-us-adapter" \
    --method verbalized

# Base model (no adapter)
python generate_answers.py \
    --input_file SimBenchTargets_US.pkl \
    --output_file results/us_base_verb.pkl \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --method verbalized

# Gemini
python generate_answers.py \
    --input_file SimBenchTargets_US.pkl \
    --output_file results/us_gemini_verb.pkl \
    --model_name "gemini-2.5-flash" \
    --method verbalized
```

**Step 2 — Score:**

```bash
python calculate_simbench_score.py --input results/us_culturellm_verb.pkl
```

## SLURM Examples

See `slurm/` for example job scripts for each country and model.
