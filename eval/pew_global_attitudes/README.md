# Pew Global Attitudes — Political/Social Value Alignment

Evaluates how well models replicate country-specific political and social attitudes from Pew Research Center surveys.

## Metric

**KL Divergence / Wasserstein Distance** between model probability distributions and Pew survey ground truth for each country.

## Data

- **Pew questions**: `pew_questions.json` and country-specific variants — generated from the ATP Master Dataset
- **Ground truth survey data**: `ga_master_<country>_long_clean.csv` files
- **Cultural norms**: `cultural-norms.json`

These files are too large for the repo. Generate them using the Pew Research data (see `data/README.md`).

## Scripts

| Script | Description |
|--------|-------------|
| `compute_global_attitudes.py` | Evaluate local CultureLLM/base model (GPU) |
| `compute_global_attitudes_flash.py` | Evaluate Gemini Flash (API, no GPU) |

## Usage

```bash
# Local model
python compute_global_attitudes.py \
    --country "India" \
    --questions_json pew_questions_india.json \
    --ga_csv ga_master_india_long_clean.csv \
    --norms_json cultural-norms.json \
    --chunk_size 4

# Gemini Flash
export GOOGLE_API_KEY="your-key"
python compute_global_attitudes_flash.py \
    --country "Italy" \
    --questions_json pew_questions.json \
    --ga_csv ga_master_italy_long_clean.csv \
    --norms_json cultural-norms.json \
    --chunk_size 4
```
