# WVS Morality — Moral Value Alignment

Evaluates how well models replicate country-specific moral value distributions from the World Values Survey.

## Metric

**KL Divergence / Wasserstein Distance** between model probability distributions over WVS morality response options and the ground truth human distributions from WVS Wave 7.

## Data

- **Ground truth**: `wvs_w7.csv` (WVS Wave 7 responses) — see `data/README.md` for download
- **Cultural norms**: `cultural-norms.json` (question mappings and cultural context)

## Scripts

| Script | Description |
|--------|-------------|
| `compute_morality.py` | Evaluate local CultureLLM/base model using log-probabilities (GPU) |
| `compute_morality_gemini.py` | Evaluate Gemini Flash (API, no GPU) |

## Usage

```bash
# Local model
python compute_morality.py \
    --checkpoint_dir models/culturellm-india-8b-morality \
    --country "India" \
    --gt_csv wvs_w7.csv \
    --norms_json cultural-norms.json

# Gemini Flash
export GOOGLE_API_KEY="your-key"
python compute_morality_gemini.py \
    --country "France" \
    --gt_csv wvs_w7.csv \
    --norms_json cultural-norms.json \
    --out_dir ./morality_outputs
```

Questions are batched (chunk_size=20) for efficiency. Results include per-question probability distributions and summary statistics.
