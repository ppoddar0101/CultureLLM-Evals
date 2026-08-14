# IPIP — Big Five Personality Evaluation

Evaluates whether culturally fine-tuned models exhibit personality trait distributions that match the target country's population.

## Metric

**Wasserstein Distance** per Big Five trait (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) between model response distribution and human ground truth from Johnson's IPIP-NEO dataset.

## Data

- **IPIP-NEO 120 questions**: 120 items, 24 per trait, each with a 5-point Likert scale
- **Ground truth**: Johnson's IPIP-NEO dataset filtered to the target country

## Scripts

| Script | Description |
|--------|-------------|
| `personality_eval_verbalize.py` | Evaluate local CultureLLM/base model (GPU) |
| `personality_eval_gemini.py` | Evaluate Gemini Flash (API, no GPU) |

## Usage

```bash
# Local model
python personality_eval_verbalize.py --country "United States"
python personality_eval_verbalize.py --country "India"

# Gemini Flash
export GOOGLE_API_KEY="your-key"
python personality_eval_gemini.py --country "United States"
python personality_eval_gemini.py --country "France"
```

The `--country` flag controls both the persona prompt and which ground truth population to compare against.
