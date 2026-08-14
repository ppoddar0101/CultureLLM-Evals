# Data Processing

Scripts to generate country-specific training data from World Values Survey / European Values Study responses.

## Pipeline

1. **`data_process.py`** — Reads raw WVS/EVS CSVs, extracts country-specific answers for 50 WVQ questions, and generates augmented training data using GPT-4 paraphrases.

2. **`convert_to_llama.py`** — Converts the generated JSONL data into Llama-3 instruction format and optionally removes morality questions (Q196-Q198).

## Usage

```bash
# Generate country data from WVS
python data_process.py --country "India" --country_code "356"
python data_process.py --country "Brazil" --country_code "76"

# Convert to Llama format
python convert_to_llama.py \
    --input data/India/Finetune/WVQ_India_v2.jsonl \
    --output data/India/Finetune/WVQ_India_v2_llama_minus_morality.jsonl
```

## Country Codes

| Country | WVS Code | Notes |
|---------|----------|-------|
| US | 840 | WVS Wave 7 (50 questions) |
| India | 356 | WVS Wave 7 (50 questions) |
| Brazil | 76 | WVS Wave 7 (50 questions) |
| France | 250 | EVS Joint dataset (24 questions available) |
| Italy | 380 | EVS Joint dataset (24 questions available) |
