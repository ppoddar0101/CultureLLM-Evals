# Data

Seed WVS questions and GPT-4 augmented paraphrases. Large survey datasets are not included.

## Included Files

| File | Description |
|------|-------------|
| `WVQ.jsonl` | 50 seed World Values Survey questions |
| `WVQ.csv` | Same questions in CSV format |
| `new_WVQ_500.jsonl` | 500 GPT-4 generated paraphrases |
| `new_WVQ_1000.jsonl` | 1000 GPT-4 generated paraphrases |
| `culture_context.jsonl` | Cultural context prompts for evaluation |

## Required Downloads

1. **World Values Survey Wave 7** (US, India, Brazil): https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
2. **European Values Study** (France, Italy): https://europeanvaluesstudy.eu/
3. **SimBench**: https://huggingface.co/datasets/pitehu/SimBench
4. **Social IQA**: https://huggingface.co/datasets/allenai/social_i_qa
5. **CultureBank**: Auto-loaded via `datasets` library
6. **PRISM / PersonalizedRewardBench**: Auto-loaded via `datasets` library
7. **IPIP-NEO 120**: https://ipip.ori.org/
