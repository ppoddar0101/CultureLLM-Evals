import os
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from scipy.stats import ttest_ind
from scipy.stats import wasserstein_distance
import json
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--country", type=str, default="United States")
args = parser.parse_args()
COUNTRY = args.country

# -------------------------------------------------
# Config
# -------------------------------------------------

COUNTRY_TO_CULTURE_KEY = {
    "United States": "English",
    "Spain": "Spanish",
    "Germany": "Germany",
    "South Korea": "Korean",
    "Brazil": "Brazilian",
    "India": "Indian",
    "France": "French",
    "Italy": "Italian",
    "Turkey": "Turkish",
    "Portugal": "Portuguese",
    "China": "Chinese",
    "Bangladesh": "Bengali",
}

COUNTRY_TO_GT_LABEL = {
    "United States": "USA",   # <- replace with whatever shows up
    "Spain": "Spain",
    "Germany": "Germany",
    "Brazil": "Brazil",
    "India": "India",
    "France": "France",
    "Italy": "Italy",
    "South Korea": "South Korea",
    "Turkey": "Turkey",
    "Portugal": "Portugal",
    "China": "China",
    "Bangladesh": "Bangladesh",
}

CULTURE_JSON_PATH = "data/culture_context.jsonl"  # your json file name

def load_culture_context(country: str) -> str:
    culture_key = COUNTRY_TO_CULTURE_KEY.get(country)
    if culture_key is None:
        raise ValueError(f"No culture key mapped for country: {country!r}")
    
    with open(CULTURE_JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if culture_key in entry:
                return entry[culture_key]
    
    raise ValueError(f"No culture context found for key: {culture_key!r}")

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "models/culturellm-france-8b-morality"

USE_ADAPTER = True
N_USERS = 1000

GROUND_TRUTH_PATH = "gt.csv"  # <-- change if needed

# -------------------------------------------------
# Load Model
# -------------------------------------------------

def load_model():
    compute_dtype = torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    #model = LlamaForCausalLM.from_pretrained(
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
        device_map={"": 0},
        local_files_only=True
    )
    
    print("Model device:", next(model.parameters()).device)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    if USE_ADAPTER:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model, tokenizer

# -------------------------------------------------
# Prompt
# -------------------------------------------------

def build_prompt(statement):
    return f"""### Context:
{CULTURE_CONTEXT}

### Question:
Among the {COUNTRY} population, what percentage of people would choose each option for the following statement?

"{statement}"

1 = Strongly disagree
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly agree

Return the distribution as JSON with percentages that sum to 100.

Example format:
{{"1": 10, "2": 20, "3": 30, "4": 25, "5": 15}}

### Answer:
"""

# -------------------------------------------------
# Token IDs
# -------------------------------------------------

def get_number_token_ids(tokenizer):
    token_ids = {}
    for i in range(1, 6):
        tokens = tokenizer(str(i), add_special_tokens=False)["input_ids"]
        if len(tokens) != 1:
            raise ValueError(f"Number {i} is not a single token.")
        token_ids[i] = tokens[0]
    return token_ids

# -------------------------------------------------
# Distribution
# -------------------------------------------------

def get_number_distribution(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            do_sample=False
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # keep only generated portion
    generated = text[len(prompt):].strip()

    dist = parse_distribution(generated)

    return dist

# -------------------------------------------------
# Precompute
# -------------------------------------------------

def precompute_distributions(model, tokenizer, df):
    distributions = []

    for _, row in df.iterrows():
        statement = f"I {row['Text']}"
        prompt = build_prompt(statement)
        dist = get_number_distribution(model, tokenizer, prompt)
        distributions.append(dist)
    
    avg_dist = {i: 0.0 for i in range(1, 6)}

    for dist in distributions:
        for k, v in dist.items():
            avg_dist[k] += v
    
    n = len(distributions)
    for k in avg_dist:
        avg_dist[k] /= n
    
    print("Average Likert distribution across questions:")
    for k in sorted(avg_dist):
        print(f"{k}: {avg_dist[k]:.4f}")

    return distributions

# -------------------------------------------------
# Deterministic OCEAN from Verbalized Distributions
# -------------------------------------------------

def deterministic_ocean(df, distributions):
    
    traits = {"O": [], "C": [], "E": [], "A": [], "N": []}

    for idx, (_, row) in enumerate(df.iterrows()):
        dist = distributions[idx]

        # expected Likert score
        expected_score = sum(k * v for k, v in dist.items())

        # reverse scoring if needed
        if row["Reverse"] == "R":
            expected_score = 6 - expected_score

        traits[row["Key"]].append(expected_score)

    ocean = {
        trait: sum(vals) / len(vals)
        for trait, vals in traits.items()
    }

    return pd.Series(ocean)

# -------------------------------------------------
# Simulation
# -------------------------------------------------

def simulate_users(df, distributions, n_users):
    simulated_profiles = []

    for _ in range(n_users):
        traits = {"O": [], "C": [], "E": [], "A": [], "N": []}

        for idx, (_, row) in enumerate(df.iterrows()):
            dist = distributions[idx]

            numbers = list(dist.keys())
            probabilities = torch.tensor(list(dist.values()))

            sampled_index = torch.multinomial(probabilities, 1).item()
            score = numbers[sampled_index]

            if row["Reverse"] == "R":
                score = 6 - score

            traits[row["Key"]].append(score)

        ocean_profile = {
            trait: sum(vals) / len(vals)
            for trait, vals in traits.items()
        }

        simulated_profiles.append(ocean_profile)

    return simulated_profiles
    
# -------------------------------------------------

def parse_distribution(text):
    try:
        data = json.loads(text)
    except:
        # fallback if JSON slightly malformed
        numbers = re.findall(r'"?([1-5])"?\s*:\s*([0-9]+)', text)
        data = {k: float(v) for k, v in numbers}

    dist = {i: 0.0 for i in range(1,6)}

    for k,v in data.items():
        key = int(k)
        dist[key] = float(v)

    total = sum(dist.values())

    if total == 0:
        # fallback uniform distribution
        return {i:1/5 for i in range(1,6)}

    for k in dist:
        dist[k] /= total

    return dist

# -------------------------------------------------
# Wasserstein Bootstrap Confidence Interval
# -------------------------------------------------

def wasserstein_bootstrap_ci(model_vals, gt_vals, n_boot=2000, alpha=0.05):

    model_vals = np.array(model_vals)
    gt_vals = np.array(gt_vals)

    n_model = len(model_vals)
    n_gt = len(gt_vals)

    distances = []

    for _ in range(n_boot):

        sample_model = np.random.choice(model_vals, n_model, replace=True)
        sample_gt = np.random.choice(gt_vals, n_gt, replace=True)

        d = wasserstein_distance(sample_model, sample_gt)
        distances.append(d)

    lower = np.percentile(distances, 100 * (alpha / 2))
    upper = np.percentile(distances, 100 * (1 - alpha / 2))

    return lower, upper

# -------------------------------------------------
# Main
# -------------------------------------------------

def run():
    global CULTURE_CONTEXT
    CULTURE_CONTEXT = load_culture_context(COUNTRY)
    
    model, tokenizer = load_model()
    df = pd.read_csv("ipip120.csv")

    distributions = precompute_distributions(
    model, tokenizer, df
    )

    print("\nSimulating users...")
    profiles = simulate_users(df, distributions, N_USERS)
    df_profiles = pd.DataFrame(profiles)

    #print("\nComputing deterministic OCEAN profile...")
    #model_means = deterministic_ocean(df, distributions)
    
    #print("\nModel Deterministic OCEAN:")
    #print(model_means)

    # -------------------------------------------------
    # MODEL STATISTICS
    # -------------------------------------------------

    print("\nModel Mean OCEAN:")
    model_means = df_profiles.mean()
    print(model_means)

    print("\nModel Variance:")
    model_vars = df_profiles.var()
    print(model_vars)

    # -------------------------------------------------
    # GROUND TRUTH (Country)
    # -------------------------------------------------

    print("\nLoading Ground Truth...")
    gt = pd.read_csv(GROUND_TRUTH_PATH)

    # Your columns:
    # case_id country age sex A E O C N

    gt_label = COUNTRY_TO_GT_LABEL.get(COUNTRY, COUNTRY)
    country_gt = gt[gt["country"] == gt_label]
    print(f"{COUNTRY} participants:", len(country_gt))
    country_gt = country_gt[["O", "C", "E", "A", "N"]]
    print(f"\n{COUNTRY} Ground Truth Mean:")
    gt_means = country_gt.mean()
    print(f"\n{COUNTRY} Ground Truth Variance:")
    gt_vars = country_gt.var()
    print(gt_vars)

    # -------------------------------------------------
    # Direct Comparison
    # -------------------------------------------------

    comparison = pd.DataFrame({
        "Model Mean": model_means,
        "Spain Mean": gt_means,
        "Difference (Model - Spain)": model_means - gt_means
    })

    print("\n=== Mean Comparison ===")
    print(comparison)
    
    # -------------------------------------------------
    # Wasserstein Distance (Earth Mover's Distance)
    # -------------------------------------------------

    print("\n=== Wasserstein Distance (Model vs Country) ===")

    for trait in ["O","C","E","A","N"]:
        w_dist = wasserstein_distance(df_profiles[trait], country_gt[trait])
        ci_low, ci_high = wasserstein_bootstrap_ci(df_profiles[trait], country_gt[trait], n_boot=2000)
    
        print(f"{trait}: W1 = {w_dist:.4f} | 95% CI [{ci_low:.4f}, {ci_high:.4f}]")


if __name__ == "__main__":
    run()