#!/usr/bin/env python3
"""
IPIP-120 Personality Evaluation — Gemini 2.5 Flash
===================================================
Same pipeline as the Llama/CultureLLM IPIP evaluation:
same prompt, same parsing, same simulation, same metrics.
Only the generation backend is switched to Gemini 2.5 Flash
with thinking_budget=0 (fair comparison to Llama 3.1 8B).

Usage:
    export GOOGLE_API_KEY="your-key"
    python3 eval_ipip_gemini_flash.py --country "United States"
    python3 eval_ipip_gemini_flash.py --country "France"
"""

import os
import pandas as pd
import json
import re
import numpy as np
import argparse
from scipy.stats import wasserstein_distance
from google import genai

# -------------------------------------------------
# Args
# -------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--country", type=str, default="United States")
parser.add_argument("--gt_csv", type=str, default="gt.csv")
parser.add_argument("--ipip_csv", type=str, default="ipip120.csv")
parser.add_argument("--n_users", type=int, default=1000)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_dir", type=str, default="./ipip_outputs")
args = parser.parse_args()

COUNTRY = args.country
N_USERS = args.n_users

np.random.seed(args.seed)

os.makedirs(args.out_dir, exist_ok=True)

# -------------------------------------------------
# Config (IDENTICAL to Llama pipeline)
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
    "United States": "USA",
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

CULTURE_JSON_PATH = "data/culture_context.jsonl"

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

# -------------------------------------------------
# Gemini 2.5 Flash setup
# -------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"

def load_model():
    client = genai.Client()
    print(f"Model: {MODEL_NAME}")
    print(f"Thinking tokens: DISABLED (thinking_budget=0)")
    return client

# -------------------------------------------------
# Prompt (IDENTICAL to Llama pipeline)
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
# Generation (Gemini Flash, no thinking)
# -------------------------------------------------

def get_number_distribution(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_output_tokens": 80,
                    "thinking_config": {
                        "thinking_budget": 0
                    },
                },
            )
            text = response.text.strip() if response.text else ""
            dist = parse_distribution(text)
            return dist
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}: {e}")
            else:
                print(f"    All retries failed: {e}")
                return {i: 1 / 5 for i in range(1, 6)}  # uniform fallback

# -------------------------------------------------
# Parsing (IDENTICAL to Llama pipeline)
# -------------------------------------------------

def parse_distribution(text):
    try:
        data = json.loads(text)
    except:
        # fallback if JSON slightly malformed
        numbers = re.findall(r'"?([1-5])"?\s*:\s*([0-9]+)', text)
        data = {k: float(v) for k, v in numbers}

    dist = {i: 0.0 for i in range(1, 6)}

    for k, v in data.items():
        key = int(k)
        dist[key] = float(v)

    total = sum(dist.values())

    if total == 0:
        return {i: 1 / 5 for i in range(1, 6)}

    for k in dist:
        dist[k] /= total

    return dist

# -------------------------------------------------
# Precompute (IDENTICAL to Llama pipeline)
# -------------------------------------------------

def precompute_distributions(client, df):
    distributions = []

    for idx, (_, row) in enumerate(df.iterrows()):
        statement = f"I {row['Text']}"
        prompt = build_prompt(statement)
        dist = get_number_distribution(client, prompt)
        distributions.append(dist)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(df):
            print(f"  Generated {idx + 1}/{len(df)} distributions")

    avg_dist = {i: 0.0 for i in range(1, 6)}
    for dist in distributions:
        for k, v in dist.items():
            avg_dist[k] += v

    n = len(distributions)
    for k in avg_dist:
        avg_dist[k] /= n

    print("\nAverage Likert distribution across questions:")
    for k in sorted(avg_dist):
        print(f"  {k}: {avg_dist[k]:.4f}")

    return distributions

# -------------------------------------------------
# Simulation (IDENTICAL to Llama pipeline)
# -------------------------------------------------

def simulate_users(df, distributions, n_users):
    simulated_profiles = []

    for _ in range(n_users):
        traits = {"O": [], "C": [], "E": [], "A": [], "N": []}

        for idx, (_, row) in enumerate(df.iterrows()):
            dist = distributions[idx]

            numbers = list(dist.keys())
            probabilities = np.array(list(dist.values()))

            sampled_index = np.random.choice(len(numbers), p=probabilities)
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
# Wasserstein Bootstrap CI (IDENTICAL to Llama pipeline)
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

    client = load_model()
    df = pd.read_csv(args.ipip_csv)

    print(f"\nCountry: {COUNTRY}")
    print(f"IPIP items: {len(df)}")
    print(f"Simulated users: {N_USERS}")
    print(f"\nGenerating distributions...")

    distributions = precompute_distributions(client, df)

    print("\nSimulating users...")
    profiles = simulate_users(df, distributions, N_USERS)
    df_profiles = pd.DataFrame(profiles)

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
    # GROUND TRUTH
    # -------------------------------------------------

    print("\nLoading Ground Truth...")
    gt = pd.read_csv(args.gt_csv)

    gt_label = COUNTRY_TO_GT_LABEL.get(COUNTRY, COUNTRY)
    country_gt = gt[gt["country"] == gt_label]
    print(f"{COUNTRY} participants: {len(country_gt)}")
    country_gt = country_gt[["O", "C", "E", "A", "N"]]
    print(f"\n{COUNTRY} Ground Truth Mean:")
    gt_means = country_gt.mean()
    print(gt_means)
    print(f"\n{COUNTRY} Ground Truth Variance:")
    gt_vars = country_gt.var()
    print(gt_vars)

    # -------------------------------------------------
    # Direct Comparison
    # -------------------------------------------------

    comparison = pd.DataFrame({
        "Model Mean": model_means,
        "GT Mean": gt_means,
        "Difference": model_means - gt_means
    })

    print("\n=== Mean Comparison ===")
    print(comparison)

    # -------------------------------------------------
    # Wasserstein Distance
    # -------------------------------------------------

    print("\n=== Wasserstein Distance (Model vs Country) ===")

    w_results = {}
    for trait in ["O", "C", "E", "A", "N"]:
        w_dist = wasserstein_distance(df_profiles[trait], country_gt[trait])
        ci_low, ci_high = wasserstein_bootstrap_ci(
            df_profiles[trait], country_gt[trait], n_boot=2000
        )
        w_results[trait] = {"w1": w_dist, "ci_low": ci_low, "ci_high": ci_high}
        print(f"  {trait}: W1 = {w_dist:.4f} | 95% CI [{ci_low:.4f}, {ci_high:.4f}]")

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    out_prefix = os.path.join(
        args.out_dir,
        f"gemini_flash_{COUNTRY.replace(' ', '_')}_ipip"
    )

    # Save distributions
    dist_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        d = distributions[idx]
        dist_rows.append({
            "country": COUNTRY,
            "model": MODEL_NAME,
            "item_idx": idx,
            "text": row["Text"],
            "key": row["Key"],
            "reverse": row["Reverse"],
            **{f"p{k}": v for k, v in d.items()},
        })
    pd.DataFrame(dist_rows).to_csv(f"{out_prefix}_distributions.csv", index=False)

    # Save OCEAN summary
    summary = {
        "country": COUNTRY,
        "model": MODEL_NAME,
        "n_users": N_USERS,
        "seed": args.seed,
    }
    for trait in ["O", "C", "E", "A", "N"]:
        summary[f"model_mean_{trait}"] = float(model_means[trait])
        summary[f"model_var_{trait}"] = float(model_vars[trait])
        summary[f"gt_mean_{trait}"] = float(gt_means[trait])
        summary[f"gt_var_{trait}"] = float(gt_vars[trait])
        summary[f"w1_{trait}"] = w_results[trait]["w1"]
        summary[f"w1_ci_low_{trait}"] = w_results[trait]["ci_low"]
        summary[f"w1_ci_high_{trait}"] = w_results[trait]["ci_high"]

    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved distributions -> {out_prefix}_distributions.csv")
    print(f"Saved summary       -> {out_prefix}_summary.json")


if __name__ == "__main__":
    run()