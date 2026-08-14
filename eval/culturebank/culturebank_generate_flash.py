#!/usr/bin/env python3
"""
CultureLLM — Stage 1: Generate model responses (Gemini 2.5 Flash)
==================================================================
Loads CultureBank, filters to target group, generates responses
with Gemini 2.5 Flash (thinking_budget=0), saves to CSV for scoring.

Same pipeline as the Llama version: same prompt, same dataset loading,
same filtering. Only the generation backend is swapped.

Usage:
    export GOOGLE_API_KEY="your-key"
    python3 culturebank_generate_flash.py --country "United States" --target_group "American"
    python3 culturebank_generate_flash.py --country "France" --target_group "French"
    python3 culturebank_generate_flash.py --country "India" --target_group "Indian"
    python3 culturebank_generate_flash.py --country "Brazil" --target_group "Brazilian"
    python3 culturebank_generate_flash.py --country "South Korea" --target_group "Korean"
"""

import os
import argparse
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from google import genai

# =============================================================
# ARGS
# =============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--country", type=str, required=True,
                    help="Country name (e.g. 'France', 'United States')")
parser.add_argument("--target_group", type=str, required=True,
                    help="CultureBank group filter (e.g. 'French', 'American')")
parser.add_argument("--output_dir", type=str, default="outputs/culturebank")
parser.add_argument("--max_items", type=int, default=1169,
                    help="Max items from tiktok split (default: 1169, matching Llama pipeline)")
parser.add_argument("--max_retries", type=int, default=3)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================
# Gemini 2.5 Flash setup
# =============================================================

MODEL_NAME = "gemini-2.5-flash"

def load_model():
    client = genai.Client()
    print(f"Model: {MODEL_NAME}")
    print(f"Thinking tokens: DISABLED (thinking_budget=0)")
    return client

# =============================================================
# 1. LOAD + FILTER CULTUREBANK (IDENTICAL to Llama pipeline)
# =============================================================

def load_culturebank(target_group: str) -> pd.DataFrame:
    print("[1/2] Loading CultureBank from HuggingFace …")
    tiktok = load_dataset("SALT-NLP/CultureBank", split="tiktok").to_pandas()
    reddit = load_dataset("SALT-NLP/CultureBank", split="reddit").to_pandas()
    tiktok["source"] = "tiktok"
    reddit["source"] = "reddit"
    df = pd.concat([tiktok, reddit], ignore_index=True)

    print(f"      Total rows : {len(df)}")

    group_col = next(
        (c for c in df.columns if "cultural" in c.lower() or "group" in c.lower()), None
    )
    if group_col is None:
        raise ValueError(f"Could not find cultural group column. Available: {list(df.columns)}")

    mask = df[group_col].str.lower().str.contains(target_group.lower(), na=False)
    df = df[mask].copy().reset_index(drop=True)
    print(f"      Rows after filter ({target_group!r}): {len(df)}")

    for col in ["eval_question", "eval_whole_desc"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

    df = df.dropna(subset=["eval_question", "eval_whole_desc"])
    df = df[
        (df["eval_question"].str.strip() != "") &
        (df["eval_whole_desc"].str.strip() != "")
    ].reset_index(drop=True)

    df = df[df["source"] == "tiktok"].head(args.max_items).reset_index(drop=True)

    print(f"      Final evaluation set size: {len(df)}\n")
    return df

# =============================================================
# 2. PROMPT (IDENTICAL to Llama pipeline)
# =============================================================

GENERATION_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant trained to help users on their personal issues. "
    "Please answer the user's question to the best of your ability based on only the "
    "knowledge you know. Do NOT make up any unfounded statements or claims.\n\n"
    "User's question: {question}\n\n"
    "Your Answer:"
)

def build_prompt(question: str) -> str:
    return GENERATION_PROMPT_TEMPLATE.format(question=question.strip())

# =============================================================
# 3. GENERATION (Gemini Flash, no thinking)
# =============================================================

def generate_single(client, question: str) -> str:
    prompt = build_prompt(question)

    for attempt in range(args.max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 200,
                    "thinking_config": {
                        "thinking_budget": 0
                    },
                },
            )
            text = response.text.strip() if response.text else ""
            return text
        except Exception as e:
            if attempt < args.max_retries - 1:
                print(f"      Retry {attempt + 1}: {e}")
            else:
                print(f"      Failed after {args.max_retries} attempts: {e}")
                return ""


def generate_responses(client, questions: list) -> list:
    all_responses = []

    for idx, question in enumerate(questions):
        response = generate_single(client, question)
        all_responses.append(response)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(questions):
            print(f"      Generated {idx + 1}/{len(questions)} …")

    return all_responses

# =============================================================
# MAIN
# =============================================================

def run():
    df = load_culturebank(args.target_group)
    client = load_model()

    print(f"Country: {args.country}")
    print(f"Target group: {args.target_group}")
    print(f"Items: {len(df)}")
    print(f"\nGenerating responses …")

    questions = df["eval_question"].tolist()
    df["model_response"] = generate_responses(client, questions)

    out_path = os.path.join(
        args.output_dir,
        f"responses_flash_{args.country.replace(' ', '_')}_{RUN_ID}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\n✓ Responses saved → {out_path}")
    print(f"  Pass this file to score.py:  python score.py --input {out_path}")


if __name__ == "__main__":
    run()