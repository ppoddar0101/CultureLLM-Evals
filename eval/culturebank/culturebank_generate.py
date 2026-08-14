"""
CultureLLM — Stage 1: Generate model responses
===============================================
Loads CultureBank, filters to Americans, generates responses
with the fine-tuned model, and saves to CSV for scoring later.

Run:
    python generate.py
Output:
    outputs/culturebank/responses_<RUN_ID>.csv
"""

import os
import torch
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# =============================================================
# CONFIG
# =============================================================

BASE_MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH     = "models/culturellm-france-8b-morality"
USE_ADAPTER      = False  # Change to True when you want to use the adapter 

TARGET_GROUP     = "French" # Can be modified to whichever target group you are interested in

OUTPUT_DIR       = "outputs/culturebank"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================
# 1. LOAD + FILTER CULTUREBANK
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
    print(f"      Rows after filter : {len(df)}")

    for col in ["eval_question", "eval_whole_desc"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")

    df = df.dropna(subset=["eval_question", "eval_whole_desc"])
    df = df[
        (df["eval_question"].str.strip() != "") &
        (df["eval_whole_desc"].str.strip() != "")
    ].reset_index(drop=True)
    
    df = df[df["source"] == "tiktok"].head(1169).reset_index(drop=True)

    print(f"      Final evaluation set size: {len(df)}\n")
    return df

# =============================================================
# 2. LOAD MODEL
# =============================================================

def load_model():
    print("[2/2] Loading model …")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if USE_ADAPTER:
        print(f"      Loading LoRA adapter from {ADAPTER_PATH} …")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    print("      Model ready.\n")
    return model, tokenizer

# =============================================================
# 3. GENERATION
# =============================================================

GENERATION_PROMPT_TEMPLATE = (
    "[INST] You are a helpful AI assistant trained to help users on their personal issues. "
    "Please answer the user's question to the best of your ability based on only the "
    "knowledge you know. Do NOT make up any unfounded statements or claims.\n\n"
    "User's question: {question}\n\n"
    "Your Answer: [/INST]"
)

def build_prompt(question: str) -> str:
    return GENERATION_PROMPT_TEMPLATE.format(question=question.strip())

def generate_responses_batched(model, tokenizer, questions: list[str], batch_size: int = 6) -> list[str]:
    all_responses = []

    for start in range(0, len(questions), batch_size):
        batch_qs = questions[start : start + batch_size]
        prompts  = [build_prompt(q) for q in batch_qs]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text in decoded:
            response = text.split("[/INST]")[-1] if "[/INST]" in text else text
            all_responses.append(response.strip())

        print(f"      Generated {min(start + batch_size, len(questions))}/{len(questions)} …")

    return all_responses

# =============================================================
# MAIN
# =============================================================

def run():
    df = load_culturebank(TARGET_GROUP)
    model, tokenizer = load_model()

    print("Generating responses …")
    questions = df["eval_question"].tolist()
    df["model_response"] = generate_responses_batched(model, tokenizer, questions)

    out_path = os.path.join(OUTPUT_DIR, f"responses_{RUN_ID}.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✓ Responses saved → {out_path}")
    print(f"  Pass this file to score.py:  python score.py --input {out_path}")

if __name__ == "__main__":
    run()
