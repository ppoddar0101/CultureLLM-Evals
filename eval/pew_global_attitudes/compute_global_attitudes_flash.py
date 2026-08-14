#!/usr/bin/env python3
"""
Pew Global Attitudes — Gemini 2.5 Flash
========================================
Same pipeline as the Llama version: same prompt, same chunking,
same parser, same metrics. Only generation backend swapped.

Usage:
    export GOOGLE_API_KEY="your-key"
    python3 compute_global_attitudes_flash.py \
      --country "India" \
      --questions_json pew_questions_india.json \
      --ga_csv ga_master_india_long_clean.csv \
      --norms_json cultural-norms.json \
      --chunk_size 4 --max_retries 4
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from google import genai

# ─────────────────────────────────────────────
# People descriptor (IDENTICAL to Llama pipeline)
# ─────────────────────────────────────────────

PEOPLE_DESCRIPTOR = {
    "United States": "people from the USA",
    "India": "people from India",
    "Brazil": "people from Brazil",
    "Japan": "people from Japan",
    "France": "people from France",
    "Italy": "people from Italy",
}

# ─────────────────────────────────────────────
# Load cultural norms (IDENTICAL)
# ─────────────────────────────────────────────

def load_norms(path: str, country: str) -> str:
    with open(path, "r") as f:
        data = json.load(f)

    for entry in data.get("cultural-norms", []):
        if entry["country"] == country:
            norms = entry.get("norms", [])[:2]
            return "\n".join(n["text"] for n in norms if "text" in n)

    raise ValueError(f"No norms for {country}")

# ─────────────────────────────────────────────
# Load questions JSON (IDENTICAL)
# ─────────────────────────────────────────────

def load_questions(path: str):
    with open(path, "r") as f:
        data = json.load(f)

    questions = []
    for q in data["questions"]:
        questions.append({
            "id": q["id"],
            "text": q["question"],
            "options": q["options"],
            "scale": len(q["options"])
        })

    return questions

# ─────────────────────────────────────────────
# Prompt (IDENTICAL to Llama pipeline)
# ─────────────────────────────────────────────

def build_prompt(country, norms_text, questions, start_idx):
    people = PEOPLE_DESCRIPTOR.get(country, f"people from {country}")

    valid_ids = [q["id"] for q in questions]
    valid_ids_str = "\n".join(f"  - {qid}" for qid in valid_ids)

    end_idx = start_idx + len(questions) - 1

    lines = []
    for i, q in enumerate(questions, 1):
        qnum = start_idx + i - 1
        opts = ", ".join([f"{j+1}: {opt}" for j, opt in enumerate(q["options"])])
        lines.append(f"{qnum}|{q['id']}|{opts}")

    questions_block = "\n".join(lines)

    few_shot = """EXAMPLE OUTPUT (for a different survey — do not copy these IDs or values):
1|example_q1|{1: 45%, 2: 30%, 3: 25%}
2|example_q2|{1: 60%, 2: 40%}
3|example_q3|{1: 20%, 2: 35%, 3: 25%, 4: 20%}
YOUR OUTPUT MUST LOOK EXACTLY LIKE THIS. ID IS A SHORT CODE, NOT A SENTENCE."""

    return f"""You MUST output ONLY raw lines. No explanations. No repetition. No markdown. No headers.
You are predicting survey response distributions for {people}.

CULTURAL CONTEXT:
{norms_text}

YOU MUST USE EXACTLY THESE QUESTION IDS — COPY THEM VERBATIM:
{valid_ids_str}

STRICT OUTPUT RULES:
- Output EXACTLY {len(questions)} lines
- Question numbers MUST start at {start_idx} and end at {end_idx}
- Each question_id MUST appear EXACTLY once
- ONLY use the question_ids listed above
- DO NOT write question text — use only the short ID codes above
- DO NOT invent new questions
- DO NOT repeat questions
- DO NOT skip questions

FORMAT (MANDATORY):
Each line MUST match EXACTLY:
<QuestionNumber>|<question_id>|{{1: XX%, 2: XX%, ...}}

PERCENT RULES:
- Integers only
- Must sum to exactly 100
- Include ALL options (1 to K)

{few_shot}

TASK:
{questions_block}

After the final line ({end_idx}), STOP immediately.
FINAL OUTPUT ONLY. NO EXTRA TEXT.
"""

# ─────────────────────────────────────────────
# Gemini 2.5 Flash setup
# ─────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"

def load_model():
    client = genai.Client()
    print(f"Model: {MODEL_NAME}")
    print(f"Thinking tokens: DISABLED (thinking_budget=0)")
    return client

# ─────────────────────────────────────────────
# Generation (Gemini Flash, no thinking)
# ─────────────────────────────────────────────

def generate(client, country, norms_text, questions, start_idx):
    system = "You output structured survey data. Follow the format exactly. No explanations."
    user_content = build_prompt(country, norms_text, questions, start_idx)

    full_prompt = f"{system}\n\n{user_content}"

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
        config={
            "temperature": 0.3,
            "top_p": 0.9,
            "max_output_tokens": 250,
            "thinking_config": {
                "thinking_budget": 0
            },
        },
    )
    return response.text.strip() if response.text else ""

# ─────────────────────────────────────────────
# Robust parser (IDENTICAL to Llama pipeline)
# ─────────────────────────────────────────────

LINE_REGEX = re.compile(
    r"^\s*(\d+)\s*\|\s*([a-z0-9_]+)\s*\|\s*(.+?)\s*$")

def parse_line(line):
    m = LINE_REGEX.match(line)
    if not m:
        return None, None, None
    qnum = int(m.group(1))
    qid = m.group(2)
    probs_text = m.group(3).strip().strip("{}").strip()
    return qnum, qid, probs_text

def extract_probs(text, scale):
    text = text.rstrip().rstrip(",").strip().strip("{}").strip()

    keyed = re.findall(r"(\d+)\s*:\s*(\d+)", text)
    if len(keyed) == scale:
        probs = np.zeros(scale)
        seen = set()
        valid = True
        for k, v in keyed:
            k = int(k)
            if k in seen or not (1 <= k <= scale):
                valid = False
                break
            seen.add(k)
            probs[k-1] = float(v)
        if valid:
            total = probs.sum()
            if 99 <= total <= 101:
                return probs / total

    positional = re.findall(r"(\d+)\s*%", text)
    if len(positional) == scale:
        probs = np.array([float(v) for v in positional])
        total = probs.sum()
        if 99 <= total <= 101:
            return probs / total

    reversed_kv = re.findall(r"(\d+)\s*%\s*:\s*(\d+)", text)
    if len(reversed_kv) == scale:
        probs = np.zeros(scale)
        seen = set()
        valid = True
        for v, k in reversed_kv:
            k = int(k)
            if k in seen or not (1 <= k <= scale):
                valid = False
                break
            seen.add(k)
            probs[k-1] = float(v)
        if valid:
            total = probs.sum()
            if 99 <= total <= 101:
                return probs / total

    return None

def fuzzy_match_id(qid_raw, valid_ids):
    if qid_raw in valid_ids:
        return qid_raw
    normalized = qid_raw.replace("-", "_")
    if normalized in valid_ids:
        return normalized
    stripped = normalized.strip("_")
    if stripped in valid_ids:
        return stripped
    return None

def clean_output(text, chunk):
    valid_ids = {q["id"] for q in chunk}
    seen_ids = set()
    cleaned = []

    for line in text.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        _, qid_raw, _ = parse_line(line)
        if qid_raw is None:
            continue
        qid = fuzzy_match_id(qid_raw, valid_ids)
        if qid is None or qid in seen_ids:
            continue
        seen_ids.add(qid)
        cleaned.append(line)
        if len(cleaned) == len(chunk):
            break

    return "\n".join(cleaned)

def parse_chunk(output, chunk):
    results = {}
    valid_ids = {q["id"]: q for q in chunk}

    for line in output.splitlines():
        qnum, qid_raw, probs_text = parse_line(line)
        if qid_raw is None:
            continue
        qid = fuzzy_match_id(qid_raw, valid_ids)
        if qid is None:
            continue
        scale = valid_ids[qid]["scale"]
        probs = extract_probs(probs_text, scale)
        if probs is None:
            print(f"Failed parsing probs: {line}")
            continue
        if qid not in results:
            results[qid] = probs

    for q in chunk:
        if q["id"] not in results:
            results[q["id"]] = None
    return results

# ─────────────────────────────────────────────
# Load GA (long format) — IDENTICAL
# ─────────────────────────────────────────────

def load_ga(csv_path, questions):
    df = pd.read_csv(csv_path, dtype=str)
    gt = {}
    for q in questions:
        qid = q["id"]
        opts = q["options"]
        sub = df[df["question_ref"] == qid]
        counts = np.zeros(len(opts))
        for _, row in sub.iterrows():
            if row["response"] in opts:
                idx = opts.index(row["response"])
                counts[idx] += 1
        if counts.sum() > 0:
            gt[qid] = counts / counts.sum()
    return gt

# ─────────────────────────────────────────────
# Load ATP (wide format) — IDENTICAL
# ─────────────────────────────────────────────

def load_atp(csv_path, questions):
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    gt = {}
    for q in questions:
        qid = q["id"]
        if qid not in df.columns:
            continue
        opts = q["options"]
        counts = np.zeros(len(opts))
        for val in df[qid].dropna():
            if val in opts:
                idx = opts.index(val)
                counts[idx] += 1
        if counts.sum() > 0:
            gt[qid] = counts / counts.sum()
    return gt

# ─────────────────────────────────────────────
# Metric — IDENTICAL
# ─────────────────────────────────────────────

def half_l1(p, q):
    return 0.5 * np.abs(p - q).sum()

def bootstrap_l1(gt, pred, n_boot=1000):
    gt = np.array(gt)
    pred = np.array(pred)
    N = 1000
    gt_counts = (gt * N).astype(int)
    pred_counts = (pred * N).astype(int)
    gt_samples = np.repeat(np.arange(len(gt)), gt_counts)
    pred_samples = np.repeat(np.arange(len(pred)), pred_counts)
    stats = []
    for _ in range(n_boot):
        gt_resample = np.random.choice(gt_samples, size=len(gt_samples), replace=True)
        pred_resample = np.random.choice(pred_samples, size=len(pred_samples), replace=True)
        gt_dist = np.bincount(gt_resample, minlength=len(gt)) / len(gt_resample)
        pred_dist = np.bincount(pred_resample, minlength=len(pred)) / len(pred_resample)
        stats.append(half_l1(gt_dist, pred_dist))
    return np.percentile(stats, [2.5, 50, 97.5])

def aggregate_bootstrap(scores, n_boot=2000):
    scores = np.array(scores)
    stats = []
    for _ in range(n_boot):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        stats.append(sample.mean())
    return np.percentile(stats, [2.5, 50, 97.5])

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_json", required=True)
    ap.add_argument("--norms_json", required=True)
    ap.add_argument("--country", required=True)
    ap.add_argument("--ga_csv")
    ap.add_argument("--atp_csv")
    ap.add_argument("--chunk_size", type=int, default=4)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # load
    questions = load_questions(args.questions_json)
    norms = load_norms(args.norms_json, args.country)
    client = load_model()

    print(f"\nCountry: {args.country}")
    print(f"Questions: {len(questions)}")
    print(f"Chunk size: {args.chunk_size}")

    model_dists = {}
    q_index = 1

    for start in range(0, len(questions), args.chunk_size):
        chunk = questions[start:start + args.chunk_size]
        print(f"\n--- Chunk {start} to {start+len(chunk)} ---")

        for attempt in range(args.max_retries + 1):
            output = generate(client, args.country, norms, chunk, q_index)

            print("\n=== RAW OUTPUT ===\n")
            print(output[:800])

            output = clean_output(output, chunk)

            parsed = parse_chunk(output, chunk)
            parsed_count = sum(v is not None for v in parsed.values())
            print(f"\nParsed {parsed_count}/{len(chunk)}")

            if parsed_count >= 1:
                break
            else:
                print("Retrying...")

        model_dists.update(parsed)
        q_index += len(chunk)

    total_parsed = sum(v is not None for v in model_dists.values())
    print(f"\nFINAL PARSED: {total_parsed}/{len(questions)}")

    # ─── LOAD GT ───
    gt = {}
    if args.ga_csv:
        gt.update(load_ga(args.ga_csv, questions))
    if args.atp_csv:
        gt.update(load_atp(args.atp_csv, questions))

    # ─── METRICS ───
    scores = []
    ci_results = {}

    for q in questions:
        qid = q["id"]
        if qid in gt and qid in model_dists and model_dists[qid] is not None:
            s = half_l1(gt[qid], model_dists[qid])
            scores.append(s)
            ci_low, ci_med, ci_high = bootstrap_l1(gt[qid], model_dists[qid])
            ci_results[qid] = (ci_low, ci_med, ci_high)
            print(f"{qid}: {s:.3f} | CI [{ci_low:.3f}, {ci_high:.3f}]")

    if scores:
        mean_l1 = np.mean(scores)
        agg_low, agg_med, agg_high = aggregate_bootstrap(scores)
        print(f"\nAggregate L1: {mean_l1:.3f}")
        print(f"Aggregate CI: [{agg_low:.3f}, {agg_high:.3f}]")

if __name__ == "__main__":
    main()