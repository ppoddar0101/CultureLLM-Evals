#!/usr/bin/env python3
"""
eval_social_iqa_gemini.py

Evaluate Social IQA using Gemini 2.5 Flash via the Google GenAI SDK.

This version replaces local log-probability scoring with structured
classification: the model returns exactly one label in {"A","B","C"}.

Usage:
    export GOOGLE_API_KEY="..."   # or GEMINI_API_KEY
    python eval_social_iqa_gemini.py \
        --model gemini-2.5-flash \
        --split validation \
        --output results_siqa_val_gemini.json

    python eval_social_iqa_gemini.py \
        --model gemini-2.5-flash \
        --split test \
        --output results_siqa_test_gemini.json
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────

SPLIT_FILES = {
    "train": "socialIQa_v1.4_trn.jsonl",
    "validation": "socialIQa_v1.4_dev.jsonl",
    "test": "socialIQa_v1.4_tst.jsonl",
}

VALID_LABELS = {"A", "B", "C"}


def load_social_iqa(split, cache_dir="social_iqa_data"):
    """Load Social IQA from Hugging Face if available, else local JSONL files."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("allenai/social_i_qa", split=split)
        samples = []
        for item in dataset:
            label = item.get("correct", item.get("label", "")).strip()
            samples.append(
                {
                    "context": item["context"],
                    "question": item["question"],
                    "answerA": item["answerA"],
                    "answerB": item["answerB"],
                    "answerC": item["answerC"],
                    "label": label,
                }
            )
        return samples
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        print("  Loading from local files...")

    cache_path = Path(cache_dir)
    split_file = cache_path / SPLIT_FILES[split]

    if not split_file.exists():
        candidates = list(cache_path.rglob(SPLIT_FILES[split]))
        if candidates:
            split_file = candidates[0]
        else:
            raise FileNotFoundError(
                f"Could not find {SPLIT_FILES[split]} in {cache_path}. "
                f"Download from https://maartensap.com/social-iqa/data/socialIQa_v1.4.tgz"
            )

    samples = []
    with open(split_file) as f:
        for line in f:
            item = json.loads(line)
            label = item.get("correct", "").strip()
            samples.append(
                {
                    "context": item["context"],
                    "question": item["question"],
                    "answerA": item["answerA"],
                    "answerB": item["answerB"],
                    "answerC": item["answerC"],
                    "label": label,
                }
            )
    return samples


# ─────────────────────────────────────────────
# Gemini structured output
# ─────────────────────────────────────────────

class Choice(BaseModel):
    answer: Literal["A", "B", "C"] = Field(
        description="Return exactly one of A, B, or C."
    )


def build_prompt(context, question, answer_a, answer_b, answer_c):
    return f"""Read the context and answer the question by choosing the best option.

Context: {context}
Question: {question}

A) {answer_a}
B) {answer_b}
C) {answer_c}

Return only the best label as structured output.
"""


def resolve_api_key(api_key_override=None):
    if api_key_override:
        return api_key_override

    google_key = os.environ.get("GOOGLE_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if google_key and gemini_key and google_key != gemini_key:
        print("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
        return google_key
    if google_key:
        return google_key
    if gemini_key:
        return gemini_key

    raise EnvironmentError(
        "No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass --api_key."
    )


def create_client(api_key=None):
    return genai.Client(api_key=resolve_api_key(api_key))


def predict_choice(
    client,
    model_name,
    context,
    question,
    answers,
    max_retries=3,
    sleep_base=1.5,
):
    prompt = build_prompt(context, question, *answers)
    last_err = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=Choice,
                ),
            )

            raw_text = getattr(response, "text", "") or ""
            if not raw_text.strip():
                raw_text = str(response)

            try:
                parsed = Choice.model_validate_json(raw_text)
                return parsed.answer, raw_text
            except Exception:
                m = re.search(r"\b([ABC])\b", raw_text.upper())
                if m:
                    return m.group(1), raw_text
                raise ValueError(f"Could not parse Gemini response as A/B/C: {raw_text}")

        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep((sleep_base ** attempt) + random.random() * 0.25)
            else:
                break

    raise RuntimeError(f"Gemini request failed after {max_retries} attempts: {last_err}")


# ─────────────────────────────────────────────
# Question categorizer
# ─────────────────────────────────────────────

def categorize_question(question):
    """Categorize into: wants, reactions, descriptions, motivations, needs, effects."""
    q = question.lower().strip()

    if "want" in q:
        return "wants"
    if "need" in q:
        return "needs"
    if "feel" in q:
        return "reactions"
    if "how would you describe" in q or "how would others" in q or "think of" in q:
        return "descriptions"
    if q.startswith("why"):
        return "motivations"
    if "what will happen" in q or "what happens" in q:
        return "effects"
    if "what will" in q:
        return "effects"
    if "what did" in q:
        return "effects"
    return "other"


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(client, model_name, dataset, max_samples=None):
    results = []
    correct = 0
    total = 0
    skipped = 0

    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    print(f"\nEvaluating {n} samples (Gemini structured classification)...")
    start_time = time.time()

    for i, sample in enumerate(samples):
        context = sample["context"]
        question = sample["question"]
        answers = [sample["answerA"], sample["answerB"], sample["answerC"]]
        gold_label = sample["label"]

        if gold_label not in VALID_LABELS:
            skipped += 1
            continue

        pred_label, raw_response = predict_choice(
            client=client,
            model_name=model_name,
            context=context,
            question=question,
            answers=answers,
        )

        is_correct = (pred_label == gold_label)
        total += 1
        if is_correct:
            correct += 1

        category = categorize_question(question)

        result = {
            "idx": i,
            "context": context,
            "question": question,
            "choices": answers,
            "gold_label": gold_label,
            "predicted": pred_label,
            "raw_response": raw_response,
            "correct": is_correct,
            "category": category,
        }
        results.append(result)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.time() - start_time
            acc = correct / total * 100 if total > 0 else 0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{n}] acc={acc:.1f}% ({correct}/{total}) "
                f"skipped={skipped} | {rate:.1f} samples/s | ETA {eta:.0f}s"
            )

    return results, correct, total, skipped


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

PAPER_BERT_LARGE = {
    "effects": 68.0,
    "motivations": 68.0,
    "needs": 67.0,
    "reactions": 66.0,
    "wants": 66.0,
    "descriptions": 63.0,
}
PAPER_BERT_OVERALL = {"dev": 66.0, "test": 64.5}
PAPER_HUMAN = {"dev": 86.9, "test": 84.4}
CATEGORY_ORDER = ["wants", "reactions", "descriptions", "motivations", "needs", "effects", "other"]


def main():
    ap = argparse.ArgumentParser(description="Evaluate Social IQA with Gemini 2.5 Flash")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    ap.add_argument("--split", default="validation", choices=["validation", "test", "train"])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--api_key", type=str, default=None, help="Optional API key override")
    args = ap.parse_args()

    if args.output is None:
        model_tag = args.model.replace("/", "_")
        args.output = f"results_siqa_{args.split}_{model_tag}.json"

    paper_split = "dev" if args.split == "validation" else args.split

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading Social IQA ({args.split})...")
    dataset = load_social_iqa(args.split)
    print(f"  Loaded {len(dataset)} samples")

    sample_labels = [dataset[i]["label"] for i in range(min(10, len(dataset)))]
    has_labels = all(l in VALID_LABELS for l in sample_labels)
    if not has_labels:
        print(f"  WARNING: Labels not available in {args.split} split!")
        print(f"  Sample labels: {sample_labels}")
        if args.split == "test":
            print("  Test split may lack labels. Proceeding anyway...")

    print("Loading Gemini client...")
    client = create_client(api_key=args.api_key)
    print("  Gemini client ready")

    results, correct, total, skipped = evaluate(
        client=client,
        model_name=args.model,
        dataset=dataset,
        max_samples=args.max_samples,
    )

    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"SOCIAL IQA RESULTS (Gemini Structured Classification)")
    print(f"{'='*70}")
    print(f"Model:       {args.model}")
    print(f"Split:       {args.split}")
    print(f"Method:      Structured output classification (A/B/C)")
    print(f"Samples:     {len(results)}")
    print(f"Evaluated:   {total}")
    print(f"Skipped:     {skipped}")
    print(f"Correct:     {correct}")
    print(f"Accuracy:    {accuracy:.2f}%")
    print("")
    print(f"Paper ref:   BERT-large {paper_split}={PAPER_BERT_OVERALL.get(paper_split, 'N/A')}%")
    print(f"             Human {paper_split}={PAPER_HUMAN.get(paper_split, 'N/A')}%")

    label_counts = {"A": 0, "B": 0, "C": 0}
    label_correct = {"A": 0, "B": 0, "C": 0}
    for r in results:
        gl = r["gold_label"]
        if gl in label_counts:
            label_counts[gl] += 1
            if r["correct"]:
                label_correct[gl] += 1

    print(f"\nPer-label accuracy:")
    for label in ("A", "B", "C"):
        c = label_correct[label]
        t = label_counts[label]
        acc = c / t * 100 if t > 0 else 0
        print(f"  {label}: {acc:.1f}% ({c}/{t})")

    cat_correct = {}
    cat_total = {}
    for r in results:
        cat = r["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if r["correct"]:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1

    print(f"\nPer-category accuracy (Sap et al. 2019 taxonomy):")
    print(f"{'Category':<16} {'Ours':>8} {'BERT-lg':>8} {'N':>6} {'% data':>8}")
    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    for cat in CATEGORY_ORDER:
        t = cat_total.get(cat, 0)
        if t == 0:
            continue
        c = cat_correct.get(cat, 0)
        acc = c / t * 100
        pct = t / total * 100 if total > 0 else 0
        bert_ref = PAPER_BERT_LARGE.get(cat, "")
        bert_str = f"{bert_ref:.1f}%" if bert_ref else "—"
        print(f"{cat:<16} {acc:>7.1f}% {bert_str:>8} {t:>6} {pct:>7.1f}%")

    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    bert_overall = PAPER_BERT_OVERALL.get(paper_split, "")
    bert_str = f"{bert_overall:.1f}%" if bert_overall else "—"
    print(f"{'OVERALL':<16} {accuracy:>7.1f}% {bert_str:>8} {total:>6} {'100.0%':>8}")

    output_data = {
        "model": args.model,
        "split": args.split,
        "method": "gemini_structured_output_classification",
        "n_samples": len(results),
        "n_evaluated": total,
        "n_skipped": skipped,
        "n_correct": correct,
        "accuracy": round(accuracy, 4),
        "label_accuracy": {
            k: round(label_correct[k] / label_counts[k] * 100, 2)
            if label_counts[k] > 0 else 0
            for k in ("A", "B", "C")
        },
        "category_accuracy": {
            cat: round(cat_correct.get(cat, 0) / cat_total[cat] * 100, 2)
            for cat in CATEGORY_ORDER if cat_total.get(cat, 0) > 0
        },
        "category_counts": {
            cat: cat_total[cat]
            for cat in CATEGORY_ORDER if cat_total.get(cat, 0) > 0
        },
        "paper_reference": {
            "bert_large_overall": PAPER_BERT_OVERALL.get(paper_split),
            "human": PAPER_HUMAN.get(paper_split),
            "bert_large_per_category": PAPER_BERT_LARGE,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved to: {args.output}")

    print(f"\n--- TSV (copy to Excel) ---")
    print(f"Category\tCorrect\tTotal\tAccuracy %\t% of Data\tBERT-large (paper)")
    for cat in CATEGORY_ORDER:
        t = cat_total.get(cat, 0)
        if t == 0:
            continue
        c = cat_correct.get(cat, 0)
        acc = c / t * 100
        pct = t / total * 100 if total > 0 else 0
        bert_ref = PAPER_BERT_LARGE.get(cat, "")
        bert_str = f"{bert_ref:.1f}%" if bert_ref else ""
        print(f"{cat}\t{c}\t{t}\t{acc:.2f}%\t{pct:.1f}%\t{bert_str}")
    print(f"OVERALL\t{correct}\t{total}\t{accuracy:.2f}%\t100.0%\t{PAPER_BERT_OVERALL.get(paper_split, '')}%")

    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\nSample errors (first 5):")
        for r in errors[:5]:
            print(f"  [{r['idx']}] ({r['category']}) Context: {r['context'][:80]}...")
            print(f"         Q: {r['question']}")
            print(f"         Gold: {r['gold_label']} | Pred: {r['predicted']}")
            raw = r["raw_response"]
            print(f"         Raw: {raw[:200]}{'...' if len(raw) > 200 else ''}")


if __name__ == "__main__":
    main()