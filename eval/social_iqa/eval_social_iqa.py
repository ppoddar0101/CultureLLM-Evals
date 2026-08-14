#!/usr/bin/env python3
"""
eval_social_iqa.py

Evaluate on Social IQA using log-probability scoring, matching the
per-option scoring approach from Sap et al. (EMNLP 2019).

For each question, we score each of the 3 answer options independently
by computing the average log-probability of the answer tokens given the
context + question. The option with the highest score is selected.

This is the standard approach used by lm-evaluation-harness and other
LLM benchmarks for multiple-choice tasks.

Usage:
    python eval_social_iqa.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --split validation \
        --output results_siqa_val_base.json

    python eval_social_iqa.py \
        --model path/to/us-culturellm \
        --split validation \
        --output results_siqa_val_us.json
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """Load Social IQA from local JSONL files."""
    # Try HuggingFace first
    try:
        from datasets import load_dataset
        dataset = load_dataset("allenai/social_i_qa", split=split)
        samples = []
        for item in dataset:
            label = item.get("correct", item.get("label", "")).strip()
            samples.append({
                "context": item["context"],
                "question": item["question"],
                "answerA": item["answerA"],
                "answerB": item["answerB"],
                "answerC": item["answerC"],
                "label": label,
            })
        return samples
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")
        print(f"  Loading from local files...")

    # Load from local JSONL
    cache_path = Path(cache_dir)
    split_file = cache_path / SPLIT_FILES[split]

    if not split_file.exists():
        # Search subdirectories
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
            samples.append({
                "context": item["context"],
                "question": item["question"],
                "answerA": item["answerA"],
                "answerB": item["answerB"],
                "answerC": item["answerC"],
                "label": label,
            })
    return samples


# ─────────────────────────────────────────────
# Log-probability scoring (chat-template-aware)
# ─────────────────────────────────────────────

def build_scoring_messages(context, question, answer_a, answer_b, answer_c):
    """Build the chat messages presenting the multiple-choice question."""
    system = "You are answering a multiple-choice social commonsense question. Reply with only A, B, or C."
    user = (
        f"Read the following context and answer the question by choosing A, B, or C.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n\n"
        f"A) {answer_a}\n"
        f"B) {answer_b}\n"
        f"C) {answer_c}\n\n"
        f"Answer:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@torch.inference_mode()
def score_options(model, tokenizer, context, question, answers):
    """
    Score each answer option using the chat template.

    For each option (A/B/C), we:
    1. Build the full chat prompt (system + user with all 3 choices)
    2. Apply the chat template with generation prompt (adds assistant header)
    3. Compute the log-probability of the answer letter token ("A", "B", or "C")
    4. Pick the option with the highest log-prob

    This works correctly with instruct models because the probabilities
    are evaluated in the distribution the model was trained on (after
    the assistant header token).
    """
    messages = build_scoring_messages(context, question, *answers)

    # Apply chat template — this adds the assistant header at the end
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize the prompt
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)

    # Forward pass
    outputs = model(prompt_ids)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Get log-probs at the last position (where the model predicts the next token)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1)

    # Get the token IDs for "A", "B", "C"
    scores = []
    for letter in ["A", "B", "C"]:
        # Try multiple tokenizations (some tokenizers add space prefix)
        token_ids = []
        for variant in [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                token_ids.extend(ids)

        if not token_ids:
            scores.append(float("-inf"))
            continue

        # Take the best (highest log-prob) tokenization
        best_score = max(log_probs[tid].item() for tid in token_ids)
        scores.append(best_score)

    # Predicted answer = highest scoring option
    pred_idx = int(np.argmax(scores))
    pred_label = ["A", "B", "C"][pred_idx]

    return scores, pred_label


# ─────────────────────────────────────────────
# Question categorizer (Sap et al. 2019, Figure 3)
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

def evaluate(model, tokenizer, dataset, max_samples=None):
    """Run log-probability scoring evaluation."""
    results = []
    correct = 0
    total = 0
    skipped = 0

    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    print(f"\nEvaluating {n} samples (log-probability scoring)...")
    start_time = time.time()

    for i, sample in enumerate(samples):
        context = sample["context"]
        question = sample["question"]
        answers = [sample["answerA"], sample["answerB"], sample["answerC"]]
        gold_label = sample["label"]

        # Skip if no label
        if gold_label not in VALID_LABELS:
            skipped += 1
            continue

        # Score all options
        scores, pred_label = score_options(model, tokenizer, context, question, answers)

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
            "scores": scores,
            "correct": is_correct,
            "category": category,
        }
        results.append(result)

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.time() - start_time
            acc = correct / total * 100 if total > 0 else 0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n}] acc={acc:.1f}% ({correct}/{total}) "
                  f"skipped={skipped} | {rate:.1f} samples/s | ETA {eta:.0f}s")

    return results, correct, total, skipped


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

# Paper reference numbers
PAPER_BERT_LARGE = {"effects": 68.0, "motivations": 68.0, "needs": 67.0,
                    "reactions": 66.0, "wants": 66.0, "descriptions": 63.0}
PAPER_BERT_OVERALL = {"dev": 66.0, "test": 64.5}
PAPER_HUMAN = {"dev": 86.9, "test": 84.4}
CATEGORY_ORDER = ["wants", "reactions", "descriptions", "motivations", "needs", "effects", "other"]


def main():
    ap = argparse.ArgumentParser(description="Evaluate on Social IQA (log-prob scoring)")
    ap.add_argument("--model", required=True, help="Model path or HF ID")
    ap.add_argument("--split", default="validation",
                    choices=["validation", "test", "train"])
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.output is None:
        model_tag = Path(args.model).name.replace("/", "_")
        args.output = f"results_siqa_{args.split}_{model_tag}.json"

    # Map split name for paper reference
    paper_split = "dev" if args.split == "validation" else args.split

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    print(f"Loading Social IQA ({args.split})...")
    dataset = load_social_iqa(args.split)
    print(f"  Loaded {len(dataset)} samples")

    # Check labels
    sample_labels = [dataset[i]["label"] for i in range(min(10, len(dataset)))]
    has_labels = all(l in VALID_LABELS for l in sample_labels)
    if not has_labels:
        print(f"  WARNING: Labels not available in {args.split} split!")
        print(f"  Sample labels: {sample_labels}")
        if args.split == "test":
            print("  Test split may lack labels. Proceeding anyway...")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"  Device: {model.device}")

    # Evaluate
    results, correct, total, skipped = evaluate(
        model, tokenizer, dataset, args.max_samples
    )

    # ─── Overall results ───
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"SOCIAL IQA RESULTS (Log-Probability Scoring)")
    print(f"{'='*70}")
    print(f"Model:       {args.model}")
    print(f"Split:       {args.split}")
    print(f"Method:      Log-probability scoring (chat-template, letter token)")
    print(f"Samples:     {len(results)}")
    print(f"Evaluated:   {total}")
    print(f"Skipped:     {skipped}")
    print(f"Correct:     {correct}")
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"")
    print(f"Paper ref:   BERT-large {paper_split}={PAPER_BERT_OVERALL.get(paper_split, 'N/A')}%")
    print(f"             Human {paper_split}={PAPER_HUMAN.get(paper_split, 'N/A')}%")

    # ─── Per-label accuracy ───
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

    # ─── Per-category accuracy ───
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
        pct = t / total * 100
        bert_ref = PAPER_BERT_LARGE.get(cat, "")
        bert_str = f"{bert_ref:.1f}%" if bert_ref else "—"
        print(f"{cat:<16} {acc:>7.1f}% {bert_str:>8} {t:>6} {pct:>7.1f}%")

    print(f"{'-'*16} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
    bert_overall = PAPER_BERT_OVERALL.get(paper_split, "")
    bert_str = f"{bert_overall:.1f}%" if bert_overall else "—"
    print(f"{'OVERALL':<16} {accuracy:>7.1f}% {bert_str:>8} {total:>6} {'100.0%':>8}")

    # ─── Save results ───
    output_data = {
        "model": args.model,
        "split": args.split,
        "method": "log_probability_chat_template",
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

    # ─── TSV for Excel ───
    print(f"\n--- TSV (copy to Excel) ---")
    print(f"Category\tCorrect\tTotal\tAccuracy %\t% of Data\tBERT-large (paper)")
    for cat in CATEGORY_ORDER:
        t = cat_total.get(cat, 0)
        if t == 0:
            continue
        c = cat_correct.get(cat, 0)
        acc = c / t * 100
        pct = t / total * 100
        bert_ref = PAPER_BERT_LARGE.get(cat, "")
        bert_str = f"{bert_ref:.1f}%" if bert_ref else ""
        print(f"{cat}\t{c}\t{t}\t{acc:.2f}%\t{pct:.1f}%\t{bert_str}")
    print(f"OVERALL\t{correct}\t{total}\t{accuracy:.2f}%\t100.0%\t{PAPER_BERT_OVERALL.get(paper_split, '')}%")

    # ─── Sample errors ───
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\nSample errors (first 5):")
        for r in errors[:5]:
            print(f"  [{r['idx']}] ({r['category']}) Context: {r['context'][:80]}...")
            print(f"         Q: {r['question']}")
            print(f"         Gold: {r['gold_label']} | Pred: {r['predicted']} | "
                  f"Scores: A={r['scores'][0]:.3f} B={r['scores'][1]:.3f} C={r['scores'][2]:.3f}")


if __name__ == "__main__":
    main()