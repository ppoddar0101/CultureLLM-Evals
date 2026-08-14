import os
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

from datasets import load_dataset
from google import genai
from google.genai import types
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Literal


SUBSET_SHORT = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}

HF_DATASET = "QiyaoMa/Personalized-RewardBench"

SYSTEM_PROMPT = """You are a fair and insightful judge. Your task is to evaluate a user's question and two candidate answers. Select the answer that best aligns with the user's expected intent.

Rules:
1. Do not be biased by response order.
2. Do not be biased by response length.
3. Judge only which answer is better for the user's question.

Return only valid JSON matching the required schema.
"""

USER_TEMPLATE = """User's current question:
```
{question}
```

Response A:
```
{response_a}
```

Response B:
```
{response_b}
```"""


def load_model(model_name: str) -> Tuple[genai.Client, str]:
    """
    Create a Gemini client and return it with the model name.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY is not set. Export one before launching the job."
        )

    client = genai.Client(api_key=api_key)
    return client, model_name


def prepare_pairwise_prompts(dataset, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    prompts: List[Dict] = []

    for item in dataset:
        pos_in_a = random.random() < 0.5
        if pos_in_a:
            response_a, response_b = item["chosen"], item["rejected"]
            correct_answer = "A"
        else:
            response_a, response_b = item["rejected"], item["chosen"]
            correct_answer = "B"

        user_content = USER_TEMPLATE.format(
            question=item["question"],
            response_a=response_a,
            response_b=response_b,
        )

        prompts.append({
            "id": item.get("id", ""),
            "query": item["question"],
            "response_a": response_a,
            "response_b": response_b,
            "correct_answer": correct_answer,
            "pos_in_a": pos_in_a,
            "prompt": user_content,
        })

    return prompts
    
class PairwiseDecision(BaseModel):
    reasoning: str = Field(description="Brief evaluation, <= 100 words.")
    result: Literal["A", "B"] = Field(description="The better answer.")


def generation_config():
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )


def generate_single(client: genai.Client, model_name: str, prompt_text: str) -> Tuple[Optional[str], Optional[str], str]:
    import time as _time
    for _attempt in range(8):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=generation_config(),
            )
            break
        except Exception as _e:
            if '429' in str(_e) or 'RESOURCE_EXHAUSTED' in str(_e):
                _wait = min(2 ** _attempt * 5, 120)
                print(f"  Rate limited, waiting {_wait}s (attempt {_attempt+1}/8)")
                _time.sleep(_wait)
            else:
                raise
    else:
        return None, None, ""

    reasoning = None
    result = None

    # Try structured output first (response.parsed is a PairwiseDecision object)
    try:
        if response.parsed:
            reasoning = response.parsed.reasoning
            result = response.parsed.result
            if result not in ("A", "B"):
                result = None
            raw_text = json.dumps({"reasoning": reasoning, "result": result})
            return reasoning, result, raw_text
    except Exception:
        pass

    # Fallback to text parsing
    raw_text = (response.text or "").strip()

    # Strip markdown code fences
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`").strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        data = json.loads(raw_text)
        reasoning = data.get("reasoning") or data.get("reason") or data.get("explanation")
        result = data.get("result") or data.get("answer") or data.get("preference") or data.get("choice")
        if isinstance(result, str):
            result = result.strip().upper()
        if result not in ("A", "B"):
            result = None
    except Exception:
        lowered = raw_text.lower()
        for key in ['"result"', '"answer"', '"preference"', '"choice"']:
            if key in lowered:
                if '"a"' in lowered:
                    result = "A"
                elif '"b"' in lowered:
                    result = "B"
                break

    return reasoning, result, raw_text


def evaluate_pairwise(client: genai.Client, model_name: str, prompts: List[Dict]) -> tuple:
    print(f"Evaluating {len(prompts)} pairwise comparisons...")
    results = []
    failed_parse_count = 0

    for item in tqdm(prompts, desc="Evaluating pairs"):
        reasoning, prediction, raw_text = generate_single(client, model_name, item["prompt"])

        parse_failed = prediction is None
        if parse_failed:
            prediction = "A"
            failed_parse_count += 1
            print(f"  Warning: parse failed for {item['id']}: '{raw_text[:120]}'...")

        results.append({
            "id": item["id"],
            "query": item["query"],
            "pos_in_a": item["pos_in_a"],
            "prediction": prediction,
            "correct_answer": item["correct_answer"],
            "correct": prediction == item["correct_answer"],
            "parse_failed": parse_failed,
            "reasoning": reasoning,
            "raw_output": raw_text,
        })

    return results, failed_parse_count


def calculate_accuracy(results: List[Dict]) -> Dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    pos_in_a = [r for r in results if r["pos_in_a"]]
    pos_in_b = [r for r in results if not r["pos_in_a"]]

    def acc(subset):
        return sum(1 for r in subset if r["correct"]) / len(subset) if subset else 0.0

    pred_a = sum(1 for r in results if r["prediction"] == "A")
    pred_b = sum(1 for r in results if r["prediction"] == "B")

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "pos_in_a_count": len(pos_in_a),
        "pos_in_a_accuracy": acc(pos_in_a),
        "pos_in_b_count": len(pos_in_b),
        "pos_in_b_accuracy": acc(pos_in_b),
        "pred_a_count": pred_a,
        "pred_b_count": pred_b,
    }


def main():
    parser = argparse.ArgumentParser(description="Pairwise evaluation on Personalized RewardBench with Gemini")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash",
                        help="Gemini model name")
    parser.add_argument("--subset", type=str, required=True,
                        choices=list(SUBSET_SHORT.keys()),
                        help="Dataset subset name")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for position shuffling")
    parser.add_argument("--output_dir", type=str, default="prb_results",
                        help="Directory to save results")
    args = parser.parse_args()

    subset_short = SUBSET_SHORT[args.subset]
    model_label = args.model_name.replace("/", "_")

    print("=" * 70)
    print("PAIRWISE EVALUATION — Personalized RewardBench")
    print(f"  Model:       {args.model_name}")
    print(f"  Model label: {model_label}")
    print(f"  Subset:      {args.subset}")
    print(f"  Seed:        {args.seed}")
    print("=" * 70)

    dataset = load_dataset(HF_DATASET, args.subset, split=args.split)
    print(f"Loaded {len(dataset)} examples")

    client, model_name = load_model(args.model_name)

    prompts = prepare_pairwise_prompts(dataset, seed=args.seed)

    results, failed_parses = evaluate_pairwise(client, model_name, prompts)
    metrics = calculate_accuracy(results)

    position_bias = abs(metrics["pos_in_a_accuracy"] - metrics["pos_in_b_accuracy"])
    total = metrics["total"] if metrics["total"] else 1

    summary = [
        "=" * 70,
        "PAIRWISE EVALUATION COMPLETE",
        "=" * 70,
        f"Subset:      {args.subset}",
        f"Model:       {model_label}",
        f"Seed:        {args.seed}",
        "",
        f"Overall Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})",
        "",
        "Position-specific Accuracy:",
        f"  Chosen in A: {metrics['pos_in_a_accuracy']:.2%} ({metrics['pos_in_a_count']} pairs)",
        f"  Chosen in B: {metrics['pos_in_b_accuracy']:.2%} ({metrics['pos_in_b_count']} pairs)",
        f"  Position bias: {position_bias:.2%}",
        "",
        "Prediction Distribution:",
        f"  Predicted A: {metrics['pred_a_count']} ({metrics['pred_a_count']/total:.2%})",
        f"  Predicted B: {metrics['pred_b_count']} ({metrics['pred_b_count']/total:.2%})",
        "",
        f"Failed parses: {failed_parses}",
        "=" * 70,
    ]

    print("\n" + "\n".join(summary))

    os.makedirs(args.output_dir, exist_ok=True)
    output_base = f"{args.output_dir}/{subset_short}_{model_label}_pairwise_seed{args.seed}"

    with open(f"{output_base}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\nSummary saved to: {output_base}.txt")

    json_out = {
        "model": args.model_name,
        "model_label": model_label,
        "subset": args.subset,
        "subset_short": subset_short,
        "seed": args.seed,
        "metrics": metrics,
        "failed_parses": failed_parses,
        "position_bias": position_bias,
    }
    with open(f"{output_base}.json", "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON results saved to: {output_base}.json")

    with open(f"{output_base}_details.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Per-example details saved to: {output_base}_details.json")


if __name__ == "__main__":
    main()
