import os
import json
import random
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from tqdm import tqdm
 
 
SUBSET_SHORT = {
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Art_and_Entertainment": "Art",
    "Society_and_Culture": "Society",
}
 
HF_DATASET = "QiyaoMa/Personalized-RewardBench"
 
# Prompts from the original repo (inlined so no separate files needed)
SYSTEM_PROMPT = """You are a fair and insightful judge. Your task is to evaluate a user's question and two candidate answers. Select the answer that best aligns with the user's expected espects.
 
Here are some rules of the evaluation:
1. Responses should NOT contain more/less than what the instruction asks for, as such responses do NOT precisely execute the instruction.
2. You should avoid any potential bias and your judgment should be as objective as possible. Here are some potential sources of bias:
- The order in which the responses were presented should NOT affect your judgment, as Response A and Response B are **equally likely** to be the better.
- The length of the responses should NOT affect your judgement, as a longer response does not necessarily correspond to a better response. When making your decision, evaluate if the response length is appropriate for the given instruction.
 
Your reply should strictly follow this format:
**Reasoning:** <feedback evaluating the responses within 100 words>
**Result:** <A or B>"""
 
USER_TEMPLATE = """User's current question:
```
{question}
```
 
Response A:
```
{Response_A}
```
 
Response B:
```
{Response_B}
```"""
 
 
def load_model(model_name: str, adapter_path: str = None):
    """Load base model + optional LoRA adapter via PEFT."""
    print(f"Loading model: {model_name}")
    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
 
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
 
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
 
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged.")
 
    model.eval()
    return model, tokenizer
 
 
def prepare_pairwise_prompts(dataset, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    prompts = []
 
    for item in dataset:
        pos_in_a = random.random() < 0.5
        if pos_in_a:
            response_a, response_b = item['chosen'], item['rejected']
            correct_answer = "A"
        else:
            response_a, response_b = item['rejected'], item['chosen']
            correct_answer = "B"
 
        user_content = USER_TEMPLATE.replace("{question}", item['question'])
        user_content = user_content.replace("{Response_A}", response_a)
        user_content = user_content.replace("{Response_B}", response_b)
 
        prompts.append({
            'id': item['id'],
            'query': item['question'],
            'response_a': response_a,
            'response_b': response_b,
            'correct_answer': correct_answer,
            'pos_in_a': pos_in_a,
            'messages': [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        })
 
    return prompts
 
 
def generate_single(model, tokenizer, messages, max_new_tokens=256):
    """Generate a response for a single set of messages."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
 
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
 
    # Decode only the newly generated tokens
    generated_ids = output_ids[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
 
 
def parse_prediction(text: str):
    """Parse model output for Result: A or B."""
    text = text.split("</think>")[-1].strip()
    text = text.replace("\"", "").replace("*", "").strip()
 
    if "result:" in text.lower():
        result_section = text.lower().split("result:")[-1].strip()
        if result_section.startswith("a"):
            return "A"
        elif result_section.startswith("b"):
            return "B"
    return None
 
 
def evaluate_pairwise(model, tokenizer, prompts: List[Dict]) -> tuple:
    print(f"Evaluating {len(prompts)} pairwise comparisons...")
    results = []
    failed_parse_count = 0
 
    for item in tqdm(prompts, desc="Evaluating pairs"):
        prediction_text = generate_single(model, tokenizer, item['messages'])
        prediction = parse_prediction(prediction_text)
 
        parse_failed = prediction is None
        if parse_failed:
            prediction = "A"
            failed_parse_count += 1
            print(f"  Warning: parse failed for {item['id']}: '{prediction_text[:100]}'...")
 
        results.append({
            'id': item['id'],
            'query': item['query'],
            'pos_in_a': item['pos_in_a'],
            'prediction': prediction,
            'correct_answer': item['correct_answer'],
            'correct': prediction == item['correct_answer'],
            'parse_failed': parse_failed,
            'raw_output': prediction_text,
        })
 
    return results, failed_parse_count
 
 
def calculate_accuracy(results: List[Dict]) -> Dict:
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
 
    pos_in_a = [r for r in results if r['pos_in_a']]
    pos_in_b = [r for r in results if not r['pos_in_a']]
 
    def acc(subset): return sum(1 for r in subset if r['correct']) / len(subset) if subset else 0.0
 
    pred_a = sum(1 for r in results if r['prediction'] == 'A')
    pred_b = sum(1 for r in results if r['prediction'] == 'B')
 
    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0.0,
        'pos_in_a_count': len(pos_in_a),
        'pos_in_a_accuracy': acc(pos_in_a),
        'pos_in_b_count': len(pos_in_b),
        'pos_in_b_accuracy': acc(pos_in_b),
        'pred_a_count': pred_a,
        'pred_b_count': pred_b,
    }
 
 
def main():
    parser = argparse.ArgumentParser(description="Pairwise evaluation on Personalized RewardBench")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base model (HuggingFace name or local path)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional LoRA adapter path")
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
 
    # Build a readable model label
    if args.adapter_path:
        model_label = os.path.basename(args.adapter_path.rstrip("/"))
    else:
        model_label = os.path.basename(args.model_name.rstrip("/"))
 
    print("=" * 70)
    print("PAIRWISE EVALUATION — Personalized RewardBench")
    print(f"  Base model:  {args.model_name}")
    print(f"  Adapter:     {args.adapter_path or 'None'}")
    print(f"  Model label: {model_label}")
    print(f"  Subset:      {args.subset}")
    print(f"  Seed:        {args.seed}")
    print("=" * 70)
 
    # Load dataset
    dataset = load_dataset(HF_DATASET, args.subset, split=args.split)
    print(f"Loaded {len(dataset)} examples")
 
    # Load model
    model, tokenizer = load_model(args.model_name, args.adapter_path)
 
    # Prepare prompts with position shuffling
    prompts = prepare_pairwise_prompts(dataset, seed=args.seed)
 
    # Run evaluation
    results, failed_parses = evaluate_pairwise(model, tokenizer, prompts)
    metrics = calculate_accuracy(results)
 
    position_bias = abs(metrics['pos_in_a_accuracy'] - metrics['pos_in_b_accuracy'])
 
    # Print summary
    summary = [
        "=" * 70,
        "PAIRWISE EVALUATION COMPLETE",
        "=" * 70,
        f"Subset:      {args.subset}",
        f"Model:       {model_label}",
        f"Adapter:     {args.adapter_path or 'None'}",
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
        f"  Predicted A: {metrics['pred_a_count']} ({metrics['pred_a_count']/metrics['total']:.2%})",
        f"  Predicted B: {metrics['pred_b_count']} ({metrics['pred_b_count']/metrics['total']:.2%})",
        "",
        f"Failed parses: {failed_parses}",
        "=" * 70,
    ]
 
    print("\n" + "\n".join(summary))
 
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_base = f"{args.output_dir}/{subset_short}_{model_label}_pairwise_seed{args.seed}"
 
    with open(f"{output_base}.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(summary))
    print(f"\nSummary saved to: {output_base}.txt")
 
    json_out = {
        'model': args.model_name,
        'adapter_path': args.adapter_path,
        'model_label': model_label,
        'subset': args.subset,
        'subset_short': subset_short,
        'seed': args.seed,
        'metrics': metrics,
        'failed_parses': failed_parses,
        'position_bias': position_bias,
    }
    with open(f"{output_base}.json", 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON results saved to: {output_base}.json")
 
    # Also save per-example results for debugging
    with open(f"{output_base}_details.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Per-example details saved to: {output_base}_details.json")
 
 
if __name__ == "__main__":
    main()