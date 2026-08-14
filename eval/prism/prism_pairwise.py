#!/usr/bin/env python3
"""
PRISM Pairwise Evaluation for CultureLLM models (BATCHED).

Replicates the SynthesizeMe DefaultJudge evaluation:
- Dataset: MichaelR207/prism_personalized_0125
- Per-user evaluation on test split
- Both orderings per test pair (flip augmentation)
- Win rate with 95% confidence intervals
- Prompt format matches their DSPy LLMAsAJudge signature
- Batched generation for ~4-6x speedup over sequential

Usage:
    python prism_pairwise.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path /path/to/culturellm-adapter \
        --batch_size 8 \
        --output_dir prism_results
"""

import os
import json
import random
import re
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import bootstrap
from tqdm import tqdm


# ─── Dataset ───
PRISM_PERSONALIZED = "MichaelR207/prism_personalized_0125"


# ─── Prompt (replicating DSPy LLMAsAJudge signature) ───
SYSTEM_PROMPT = """Your input fields are:
1. `conversation` (str): The conversation context leading up to the completions.
2. `first_completion` (str): The first of the two possible completions to judge between.
3. `second_completion` (str): The second of the two possible completions to judge between.

Your output fields are:
1. `reasoning` (str)
2. `preference` (Literal['First', 'Second']): The completion that the judge is more likely to prefer. Possible values are 'First' and 'Second'.

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## conversation ## ]]
{conversation}

[[ ## first_completion ## ]]
{first_completion}

[[ ## second_completion ## ]]
{second_completion}

Outputs will be a JSON object with the following fields.

{
  "reasoning": "{reasoning}",
  "preference": "{preference}        # note: the value you produce must be one of: First; Second"
}

In adhering to this structure, your objective is:
Given a conversation and two completions from different models, determine which completion the human judge is more likely to prefer. Use any provided context to learn about the personal preferences of the judge before making a decision. If no context is provided it can be useful to speculate about the preferences of the judge. It's okay to be wrong, let's explore the space of possibilities and hypothesize about what might be true. Please hypothesize between 1-3 speculations about the judge's preferences or persona when reasoning. Draw from the context of the conversation and the completions as well as the user written statements to make your decision."""

USER_TEMPLATE = """[[ ## conversation ## ]]
{conversation}

[[ ## first_completion ## ]]
{first_completion}

[[ ## second_completion ## ]]
{second_completion}

Respond with a JSON object in the following order of fields: `reasoning`, then `preference` (must be formatted as a valid Python Literal['First', 'Second'])."""


def format_conversation(conversation):
    """Format conversation turns — matches their format_conv.py."""
    if isinstance(conversation, np.ndarray):
        conversation = conversation.tolist()
    if not isinstance(conversation, list):
        conversation = [conversation]

    output = []
    for turn in conversation:
        if isinstance(turn, dict):
            output.append(f"[{turn['role']}]: {turn['content']}")
        else:
            output.append(str(turn))
    return '\n\n'.join(output)


def load_prism_data():
    """Load PRISM PersonalRewardBench and organize by user."""
    print("Loading PRISM PersonalRewardBench...")
    ds = load_dataset(PRISM_PERSONALIZED)

    dfs = []
    for split_name in ds.keys():
        dfs.append(ds[split_name].to_pandas())
    all_data = pd.concat(dfs, ignore_index=True)

    users = {}
    for user_id, group in all_data.groupby('user_id'):
        test = group[group['split'] == 'test']
        if len(test) > 0:
            users[user_id] = {
                'test': test.to_dict('records'),
                'n_test': len(test),
            }

    total_test_pairs = sum(u['n_test'] for u in users.values())
    print(f"  Total users with test data: {len(users)}")
    print(f"  Total test pairs: {total_test_pairs}")
    print(f"  Total judgments (2x flip): {total_test_pairs * 2}")

    return users


def load_model(model_name, adapter_path=None):
    """Load base model + optional LoRA adapter."""
    print(f"Loading model: {model_name}")
    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-padding is required for batched generation
    tokenizer.padding_side = "left"

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


def build_all_prompts(users, user_ids, seed=42):
    """Pre-compute ALL prompts (normal + flipped) for batched generation.
    Returns a flat list of prompt dicts with metadata for reassembly."""
    prompts = []

    for user_id in user_ids:
        rand = random.Random(seed)
        for pair_idx, row in enumerate(users[user_id]['test']):
            conversation_str = format_conversation(row['context'])
            flip = row.get('flip', rand.choice([True, False]))

            if flip:
                comp_one = format_conversation(row['chosen'])
                comp_two = format_conversation(row['rejected'])
                chosen_label = 'First'
            else:
                comp_one = format_conversation(row['rejected'])
                comp_two = format_conversation(row['chosen'])
                chosen_label = 'Second'

            flipped_chosen = 'Second' if chosen_label == 'First' else 'First'

            # Normal ordering
            user_content_normal = USER_TEMPLATE.format(
                conversation=conversation_str,
                first_completion=comp_one,
                second_completion=comp_two,
            )
            prompts.append({
                'user_id': user_id,
                'pair_idx': pair_idx,
                'ordering': 'normal',
                'chosen_label': chosen_label,
                'messages': [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content_normal},
                ],
            })

            # Flipped ordering
            user_content_flipped = USER_TEMPLATE.format(
                conversation=conversation_str,
                first_completion=comp_two,
                second_completion=comp_one,
            )
            prompts.append({
                'user_id': user_id,
                'pair_idx': pair_idx,
                'ordering': 'flipped',
                'chosen_label': flipped_chosen,
                'messages': [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content_flipped},
                ],
            })

    return prompts


def generate_batch(model, tokenizer, messages_list, max_new_tokens=512):
    """Generate responses for a batch of message lists."""
    # Apply chat template to each
    texts = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    # Tokenize with left-padding for batched generation
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=7680,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,  # required when do_sample=False to avoid warning
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens for each example
    results = []
    attention_mask = inputs["attention_mask"]
    for i in range(len(messages_list)):
        input_len = attention_mask[i].sum().item()
        generated = output_ids[i][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(text)

    return results


def parse_preference(text):
    """Parse model output for First or Second preference."""
    text = text.split("</think>")[-1].strip()

    # Try JSON parsing first
    try:
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            pref = parsed.get('preference', '')
            if 'First' in pref and 'Second' not in pref:
                return 'First'
            elif 'Second' in pref and 'First' not in pref:
                return 'Second'
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: look for First/Second
    has_first = 'first' in text.lower()
    has_second = 'second' in text.lower()

    if has_first and not has_second:
        return 'First'
    elif has_second and not has_first:
        return 'Second'

    # Both appear — use last occurrence
    last_first = text.lower().rfind('first')
    last_second = text.lower().rfind('second')
    if last_first > last_second:
        return 'First'
    elif last_second > last_first:
        return 'Second'

    return 'Tie'


def is_correct(preference, chosen_label):
    """Check if the parsed preference matches the chosen label."""
    if preference == 'First' and chosen_label == 'First':
        return True
    if preference == 'Second' and chosen_label == 'Second':
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="PRISM pairwise eval (batched)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="prism_results")
    parser.add_argument("--max_users", type=int, default=None,
                        help="Limit number of users (for debugging)")
    args = parser.parse_args()

    # Model label
    if args.adapter_path:
        model_label = os.path.basename(args.adapter_path.rstrip("/"))
    else:
        model_label = os.path.basename(args.model_name.rstrip("/"))

    print("=" * 70)
    print("PRISM PAIRWISE EVALUATION (DefaultJudge replication, BATCHED)")
    print(f"  Base model:  {args.model_name}")
    print(f"  Adapter:     {args.adapter_path or 'None'}")
    print(f"  Model label: {model_label}")
    print(f"  Seed:        {args.seed}")
    print(f"  Batch size:  {args.batch_size}")
    print("=" * 70)

    # Load data
    users = load_prism_data()
    user_ids = sorted(users.keys())

    if args.max_users:
        user_ids = user_ids[:args.max_users]
        print(f"  Limited to {args.max_users} users")

    # Load model
    model, tokenizer = load_model(args.model_name, args.adapter_path)

    # Build all prompts upfront
    print("Building all prompts...")
    all_prompts = build_all_prompts(users, user_ids, seed=args.seed)
    print(f"  Total prompts to process: {len(all_prompts)}")

    # Process in batches
    all_outputs = []
    num_batches = (len(all_prompts) + args.batch_size - 1) // args.batch_size

    for batch_start in tqdm(range(0, len(all_prompts), args.batch_size),
                            total=num_batches, desc="Batches"):
        batch = all_prompts[batch_start:batch_start + args.batch_size]
        messages_list = [p['messages'] for p in batch]

        try:
            outputs = generate_batch(model, tokenizer, messages_list)
        except torch.cuda.OutOfMemoryError:
            # Fall back to smaller batch or one-at-a-time
            print(f"  OOM at batch_size={len(batch)}, falling back to sequential...")
            torch.cuda.empty_cache()
            outputs = []
            for p in batch:
                try:
                    out = generate_batch(model, tokenizer, [p['messages']])
                    outputs.append(out[0])
                except torch.cuda.OutOfMemoryError:
                    print(f"  OOM on single example, skipping...")
                    torch.cuda.empty_cache()
                    outputs.append("")

        all_outputs.extend(outputs)

    # Parse results and compute scores
    assert len(all_outputs) == len(all_prompts), \
        f"Output count mismatch: {len(all_outputs)} vs {len(all_prompts)}"

    per_user_scores = {}
    overall_scores = []
    total_ties = 0

    for prompt, output in zip(all_prompts, all_outputs):
        preference = parse_preference(output)
        if preference == 'Tie':
            total_ties += 1
        correct = is_correct(preference, prompt['chosen_label'])
        score = 1 if correct else 0
        overall_scores.append(score)

        uid = prompt['user_id']
        if uid not in per_user_scores:
            per_user_scores[uid] = []
        per_user_scores[uid].append(score)

    # Compute overall metrics with confidence intervals
    overall_arr = np.array(overall_scores)
    overall_mean = float(np.mean(overall_arr))

    ci = bootstrap(
        (overall_arr,),
        np.mean,
        confidence_level=0.95,
        method='basic'
    )

    # Per-user win rates
    per_user_results = {}
    for uid, scores in per_user_scores.items():
        per_user_results[uid] = {
            'n_test': users[uid]['n_test'],
            'n_scores': len(scores),
            'win_rate': float(np.mean(scores)),
        }

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model:              {model_label}")
    print(f"Seed:               {args.seed}")
    print(f"Users evaluated:    {len(user_ids)}")
    print(f"Total judgments:    {len(overall_scores)} (2x per test pair)")
    print(f"Ties (unparsed):    {total_ties}")
    print(f"")
    print(f"Win Rate:           {overall_mean:.4f} ({overall_mean*100:.2f}%)")
    print(f"95% CI:             [{ci.confidence_interval.low:.4f}, {ci.confidence_interval.high:.4f}]")
    print(f"                    [{ci.confidence_interval.low*100:.2f}%, {ci.confidence_interval.high*100:.2f}%]")
    print("=" * 70)

    print(f"\nReference (SynthesizeMe paper, Llama-3.1-8B on PRISM):")
    print(f"  Default Judge:    52.80%")
    print(f"  Demographics:     54.06%")
    print(f"  Memory:           54.17%")
    print(f"  SynthesizeMe:     55.24%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_base = f"{args.output_dir}/prism_{model_label}_seed{args.seed}"

    json_out = {
        'model': args.model_name,
        'adapter_path': args.adapter_path,
        'model_label': model_label,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'win_rate': overall_mean,
        'ci_lower': float(ci.confidence_interval.low),
        'ci_upper': float(ci.confidence_interval.high),
        'n_users': len(user_ids),
        'n_judgments': len(overall_scores),
        'n_ties': total_ties,
        'per_user': per_user_results,
    }
    with open(f"{output_base}.json", 'w') as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\nResults saved to: {output_base}.json")


if __name__ == "__main__":
    main()