#!/usr/bin/env python3
"""
PRISM Original Dataset — Pairwise Evaluation for CultureLLM.

Constructs preference pairs from the original PRISM utterances (scored responses),
evaluates CultureLLM models as judges, and breaks down accuracy by user country.

Usage:
    python prism_original_pairwise.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --adapter_path /path/to/culturellm-adapter \
        --batch_size 8 \
        --output_dir prism_original_results
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
from tqdm import tqdm


# ─── Prompt (with demographic context) ───
SYSTEM_PROMPT = """Your input fields are:
1. `user_profile` (str): Demographic background of the human judge.
2. `conversation` (str): The conversation context leading up to the completions.
3. `first_completion` (str): The first of the two possible completions to judge between.
4. `second_completion` (str): The second of the two possible completions to judge between.

Your output fields are:
1. `reasoning` (str)
2. `preference` (Literal['First', 'Second']): The completion that the judge is more likely to prefer. Possible values are 'First' and 'Second'.

All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## user_profile ## ]]
{user_profile}

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
Given a user's demographic profile, a conversation and two completions from different models, determine which completion the human judge is more likely to prefer. Use the user profile and any provided context to learn about the personal preferences of the judge before making a decision. Consider how the user's cultural background, values, and life experience may shape their preferences. It's okay to be wrong, let's explore the space of possibilities and hypothesize about what might be true. Please hypothesize between 1-3 speculations about the judge's preferences or persona when reasoning. Draw from the user profile, context of the conversation and the completions as well as the user written statements to make your decision."""

USER_TEMPLATE = """[[ ## user_profile ## ]]
{user_profile}

[[ ## conversation ## ]]
{conversation}

[[ ## first_completion ## ]]
{first_completion}

[[ ## second_completion ## ]]
{second_completion}

Respond with a JSON object in the following order of fields: `reasoning`, then `preference` (must be formatted as a valid Python Literal['First', 'Second'])."""


def load_and_build_pairs(min_score_diff=5):
    """Load original PRISM data and construct preference pairs from utterances."""
    print("Loading original PRISM data...")

    # Load utterances
    utts = load_dataset("HannahRoseKirk/prism-alignment", "utterances", split="train")
    utts_df = utts.to_pandas()
    print(f"  Utterances: {len(utts_df)}")

    # Load survey for demographics
    survey = load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train")
    survey_df = survey.to_pandas()

    # Flatten location
    if 'location' in survey_df.columns and isinstance(survey_df['location'].iloc[0], dict):
        loc_df = pd.json_normalize(survey_df['location'])
        loc_df.columns = ['loc_' + c for c in loc_df.columns]
        survey_df = pd.concat([survey_df.drop(columns=['location']), loc_df], axis=1)

    # Flatten religion
    if 'religion' in survey_df.columns and isinstance(survey_df['religion'].iloc[0], dict):
        rel_df = pd.json_normalize(survey_df['religion'])
        rel_df.columns = ['religion_' + c for c in rel_df.columns]
        survey_df = pd.concat([survey_df.drop(columns=['religion']), rel_df], axis=1)

    # Flatten ethnicity
    if 'ethnicity' in survey_df.columns and isinstance(survey_df['ethnicity'].iloc[0], dict):
        eth_df = pd.json_normalize(survey_df['ethnicity'])
        eth_df.columns = ['ethnicity_' + c for c in eth_df.columns]
        survey_df = pd.concat([survey_df.drop(columns=['ethnicity']), eth_df], axis=1)

    print(f"  Survey entries: {len(survey_df)}")

    # Build pairs from interactions
    # Group by interaction_id: each interaction has multiple responses to the same prompt
    pairs = []
    for interaction_id, group in utts_df.groupby('interaction_id'):
        if len(group) < 2:
            continue

        # Get chosen and rejected
        chosen_rows = group[group['if_chosen'] == True]
        rejected_rows = group[group['if_chosen'] == False]

        if len(chosen_rows) == 0 or len(rejected_rows) == 0:
            continue

        chosen = chosen_rows.iloc[0]
        rejected = rejected_rows.iloc[0]

        # Optional: filter by score difference for cleaner pairs
        score_diff = chosen['score'] - rejected['score']
        if score_diff < min_score_diff:
            continue

        pairs.append({
            'interaction_id': interaction_id,
            'conversation_id': chosen['conversation_id'],
            'user_id': chosen['user_id'],
            'user_prompt': chosen['user_prompt'],
            'chosen_response': chosen['model_response'],
            'rejected_response': rejected['model_response'],
            'chosen_score': int(chosen['score']),
            'rejected_score': int(rejected['score']),
            'score_diff': int(score_diff),
            'chosen_model': chosen['model_name'],
            'rejected_model': rejected['model_name'],
            'conversation_type': chosen['conversation_type'],
            'turn': int(chosen['turn']),
        })

    pairs_df = pd.DataFrame(pairs)
    print(f"  Constructed pairs (score_diff >= {min_score_diff}): {len(pairs_df)}")

    # Join with survey demographics
    demo_cols = ['user_id', 'loc_birth_country', 'loc_reside_country',
                 'loc_birth_region', 'loc_reside_subregion',
                 'age', 'gender', 'employment_status', 'education',
                 'marital_status', 'english_proficiency', 'self_description']
    # Add religion/ethnicity if they were flattened
    for col in ['religion_simplified', 'religion_categorised',
                'ethnicity_simplified']:
        if col in survey_df.columns:
            demo_cols.append(col)

    # Only keep columns that exist
    demo_cols = [c for c in demo_cols if c in survey_df.columns]
    pairs_df = pairs_df.merge(survey_df[demo_cols], on='user_id', how='left')

    # Print country coverage
    print(f"\n  Pairs per birth country (CultureLLM targets):")
    targets = ['United States', 'United Kingdom', 'France', 'Brazil', 'Italy', 'India']
    for t in targets:
        n = len(pairs_df[pairs_df['loc_birth_country'] == t])
        n_users = pairs_df[pairs_df['loc_birth_country'] == t]['user_id'].nunique()
        print(f"    {t:<25} {n:>6} pairs from {n_users:>4} users")

    print(f"\n  Pairs per conversation type:")
    for ct, count in pairs_df['conversation_type'].value_counts().items():
        print(f"    {ct:<25} {count:>6}")

    return pairs_df


def load_model(model_name, adapter_path=None):
    """Load base model + optional LoRA adapter."""
    print(f"\nLoading model: {model_name}")
    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
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


def generate_batch(model, tokenizer, messages_list, max_new_tokens=512):
    """Generate responses for a batch of message lists."""
    texts = []
    for messages in messages_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

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
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(messages_list)):
        input_len = inputs['input_ids'][i].shape[0]
        generated = output_ids[i][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(text)

    return results


def parse_preference(text):
    """Parse model output for First or Second preference."""
    text = text.split("</think>")[-1].strip()

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

    has_first = 'first' in text.lower()
    has_second = 'second' in text.lower()

    if has_first and not has_second:
        return 'First'
    elif has_second and not has_first:
        return 'Second'

    last_first = text.lower().rfind('first')
    last_second = text.lower().rfind('second')
    if last_first > last_second:
        return 'First'
    elif last_second > last_first:
        return 'Second'

    return 'Tie'


def format_user_profile(row):
    """Format demographic fields into a readable profile string."""
    parts = []

    field_map = [
        ('age', 'Age'),
        ('gender', 'Gender'),
        ('loc_birth_country', 'Birth Country'),
        ('loc_reside_country', 'Residing Country'),
        ('education', 'Education'),
        ('employment_status', 'Employment'),
        ('marital_status', 'Marital Status'),
        ('english_proficiency', 'English Proficiency'),
        ('religion_simplified', 'Religion'),
        ('ethnicity_simplified', 'Ethnicity'),
    ]

    for field, label in field_map:
        val = row.get(field, None)
        if pd.notna(val) and str(val).strip() and str(val) != 'Prefer not to say':
            parts.append(f"{label}: {val}")

    # Include self-description if available
    self_desc = row.get('self_description', None)
    if pd.notna(self_desc) and str(self_desc).strip():
        parts.append(f"Self-description: {self_desc}")

    return '\n'.join(parts) if parts else 'No profile available'


def build_prompts(pairs_df, seed=42):
    """Build all prompts with position randomization + flipped versions."""
    random.seed(seed)
    prompts = []

    for _, row in pairs_df.iterrows():
        conversation = f"[user]: {row['user_prompt']}"
        user_profile = format_user_profile(row)

        for ordering in ['normal', 'flipped']:
            if ordering == 'normal':
                first = row['chosen_response']
                second = row['rejected_response']
                chosen_label = 'First'
            else:
                first = row['rejected_response']
                second = row['chosen_response']
                chosen_label = 'Second'

            user_content = USER_TEMPLATE.format(
                user_profile=user_profile,
                conversation=conversation,
                first_completion=first,
                second_completion=second,
            )

            prompts.append({
                'interaction_id': row['interaction_id'],
                'user_id': row['user_id'],
                'birth_country': row.get('loc_birth_country', 'Unknown'),
                'reside_country': row.get('loc_reside_country', 'Unknown'),
                'region': row.get('loc_reside_subregion', 'Unknown'),
                'conversation_type': row['conversation_type'],
                'score_diff': row['score_diff'],
                'ordering': ordering,
                'chosen_label': chosen_label,
                'messages': [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            })

    return prompts


def main():
    parser = argparse.ArgumentParser(description="PRISM original pairwise eval")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--min_score_diff", type=int, default=5,
                        help="Minimum score difference for a valid pair")
    parser.add_argument("--output_dir", type=str, default="prism_original_results")
    args = parser.parse_args()

    # Model label
    if args.adapter_path:
        model_label = os.path.basename(args.adapter_path.rstrip("/"))
    else:
        model_label = os.path.basename(args.model_name.rstrip("/"))

    print("=" * 70)
    print("PRISM ORIGINAL — PAIRWISE EVALUATION")
    print(f"  Base model:  {args.model_name}")
    print(f"  Adapter:     {args.adapter_path or 'None'}")
    print(f"  Model label: {model_label}")
    print(f"  Seed:        {args.seed}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Min score diff: {args.min_score_diff}")
    print("=" * 70)

    # Load data and build pairs
    pairs_df = load_and_build_pairs(min_score_diff=args.min_score_diff)

    # Load model
    model, tokenizer = load_model(args.model_name, args.adapter_path)

    # Build all prompts (normal + flipped for each pair)
    print("\nBuilding prompts...")
    all_prompts = build_prompts(pairs_df, seed=args.seed)
    print(f"  Total prompts: {len(all_prompts)} ({len(pairs_df)} pairs x 2 orderings)")

    # Batched inference
    all_outputs = []
    num_batches = (len(all_prompts) + args.batch_size - 1) // args.batch_size

    for batch_start in tqdm(range(0, len(all_prompts), args.batch_size),
                            total=num_batches, desc="Batches"):
        batch = all_prompts[batch_start:batch_start + args.batch_size]
        messages_list = [p['messages'] for p in batch]

        try:
            outputs = generate_batch(model, tokenizer, messages_list)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch_size={len(batch)}, falling back to sequential...")
            torch.cuda.empty_cache()
            outputs = []
            for p in batch:
                try:
                    out = generate_batch(model, tokenizer, [p['messages']])
                    outputs.append(out[0])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    outputs.append("")

        all_outputs.extend(outputs)

    # Parse and score
    results = []
    for prompt, output in zip(all_prompts, all_outputs):
        preference = parse_preference(output)
        correct = (preference == 'First' and prompt['chosen_label'] == 'First') or \
                  (preference == 'Second' and prompt['chosen_label'] == 'Second')

        results.append({
            'interaction_id': prompt['interaction_id'],
            'user_id': prompt['user_id'],
            'birth_country': prompt['birth_country'],
            'reside_country': prompt['reside_country'],
            'region': prompt['region'],
            'conversation_type': prompt['conversation_type'],
            'score_diff': prompt['score_diff'],
            'ordering': prompt['ordering'],
            'correct': correct,
            'preference': preference,
        })

    results_df = pd.DataFrame(results)

    # ── Compute metrics ──
    overall_acc = results_df['correct'].mean()
    n_ties = (results_df['preference'] == 'Tie').sum()

    # Per birth country
    country_metrics = {}
    for country, group in results_df.groupby('birth_country'):
        if group['user_id'].nunique() >= 2:
            country_metrics[country] = {
                'accuracy': group['correct'].mean(),
                'n_pairs': len(group) // 2,
                'n_judgments': len(group),
                'n_users': group['user_id'].nunique(),
            }

    # Per conversation type
    type_metrics = {}
    for ctype, group in results_df.groupby('conversation_type'):
        type_metrics[ctype] = {
            'accuracy': group['correct'].mean(),
            'n_judgments': len(group),
        }

    # ── Print results ──
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Model:              {model_label}")
    print(f"Total pairs:        {len(pairs_df)}")
    print(f"Total judgments:     {len(results)} (2x per pair)")
    print(f"Ties (unparsed):    {n_ties}")
    print(f"Overall Win Rate:   {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    print(f"\nPer Conversation Type:")
    for ctype in sorted(type_metrics.keys()):
        m = type_metrics[ctype]
        print(f"  {ctype:<25} {m['accuracy']*100:.2f}%  (n={m['n_judgments']})")

    print(f"\nPer Birth Country (CultureLLM targets):")
    targets = ['United States', 'United Kingdom', 'France', 'Italy', 'India', 'Brazil']
    for t in targets:
        if t in country_metrics:
            m = country_metrics[t]
            print(f"  {t:<25} {m['accuracy']*100:.2f}%  (pairs={m['n_pairs']}, users={m['n_users']})")
        else:
            print(f"  {t:<25} N/A (too few users)")

    print(f"\nPer Birth Country (all with 2+ users, sorted by count):")
    for country in sorted(country_metrics.keys(),
                          key=lambda c: country_metrics[c]['n_users'], reverse=True):
        m = country_metrics[country]
        marker = " ★" if country in targets else ""
        print(f"  {country:<25} {m['accuracy']*100:.2f}%  (pairs={m['n_pairs']}, users={m['n_users']}){marker}")

    # ── Save results ──
    os.makedirs(args.output_dir, exist_ok=True)
    output_base = f"{args.output_dir}/prism_orig_{model_label}_seed{args.seed}"

    json_out = {
        'model': args.model_name,
        'adapter_path': args.adapter_path,
        'model_label': model_label,
        'seed': args.seed,
        'min_score_diff': args.min_score_diff,
        'overall_accuracy': float(overall_acc),
        'n_pairs': len(pairs_df),
        'n_judgments': len(results),
        'n_ties': int(n_ties),
        'per_birth_country': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                   for kk, vv in v.items()}
                               for k, v in country_metrics.items()},
        'per_conversation_type': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                       for kk, vv in v.items()}
                                   for k, v in type_metrics.items()},
    }
    with open(f"{output_base}.json", 'w') as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\nResults saved to: {output_base}.json")


if __name__ == "__main__":
    main()