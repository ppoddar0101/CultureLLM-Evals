#!/usr/bin/env python3
"""
Break down pairwise evaluation results by fine-grained category.
Cross-references per-example detail files with the HuggingFace dataset
to get category labels, then computes accuracy per category.

Usage:
    python category_breakdown.py
    python category_breakdown.py --results_dir prb_results
"""

import os
import json
import glob
import argparse
from collections import defaultdict
from datasets import load_dataset


HF_DATASET = "QiyaoMa/Personalized-RewardBench"

SUBSETS = [
    "Art_and_Entertainment",
    "Lifestyle_and_Personal_Development",
    "Society_and_Culture",
]

SUBSET_SHORT = {
    "Art_and_Entertainment": "Art",
    "Lifestyle_and_Personal_Development": "Lifestyle",
    "Society_and_Culture": "Society",
}


def build_category_map():
    """Load all subsets and build id -> (subset, category) mapping."""
    print("Loading dataset from HuggingFace to get category labels...")
    id_to_cat = {}
    subset_categories = defaultdict(set)

    for subset in SUBSETS:
        ds = load_dataset(HF_DATASET, subset, split="test")
        for item in ds:
            id_to_cat[item['id']] = {
                'category': item['category'],
                'subset': subset,
                'subset_short': SUBSET_SHORT[subset],
            }
            subset_categories[SUBSET_SHORT[subset]].add(item['category'])

    print(f"Loaded {len(id_to_cat)} examples across {len(SUBSETS)} subsets")
    for s, cats in sorted(subset_categories.items()):
        print(f"  {s}: {len(cats)} categories — {', '.join(sorted(cats))}")
    print()

    return id_to_cat


def load_detail_files(results_dir):
    """Load all per-example detail JSON files."""
    pattern = os.path.join(results_dir, "*_details.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No detail files found in {results_dir}/")
        print("Expected pattern: *_pairwise_seed*_details.json")
        return []

    all_runs = []
    for f in files:
        # Parse model and seed from filename
        # Format: {subset}_{model}_pairwise_seed{seed}_details.json
        basename = os.path.basename(f).replace("_details.json", "")
        parts = basename.split("_pairwise_seed")
        if len(parts) != 2:
            print(f"  Skipping unrecognized file: {f}")
            continue

        seed = int(parts[1])
        # The first part is {subset_short}_{model_label}
        # subset_short is the first token before the first underscore that matches known subsets
        prefix = parts[0]
        model_label = None
        subset_short = None
        for ss in SUBSET_SHORT.values():
            if prefix.startswith(ss + "_"):
                subset_short = ss
                model_label = prefix[len(ss) + 1:]
                break

        if not model_label:
            print(f"  Skipping unrecognized file: {f}")
            continue

        with open(f, 'r') as fh:
            examples = json.load(fh)

        all_runs.append({
            'file': f,
            'model_label': model_label,
            'subset_short': subset_short,
            'seed': seed,
            'examples': examples,
        })

    print(f"Loaded {len(all_runs)} detail files")
    return all_runs


def compute_category_breakdown(all_runs, id_to_cat):
    """Compute per-category accuracy, averaged across seeds."""
    # Group: (model, subset, category) -> list of (correct, total) per seed
    raw = defaultdict(lambda: defaultdict(list))

    for run in all_runs:
        model = run['model_label']
        seed = run['seed']

        # Group examples by category for this run
        cat_correct = defaultdict(int)
        cat_total = defaultdict(int)

        for ex in run['examples']:
            info = id_to_cat.get(ex['id'])
            if not info:
                continue
            cat = info['category']
            subset = info['subset_short']
            key = (model, subset, cat)
            cat_total[key] += 1
            if ex['correct']:
                cat_correct[key] += 1

        for key in cat_total:
            raw[key][seed] = {
                'correct': cat_correct[key],
                'total': cat_total[key],
                'accuracy': cat_correct[key] / cat_total[key] if cat_total[key] > 0 else 0.0,
            }

    # Average across seeds
    aggregated = {}
    for (model, subset, cat), seed_data in raw.items():
        accs = [v['accuracy'] for v in seed_data.values()]
        totals = [v['total'] for v in seed_data.values()]
        aggregated[(model, subset, cat)] = {
            'n_seeds': len(seed_data),
            'accuracy_mean': sum(accs) / len(accs),
            'accuracy_per_seed': {str(s): v['accuracy'] for s, v in seed_data.items()},
            'count': totals[0],  # same count across seeds
        }

    return aggregated


def print_breakdown(aggregated):
    """Print formatted category breakdown."""
    models = sorted(set(m for m, _, _ in aggregated.keys()))
    subsets_present = sorted(set(s for _, s, _ in aggregated.keys()),
                             key=lambda x: ["Art", "Lifestyle", "Society"].index(x)
                             if x in ["Art", "Lifestyle", "Society"] else 99)

    print("\n" + "=" * 100)
    print("PERSONALIZED REWARDBENCH — CATEGORY BREAKDOWN")
    print("=" * 100)

    for subset in subsets_present:
        print(f"\n{'─' * 100}")
        print(f"  {subset}")
        print(f"{'─' * 100}")

        # Get all categories in this subset
        categories = sorted(set(c for m, s, c in aggregated if s == subset))

        # Header
        header = f"  {'Category':<30} {'Count':>6}"
        for model in models:
            header += f"  {model:>20}"
        if len(models) == 2:
            header += f"  {'Δ':>8}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")

        subset_totals = {m: {'correct': 0, 'total': 0} for m in models}

        for cat in categories:
            row = f"  {cat:<30}"
            accs = {}
            count = 0
            for model in models:
                key = (model, subset, cat)
                if key in aggregated:
                    entry = aggregated[key]
                    acc = entry['accuracy_mean']
                    count = entry['count']
                    accs[model] = acc
                    row += f"  {acc * 100:>18.2f}%"
                    subset_totals[model]['correct'] += acc * count
                    subset_totals[model]['total'] += count
                else:
                    row += f"  {'N/A':>20}"

            row = f"  {cat:<30} {count:>6}" + row.split(f"{cat:<30}")[1]

            # Delta column
            if len(models) == 2 and len(accs) == 2:
                vals = list(accs.values())
                delta = vals[1] - vals[0]
                sign = "+" if delta >= 0 else ""
                row += f"  {sign}{delta * 100:>5.2f}%"

            print(row)

        # Subset total
        print(f"  {'-' * (len(header) - 2)}")
        total_row = f"  {'SUBSET TOTAL':<30} {'':>6}"
        for model in models:
            t = subset_totals[model]
            if t['total'] > 0:
                acc = t['correct'] / t['total']
                total_row += f"  {acc * 100:>18.2f}%"
            else:
                total_row += f"  {'N/A':>20}"
        print(total_row)

    print(f"\n{'=' * 100}")


def save_breakdown(aggregated, results_dir):
    """Save category breakdown as JSON."""
    serializable = {}
    for (model, subset, cat), v in aggregated.items():
        key = f"{model}__{subset}__{cat}"
        serializable[key] = v

    out_path = os.path.join(results_dir, "category_breakdown.json")
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nCategory breakdown saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="prb_results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()

    id_to_cat = build_category_map()
    all_runs = load_detail_files(args.results_dir)
    if not all_runs:
        return

    aggregated = compute_category_breakdown(all_runs, id_to_cat)
    print_breakdown(aggregated)
    save_breakdown(aggregated, args.results_dir)


if __name__ == "__main__":
    main()