#!/usr/bin/env python3
"""
Aggregate pairwise evaluation results across models, subsets, and seeds.
 
Usage:
    python aggregate_results.py
    python aggregate_results.py --results_dir prb_results
"""
 
import os
import json
import glob
import argparse
from collections import defaultdict
 
 
SUBSET_ORDER = ["Art", "Lifestyle", "Society"]
 
 
def load_all_results(results_dir):
    """Load all JSON result files (exclude _details files)."""
    pattern = os.path.join(results_dir, "*_pairwise_seed*.json")
    files = sorted(glob.glob(pattern))
    # Filter out the per-example detail files
    files = [f for f in files if not f.endswith("_details.json")]
 
    if not files:
        print(f"No result files found in {results_dir}/")
        print("Expected pattern: *_pairwise_seed*.json")
        return []
 
    results = []
    for f in files:
        with open(f, 'r') as fh:
            results.append(json.load(fh))
    return results
 
 
def aggregate(results):
    """Group by (model_label, subset) and average across seeds."""
    grouped = defaultdict(list)
    for r in results:
        key = (r['model_label'], r['subset_short'])
        grouped[key].append(r)
 
    table = {}
    for (model, subset), runs in grouped.items():
        accs = [r['metrics']['accuracy'] for r in runs]
        pos_a_accs = [r['metrics']['pos_in_a_accuracy'] for r in runs]
        pos_b_accs = [r['metrics']['pos_in_b_accuracy'] for r in runs]
        biases = [r['position_bias'] for r in runs]
        failed = [r['failed_parses'] for r in runs]
 
        table[(model, subset)] = {
            'n_seeds': len(runs),
            'seeds': [r['seed'] for r in runs],
            'accuracy_mean': sum(accs) / len(accs),
            'accuracy_per_seed': accs,
            'pos_a_accuracy_mean': sum(pos_a_accs) / len(pos_a_accs),
            'pos_b_accuracy_mean': sum(pos_b_accs) / len(pos_b_accs),
            'position_bias_mean': sum(biases) / len(biases),
            'failed_parses_total': sum(failed),
        }
 
    return table
 
 
def print_table(table):
    """Print a formatted comparison table."""
    models = sorted(set(m for m, _ in table.keys()))
    subsets = [s for s in SUBSET_ORDER if any(s == sub for _, sub in table.keys())]
 
    print("\n" + "=" * 90)
    print("PERSONALIZED REWARDBENCH — PAIRWISE EVALUATION RESULTS")
    print("=" * 90)
 
    # Header
    header = f"{'Model':<40}"
    for s in subsets:
        header += f"  {s:>12}"
    header += f"  {'Average':>12}"
    print(header)
    print("-" * 90)
 
    for model in models:
        row = f"{model:<40}"
        subset_accs = []
        for s in subsets:
            key = (model, s)
            if key in table:
                acc = table[key]['accuracy_mean']
                row += f"  {acc*100:>10.2f}%"
                subset_accs.append(acc)
            else:
                row += f"  {'N/A':>12}"
        if subset_accs:
            avg = sum(subset_accs) / len(subset_accs)
            row += f"  {avg*100:>10.2f}%"
        else:
            row += f"  {'N/A':>12}"
        print(row)
 
    print("-" * 90)
 
    # Position bias breakdown
    print(f"\nPOSITION BIAS ANALYSIS")
    print("-" * 90)
    header2 = f"{'Model':<40}"
    for s in subsets:
        header2 += f"  {s+' bias':>12}"
    print(header2)
    print("-" * 90)
 
    for model in models:
        row = f"{model:<40}"
        for s in subsets:
            key = (model, s)
            if key in table:
                bias = table[key]['position_bias_mean']
                row += f"  {bias*100:>10.2f}%"
            else:
                row += f"  {'N/A':>12}"
        print(row)
 
    print("-" * 90)
 
    # Per-seed detail
    print(f"\nPER-SEED ACCURACY DETAIL")
    print("-" * 90)
    for model in models:
        print(f"\n  {model}:")
        for s in subsets:
            key = (model, s)
            if key in table:
                entry = table[key]
                seed_strs = [f"seed {sd}: {ac*100:.2f}%"
                             for sd, ac in zip(entry['seeds'], entry['accuracy_per_seed'])]
                print(f"    {s:<12} — {', '.join(seed_strs)}  (mean: {entry['accuracy_mean']*100:.2f}%)")
                print(f"    {'':12}   pos_A acc: {entry['pos_a_accuracy_mean']*100:.2f}%, "
                      f"pos_B acc: {entry['pos_b_accuracy_mean']*100:.2f}%, "
                      f"failed_parses: {entry['failed_parses_total']}")
 
    print("\n" + "=" * 90)
 
 
def save_summary(table, results_dir):
    """Save a summary JSON."""
    serializable = {}
    for (model, subset), v in table.items():
        serializable[f"{model}__{subset}"] = v
 
    out_path = os.path.join(results_dir, "aggregated_summary.json")
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAggregated summary saved to: {out_path}")
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="prb_results",
                        help="Directory containing result JSON files")
    args = parser.parse_args()
 
    results = load_all_results(args.results_dir)
    if not results:
        return
 
    print(f"Loaded {len(results)} result file(s)")
    table = aggregate(results)
    print_table(table)
    save_summary(table, args.results_dir)
 
 
if __name__ == "__main__":
    main()