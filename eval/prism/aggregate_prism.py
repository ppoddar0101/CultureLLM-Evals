#!/usr/bin/env python3
"""
Aggregate PRISM pairwise results across CultureLLM models.

Usage:
    python aggregate_prism.py
    python aggregate_prism.py --results_dir prism_results
"""

import os
import json
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="prism_results")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "prism_*_seed*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No result files found in {args.results_dir}/")
        return

    results = []
    for f in files:
        with open(f, 'r') as fh:
            results.append(json.load(fh))

    print(f"Loaded {len(results)} result file(s)")
    print()
    print("=" * 80)
    print("PRISM PAIRWISE — WIN RATE COMPARISON")
    print("=" * 80)
    print(f"{'Model':<45} {'Win Rate':>10} {'95% CI':>20} {'Users':>8} {'Judgments':>10}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x['win_rate'], reverse=True):
        ci_str = f"[{r['ci_lower']*100:.2f}%, {r['ci_upper']*100:.2f}%]"
        print(f"{r['model_label']:<45} {r['win_rate']*100:>8.2f}% {ci_str:>20} {r['n_users']:>8} {r['n_judgments']:>10}")

    print("-" * 80)
    print()
    print("Reference (SynthesizeMe paper, Llama-3.1-8B on PRISM):")
    print(f"  Default Judge:    52.80%")
    print(f"  Demographics:     54.06%")
    print(f"  Memory:           54.17%")
    print(f"  SynthesizeMe:     55.24%")
    print("=" * 80)


if __name__ == "__main__":
    main()