#!/usr/bin/env python3
"""
Aggregate PRISM original results into a cross-cultural matrix.

Usage:
    python aggregate_prism_original.py
    python aggregate_prism_original.py --results_dir prism_original_results
"""

import os
import json
import glob
import argparse
import pandas as pd


MODEL_SHORT = {
    'llama8b-english-wvq-morality': 'CultureLLM-US',
    'culturellm-france-8b-morality': 'CultureLLM-FR',
    'culturellm-brazil-8b-morality': 'CultureLLM-BR',
    'culturellm-italy-8b-morality': 'CultureLLM-IT',
    'culturellm-india-8b-morality': 'CultureLLM-IN',
    'Llama-3.1-8B-Instruct': 'Base-Instruct',
}

TARGET_MATCH = {
    'CultureLLM-US': ['United States', 'United Kingdom'],
    'CultureLLM-FR': ['France'],
    'CultureLLM-BR': ['Brazil'],
    'CultureLLM-IT': ['Italy'],
    'CultureLLM-IN': ['India'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="prism_original_results")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "prism_orig_*_seed*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No result files found in {args.results_dir}/")
        return

    results = []
    for f in files:
        with open(f, 'r') as fh:
            results.append(json.load(fh))

    print(f"Loaded {len(results)} result file(s)")

    # ── Overall comparison ──
    print("\n" + "=" * 90)
    print("PRISM ORIGINAL — OVERALL WIN RATE")
    print("=" * 90)
    print(f"{'Model':<40} {'Win Rate':>10} {'Pairs':>8} {'Ties':>6}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x['overall_accuracy'], reverse=True):
        short = MODEL_SHORT.get(r['model_label'], r['model_label'])
        print(f"{short:<40} {r['overall_accuracy']*100:>8.2f}% {r['n_pairs']:>8} {r['n_ties']:>6}")
    print("-" * 90)

    # ── Cross-cultural matrix ──
    # Collect all countries across all models
    all_countries = {}
    for r in results:
        for country, info in r.get('per_birth_country', {}).items():
            if info['n_users'] >= 2:
                all_countries[country] = max(all_countries.get(country, 0), info['n_users'])

    top_countries = sorted(all_countries.keys(), key=lambda c: all_countries[c], reverse=True)

    print("\n" + "=" * 130)
    print("CROSS-CULTURAL MATRIX — Model × User Birth Country")
    print("(★ = CultureLLM target match)")
    print("=" * 130)

    model_labels = [MODEL_SHORT.get(r['model_label'], r['model_label']) for r in results]
    model_labels_sorted = sorted(model_labels)

    header = f"{'Country':<25} {'Users':>6}"
    for m in model_labels_sorted:
        header += f"  {m:>15}"
    print(header)
    print("-" * 130)

    tsv_rows = []
    for country in top_countries:
        n_users = all_countries[country]
        row_str = f"{country:<25} {n_users:>6}"
        tsv_row = {'Country': country, 'Users': n_users}

        for r in sorted(results, key=lambda x: MODEL_SHORT.get(x['model_label'], x['model_label'])):
            short = MODEL_SHORT.get(r['model_label'], r['model_label'])
            info = r.get('per_birth_country', {}).get(country)

            if info:
                acc = info['accuracy']
                is_match = country in TARGET_MATCH.get(short, [])
                if is_match:
                    row_str += f"  {acc*100:>12.2f}%★"
                else:
                    row_str += f"  {acc*100:>13.2f}%"
                tsv_row[short] = f"{acc*100:.2f}"
            else:
                row_str += f"  {'N/A':>15}"
                tsv_row[short] = 'N/A'

        print(row_str)
        tsv_rows.append(tsv_row)

    # Overall row
    print("-" * 130)
    row_str = f"{'OVERALL':<25} {'':>6}"
    for r in sorted(results, key=lambda x: MODEL_SHORT.get(x['model_label'], x['model_label'])):
        short = MODEL_SHORT.get(r['model_label'], r['model_label'])
        row_str += f"  {r['overall_accuracy']*100:>13.2f}%"
    print(row_str)
    print("=" * 130)

    # ── Conversation type breakdown ──
    print("\n" + "=" * 130)
    print("CONVERSATION TYPE BREAKDOWN")
    print("=" * 130)
    header = f"{'Type':<25}"
    for m in model_labels_sorted:
        header += f"  {m:>15}"
    print(header)
    print("-" * 130)

    all_types = set()
    for r in results:
        all_types.update(r.get('per_conversation_type', {}).keys())

    for ctype in sorted(all_types):
        row_str = f"{ctype:<25}"
        for r in sorted(results, key=lambda x: MODEL_SHORT.get(x['model_label'], x['model_label'])):
            info = r.get('per_conversation_type', {}).get(ctype)
            if info:
                row_str += f"  {info['accuracy']*100:>13.2f}%"
            else:
                row_str += f"  {'N/A':>15}"
        print(row_str)
    print("=" * 130)

    # ── Save TSV ──
    tsv_df = pd.DataFrame(tsv_rows)
    tsv_path = os.path.join(args.results_dir, "prism_original_cross_cultural.tsv")
    tsv_df.to_csv(tsv_path, sep='\t', index=False)
    print(f"\nTSV saved to: {tsv_path}")


if __name__ == "__main__":
    main()