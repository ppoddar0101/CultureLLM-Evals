## SimBench: Double-blind compliant code — with per-dataset breakdown

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def calc_total_variation(human_dist, model_dist):
    """Calculate Total Variation distance between two distributions."""
    if not isinstance(human_dist, (list, np.ndarray)) or not isinstance(model_dist, (list, np.ndarray)):
        return np.nan
    
    p = np.array(human_dist, dtype=float)
    q = np.array(model_dist, dtype=float)
    
    if len(p) != len(q) or len(p) == 0:
        return np.nan
    
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    
    return 0.5 * np.sum(np.abs(p - q))


def uniform_distribution(n):
    """Create uniform distribution with n options."""
    if n <= 0:
        return []
    return [1.0 / n] * n


def process_results_file(file_path):
    """Process a single results pickle file."""
    try:
        df = pd.read_pickle(file_path)
        print(f"Loaded {len(df)} results from {Path(file_path).name}")
        
        if 'human_answer' in df.columns:
            df['Human_Distribution'] = df['human_answer'].apply(
                lambda x: list(x.values()) if isinstance(x, dict) else []
            )
        
        df['Total_Variation'] = df.apply(
            lambda row: calc_total_variation(
                row['Human_Distribution'], 
                row['Response_Distribution']
            ), axis=1
        )
        
        df['TV_Uniform'] = df.apply(
            lambda row: calc_total_variation(
                row['Human_Distribution'],
                uniform_distribution(len(row['Human_Distribution']))
            ) if len(row['Human_Distribution']) > 1 else np.nan, axis=1
        )
        
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_simbench_score.py <results_folder>")
        print("\nExample: python calculate_simbench_score.py ./results/")
        sys.exit(1)
    
    results_folder = sys.argv[1]
    
    if not os.path.exists(results_folder):
        print(f"Error: Results folder '{results_folder}' does not exist.")
        sys.exit(1)
    
    pickle_files = glob.glob(os.path.join(results_folder, "*.pkl"))
    
    if not pickle_files:
        print(f"No .pkl files found in {results_folder}")
        sys.exit(1)
    
    print(f"Found {len(pickle_files)} result files")
    print("=" * 60)
    
    all_results = []
    
    for file_path in pickle_files:
        df = process_results_file(file_path)
        if df is not None:
            all_results.append(df)
    
    if not all_results:
        print("No valid results to process.")
        sys.exit(1)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    if 'dataset_name' not in combined_df.columns:
        print("ERROR: 'dataset_name' column not found in results!")
        print("Available columns:", list(combined_df.columns))
        sys.exit(1)
    
    # Calculate dataset-specific uniform baselines
    print("Calculating dataset-specific uniform baselines...")
    dataset_norms = combined_df.groupby('dataset_name')['TV_Uniform'].mean()
    print("Dataset norms:")
    for dataset, norm in dataset_norms.items():
        print(f"  {dataset}: {norm:.4f}")
    
    # Calculate SimBench scores
    combined_df['SimBench_Score'] = combined_df.apply(
        lambda row: 100 * (1 - (row['Total_Variation'] / dataset_norms[row['dataset_name']])) 
        if not pd.isna(row['Total_Variation']) and row['dataset_name'] in dataset_norms and dataset_norms[row['dataset_name']] > 0
        else np.nan, axis=1
    )
    
    # ── OVERALL SCORES ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL SIMBENCH SCORES (Higher is Better)")
    print("=" * 80)
    
    model_scores = combined_df.groupby('Model')['SimBench_Score'].agg(['mean', 'std', 'count']).round(2)
    model_scores.columns = ['SimBench_Score', 'Std', 'Num_Questions']
    model_scores = model_scores.sort_values('SimBench_Score', ascending=False)
    
    print(f"{'Model':<40} {'Score':<12} {'Std':<10} {'N':<6}")
    print("-" * 68)
    for model, row in model_scores.iterrows():
        print(f"{model:<40} {row['SimBench_Score']:<12.2f} {row['Std']:<10.2f} {int(row['Num_Questions']):<6}")
    
    # ── PER-DATASET BREAKDOWN ────────────────────────────────────────
    models = sorted(combined_df['Model'].unique())
    datasets = sorted(combined_df['dataset_name'].unique())
    
    print("\n" + "=" * 80)
    print("PER-DATASET BREAKDOWN")
    print("=" * 80)
    
    # Build per-dataset table
    per_dataset_rows = []
    
    for ds in datasets:
        ds_data = combined_df[combined_df['dataset_name'] == ds]
        row_data = {'Dataset': ds}
        
        for model in models:
            model_ds = ds_data[ds_data['Model'] == model]
            if len(model_ds) > 0:
                row_data[f'{model}_score'] = model_ds['SimBench_Score'].mean()
                row_data[f'{model}_n'] = len(model_ds)
                row_data[f'{model}_tv'] = model_ds['Total_Variation'].mean()
            else:
                row_data[f'{model}_score'] = np.nan
                row_data[f'{model}_n'] = 0
                row_data[f'{model}_tv'] = np.nan
        
        per_dataset_rows.append(row_data)
    
    # Print per-dataset table for each model pair
    if len(models) == 2:
        m1, m2 = models[0], models[1]
        print(f"\n  {'Dataset':<25} {m1[:20]:<22} {m2[:20]:<22} {'Δ':>8}  {'N':>4}")
        print(f"  {'-'*85}")
        
        for row_data in per_dataset_rows:
            ds = row_data['Dataset']
            s1 = row_data.get(f'{m1}_score', np.nan)
            s2 = row_data.get(f'{m2}_score', np.nan)
            n = row_data.get(f'{m1}_n', 0)
            delta = s1 - s2 if not (np.isnan(s1) or np.isnan(s2)) else np.nan
            
            s1_str = f"{s1:.2f}" if not np.isnan(s1) else "N/A"
            s2_str = f"{s2:.2f}" if not np.isnan(s2) else "N/A"
            d_str = f"{delta:+.2f}" if not np.isnan(delta) else "N/A"
            
            print(f"  {ds:<25} {s1_str:<22} {s2_str:<22} {d_str:>8}  {n:>4}")
    else:
        # Generic: one column per model
        header = f"  {'Dataset':<25}"
        for m in models:
            header += f" {m[:18]:<20}"
        header += f" {'N':>4}"
        print(header)
        print(f"  {'-' * (25 + 20*len(models) + 6)}")
        
        for row_data in per_dataset_rows:
            line = f"  {row_data['Dataset']:<25}"
            for m in models:
                s = row_data.get(f'{m}_score', np.nan)
                line += f" {s:<20.2f}" if not np.isnan(s) else f" {'N/A':<20}"
            n = row_data.get(f'{models[0]}_n', 0)
            line += f" {n:>4}"
            print(line)
    
    # ── MEAN TV DISTANCES ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("MEAN TOTAL VARIATION DISTANCES (Lower is Better)")
    print(f"{'='*80}")
    
    if len(models) == 2:
        m1, m2 = models[0], models[1]
        print(f"\n  {'Dataset':<25} {m1[:20]:<22} {m2[:20]:<22} {'Uniform':>10}")
        print(f"  {'-'*80}")
        
        for row_data in per_dataset_rows:
            ds = row_data['Dataset']
            tv1 = row_data.get(f'{m1}_tv', np.nan)
            tv2 = row_data.get(f'{m2}_tv', np.nan)
            unif = dataset_norms.get(ds, np.nan)
            
            tv1_str = f"{tv1:.4f}" if not np.isnan(tv1) else "N/A"
            tv2_str = f"{tv2:.4f}" if not np.isnan(tv2) else "N/A"
            unif_str = f"{unif:.4f}" if not np.isnan(unif) else "N/A"
            
            print(f"  {ds:<25} {tv1_str:<22} {tv2_str:<22} {unif_str:>10}")
    
    # ── SAVE RESULTS ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions processed: {len(combined_df)}")
    print(f"Models evaluated: {len(model_scores)}")
    print(f"Datasets covered: {len(datasets)}")
    print(f"Best performing model: {model_scores.index[0]} ({model_scores.iloc[0]['SimBench_Score']:.2f})")
    
    # Save overall scores
    output_file = os.path.join(results_folder, "simbench_scores.csv")
    model_scores.to_csv(output_file)
    
    # Save per-dataset breakdown
    per_dataset_df = pd.DataFrame(per_dataset_rows)
    per_dataset_file = os.path.join(results_folder, "simbench_per_dataset.csv")
    per_dataset_df.to_csv(per_dataset_file, index=False)
    
    # Save full detailed results
    detail_file = os.path.join(results_folder, "simbench_detailed.csv")
    combined_df[['Model', 'dataset_name', 'SimBench_Score', 'Total_Variation', 'TV_Uniform']].to_csv(
        detail_file, index=False
    )
    
    print(f"\nResults saved:")
    print(f"  Overall scores:        {output_file}")
    print(f"  Per-dataset breakdown: {per_dataset_file}")
    print(f"  Detailed (per-row):    {detail_file}")


if __name__ == "__main__":
    main()