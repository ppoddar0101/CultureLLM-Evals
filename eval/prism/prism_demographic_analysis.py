#!/usr/bin/env python3
"""
Demographic breakdown of PRISM results.

Joins per-user win rates (already computed) with PRISM survey demographics
to analyze: does CultureLLM-India do better on Indian users? etc.

No GPU needed — runs on login node in seconds.

Usage:
    python prism_demographic_analysis.py
    python prism_demographic_analysis.py --results_dir prism_results
"""

import os
import json
import glob
import argparse
import pandas as pd
from datasets import load_dataset


PRISM_SURVEY = "HannahRoseKirk/prism-alignment"

# Map CultureLLM models to their target countries
MODEL_TARGET_COUNTRIES = {
    'llama8b-english-wvq-morality': ['United States', 'United Kingdom', 'Canada', 'Australia'],
    'culturellm-france-8b-morality': ['France'],
    'culturellm-brazil-8b-morality': ['Brazil'],
    'culturellm-italy-8b-morality': ['Italy'],
    'culturellm-india-8b-morality': ['India'],
    'Llama-3.1-8B-Instruct': [],
}


def load_and_flatten_survey():
    """Load PRISM survey and flatten nested dict columns."""
    print("Loading PRISM survey demographics...")
    survey = load_dataset(PRISM_SURVEY, "survey", split="train")
    df = survey.to_pandas()
    print(f"  Raw survey entries: {len(df)}")

    # Flatten 'location' dict -> individual columns
    if 'location' in df.columns:
        location_df = pd.json_normalize(df['location'])
        # Prefix columns to be clear
        location_df.columns = ['loc_' + c for c in location_df.columns]
        df = pd.concat([df.drop(columns=['location']), location_df], axis=1)
        print(f"  Flattened location -> {list(location_df.columns)}")

    # Flatten 'religion' dict -> individual columns
    if 'religion' in df.columns:
        # Handle case where religion might be a string already
        if isinstance(df['religion'].iloc[0], dict):
            religion_df = pd.json_normalize(df['religion'])
            religion_df.columns = ['religion_' + c for c in religion_df.columns]
            df = pd.concat([df.drop(columns=['religion']), religion_df], axis=1)
            print(f"  Flattened religion -> {list(religion_df.columns)}")

    # Flatten 'ethnicity' dict
    if 'ethnicity' in df.columns:
        if isinstance(df['ethnicity'].iloc[0], dict):
            eth_df = pd.json_normalize(df['ethnicity'])
            eth_df.columns = ['ethnicity_' + c for c in eth_df.columns]
            df = pd.concat([df.drop(columns=['ethnicity']), eth_df], axis=1)
            print(f"  Flattened ethnicity -> {list(eth_df.columns)}")

    # Print what we have now
    print(f"\n  Key demographic columns:")
    for col in ['loc_reside_country', 'loc_birth_country', 'loc_birth_region',
                'loc_reside_region', 'loc_birth_subregion', 'loc_reside_subregion',
                'study_locale', 'religion_simplified', 'religion_categorised',
                'age', 'gender', 'ethnicity_simplified']:
        if col in df.columns:
            top = df[col].value_counts().head(5)
            print(f"    {col}: {dict(top)}")

    return df


def load_per_user_results(results_dir):
    """Load per-user win rates from all model result files."""
    pattern = os.path.join(results_dir, "prism_*_seed*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No result files found in {results_dir}/")
        return {}

    models = {}
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)

        model_label = data['model_label']
        per_user = data.get('per_user', {})

        rows = []
        for user_id, info in per_user.items():
            rows.append({
                'user_id': user_id,
                'win_rate': info['win_rate'],
                'n_test': info['n_test'],
                'n_scores': info['n_scores'],
            })

        models[model_label] = {
            'overall_win_rate': data['win_rate'],
            'df': pd.DataFrame(rows),
        }

    print(f"Loaded results for {len(models)} models: {list(models.keys())}")
    return models


def join_demographics(models, survey_df):
    """Join per-user results with flattened demographics."""
    survey_df['user_id'] = survey_df['user_id'].astype(str)

    # Check overlap
    sample_model = list(models.values())[0]
    model_ids = set(sample_model['df']['user_id'].astype(str))
    survey_ids = set(survey_df['user_id'])
    overlap = model_ids & survey_ids
    print(f"\n  User ID overlap: {len(overlap)}/{len(model_ids)} model users found in survey")

    if len(overlap) == 0:
        # Try without 'user' prefix
        print("  Trying ID format variants...")
        survey_df['user_id_alt'] = 'user' + survey_df['user_id'].astype(str)
        overlap_alt = model_ids & set(survey_df['user_id_alt'])
        if len(overlap_alt) > len(overlap):
            survey_df['user_id'] = survey_df['user_id_alt']
            overlap = overlap_alt
            print(f"  With 'user' prefix: {len(overlap)} matches")
        survey_df = survey_df.drop(columns=['user_id_alt'], errors='ignore')

    results = {}
    for model_label, model_data in models.items():
        user_df = model_data['df'].copy()
        user_df['user_id'] = user_df['user_id'].astype(str)
        merged = user_df.merge(survey_df, on='user_id', how='left')

        # Check match rate
        country_col = 'loc_reside_country' if 'loc_reside_country' in merged.columns else None
        if country_col:
            matched = merged[country_col].notna().sum()
            print(f"  {model_label}: matched {matched}/{len(user_df)} users")

        results[model_label] = merged

    return results


def print_country_matrix(results, models):
    """Print model × country win rate matrix."""
    country_col = 'loc_reside_country'
    if country_col not in list(results.values())[0].columns:
        print("No country column found after join.")
        return

    model_labels = sorted(results.keys())

    # Get countries with enough users
    all_countries = {}
    for m, df in results.items():
        for country, count in df[country_col].value_counts().items():
            if pd.notna(country) and count >= 3:
                all_countries[country] = max(all_countries.get(country, 0), count)

    top_countries = sorted(all_countries.keys(), key=lambda c: all_countries[c], reverse=True)

    print("\n" + "=" * 130)
    print("CROSS-CULTURAL WIN RATE MATRIX — Model × User's Residing Country")
    print("(★ = CultureLLM model's target culture)")
    print("=" * 130)

    # Build short model names for header
    short_names = {}
    for m in model_labels:
        if 'english' in m:
            short_names[m] = 'CultureLLM-US'
        elif 'france' in m:
            short_names[m] = 'CultureLLM-FR'
        elif 'brazil' in m:
            short_names[m] = 'CultureLLM-BR'
        elif 'italy' in m:
            short_names[m] = 'CultureLLM-IT'
        elif 'india' in m:
            short_names[m] = 'CultureLLM-IN'
        else:
            short_names[m] = 'Base-Instruct'

    header = f"{'Country':<25} {'n':>5}"
    for m in model_labels:
        header += f"  {short_names[m]:>15}"
    print(header)
    print("-" * 130)

    tsv_rows = []

    for country in top_countries:
        n = all_countries[country]
        row_str = f"{country:<25} {n:>5}"
        row_data = {'Country': country, 'n': n}

        for m in model_labels:
            df = results[m]
            users = df[df[country_col] == country]

            if len(users) >= 3:
                acc = users['win_rate'].mean()
                targets = MODEL_TARGET_COUNTRIES.get(m, [])
                is_match = any(t.lower() == country.lower() for t in targets)

                if is_match:
                    row_str += f"  {acc*100:>12.2f}%★"
                else:
                    row_str += f"  {acc*100:>13.2f}%"
                row_data[short_names[m]] = f"{acc*100:.2f}"
            else:
                row_str += f"  {'N/A':>15}"
                row_data[short_names[m]] = "N/A"

        print(row_str)
        tsv_rows.append(row_data)

    print("-" * 130)

    # Overall
    row_str = f"{'OVERALL':<25} {'':>5}"
    for m in model_labels:
        wr = models[m]['overall_win_rate']
        row_str += f"  {wr*100:>13.2f}%"
    print(row_str)
    print("=" * 130)

    return tsv_rows


def print_region_matrix(results, models):
    """Print model × region win rate matrix (broader grouping)."""
    region_col = 'loc_reside_subregion'
    if region_col not in list(results.values())[0].columns:
        region_col = 'loc_birth_region'
    if region_col not in list(results.values())[0].columns:
        return

    model_labels = sorted(results.keys())

    short_names = {}
    for m in model_labels:
        if 'english' in m: short_names[m] = 'CultureLLM-US'
        elif 'france' in m: short_names[m] = 'CultureLLM-FR'
        elif 'brazil' in m: short_names[m] = 'CultureLLM-BR'
        elif 'italy' in m: short_names[m] = 'CultureLLM-IT'
        elif 'india' in m: short_names[m] = 'CultureLLM-IN'
        else: short_names[m] = 'Base-Instruct'

    all_regions = {}
    for m, df in results.items():
        for region, count in df[region_col].value_counts().items():
            if pd.notna(region) and count >= 5:
                all_regions[region] = max(all_regions.get(region, 0), count)

    print("\n" + "=" * 130)
    print(f"REGIONAL BREAKDOWN ({region_col})")
    print("=" * 130)

    header = f"{'Region':<30} {'n':>5}"
    for m in model_labels:
        header += f"  {short_names[m]:>15}"
    print(header)
    print("-" * 130)

    for region in sorted(all_regions.keys(), key=lambda r: all_regions[r], reverse=True):
        n = all_regions[region]
        row_str = f"{region:<30} {n:>5}"
        for m in model_labels:
            df = results[m]
            users = df[df[region_col] == region]
            if len(users) >= 3:
                acc = users['win_rate'].mean()
                row_str += f"  {acc*100:>13.2f}%"
            else:
                row_str += f"  {'N/A':>15}"
        print(row_str)

    print("=" * 130)


def print_religion_matrix(results, models):
    """Print model × religion win rate matrix."""
    religion_col = 'religion_simplified'
    if religion_col not in list(results.values())[0].columns:
        religion_col = 'religion_categorised'
    if religion_col not in list(results.values())[0].columns:
        print("No religion column found.")
        return

    model_labels = sorted(results.keys())

    short_names = {}
    for m in model_labels:
        if 'english' in m: short_names[m] = 'CultureLLM-US'
        elif 'france' in m: short_names[m] = 'CultureLLM-FR'
        elif 'brazil' in m: short_names[m] = 'CultureLLM-BR'
        elif 'italy' in m: short_names[m] = 'CultureLLM-IT'
        elif 'india' in m: short_names[m] = 'CultureLLM-IN'
        else: short_names[m] = 'Base-Instruct'

    all_religions = {}
    for m, df in results.items():
        for r, count in df[religion_col].value_counts().items():
            if pd.notna(r) and count >= 5:
                all_religions[r] = max(all_religions.get(r, 0), count)

    print("\n" + "=" * 130)
    print(f"RELIGION BREAKDOWN ({religion_col})")
    print("=" * 130)

    header = f"{'Religion':<30} {'n':>5}"
    for m in model_labels:
        header += f"  {short_names[m]:>15}"
    print(header)
    print("-" * 130)

    for religion in sorted(all_religions.keys(), key=lambda r: all_religions[r], reverse=True):
        n = all_religions[religion]
        row_str = f"{religion:<30} {n:>5}"
        for m in model_labels:
            df = results[m]
            users = df[df[religion_col] == religion]
            if len(users) >= 3:
                acc = users['win_rate'].mean()
                row_str += f"  {acc*100:>13.2f}%"
            else:
                row_str += f"  {'N/A':>15}"
        print(row_str)

    print("=" * 130)


def save_tsv(tsv_rows, results_dir):
    """Save cross-cultural matrix as TSV for Google Sheets."""
    if not tsv_rows:
        return
    out_df = pd.DataFrame(tsv_rows)
    tsv_path = os.path.join(results_dir, "prism_cross_cultural.tsv")
    out_df.to_csv(tsv_path, sep='\t', index=False)
    print(f"\nTSV saved to: {tsv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="prism_results")
    args = parser.parse_args()

    models = load_per_user_results(args.results_dir)
    if not models:
        return

    survey_df = load_and_flatten_survey()
    results = join_demographics(models, survey_df)

    tsv_rows = print_country_matrix(results, models)
    print_region_matrix(results, models)
    print_religion_matrix(results, models)
    save_tsv(tsv_rows, args.results_dir)


if __name__ == "__main__":
    main()