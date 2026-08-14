"""
CultureLLM — Stage 2: Score responses with LLM judge
=====================================================
Reads the responses CSV produced by generate.py and scores
each row using GPT-4o (Listing 7 entailment prompt).
No GPU required.

Run:
    python score.py --input outputs/culturebank/responses_<RUN_ID>.csv
Output:
    outputs/culturebank/results_<RUN_ID>.csv
    outputs/culturebank/summary_<RUN_ID>.json
"""

import os
import time
import json
import argparse
import pandas as pd
from datetime import datetime
from openai import OpenAI

# =============================================================
# CONFIG
# =============================================================

OPENAI_MODEL   = "gpt-4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

OUTPUT_DIR     = "outputs/culturebank"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================
# ENTAILMENT SCORING  (Listing 7 — exact paper prompts)
# =============================================================

JUDGE_SYSTEM = (
    "You are an AI assistant trained to classify whether a given response from a language "
    "agent entails a specific cultural knowledge. Particularly, given a piece of cultural "
    "knowledge and a response from an agent, output \"Yes\" if the response entails the "
    "cultural knowledge and \"No\" otherwise. Your output should be Yes/No only. Even if "
    "you are uncertain, you must pick either \"Yes\" or \"No\" without using any other words"
)

JUDGE_USER_TEMPLATE = (
    "Response: {model_response}\n\n"
    "Knowledge: {cultural_knowledge_description}\n\n"
    "Does the given response entail the provided knowledge? Entailment (Yes/No):"
)

def score_with_llm_judge(client: OpenAI, response: str, ground_truth: str, retries: int = 3) -> dict:
    result = {"entails": None, "judge_raw": None, "judge_error": None}

    user_msg = JUDGE_USER_TEMPLATE.format(
        model_response=str(response).strip() if pd.notna(response) else "",
        cultural_knowledge_description=str(ground_truth).strip() if pd.notna(ground_truth) else "",
    )

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = r.choices[0].message.content.strip()
            result["judge_raw"] = raw
            result["entails"]   = raw.lower().startswith("yes")
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                result["judge_error"] = str(e)

    return result

# =============================================================
# MAIN
# =============================================================

def run(input_path: str):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Export it before running score.py.")

    print(f"Loading responses from {input_path} …")
    df = pd.read_csv(input_path)
    print(f"      {len(df)} rows to score\n")

    # ── resume support: skip already-scored rows ──────────────
    checkpoint_path = input_path.replace(".csv", "_checkpoint.csv")

    if os.path.exists(checkpoint_path):
        done_df = pd.read_csv(checkpoint_path)
        n_done  = len(done_df)
        df_todo = df.iloc[n_done:].copy()
        print(f"      Resuming from checkpoint: {n_done} already scored, {len(df_todo)} remaining\n")
    else:
        done_df = pd.DataFrame()
        df_todo = df.copy()

    client = OpenAI(api_key=OPENAI_API_KEY)
    new_scores = []

    for i, (_, row) in enumerate(df_todo.iterrows()):
        score = score_with_llm_judge(
            client,
            response    = row["model_response"],
            ground_truth= row["eval_whole_desc"],
        )
        row_result = row.to_dict()
        row_result.update(score)
        new_scores.append(row_result)

        # save checkpoint every 50 rows
        if (i + 1) % 50 == 0:
            checkpoint_df = pd.concat(
                [done_df, pd.DataFrame(new_scores)], ignore_index=True
            )
            checkpoint_df.to_csv(checkpoint_path, index=False)

            so_far = [s["entails"] for s in new_scores if s["entails"] is not None]
            rate   = sum(so_far) / len(so_far) if so_far else 0
            total_done = len(done_df) + i + 1
            print(f"      Scored {total_done}/{len(df)} | running entailment rate: {rate:.2%}")

    # ── merge checkpoint + new scores ─────────────────────────
    final_df = pd.concat([done_df, pd.DataFrame(new_scores)], ignore_index=True)

    # ── save final results ────────────────────────────────────
    csv_path  = os.path.join(OUTPUT_DIR, f"results_{RUN_ID}.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved → {csv_path}")

    # clean up checkpoint now that we're done
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # ── summary ───────────────────────────────────────────────
    entails_col = final_df["entails"].dropna()

    summary = {
        "run_id"          : RUN_ID,
        "input_file"      : input_path,
        "judge_model"     : OPENAI_MODEL,
        "n_samples"       : len(final_df),
        "entailment_rate" : round(float(entails_col.mean()), 3) if len(entails_col) else None,
        "n_yes"           : int(entails_col.sum())        if len(entails_col) else 0,
        "n_no"            : int((~entails_col).sum())     if len(entails_col) else 0,
        "n_errors"        : int(final_df["judge_error"].notna().sum()),
    }

    json_path = os.path.join(OUTPUT_DIR, f"summary_{RUN_ID}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved  → {json_path}")

    print("\n" + "="*55)
    print("EVALUATION SUMMARY")
    print("="*55)
    print(f"  Samples scored  : {len(final_df)}")
    if summary["entailment_rate"] is not None:
        print(f"  Entailment rate : {summary['entailment_rate']:.2%}")
        print(f"  Yes / No        : {summary['n_yes']} / {summary['n_no']}")
        print(f"  Errors          : {summary['n_errors']}")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to responses CSV from generate.py")
    args = parser.parse_args()
    run(args.input)