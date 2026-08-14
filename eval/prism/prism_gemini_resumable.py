#!/usr/bin/env python3
"""
prism_gemini_resumable.py  (multi-key + async edition)

PRISM Original Pairwise Evaluation using Gemini 2.5 Flash.

Features:
- Multiple API keys: rotate across keys to multiply rate limits
- Async concurrency with semaphore
- Per-result JSONL checkpoint — never redo completed work
- Resumes from previous checkpoint on rerun
- Continues past individual failures

Usage:
    # Single key (any of these):
    export GEMINI_API_KEY="key1"
    python prism_gemini_resumable.py --model gemini-2.5-flash

    # Multiple keys (comma-separated):
    export GEMINI_API_KEYS="key1,key2,key3"
    python prism_gemini_resumable.py --model gemini-2.5-flash --concurrency 30

    # Or pass directly:
    python prism_gemini_resumable.py --api_keys "key1,key2,key3" --concurrency 30
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Literal, Optional, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from pydantic import BaseModel, Field
from tqdm import tqdm
from google import genai
from google.genai import types


TARGET_COUNTRIES = ["United States", "Brazil", "India", "France", "Italy"]


SYSTEM_PROMPT = """Your task is to judge which completion a human is more likely to prefer.

You will be given:
1. user_profile
2. conversation
3. first_completion
4. second_completion

Return only the preference label.
Do not provide reasoning.
"""

USER_TEMPLATE = """[[ ## user_profile ## ]]
{user_profile}

[[ ## conversation ## ]]
{conversation}

[[ ## first_completion ## ]]
{first_completion}

[[ ## second_completion ## ]]
{second_completion}

Return only a JSON object with one field:
{{
  "preference": "First" or "Second"
}}
"""


class PairwiseDecision(BaseModel):
    preference: Literal["First", "Second"] = Field(description="Choose First or Second only.")


# ───────────────────────── API Key handling ─────────────────────────

def resolve_api_keys(api_keys_override: Optional[str] = None) -> List[str]:
    """Resolve one or more API keys from args or environment.

    Priority:
      1. --api_keys "k1,k2,k3"
      2. GEMINI_API_KEYS="k1,k2,k3"
      3. GOOGLE_API_KEY (single)
      4. GEMINI_API_KEY (single)
    """
    raw = None

    if api_keys_override:
        raw = api_keys_override
    elif os.environ.get("GEMINI_API_KEYS"):
        raw = os.environ["GEMINI_API_KEYS"]
    elif os.environ.get("GOOGLE_API_KEY"):
        raw = os.environ["GOOGLE_API_KEY"]
    elif os.environ.get("GEMINI_API_KEY"):
        raw = os.environ["GEMINI_API_KEY"]

    if not raw:
        raise EnvironmentError(
            "No API key found. Set GEMINI_API_KEYS (comma-separated), "
            "GOOGLE_API_KEY, GEMINI_API_KEY, or pass --api_keys."
        )

    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise EnvironmentError("No valid API keys after parsing.")

    return keys


def create_client_pool(api_keys: List[str]) -> List[genai.Client]:
    """Create one genai.Client per API key."""
    clients = []
    for key in api_keys:
        clients.append(genai.Client(api_key=key))
    return clients


# ───────────────────────── Data loading ─────────────────────────

def load_and_build_pairs(min_score_diff=5):
    print("Loading original PRISM data...")

    utts = load_dataset("HannahRoseKirk/prism-alignment", "utterances", split="train")
    utts_df = utts.to_pandas()
    print(f"  Utterances: {len(utts_df)}")

    survey = load_dataset("HannahRoseKirk/prism-alignment", "survey", split="train")
    survey_df = survey.to_pandas()

    if "location" in survey_df.columns and isinstance(survey_df["location"].iloc[0], dict):
        loc_df = pd.json_normalize(survey_df["location"])
        loc_df.columns = ["loc_" + c for c in loc_df.columns]
        survey_df = pd.concat([survey_df.drop(columns=["location"]), loc_df], axis=1)

    if "religion" in survey_df.columns and isinstance(survey_df["religion"].iloc[0], dict):
        rel_df = pd.json_normalize(survey_df["religion"])
        rel_df.columns = ["religion_" + c for c in rel_df.columns]
        survey_df = pd.concat([survey_df.drop(columns=["religion"]), rel_df], axis=1)

    if "ethnicity" in survey_df.columns and isinstance(survey_df["ethnicity"].iloc[0], dict):
        eth_df = pd.json_normalize(survey_df["ethnicity"])
        eth_df.columns = ["ethnicity_" + c for c in eth_df.columns]
        survey_df = pd.concat([survey_df.drop(columns=["ethnicity"]), eth_df], axis=1)

    target_users = survey_df[survey_df["loc_birth_country"].isin(TARGET_COUNTRIES)]
    target_user_ids = set(target_users["user_id"].tolist())
    print(f"  Users from target countries: {len(target_user_ids)}")
    for c in TARGET_COUNTRIES:
        n = len(target_users[target_users["loc_birth_country"] == c])
        print(f"    {c:<25} {n} users")

    target_utts = utts_df[utts_df["user_id"].isin(target_user_ids)]
    print(f"  Target utterances: {len(target_utts)}")

    pairs = []
    for interaction_id, group in target_utts.groupby("interaction_id"):
        if len(group) < 2:
            continue

        chosen_rows = group[group["if_chosen"] == True]
        rejected_rows = group[group["if_chosen"] == False]
        if len(chosen_rows) == 0 or len(rejected_rows) == 0:
            continue

        chosen = chosen_rows.iloc[0]
        rejected = rejected_rows.iloc[0]
        score_diff = chosen["score"] - rejected["score"]
        if score_diff < min_score_diff:
            continue

        pairs.append({
            "interaction_id": interaction_id,
            "user_id": chosen["user_id"],
            "user_prompt": chosen["user_prompt"],
            "chosen_response": chosen["model_response"],
            "rejected_response": rejected["model_response"],
            "score_diff": int(score_diff),
        })

    pairs_df = pd.DataFrame(pairs)

    demo_cols = [
        "user_id", "loc_birth_country", "loc_reside_country",
        "age", "gender", "employment_status", "education",
        "marital_status", "english_proficiency", "self_description",
    ]
    for col in ["religion_simplified", "ethnicity_simplified"]:
        if col in survey_df.columns:
            demo_cols.append(col)
    demo_cols = [c for c in demo_cols if c in survey_df.columns]

    pairs_df = pairs_df.merge(survey_df[demo_cols], on="user_id", how="left")

    print(f"  Total pairs (target countries, score_diff >= {min_score_diff}): {len(pairs_df)}")
    for c in TARGET_COUNTRIES:
        n = len(pairs_df[pairs_df["loc_birth_country"] == c])
        print(f"    {c:<25} {n} pairs")

    return pairs_df


def format_user_profile(row):
    field_map = [
        ("age", "Age"),
        ("gender", "Gender"),
        ("loc_birth_country", "Birth Country"),
        ("loc_reside_country", "Residing Country"),
        ("education", "Education"),
        ("employment_status", "Employment"),
        ("marital_status", "Marital Status"),
        ("english_proficiency", "English Proficiency"),
        ("religion_simplified", "Religion"),
        ("ethnicity_simplified", "Ethnicity"),
    ]

    parts = []
    for field, label in field_map:
        val = row.get(field, None)
        if pd.notna(val) and str(val).strip() and str(val) != "Prefer not to say":
            parts.append(f"{label}: {val}")

    self_desc = row.get("self_description", None)
    if pd.notna(self_desc) and str(self_desc).strip():
        parts.append(f"Self-description: {self_desc}")

    return "\n".join(parts) if parts else "No profile available"


def build_prompts(pairs_df, seed=42):
    random.seed(seed)
    prompts = []

    for _, row in pairs_df.iterrows():
        conversation = f"[user]: {row['user_prompt']}"
        user_profile = format_user_profile(row)

        for ordering in ["normal", "flipped"]:
            if ordering == "normal":
                first, second, chosen_label = row["chosen_response"], row["rejected_response"], "First"
            else:
                first, second, chosen_label = row["rejected_response"], row["chosen_response"], "Second"

            user_content = USER_TEMPLATE.format(
                user_profile=user_profile,
                conversation=conversation,
                first_completion=first,
                second_completion=second,
            )

            prompts.append({
                "prompt_id": f"{row['interaction_id']}::{ordering}",
                "interaction_id": row["interaction_id"],
                "user_id": row["user_id"],
                "birth_country": row.get("loc_birth_country", "Unknown"),
                "ordering": ordering,
                "chosen_label": chosen_label,
                "full_prompt": f"{SYSTEM_PROMPT}\n\n{user_content}",
            })

    return prompts


# ───────────────────────── Parsing helpers ─────────────────────────

def parse_preference(text):
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            pref = str(parsed.get("preference", "")).strip()
            if pref.lower().startswith("first"):
                return "First"
            if pref.lower().startswith("second"):
                return "Second"
    except Exception:
        pass

    low = text.lower()
    if "first" in low and "second" not in low:
        return "First"
    if "second" in low and "first" not in low:
        return "Second"

    lf, ls = low.rfind("first"), low.rfind("second")
    if lf > ls:
        return "First"
    if ls > lf:
        return "Second"
    return None


def extract_retry_after_seconds(error_text):
    text = error_text.lower()

    m = re.search(r"retrydelay['\"]?:\s*['\"]?(\d+)s", text)
    if m:
        return int(m.group(1))

    m = re.search(r"retry in (\d+)h(\d+)m([\d.]+)s", text)
    if m:
        return int(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)))

    m = re.search(r"retry in (\d+)m([\d.]+)s", text)
    if m:
        return int(int(m.group(1)) * 60 + float(m.group(2)))

    m = re.search(r"retry in (\d+)s", text)
    if m:
        return int(m.group(1))

    return None


def is_retryable_error_text(error_text):
    t = error_text.lower()
    markers = [
        "429", "resource_exhausted", "quota", "rate limit", "rate-limit",
        "503", "unavailable", "temporarily overloaded", "service unavailable",
        "internal", "deadline exceeded", "connection", "timeout",
    ]
    return any(m in t for m in markers)


# ───────────────────────── Checkpoint I/O ─────────────────────────

def load_jsonl_records(path):
    records = []
    if not path.exists():
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def load_existing_results(final_path, checkpoint_path):
    records_by_id = {}

    for r in load_jsonl_records(checkpoint_path):
        pid = r.get("prompt_id")
        if pid:
            records_by_id[pid] = r

    if final_path.exists():
        try:
            with open(final_path, "r") as f:
                data = json.load(f)
            for r in data.get("results", []):
                pid = r.get("prompt_id")
                if pid:
                    records_by_id[pid] = r
        except Exception:
            pass

    return records_by_id


# ───────────────────────── Async Gemini calls with key rotation ─────────────────────────

class QuotaWaitRequired(Exception):
    def __init__(self, wait_seconds: int, message: str):
        super().__init__(message)
        self.wait_seconds = wait_seconds


class ClientPool:
    """Round-robin pool of Gemini clients with per-key cooldown on rate limits."""

    def __init__(self, clients: List[genai.Client]):
        self.clients = clients
        self.n = len(clients)
        self._counter = 0
        self._lock = asyncio.Lock()
        # Track cooldown expiry per client index
        self._cooldown_until: dict[int, float] = {}

    async def get_client(self) -> tuple[genai.Client, int]:
        """Get next available client, skipping those in cooldown."""
        async with self._lock:
            now = time.time()
            # Try up to N clients to find one not in cooldown
            for _ in range(self.n):
                idx = self._counter % self.n
                self._counter += 1
                cooldown_end = self._cooldown_until.get(idx, 0)
                if now >= cooldown_end:
                    return self.clients[idx], idx

            # All in cooldown — return the one with shortest remaining wait
            soonest_idx = min(self._cooldown_until, key=self._cooldown_until.get)
            wait_s = self._cooldown_until[soonest_idx] - now
            if wait_s > 0:
                return self.clients[soonest_idx], soonest_idx
            return self.clients[soonest_idx], soonest_idx

    def set_cooldown(self, client_idx: int, seconds: float):
        """Mark a client as rate-limited for a duration."""
        self._cooldown_until[client_idx] = time.time() + seconds

    def clear_cooldown(self, client_idx: int):
        self._cooldown_until.pop(client_idx, None)


async def call_gemini_one(
    pool: ClientPool,
    model_name: str,
    prompt: str,
    max_retries: int = 6,
    base_delay_s: float = 5.0,
    max_delay_s: float = 300.0,
    stop_on_long_quota_wait_s: int = 300,
):
    """Single async Gemini call with retries and key rotation on rate limits."""
    delay_s = base_delay_s

    for attempt in range(max_retries):
        client, client_idx = await pool.get_client()

        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                ),
            )

            # Success — clear any cooldown on this key
            pool.clear_cooldown(client_idx)

            raw_text = getattr(response, "text", "") or ""
            if not raw_text.strip():
                raw_text = str(response)

            try:
                parsed = PairwiseDecision.model_validate_json(raw_text)
                return parsed.preference, raw_text, True, None
            except Exception:
                pref = parse_preference(raw_text)
                if pref in {"First", "Second"}:
                    return pref, raw_text, True, None
                raise ValueError(f"Could not parse: {raw_text[:300]}")

        except Exception as e:
            err_text = str(e)
            retry_after = extract_retry_after_seconds(err_text)

            # If this is a rate limit, cool down this specific key
            if is_retryable_error_text(err_text) and any(
                m in err_text.lower() for m in ["429", "resource_exhausted", "quota", "rate limit"]
            ):
                cooldown = retry_after if retry_after else delay_s * 2
                pool.set_cooldown(client_idx, cooldown)

            if retry_after is not None and retry_after > stop_on_long_quota_wait_s:
                raise QuotaWaitRequired(retry_after, err_text)

            if not is_retryable_error_text(err_text) and "could not parse" not in err_text.lower():
                return None, "", False, err_text

            if attempt == max_retries - 1:
                return None, "", False, err_text

            if retry_after is not None:
                sleep_s = min(float(retry_after), max_delay_s)
            else:
                sleep_s = min(delay_s * (1.0 + random.random() * 0.25), max_delay_s)

            await asyncio.sleep(sleep_s)

            if retry_after is None:
                delay_s = min(delay_s * 2.0, max_delay_s)

    return None, "", False, "max retries exhausted"


async def evaluate_pairwise_async(
    pool: ClientPool,
    model_name: str,
    prompts: list,
    checkpoint_path: Path,
    existing_results_by_id: dict,
    concurrency: int = 10,
    max_retries: int = 6,
    base_delay_s: float = 5.0,
    max_delay_s: float = 300.0,
    stop_on_long_quota_wait_s: int = 300,
):
    results_by_id = dict(existing_results_by_id)
    ok_existing = {
        pid for pid, rec in results_by_id.items()
        if rec.get("status") == "ok"
    }

    pending = [p for p in prompts if p["prompt_id"] not in ok_existing]
    print(f"  Already done: {len(ok_existing)},  pending: {len(pending)}")

    if not pending:
        ordered = [results_by_id[p["prompt_id"]] for p in prompts if p["prompt_id"] in results_by_id]
        return ordered, True

    semaphore = asyncio.Semaphore(concurrency)
    checkpoint_lock = asyncio.Lock()
    pbar = tqdm(total=len(pending), desc="Gemini calls", initial=0)
    completed_flag = True

    async def write_checkpoint(record):
        async with checkpoint_lock:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    async def process_one(prompt_data):
        nonlocal completed_flag

        async with semaphore:
            try:
                preference, raw_text, ok, err_text = await call_gemini_one(
                    pool=pool,
                    model_name=model_name,
                    prompt=prompt_data["full_prompt"],
                    max_retries=max_retries,
                    base_delay_s=base_delay_s,
                    max_delay_s=max_delay_s,
                    stop_on_long_quota_wait_s=stop_on_long_quota_wait_s,
                )
            except QuotaWaitRequired as e:
                print(f"\n  Quota wait too long ({e.wait_seconds}s) — marking failed, continuing others.")
                preference, raw_text, ok, err_text = None, "", False, str(e)
            except Exception as e:
                preference, raw_text, ok, err_text = None, "", False, str(e)

            if ok and preference in {"First", "Second"}:
                correct = (
                    (preference == "First" and prompt_data["chosen_label"] == "First")
                    or (preference == "Second" and prompt_data["chosen_label"] == "Second")
                )
                status = "ok"
            else:
                correct = False
                status = "failed"
                if not err_text:
                    err_text = "Unparseable or empty response"

            record = {
                "prompt_id": prompt_data["prompt_id"],
                "user_id": prompt_data["user_id"],
                "birth_country": prompt_data["birth_country"],
                "ordering": prompt_data["ordering"],
                "chosen_label": prompt_data["chosen_label"],
                "status": status,
                "correct": bool(correct),
                "preference": preference,
                "raw_text": raw_text,
                "error": err_text,
            }

            results_by_id[prompt_data["prompt_id"]] = record
            await write_checkpoint(record)
            pbar.update(1)

    tasks = [asyncio.create_task(process_one(p)) for p in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()

    ordered = [results_by_id[p["prompt_id"]] for p in prompts if p["prompt_id"] in results_by_id]
    return ordered, completed_flag


# ───────────────────────── Summary ─────────────────────────

def summarize_results(results, args):
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        return {
            "model_label": args.model,
            "seed": args.seed,
            "overall_accuracy": None,
            "n_judgments": 0,
            "n_failed": 0,
            "n_ties": 0,
            "per_birth_country": {},
            "results": [],
        }

    judged = results_df[results_df["status"] == "ok"].copy()
    failed = results_df[results_df["status"] != "ok"].copy()

    overall_acc = float(judged["correct"].mean()) if len(judged) > 0 else None
    n_ties = int((judged["preference"] == "Tie").sum()) if len(judged) > 0 else 0

    country_metrics = {}
    for country in TARGET_COUNTRIES:
        group = judged[judged["birth_country"] == country]
        if len(group) > 0:
            country_metrics[country] = {
                "accuracy": float(group["correct"].mean()),
                "n_judgments": int(len(group)),
                "n_users": int(group["user_id"].nunique()),
            }

    return {
        "model_label": args.model,
        "seed": args.seed,
        "split": "prism_original_pairwise",
        "min_score_diff": args.min_score_diff,
        "overall_accuracy": overall_acc,
        "n_judgments": int(len(judged)),
        "n_failed": int(len(failed)),
        "n_ties": int(n_ties),
        "per_birth_country": country_metrics,
        "results": results,
    }


# ───────────────────────── Main ─────────────────────────

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="prism_original_results")
    parser.add_argument("--min_score_diff", type=int, default=5)
    parser.add_argument("--api_keys", type=str, default=None,
                        help="Comma-separated API keys (or set GEMINI_API_KEYS env var)")

    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max parallel Gemini API calls (total across all keys)")
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--base_delay_s", type=float, default=5.0)
    parser.add_argument("--max_delay_s", type=float, default=300.0)
    parser.add_argument("--stop_on_long_quota_wait_s", type=int, default=300)

    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0-based). Use with --num_shards.")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards to split work across.")
    args = parser.parse_args()

    if (args.shard is None) != (args.num_shards is None):
        parser.error("--shard and --num_shards must be used together.")
    if args.shard is not None and args.shard >= args.num_shards:
        parser.error(f"--shard {args.shard} must be < --num_shards {args.num_shards}.")

    api_keys = resolve_api_keys(args.api_keys)
    clients = create_client_pool(api_keys)
    pool = ClientPool(clients)

    print(f"Using model: {args.model}")
    print(f"API keys:    {len(api_keys)}")
    print(f"Concurrency: {args.concurrency}")

    pairs_df = load_and_build_pairs(min_score_diff=args.min_score_diff)
    prompts = build_prompts(pairs_df, seed=args.seed)
    print(f"\nTotal prompts (all shards): {len(prompts)}")

    # Shard filtering
    if args.num_shards is not None:
        prompts = [p for i, p in enumerate(prompts) if i % args.num_shards == args.shard]
        print(f"Shard {args.shard}/{args.num_shards}: {len(prompts)} prompts")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")

    if args.num_shards is not None:
        file_base = f"prism_orig_{safe_model}_seed{args.seed}_shard{args.shard}of{args.num_shards}"
    else:
        file_base = f"prism_orig_{safe_model}_seed{args.seed}"

    final_path = out_dir / f"{file_base}.json"
    checkpoint_path = out_dir / f"{file_base}.jsonl"

    if args.no_resume:
        existing_results_by_id = {}
    else:
        existing_results_by_id = load_existing_results(final_path, checkpoint_path)
        if existing_results_by_id:
            ok_count = sum(1 for r in existing_results_by_id.values() if r.get("status") == "ok")
            fail_count = sum(1 for r in existing_results_by_id.values() if r.get("status") != "ok")
            print(f"Resuming: {ok_count} ok, {fail_count} failed from previous run")

    results, completed = await evaluate_pairwise_async(
        pool=pool,
        model_name=args.model,
        prompts=prompts,
        checkpoint_path=checkpoint_path,
        existing_results_by_id=existing_results_by_id,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        base_delay_s=args.base_delay_s,
        max_delay_s=args.max_delay_s,
        stop_on_long_quota_wait_s=args.stop_on_long_quota_wait_s,
    )

    json_out = summarize_results(results, args)
    json_out["completed"] = completed
    json_out["n_api_keys"] = len(api_keys)

    print(f"\n{'=' * 70}")
    print(f"GEMINI — PRISM ORIGINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Model:            {args.model}")
    print(f"API keys:         {len(api_keys)}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Seed:             {args.seed}")
    print(f"Min score diff:   {args.min_score_diff}")
    print(f"Total prompts:    {len(prompts)}")
    print(f"Completed:        {completed}")
    print(f"Judgments saved:  {json_out['n_judgments']}")
    print(f"Failed:           {json_out['n_failed']}")
    print(f"Ties:             {json_out['n_ties']}")
    if json_out["overall_accuracy"] is not None:
        print(f"Overall Accuracy: {json_out['overall_accuracy']:.4f} ({json_out['overall_accuracy'] * 100:.2f}%)")
    else:
        print("Overall Accuracy: N/A")

    print("\nPer Country:")
    for country in TARGET_COUNTRIES:
        if country in json_out["per_birth_country"]:
            m = json_out["per_birth_country"][country]
            print(f"  {country:<25} {m['accuracy'] * 100:.2f}%  (judgments={m['n_judgments']}, users={m['n_users']})")
        else:
            print(f"  {country:<25} N/A")

    with open(final_path, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {final_path}")
    print(f"Checkpoint: {checkpoint_path}")

    if not completed:
        print("\nRun stopped early. Re-run to resume from checkpoint.")


def merge_shards(output_dir, model, seed, num_shards):
    """Merge shard outputs into a single results file."""
    out_dir = Path(output_dir)
    safe_model = model.replace("/", "_")

    all_results = {}
    for shard_idx in range(num_shards):
        shard_base = f"prism_orig_{safe_model}_seed{seed}_shard{shard_idx}of{num_shards}"
        # Try final JSON first, then checkpoint JSONL
        json_path = out_dir / f"{shard_base}.json"
        jsonl_path = out_dir / f"{shard_base}.jsonl"

        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for r in data.get("results", []):
                pid = r.get("prompt_id")
                if pid:
                    all_results[pid] = r
            print(f"  Shard {shard_idx}: loaded {len(data.get('results', []))} from JSON")
        elif jsonl_path.exists():
            records = load_jsonl_records(jsonl_path)
            for r in records:
                pid = r.get("prompt_id")
                if pid and r.get("status") == "ok":
                    all_results[pid] = r
            print(f"  Shard {shard_idx}: loaded {len(records)} from checkpoint")
        else:
            print(f"  Shard {shard_idx}: NOT FOUND")

    results = list(all_results.values())
    ok_results = [r for r in results if r.get("status") == "ok"]
    failed_results = [r for r in results if r.get("status") != "ok"]

    results_df = pd.DataFrame(ok_results) if ok_results else pd.DataFrame()
    overall_acc = float(results_df["correct"].mean()) if len(results_df) > 0 else None

    country_metrics = {}
    for country in TARGET_COUNTRIES:
        if len(results_df) > 0:
            group = results_df[results_df["birth_country"] == country]
            if len(group) > 0:
                country_metrics[country] = {
                    "accuracy": float(group["correct"].mean()),
                    "n_judgments": int(len(group)),
                    "n_users": int(group["user_id"].nunique()),
                }

    merged = {
        "model_label": model,
        "seed": seed,
        "split": "prism_original_pairwise",
        "overall_accuracy": overall_acc,
        "n_judgments": len(ok_results),
        "n_failed": len(failed_results),
        "per_birth_country": country_metrics,
        "results": results,
    }

    merged_path = out_dir / f"prism_orig_{safe_model}_seed{seed}_MERGED.json"
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("MERGED RESULTS")
    print(f"{'=' * 70}")
    print(f"Total OK:    {len(ok_results)}")
    print(f"Failed:      {len(failed_results)}")
    if overall_acc is not None:
        print(f"Accuracy:    {overall_acc:.4f} ({overall_acc * 100:.2f}%)")

    print("\nPer Country:")
    for country in TARGET_COUNTRIES:
        if country in country_metrics:
            m = country_metrics[country]
            print(f"  {country:<25} {m['accuracy'] * 100:.2f}%  (judgments={m['n_judgments']}, users={m['n_users']})")
        else:
            print(f"  {country:<25} N/A")

    print(f"\nSaved to: {merged_path}")


def main():
    # Quick check for --merge mode
    import sys
    if "--merge" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--merge", action="store_true")
        parser.add_argument("--model", type=str, default="gemini-2.5-flash")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--output_dir", type=str, default="prism_original_results")
        parser.add_argument("--num_shards", type=int, required=True)
        args = parser.parse_args()
        merge_shards(args.output_dir, args.model, args.seed, args.num_shards)
    else:
        asyncio.run(async_main())


if __name__ == "__main__":
    main()