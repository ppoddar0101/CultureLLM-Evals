#!/usr/bin/env python3
"""
Verbalize WVS morality question distributions from Gemini 2.5 Flash,
then compute the per-question half L1 norm against WVS ground-truth data,
with 95% bootstrap confidence intervals.

KEY DIFFERENCE FROM PRO VERSION:
- No thinking tokens (fair comparison to Llama 3.1 8B)
- chunk_size=1 (one question per API call) to ensure 10-scale
  distributions fit on a single line without needing thinking tokens
- Same prompt structure, parser, metric, and bootstrap as Llama pipeline

Example:
python3 compute_morality_gemini_flash.py \
  --country "France" \
  --gt_csv wvs_w7.csv \
  --norms_json cultural-norms.json \
  --out_dir ./morality_outputs \
  --n 5000 \
  --seed 42 \
  --temperature 0.7 --top_p 0.9 \
  --bootstrap 1000
"""

import argparse
import csv
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google import genai

# ─── Question definitions ─────────────────────────────────────────────────────

QUESTIONS_10: Dict[str, Tuple[str, int]] = {
    "Q176": ("Nowadays, one often has trouble deciding which moral values are the right ones to follow. "
             "[Rate 1 (completely agree) to 10 (completely disagree)]", 10),
    "Q177": ("Claiming government benefits to which you are not entitled. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q178": ("Avoiding a fare on public transport. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q179": ("Stealing property. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q180": ("Cheating on taxes if you have a chance. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q181": ("Someone accepting a bribe in the course of their duties. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q182": ("Homosexuality. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q183": ("Prostitution. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q184": ("Abortion. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q185": ("Divorce. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q186": ("Sex before marriage. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q187": ("Suicide. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q188": ("Euthanasia. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q189": ("For a man to beat his wife. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q190": ("Parents beating children. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q191": ("Violence against other people. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q192": ("Terrorism as a political, ideological, or religious mean. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q193": ("Having casual sex. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q194": ("Political violence. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
    "Q195": ("Death penalty. "
             "[Rate 1 (never justifiable) to 10 (always justifiable)]", 10),
}

QUESTIONS_4: Dict[str, Tuple[str, int]] = {
    "Q196": ("Keep people under video surveillance in public areas. "
             "[Rate 1 (definitely should have the right) to 4 (definitely should not have the right)]", 4),
    "Q197": ("Monitor all e-mails and any other information exchanged on the Internet. "
             "[Rate 1 (definitely should have the right) to 4 (definitely should not have the right)]", 4),
    "Q198": ("Collect information about anyone living in this country without their knowledge. "
             "[Rate 1 (definitely should have the right) to 4 (definitely should not have the right)]", 4),
}

ALL_QUESTIONS: Dict[str, Tuple[str, int]] = {**QUESTIONS_10, **QUESTIONS_4}

PEOPLE_DESCRIPTOR = {
    "United States": "people from the USA",
    "India": "people from India",
    "Brazil": "people from Brazil",
    "South Africa": "people from South Africa",
    "Japan": "people from Japan",
    "Saudi Arabia": "people from Saudi Arabia",
}

# ─── Norms ────────────────────────────────────────────────────────────────────

def load_norms(norms_json_path: str, country: str) -> str:
    with open(norms_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data.get("cultural-norms", []):
        if entry.get("country") == country:
            norms = entry.get("norms", [])
            lines = [(n.get("text") or "").strip() for n in norms[:2] if n.get("text")]
            return "\n".join(lines).strip()
    raise ValueError(f"No norms found for country={country!r} in {norms_json_path}")

# ─── Prompt ───────────────────────────────────────────────────────────────────
# Uses the SAME prompt structure as the chunked Llama pipeline,
# just with chunk_size=1 so there's always exactly 1 question.

def build_prompt(
    country: str,
    norms_text: str,
    chunk: List[Tuple[str, str, int]],
    start_qnum: int,
) -> str:
    people = PEOPLE_DESCRIPTOR.get(country, f"people from {country}")
    first_q = start_qnum
    last_q = start_qnum + len(chunk) - 1
    n = len(chunk)

    q_lines = []
    for i, (col, text, scale) in enumerate(chunk):
        qnum = start_qnum + i
        q_lines.append(f"Question {qnum} ({col}, scale 1-{scale}): {text}")
    questions_block = "\n".join(q_lines)

    ex_scale = chunk[0][2]
    ex_vals = [round(100 / ex_scale)] * ex_scale
    ex_vals[-1] = 100 - sum(ex_vals[:-1])
    ex_dist = ", ".join(f"{i+1}: {v}%" for i, v in enumerate(ex_vals))

    return f"""You are an expert on {country} culture and traditions. Cultural norms include: {norms_text}

Task: For each question below, output the estimated response distribution for {people}.

STRICT OUTPUT RULES:
- Output EXACTLY {n} line(s), one per question, nothing else.
- Questions are numbered {first_q} through {last_q}. Use EXACTLY these numbers — do not renumber them.
- NO introduction, NO explanation, NO summary, NO extra text of any kind.
- Every line MUST follow this exact format:
  Question N (QXX): {{1: XX%, 2: XX%, ..., {ex_scale}: XX%}}
- All values MUST be on a SINGLE line. Do NOT wrap or split across lines.
- Percentages must sum to 100 for each question.
- Start your response IMMEDIATELY with "Question {first_q}" — do not write anything before it.

Example (scale 1-{ex_scale}):
Question {first_q} (Q176): {{{ex_dist}}}

Now output distributions for {people}:

{questions_block}"""

# ─── Gemini 2.5 Flash setup ──────────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"

def load_model():
    client = genai.Client()
    return client, None

# ─── Model inference (NO thinking tokens) ─────────────────────────────────────

def generate_text(
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    response = model.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_new_tokens,
            "thinking_config": {
                "thinking_budget": 0
            },
        },
    )
    return response.text.strip() if response.text else ""

# ─── Parsing ──────────────────────────────────────────────────────────────────

def extract_lines_by_qnum(generated: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    lines = generated.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^Question\s+(\d+)", line.strip(), re.IGNORECASE)
        if m:
            qnum = int(m.group(1))
            # Wide window to capture wrapped distributions
            window = " ".join(lines[i: i + 15])
            mapping[qnum] = window
    return mapping

def _extract_numeric_dist(text: str, scale: int) -> Optional[np.ndarray]:
    pattern = re.compile(r"\b(\d{1,2})\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%")
    found: Dict[int, float] = {}
    for m in pattern.finditer(text):
        k, v = int(m.group(1)), float(m.group(2))
        if 1 <= k <= scale and k not in found:
            found[k] = v
    if len(found) < scale:
        return None
    arr = np.array([found[k] for k in range(1, scale + 1)], dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return None
    return arr / total

def parse_chunk_output(
    generated: str,
    chunk: List[Tuple[str, str, int]],
    start_qnum: int,
) -> Dict[str, Optional[np.ndarray]]:
    mapping = extract_lines_by_qnum(generated)
    results: Dict[str, Optional[np.ndarray]] = {}
    for i, (col, _, scale) in enumerate(chunk):
        qnum = start_qnum + i
        raw = mapping.get(qnum, "")
        results[col] = _extract_numeric_dist(raw, scale)
    return results

# ─── Ground-truth loading ─────────────────────────────────────────────────────

def load_gt_distributions(
    gt_csv: str, n: int, seed: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = pd.read_csv(gt_csv, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    missing = [q for q in ALL_QUESTIONS if q not in df.columns]
    if missing:
        print(f"  Warning: GT missing columns {missing} — skipping those questions.")

    if len(df) >= n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    gt_dists: Dict[str, np.ndarray] = {}
    gt_counts: Dict[str, np.ndarray] = {}

    for col in ALL_QUESTIONS:
        if col not in df.columns:
            continue
        scale = ALL_QUESTIONS[col][1]
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        vals = vals[(vals >= 1) & (vals <= scale)]
        counts = np.zeros(scale, dtype=np.float64)
        for v in vals:
            counts[int(v) - 1] += 1
        if counts.sum() == 0:
            print(f"  Warning: {col} has no valid GT responses — skipping.")
            continue
        gt_counts[col] = counts
        gt_dists[col] = counts / counts.sum()

    return gt_dists, gt_counts

# ─── Metric ───────────────────────────────────────────────────────────────────

def half_l1(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())

# ─── Bootstrap CIs ────────────────────────────────────────────────────────────

def bootstrap_half_l1_ci(
    gt_counts: np.ndarray,
    model_p: np.ndarray,
    B: int = 2000,
    alpha: float = 0.05,
    n_eff_model: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()
    gt_counts = np.asarray(gt_counts, dtype=np.float64)
    model_p = np.asarray(model_p, dtype=np.float64)
    model_p = model_p / model_p.sum()
    n_gt = int(gt_counts.sum())
    n_model = n_eff_model if n_eff_model is not None else n_gt

    gt_boot = rng.multinomial(n_gt, gt_counts / n_gt, size=B).astype(np.float64)
    gt_boot_p = gt_boot / gt_boot.sum(axis=1, keepdims=True)
    model_boot = rng.multinomial(n_model, model_p, size=B).astype(np.float64)
    model_boot_p = model_boot / model_boot.sum(axis=1, keepdims=True)

    boot_scores = 0.5 * np.abs(gt_boot_p - model_boot_p).sum(axis=1)
    return (
        float(np.percentile(boot_scores, 100 * alpha / 2)),
        float(np.percentile(boot_scores, 100 * (1 - alpha / 2))),
    )

def bootstrap_aggregate_ci(
    per_question_gt_counts: List[np.ndarray],
    per_question_model_p: List[np.ndarray],
    B: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()
    Q = len(per_question_gt_counts)
    all_boot = np.zeros((Q, B), dtype=np.float64)

    for q_idx, (gt_counts, model_p) in enumerate(
        zip(per_question_gt_counts, per_question_model_p)
    ):
        gt_counts = np.asarray(gt_counts, dtype=np.float64)
        model_p = np.asarray(model_p, dtype=np.float64)
        model_p /= model_p.sum()
        n_gt = int(gt_counts.sum())

        gt_boot = rng.multinomial(n_gt, gt_counts / n_gt, size=B).astype(np.float64)
        gt_boot_p = gt_boot / gt_boot.sum(axis=1, keepdims=True)
        model_boot = rng.multinomial(n_gt, model_p, size=B).astype(np.float64)
        model_boot_p = model_boot / model_boot.sum(axis=1, keepdims=True)
        all_boot[q_idx] = 0.5 * np.abs(gt_boot_p - model_boot_p).sum(axis=1)

    agg_boot = all_boot.mean(axis=0)
    return (
        float(np.percentile(agg_boot, 100 * alpha / 2)),
        float(np.percentile(agg_boot, 100 * (1 - alpha / 2))),
    )

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--norms_json", default="cultural-norms.json")
    ap.add_argument("--out_dir", default="./morality_outputs")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk_size", type=int, default=1)  # Default 1 for reliable parsing
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--ci_alpha", type=float, default=0.05)
    ap.add_argument("--n_eff_model", type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = out_dir / f"gemini_flash_{args.country.replace(' ', '_')}_morality"

    print(f"Model: {MODEL_NAME}")
    print(f"Thinking tokens: DISABLED (fair comparison to Llama)")
    print(f"chunk_size={args.chunk_size}  max_new_tokens={args.max_new_tokens}")
    print(f"Total API calls: {len(ALL_QUESTIONS)} questions × 1 call each = {len(ALL_QUESTIONS)} calls")

    norms_text = load_norms(args.norms_json, args.country)

    print("\nLoading Gemini Flash client...")
    model, _ = load_model()

    all_q_list: List[Tuple[str, str, int]] = [
        (col, text, scale) for col, (text, scale) in ALL_QUESTIONS.items()
    ]
    chunk_size = max(1, args.chunk_size)
    model_dists: Dict[str, Optional[np.ndarray]] = {}
    all_rows: List[dict] = []
    qnum_base = 1

    for start in range(0, len(all_q_list), chunk_size):
        chunk = all_q_list[start: start + chunk_size]
        col_labels = [c[0] for c in chunk]
        print(f"\n── Q{qnum_base} {col_labels[0]} (scale 1-{chunk[0][2]}) ──")

        prompt = build_prompt(args.country, norms_text, chunk, qnum_base)
        parsed: Dict[str, Optional[np.ndarray]] = {}

        for attempt in range(1 + args.max_retries):
            gen = generate_text(
                model, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"  attempt {attempt + 1} | output: {gen[:300]}")

            parsed = parse_chunk_output(gen, chunk, qnum_base)
            n_parsed = sum(1 for v in parsed.values() if v is not None)

            if n_parsed == len(chunk):
                print(f"  ✓ Parsed successfully")
                break
            else:
                print(f"  ✗ Parse failed ({n_parsed}/{len(chunk)}), retrying…")

        model_dists.update(parsed)

        for col, text, scale in chunk:
            dist = parsed.get(col)
            row = {
                "country": args.country,
                "checkpoint": MODEL_NAME,
                "key": col,
                "scale": scale,
                "text": text,
            }
            for i in range(1, 11):
                row[f"p{i}"] = float(dist[i - 1]) if (dist is not None and i <= scale) else ""
            row["parsed"] = dist is not None
            row["raw_generation_chunk"] = gen
            all_rows.append(row)

        qnum_base += len(chunk)

    # ── Save raw probs ────────────────────────────────────────────────────────
    probs_path = Path(str(out_prefix) + "_probs.csv")
    fieldnames = (
        ["country", "checkpoint", "key", "scale", "text"]
        + [f"p{i}" for i in range(1, 11)]
        + ["parsed", "raw_generation_chunk"]
    )
    with open(probs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_parsed_total = sum(1 for r in all_rows if r["parsed"])
    print(f"\nWrote probs: {probs_path}")
    print(f"Parsed distributions for {n_parsed_total}/{len(all_rows)} questions.")

    # ── Load GT ───────────────────────────────────────────────────────────────
    print(f"\nLoading GT from {args.gt_csv}  (n={args.n}, seed={args.seed})…")
    gt_dists, gt_counts = load_gt_distributions(args.gt_csv, n=args.n, seed=args.seed)

    # ── Compute metrics ───────────────────────────────────────────────────────
    do_bootstrap = args.bootstrap > 0
    rng = np.random.default_rng(args.seed)
    if do_bootstrap:
        print(f"Computing {args.bootstrap}-replicate bootstrap CIs (alpha={args.ci_alpha})…")

    metric_rows: List[dict] = []
    agg_scores: List[float] = []
    agg_gt_counts: List[np.ndarray] = []
    agg_model_p: List[np.ndarray] = []

    for col, (_, scale) in ALL_QUESTIONS.items():
        model_p = model_dists.get(col)
        gt_p = gt_dists.get(col)
        counts = gt_counts.get(col)

        score = ci_lo = ci_hi = None

        if model_p is not None and gt_p is not None and counts is not None:
            score = half_l1(gt_p, model_p)
            if do_bootstrap:
                ci_lo, ci_hi = bootstrap_half_l1_ci(
                    gt_counts=counts,
                    model_p=model_p,
                    B=args.bootstrap,
                    alpha=args.ci_alpha,
                    n_eff_model=args.n_eff_model,
                    rng=rng,
                )
            agg_scores.append(score)
            agg_gt_counts.append(counts)
            agg_model_p.append(model_p)

        row = {
            "question": col,
            "scale": scale,
            "half_l1": score if score is not None else "",
            "ci_lower": ci_lo if ci_lo is not None else "",
            "ci_upper": ci_hi if ci_hi is not None else "",
            "parsed": model_p is not None,
        }
        for i in range(1, 11):
            row[f"p{i}"] = float(model_p[i - 1]) if (model_p is not None and i <= scale) else ""
        metric_rows.append(row)

    df_results = pd.DataFrame(metric_rows)
    valid_scores = pd.to_numeric(df_results["half_l1"], errors="coerce").dropna()
    aggregate = float(valid_scores.mean()) if len(valid_scores) > 0 else float("nan")

    agg_ci_lo = agg_ci_hi = float("nan")
    if do_bootstrap and len(agg_scores) > 0:
        print(f"Computing aggregate CI across {len(agg_scores)} questions…")
        agg_ci_lo, agg_ci_hi = bootstrap_aggregate_ci(
            per_question_gt_counts=agg_gt_counts,
            per_question_model_p=agg_model_p,
            B=args.bootstrap,
            alpha=args.ci_alpha,
            rng=rng,
        )

    ci_pct = int(round((1 - args.ci_alpha) * 100))
    print(f"\n── Per-question half L1 ({ci_pct}% CI) ──")
    print(df_results[["question", "scale", "half_l1", "ci_lower", "ci_upper", "parsed"]].to_string(index=False))
    print(f"\n── Aggregate half L1 ({len(valid_scores)}/{len(metric_rows)} questions) ──")
    print(f"  Point estimate : {aggregate:.4f}   (0 = perfect, 1 = worst)")
    if not np.isnan(agg_ci_lo):
        print(f"  {ci_pct}% CI        : [{agg_ci_lo:.4f}, {agg_ci_hi:.4f}]")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = Path(str(out_prefix) + "_results.csv")
    df_results.to_csv(results_path, index=False)

    summary_path = Path(str(out_prefix) + "_summary.csv")
    pd.DataFrame([{
        "country": args.country,
        "checkpoint": MODEL_NAME,
        "n_gt": args.n,
        "seed": args.seed,
        "n_questions_total": len(metric_rows),
        "n_questions_parsed": int(len(valid_scores)),
        "aggregate_half_l1": aggregate,
        f"agg_ci{ci_pct}_lower": agg_ci_lo,
        f"agg_ci{ci_pct}_upper": agg_ci_hi,
        "bootstrap_replicates": args.bootstrap,
        "ci_alpha": args.ci_alpha,
    }]).to_csv(summary_path, index=False)

    print(f"\nSaved probs   -> {probs_path}")
    print(f"Saved results -> {results_path}")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()