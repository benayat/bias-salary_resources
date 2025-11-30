#!/usr/bin/env python3
"""
SOC-diverse matched-pairs sampler (one-shot: original -> ~200 rows), WITH CASE_STATUS filtering.

Core idea:
- Filter: keep only rows where CASE_STATUS == "Certified" (configurable flags).
- AI label comes ONLY from a provided AI title list (txt or csv with AI_job==True).
- Build matched pairs within SOC_CODE: for each selected AI row, select one non-AI row from SAME SOC_CODE.
- To avoid collapsing into a few mega SOCs:
  - optional max cap per SOC (--max-pairs-per-soc)
  - SOC selection weights (--ai-soc-weight-mode), e.g. inverse_sqrt to favor smaller SOCs

Output:
- ROW_ID (unique id 0..N-1 from the ORIGINAL file row order)
- IS_AI, PAIR_ID, MATCH_LEVEL
- plus all original columns

Example:
  uv run scripts/h1b_dataset/sample_dataset.py \
    --input data/.../Combined_LCA_Disclosure_Data_FY2024.csv \
    --ai-titles data/.../ai_ml_job_titles.csv \
    --output data/.../h1b_2024_sampled.csv \
    --target-total 200 \
    --seed 42 \
    --max-pairs-per-soc 5 \
    --ai-soc-weight-mode inverse_sqrt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Set, Tuple, List

import numpy as np
import pandas as pd


# -----------------------------
# Title list loading (your semantics)
# -----------------------------

def norm_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_ai_titles(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AI titles file not found: {path}")

    if p.suffix.lower() == ".txt":
        titles: List[str] = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                titles.append(line)
        return {norm_title(t) for t in titles if norm_title(t)}

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "AI_job" not in df.columns:
            raise KeyError("AI titles CSV must contain column 'AI_job' (boolean).")
        if "job_title" not in df.columns:
            raise KeyError("AI titles CSV must contain column 'job_title'.")

        only_ai_df = df[df.get("AI_job", False) == True]
        print(f"AI titles: {len(only_ai_df)}")
        return {norm_title(t) for t in only_ai_df.get("job_title") if norm_title(t)}

    raise ValueError(f"Unsupported AI titles extension: {p.suffix} (use .txt or .csv)")


# -----------------------------
# IO + helpers
# -----------------------------

def read_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)  # supports .csv.gz too


def naics2(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    m = re.search(r"(\d{2})", s)
    return m.group(1) if m else ""


def pick_one(rng: np.random.Generator, arr: np.ndarray) -> int:
    return int(arr[int(rng.integers(0, len(arr)))])  # one element


# -----------------------------
# Main sampler
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ai-titles", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--job-title-col", default="JOB_TITLE")
    ap.add_argument("--soc-col", default="SOC_CODE")
    ap.add_argument("--state-col", default="WORKSITE_STATE")
    ap.add_argument("--naics-col", default="NAICS_CODE")
    ap.add_argument("--ft-col", default="FULL_TIME_POSITION")

    # NEW: CASE_STATUS filter
    ap.add_argument("--status-col", default="CASE_STATUS")
    ap.add_argument("--status-value", default="Certified")

    ap.add_argument("--target-total", type=int, default=200, help="Total rows in output (will use target_total//2 pairs).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min-ai-per-soc", type=int, default=1, help="Filter SOCs with fewer AI rows than this.")
    ap.add_argument(
        "--max-pairs-per-soc",
        type=int,
        default=0,
        help="Hard cap on pairs per SOC (0 = unlimited). Strongest diversity control."
    )
    ap.add_argument(
        "--ai-soc-weight-mode",
        choices=["uniform", "sqrt", "inverse_sqrt"],
        default="sqrt",
        help=(
            "How to weight SOCs when choosing where to draw the next pair.\n"
            "  uniform:      equal chance per SOC\n"
            "  sqrt:         favors larger SOCs\n"
            "  inverse_sqrt: favors smaller SOCs (more diversity)"
        ),
    )

    args = ap.parse_args()

    if args.target_total < 2:
        raise ValueError("--target-total must be >= 2")
    target_pairs = args.target_total // 2
    if target_pairs <= 0:
        raise ValueError("--target-total too small to form pairs")

    rng = np.random.default_rng(args.seed)

    ai_titles = load_ai_titles(args.ai_titles)
    if not ai_titles:
        raise SystemExit("AI titles list is empty after loading/normalization.")

    df_raw = read_df(args.input)

    # Add ROW_ID based on ORIGINAL file row order (before any filtering)
    if "ROW_ID" not in df_raw.columns:
        df_raw.insert(0, "ROW_ID", np.arange(len(df_raw), dtype=np.int64))

    # Required columns (status-col too, since we're filtering)
    for c in (args.job_title_col, args.soc_col, args.status_col):
        if c not in df_raw.columns:
            raise KeyError(f"Missing required column '{c}' in input.")

    n_before_status = len(df_raw)
    df_raw = df_raw[df_raw[args.status_col].astype(str) == str(args.status_value)]
    n_after_status = len(df_raw)

    # Reset index for safe positional indexing later, but keep original ROW_ID values
    df = df_raw.reset_index(drop=True)

    print(f"Input rows: {n_before_status:,}")
    print(f"Rows after {args.status_col} == {args.status_value!r}: {n_after_status:,}")

    # Normalize SOC + label AI by exact title list match (normalized)
    df[args.soc_col] = df[args.soc_col].astype(str)
    title_norm = df[args.job_title_col].astype(str).map(norm_title)
    is_ai = title_norm.isin(ai_titles).to_numpy(dtype=bool)

    # Helper arrays for fast matching filters
    soc_arr = df[args.soc_col].astype(str).to_numpy()
    state_arr = df[args.state_col].astype(str).to_numpy() if args.state_col in df.columns else np.array([""] * len(df))
    naics2_arr = df[args.naics_col].map(naics2).astype(str).to_numpy() if args.naics_col in df.columns else np.array([""] * len(df))
    ft_arr = df[args.ft_col].astype(str).to_numpy() if args.ft_col in df.columns else np.array([""] * len(df))

    # Index arrays
    ai_idx_all = np.flatnonzero(is_ai)
    other_idx_all = np.flatnonzero(~is_ai)

    if len(ai_idx_all) == 0:
        raise SystemExit("No AI rows found in filtered input using the provided AI title list.")
    if len(other_idx_all) == 0:
        raise SystemExit("No Other rows available (everything labeled AI) after filtering.")

    # SOCs that have AI and Other
    ai_socs = pd.Series(soc_arr[ai_idx_all]).value_counts()
    ai_socs = ai_socs[ai_socs >= args.min_ai_per_soc]
    soc_ai_set = set(ai_socs.index.tolist())

    other_socs = set(pd.Series(soc_arr[other_idx_all]).unique().tolist())
    soc_ok = sorted(list(soc_ai_set.intersection(other_socs)))
    print(f"soc_ok count: {len(soc_ok)}")

    if not soc_ok:
        raise SystemExit("No SOC_CODE has both AI and Other rows under current constraints.")

    # Precompute per-SOC AI indices + per-SOC Other indices
    soc_ok_set = set(soc_ok)
    ai_idx = ai_idx_all[np.isin(soc_arr[ai_idx_all], soc_ok)]
    other_idx = other_idx_all[np.isin(soc_arr[other_idx_all], soc_ok)]

    ai_idx_by_soc: Dict[str, np.ndarray] = {}
    other_idx_by_soc: Dict[str, np.ndarray] = {}
    for soc in soc_ok:
        ai_idx_by_soc[soc] = ai_idx[soc_arr[ai_idx] == soc]
        other_idx_by_soc[soc] = other_idx[soc_arr[other_idx] == soc]

    # Availability masks (remove used rows without rebuilding arrays)
    available_ai = np.zeros(len(df), dtype=bool)
    available_other = np.zeros(len(df), dtype=bool)
    available_ai[ai_idx] = True
    available_other[other_idx] = True

    # SOC weights (SOC-first)
    ai_counts_ok = {soc: int(len(ai_idx_by_soc[soc])) for soc in soc_ok}

    def soc_weight(soc: str) -> float:
        c = ai_counts_ok.get(soc, 0)
        if c <= 0:
            return 0.0
        if args.ai_soc_weight_mode == "uniform":
            return 1.0
        if args.ai_soc_weight_mode == "inverse_sqrt":
            return 1.0 / float(np.sqrt(c))
        return float(np.sqrt(c))  # "sqrt"

    # Pair cap per SOC
    cap = args.max_pairs_per_soc if args.max_pairs_per_soc and args.max_pairs_per_soc > 0 else 10**18
    pairs_per_soc: Dict[str, int] = {soc: 0 for soc in soc_ok}

    # Storage for selected indices + metadata
    selected_indices: List[int] = []
    meta_is_ai: List[bool] = []
    meta_pair_id: List[int] = []
    meta_match_level: List[str] = []

    def match_other(ai_i: int, soc: str) -> Tuple[int | None, str]:
        base = other_idx_by_soc[soc]
        if base.size == 0:
            return None, "no_soc_other_pool"

        base_av = base[available_other[base]]
        if base_av.size == 0:
            return None, "no_other_left"

        st = state_arr[ai_i]
        n2 = naics2_arr[ai_i]
        ft = ft_arr[ai_i]

        tiers = [
            ("soc+state+naics2+ft", (state_arr[base_av] == st) & (naics2_arr[base_av] == n2) & (ft_arr[base_av] == ft)),
            ("soc+state+naics2",    (state_arr[base_av] == st) & (naics2_arr[base_av] == n2)),
            ("soc+state",           (state_arr[base_av] == st)),
            ("soc_only",            None),
        ]

        for level, mask in tiers:
            cand = base_av if mask is None else base_av[mask]
            if cand.size > 0:
                j = pick_one(rng, cand)
                return j, level

        return None, "no_match"

    # Main loop: choose SOC -> choose available AI -> match available Other (within SOC)
    pair_id = 0
    while pair_id < target_pairs:
        eligible_socs = []
        weights = []
        for soc in soc_ok:
            if pairs_per_soc[soc] >= cap:
                continue

            ai_candidates = ai_idx_by_soc[soc]
            if ai_candidates.size == 0 or not np.any(available_ai[ai_candidates]):
                continue

            other_candidates = other_idx_by_soc[soc]
            if other_candidates.size == 0 or not np.any(available_other[other_candidates]):
                continue

            w = soc_weight(soc)
            if w <= 0:
                continue
            eligible_socs.append(soc)
            weights.append(w)

        if not eligible_socs:
            break

        weights_np = np.array(weights, dtype=float)
        weights_np = weights_np / weights_np.sum()
        soc = str(rng.choice(eligible_socs, p=weights_np))

        ai_pool = ai_idx_by_soc[soc]
        ai_pool_av = ai_pool[available_ai[ai_pool]]
        if ai_pool_av.size == 0:
            continue

        ai_i = pick_one(rng, ai_pool_av)

        other_j, match_level = match_other(ai_i, soc)
        if other_j is None:
            # discard this AI row for future attempts to prevent loops
            available_ai[ai_i] = False
            continue

        selected_indices.extend([ai_i, other_j])
        meta_is_ai.extend([True, False])
        meta_pair_id.extend([pair_id, pair_id])
        meta_match_level.extend([match_level, match_level])

        available_ai[ai_i] = False
        available_other[other_j] = False

        pairs_per_soc[soc] += 1
        pair_id += 1

    if pair_id == 0:
        raise SystemExit("Failed to create any pairs (no eligible SOCs / no matches).")

    out = df.loc[selected_indices].copy()
    out["IS_AI"] = meta_is_ai
    out["PAIR_ID"] = meta_pair_id
    out["MATCH_LEVEL"] = meta_match_level

    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Pairs created: {pair_id} => rows: {len(out)}")
    print(f"Target total: {args.target_total} (achieved {len(out)})")
    print(f"Unique SOCs in output: {out[args.soc_col].nunique()}")

    counts = out[args.soc_col].value_counts()
    print("Actual SOCs used:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
