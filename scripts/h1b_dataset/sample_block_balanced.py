#!/usr/bin/env python3
"""
Block-balanced sampler (NOT paired): sample n/2 AI + n/2 Other while controlling by block.

Block definition (default):
  BLOCK_ID = (SOC_CODE, WORKSITE_STATE, NAICS2, FULL_TIME_POSITION)

Key properties:
- No 1:1 row matching, no PAIR_ID, no MATCH_LEVEL.
- Strong control by construction: within each BLOCK_ID, sampled AI count == sampled Other count.
- Enforces quota cap: q_b <= max_per_block (optional) to prevent mega-block domination.
- Keeps your hygiene filters: CASE_STATUS + wage-unit Year/Year.

Output:
- ROW_ID (original row order id)
- IS_AI
- BLOCK_ID (for audits/debugging; you can drop later if you want)
- plus all original columns
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Title list loading (same semantics)
# -----------------------------

def norm_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _to_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x
    s = x.astype(str).str.strip().str.lower()
    return s.isin({"true", "1", "yes", "y", "t"})


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
            raise KeyError("AI titles CSV must contain column 'AI_job' (boolean-like).")
        if "job_title" not in df.columns:
            raise KeyError("AI titles CSV must contain column 'job_title'.")

        ai_mask = _to_bool_series(df["AI_job"])
        only_ai_df = df[ai_mask]
        print(f"AI titles: {len(only_ai_df)}")
        return {norm_title(t) for t in only_ai_df["job_title"].tolist() if norm_title(t)}

    raise ValueError(f"Unsupported AI titles extension: {p.suffix} (use .txt or .csv)")


# -----------------------------
# IO + helpers
# -----------------------------

def read_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def unit_is_year(series: pd.Series, year_value: str) -> pd.Series:
    y = str(year_value).strip().lower()
    return series.astype(str).str.strip().str.lower().eq(y)


def naics2(x) -> str:
    if pd.isna(x):
        return "UNK"
    s = str(x)
    m = re.search(r"(\d{2})", s)
    return m.group(1) if m else "UNK"


def clean_str_series(s: pd.Series, unk: str = "UNK") -> pd.Series:
    # Normalize missing -> UNK, strip, cast to string
    out = s.copy()
    out = out.where(~out.isna(), other=unk)
    out = out.astype(str).str.strip()
    out = out.replace({"": unk})
    return out


def soc_weight_from_capacity(capacity: int, mode: str) -> float:
    if capacity <= 0:
        return 0.0
    if mode == "uniform":
        return 1.0
    if mode == "inverse_sqrt":
        return 1.0 / float(np.sqrt(capacity))
    return float(np.sqrt(capacity))  # "sqrt"


def allocate_block_quotas(
        rng: np.random.Generator,
        blocks: List[str],
        capacity_by_block: Dict[str, int],
        target_per_group: int,
        weight_mode: str,
        max_per_block: int,
) -> Dict[str, int]:
    """
    Allocate integer quotas q_b such that:
      sum(q_b) = target_per_group
      0 <= q_b <= capacity_by_block[b]
      q_b <= max_per_block (if max_per_block>0)
    """
    quotas: Dict[str, int] = {b: 0 for b in blocks}
    cap = max_per_block if max_per_block and max_per_block > 0 else 10**18

    total = 0
    while total < target_per_group:
        eligible: List[str] = []
        weights: List[float] = []

        for b in blocks:
            raw_cap = int(capacity_by_block.get(b, 0))
            if raw_cap <= 0:
                continue
            eff_cap = min(raw_cap, cap)  # enforces q_b <= max_per_block
            if quotas[b] >= eff_cap:
                continue

            w = soc_weight_from_capacity(eff_cap, weight_mode)
            if w <= 0:
                continue

            eligible.append(b)
            weights.append(w)

        if not eligible:
            break

        wnp = np.array(weights, dtype=float)
        wnp = wnp / wnp.sum()
        chosen = str(rng.choice(eligible, p=wnp))
        quotas[chosen] += 1
        total += 1

    if total < target_per_group:
        raise RuntimeError(
            f"Could only allocate {total}/{target_per_group} balanced samples across blocks. "
            f"Try lowering --target-total, raising --max-per-block, or relaxing filters."
        )

    return quotas


def build_index_by_block(block_arr: np.ndarray, idx: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Efficiently build mapping block -> indices (without O(#blocks*N) masks).
    """
    keys = block_arr[idx].astype(object)
    order = np.argsort(keys, kind="mergesort")
    idx_sorted = idx[order]
    keys_sorted = keys[order]

    uniq, starts = np.unique(keys_sorted, return_index=True)
    ends = np.concatenate([starts[1:], np.array([len(keys_sorted)], dtype=int)])

    out: Dict[str, np.ndarray] = {}
    for u, s, e in zip(uniq, starts, ends):
        out[str(u)] = idx_sorted[s:e]
    return out


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--ai-titles", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--job-title-col", default="JOB_TITLE")

    # Block columns (defaults mirror your old script)
    ap.add_argument("--soc-col", default="SOC_CODE")
    ap.add_argument("--state-col", default="WORKSITE_STATE")
    ap.add_argument("--naics-col", default="NAICS_CODE")
    ap.add_argument("--ft-col", default="FULL_TIME_POSITION")

    # CASE_STATUS filter
    ap.add_argument("--status-col", default="CASE_STATUS")
    ap.add_argument("--status-value", default="Certified")

    # Wage-unit filters
    ap.add_argument("--pw-unit-col", default="PW_UNIT_OF_PAY")
    ap.add_argument("--wage-unit-col", default="WAGE_UNIT_OF_PAY")
    ap.add_argument("--year-unit-value", default="Year")
    ap.add_argument("--allow-missing-unit-cols", action="store_true")

    ap.add_argument("--target-total", type=int, default=1000,
                    help="Total output rows; produces target_total//2 AI + target_total//2 Other (block-balanced).")
    ap.add_argument("--seed", type=int, default=0)

    # Allocation controls
    ap.add_argument("--weight-mode", choices=["uniform", "sqrt", "inverse_sqrt"], default="inverse_sqrt",
                    help="Weights for allocating quotas across blocks (based on effective block capacity).")
    ap.add_argument("--max-per-block", type=int, default=0,
                    help="Hard cap on q_b per block (0 = unlimited). Prevents mega-block domination.")

    # Optional: drop tiny blocks early (keeps things sane)
    ap.add_argument("--min-ai-per-block", type=int, default=1)
    ap.add_argument("--min-other-per-block", type=int, default=1)

    ap.add_argument("--allow-short", action="store_true",
                    help="If set, shrink to the largest feasible balanced sample size instead of failing.")

    args = ap.parse_args()

    if args.target_total < 2:
        raise ValueError("--target-total must be >= 2")

    half = args.target_total // 2
    if half <= 0:
        raise ValueError("--target-total too small")

    if args.target_total % 2 != 0:
        print(f"[WARN] target_total={args.target_total} is odd; output will be {2*half} to keep 50/50.")

    rng = np.random.default_rng(args.seed)
    ai_titles = load_ai_titles(args.ai_titles)
    if not ai_titles:
        raise SystemExit("AI titles list is empty after loading/normalization.")

    df_raw = read_df(args.input)

    # ROW_ID from original file order
    if "ROW_ID" not in df_raw.columns:
        df_raw.insert(0, "ROW_ID", np.arange(len(df_raw), dtype=np.int64))

    # Required columns
    for c in (args.job_title_col, args.status_col, args.soc_col, args.state_col, args.naics_col, args.ft_col):
        if c not in df_raw.columns:
            raise KeyError(f"Missing required column '{c}' in input.")

    n_initial = len(df_raw)

    # CASE_STATUS
    df_raw = df_raw[df_raw[args.status_col].astype(str) == str(args.status_value)]
    n_after_status = len(df_raw)

    # Yearly wage units
    for col in (args.pw_unit_col, args.wage_unit_col):
        if not col:
            continue
        if col not in df_raw.columns:
            if args.allow_missing_unit_cols:
                print(f"[WARN] Unit column missing, skipping yearly filter for: {col}")
                continue
            raise KeyError(f"Missing required unit column '{col}' (use --allow-missing-unit-cols to ignore).")

        before = len(df_raw)
        df_raw = df_raw[unit_is_year(df_raw[col], args.year_unit_value)]
        after = len(df_raw)
        print(f"Rows after {col} == {args.year_unit_value!r} (case-insensitive): {after:,} (dropped {before-after:,})")

    n_after_units = len(df_raw)

    df = df_raw.reset_index(drop=True)

    print(f"Input rows: {n_initial:,}")
    print(f"Rows after {args.status_col} == {args.status_value!r}: {n_after_status:,}")
    print(f"Rows after yearly-unit filters: {n_after_units:,}")

    # Label AI
    title_norm = df[args.job_title_col].astype(str).map(norm_title)
    is_ai = title_norm.isin(ai_titles).to_numpy(dtype=bool)

    ai_idx_all = np.flatnonzero(is_ai)
    other_idx_all = np.flatnonzero(~is_ai)

    if ai_idx_all.size == 0:
        raise SystemExit("No AI rows found after filtering using the provided AI title list.")
    if other_idx_all.size == 0:
        raise SystemExit("No Other rows found after filtering (everything labeled AI).")

    # If globally insufficient for requested size
    if (ai_idx_all.size < half) or (other_idx_all.size < half):
        if not args.allow_short:
            raise SystemExit(
                f"Not enough rows to sample 50/50.\n"
                f"  requested per-group: {half}\n"
                f"  available AI: {ai_idx_all.size}\n"
                f"  available Other: {other_idx_all.size}\n"
                f"Reduce --target-total or set --allow-short."
            )
        half = min(half, int(ai_idx_all.size), int(other_idx_all.size))
        print(f"[WARN] allow_short enabled: sampling {half} per-group => total {2*half}")

    # Build BLOCK_ID components (with UNK normalization)
    soc_s = clean_str_series(df[args.soc_col])
    st_s = clean_str_series(df[args.state_col])
    n2_s = df[args.naics_col].map(naics2).astype(str)
    ft_s = clean_str_series(df[args.ft_col])

    block_id = (soc_s + "||" + st_s + "||" + n2_s + "||" + ft_s).to_numpy(dtype=object)
    df["BLOCK_ID"] = block_id  # helpful for audits/debug; you can drop later

    # Count per block (AI vs Other)
    ai_counts = pd.Series(block_id[ai_idx_all]).value_counts()
    other_counts = pd.Series(block_id[other_idx_all]).value_counts()

    # Eligible blocks: meet min counts in each group
    block_ok: List[str] = []
    capacity_by_block: Dict[str, int] = {}

    # Iterate over intersection only
    common_blocks = ai_counts.index.intersection(other_counts.index)
    for b in common_blocks:
        a = int(ai_counts.get(b, 0))
        o = int(other_counts.get(b, 0))
        if a < args.min_ai_per_block:
            continue
        if o < args.min_other_per_block:
            continue
        cap = min(a, o)
        if cap > 0:
            bs = str(b)
            block_ok.append(bs)
            capacity_by_block[bs] = cap

    block_ok.sort()
    print(f"block_ok count: {len(block_ok)}")
    if not block_ok:
        raise SystemExit("No blocks have both AI and Other rows under current constraints.")

    # Total balanced capacity check (respecting max_per_block)
    per_block_cap = args.max_per_block if args.max_per_block and args.max_per_block > 0 else 10**18
    total_eff_capacity = sum(min(capacity_by_block[b], per_block_cap) for b in block_ok)
    if total_eff_capacity < half:
        if not args.allow_short:
            raise SystemExit(
                f"Insufficient block-balanced capacity for requested sampling.\n"
                f"  requested per-group: {half}\n"
                f"  total effective capacity across blocks: {total_eff_capacity}\n"
                f"  (max_per_block={args.max_per_block})\n"
                f"Reduce --target-total, increase --max-per-block, or relax filters."
            )
        half = total_eff_capacity
        print(f"[WARN] allow_short enabled: using max feasible half={half} => total {2*half}")

    # Restrict indices to eligible blocks only
    block_ok_set = set(block_ok)
    ai_idx = ai_idx_all[np.isin(block_id[ai_idx_all], list(block_ok_set))]
    other_idx = other_idx_all[np.isin(block_id[other_idx_all], list(block_ok_set))]

    # Build block -> indices maps (fast)
    ai_by_block = build_index_by_block(block_id, ai_idx)
    other_by_block = build_index_by_block(block_id, other_idx)

    # Allocate quotas q_b with q_b <= min(ai,other) and q_b <= max_per_block
    quotas = allocate_block_quotas(
        rng=rng,
        blocks=block_ok,
        capacity_by_block=capacity_by_block,
        target_per_group=half,
        weight_mode=args.weight_mode,
        max_per_block=args.max_per_block,
    )

    # Sample independently within each block (NO PAIRS)
    sel_ai: List[int] = []
    sel_other: List[int] = []

    for b in block_ok:
        q = int(quotas.get(b, 0))
        if q <= 0:
            continue
        ai_pool = ai_by_block.get(b, np.array([], dtype=int))
        ot_pool = other_by_block.get(b, np.array([], dtype=int))
        if ai_pool.size < q or ot_pool.size < q:
            raise RuntimeError(f"Quota infeasible for block={b}: q={q}, ai={ai_pool.size}, other={ot_pool.size}")

        sel_ai.extend(rng.choice(ai_pool, size=q, replace=False).tolist())
        sel_other.extend(rng.choice(ot_pool, size=q, replace=False).tolist())

    if len(sel_ai) != half or len(sel_other) != half:
        raise RuntimeError(f"Internal error: sampled AI={len(sel_ai)} Other={len(sel_other)} expected={half}")

    selected = np.array(sel_ai + sel_other, dtype=int)
    out = df.loc[selected].copy()
    out["IS_AI"] = False
    out.loc[out.index.isin(sel_ai), "IS_AI"] = True

    # Shuffle output
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Sanity checks
    print(f"Saved: {args.output}")
    print(f"Rows saved: {len(out)} (AI={int(out['IS_AI'].sum())}, Other={int((~out['IS_AI']).sum())})")
    print(f"Unique BLOCK_ID in output: {out['BLOCK_ID'].nunique()}")

    # Verify block balance: AI==Other per block
    b_ai = out[out["IS_AI"]].groupby("BLOCK_ID").size()
    b_ot = out[~out["IS_AI"]].groupby("BLOCK_ID").size()
    chk = pd.concat([b_ai.rename("AI"), b_ot.rename("Other")], axis=1).fillna(0).astype(int)
    bad = chk[chk["AI"] != chk["Other"]]
    if len(bad) == 0:
        print("Block balance check: OK (AI == Other within every BLOCK_ID).")
    else:
        print("[WARN] Block balance check failed for some blocks:")
        print(bad.to_string())


if __name__ == "__main__":
    main()
