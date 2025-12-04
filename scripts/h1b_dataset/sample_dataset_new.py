#!/usr/bin/env python3
"""
SOC-controlled balanced sampler (NOT paired):

Goal:
- Sample n/2 AI rows and n/2 Other rows
- Control for SOC_CODE by construction (same SOC composition in both groups)
- Avoid matched pairs (no PAIR_ID / no row-to-row matching)

Filters (kept from your original script):
- CASE_STATUS == "Certified" (configurable)
- PW_UNIT_OF_PAY == "Year" AND WAGE_UNIT_OF_PAY == "Year" (configurable)
- AI label comes ONLY from provided AI title list (txt) or CSV (job_title + AI_job==True)

Sampling design:
- Work in SOC strata that have both AI and Other rows
- Allocate per-SOC quota q_soc (same for AI and Other in that SOC)
- Then sample q_soc AI rows and q_soc Other rows independently within SOC

Output:
- ROW_ID (original row order id)
- IS_AI
- plus all original columns
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


# -----------------------------
# Title list loading (same semantics; CSV boolean parsing made robust)
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


def soc_weight_from_capacity(capacity: int, mode: str) -> float:
    # capacity := min(ai_count, other_count) in this SOC (how many balanced draws possible)
    if capacity <= 0:
        return 0.0
    if mode == "uniform":
        return 1.0
    if mode == "inverse_sqrt":
        return 1.0 / float(np.sqrt(capacity))
    return float(np.sqrt(capacity))  # "sqrt"


def allocate_soc_quotas(
        rng: np.random.Generator,
        soc_list: List[str],
        capacity_by_soc: Dict[str, int],
        target_per_group: int,
        weight_mode: str,
        cap_per_soc: int,
) -> Dict[str, int]:
    """
    Allocate q_soc counts such that sum(q_soc)=target_per_group and
    q_soc <= capacity_by_soc[soc] and q_soc <= cap_per_soc (if cap_per_soc>0).
    """
    quotas: Dict[str, int] = {soc: 0 for soc in soc_list}
    cap = cap_per_soc if cap_per_soc and cap_per_soc > 0 else 10**18

    total = 0
    while total < target_per_group:
        eligible = []
        weights = []
        for soc in soc_list:
            c = int(capacity_by_soc.get(soc, 0))
            if c <= 0:
                continue
            if quotas[soc] >= c:
                continue
            if quotas[soc] >= cap:
                continue
            w = soc_weight_from_capacity(c, weight_mode)
            if w <= 0:
                continue
            eligible.append(soc)
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
            f"Could only allocate {total}/{target_per_group} balanced samples across SOCs. "
            f"Try lowering --target-total, lowering --min-ai-per-soc, or increasing/clearing --max-per-soc."
        )
    return quotas


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

    # CASE_STATUS filter
    ap.add_argument("--status-col", default="CASE_STATUS")
    ap.add_argument("--status-value", default="Certified")

    # Yearly wage filters
    ap.add_argument("--pw-unit-col", default="PW_UNIT_OF_PAY")
    ap.add_argument("--wage-unit-col", default="WAGE_UNIT_OF_PAY")
    ap.add_argument("--year-unit-value", default="Year")
    ap.add_argument("--allow-missing-unit-cols", action="store_true")

    ap.add_argument("--target-total", type=int, default=200,
                    help="Total output rows. Will sample target_total//2 AI + target_total//2 Other (SOC-balanced).")
    ap.add_argument("--seed", type=int, default=0)

    # SOC control knobs (kept similar spirit to your old script, but now for quotas not pairs)
    ap.add_argument("--min-ai-per-soc", type=int, default=1,
                    help="Require at least this many AI rows in a SOC to be considered.")
    ap.add_argument("--max-per-soc", type=int, default=0,
                    help="Hard cap for q_soc (0 = unlimited). This is the strongest SOC diversity control.")
    ap.add_argument("--soc-weight-mode", choices=["uniform", "sqrt", "inverse_sqrt"], default="inverse_sqrt",
                    help="How to weight SOCs when allocating q_soc. Uses capacity=min(ai,other) per SOC.")

    ap.add_argument("--allow-short", action="store_true",
                    help="If set and one group is too small, shrink to the largest feasible balanced size.")

    args = ap.parse_args()

    if args.target_total < 2:
        raise ValueError("--target-total must be >= 2")

    half = args.target_total // 2
    if half == 0:
        raise ValueError("--target-total too small")
    if args.target_total % 2 != 0:
        print(f"[WARN] target_total={args.target_total} is odd; output will be {2*half} to keep 50/50.")

    rng = np.random.default_rng(args.seed)
    ai_titles = load_ai_titles(args.ai_titles)
    if not ai_titles:
        raise SystemExit("AI titles list is empty after loading/normalization.")

    df_raw = read_df(args.input)

    # Add ROW_ID based on ORIGINAL file row order (before any filtering)
    if "ROW_ID" not in df_raw.columns:
        df_raw.insert(0, "ROW_ID", np.arange(len(df_raw), dtype=np.int64))

    # Required columns
    for c in (args.job_title_col, args.soc_col, args.status_col):
        if c not in df_raw.columns:
            raise KeyError(f"Missing required column '{c}' in input.")

    n_initial = len(df_raw)

    # CASE_STATUS filter
    df_raw = df_raw[df_raw[args.status_col].astype(str) == str(args.status_value)]
    n_after_status = len(df_raw)

    # Yearly unit filters
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

    # Label AI by exact normalized title list match
    title_norm = df[args.job_title_col].astype(str).map(norm_title)
    is_ai = title_norm.isin(ai_titles).to_numpy(dtype=bool)

    soc_arr = df[args.soc_col].astype(str).to_numpy()
    ai_idx_all = np.flatnonzero(is_ai)
    other_idx_all = np.flatnonzero(~is_ai)

    if ai_idx_all.size == 0:
        raise SystemExit("No AI rows found in filtered input using the provided AI title list.")
    if other_idx_all.size == 0:
        raise SystemExit("No Other rows available (everything labeled AI) after filtering.")

    # Build per-SOC pools
    ai_socs_counts = pd.Series(soc_arr[ai_idx_all]).value_counts().to_dict()
    other_socs_counts = pd.Series(soc_arr[other_idx_all]).value_counts().to_dict()

    # SOCs eligible: have both AI and Other, and meet min_ai_per_soc
    soc_ok: List[str] = []
    capacity_by_soc: Dict[str, int] = {}
    for soc, a_cnt in ai_socs_counts.items():
        if int(a_cnt) < int(args.min_ai_per_soc):
            continue
        o_cnt = int(other_socs_counts.get(soc, 0))
        if o_cnt <= 0:
            continue
        cap = min(int(a_cnt), int(o_cnt))
        if cap > 0:
            soc_ok.append(str(soc))
            capacity_by_soc[str(soc)] = cap

    soc_ok.sort()
    print(f"soc_ok count: {len(soc_ok)}")
    if not soc_ok:
        raise SystemExit("No SOC_CODE has both AI and Other rows under current constraints.")

    # If one group is too small globally for requested half, handle
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
        print(f"[WARN] allow_short enabled: sampling {half} per-group => total {2 * half}")

    # Also ensure SOC capacity can support requested half
    total_capacity = sum(capacity_by_soc[s] for s in soc_ok)
    if total_capacity < half:
        if not args.allow_short:
            raise SystemExit(
                f"Insufficient SOC-balanced capacity for requested sampling.\n"
                f"  requested per-group: {half}\n"
                f"  total balanced capacity across SOCs: {total_capacity}\n"
                f"Reduce --target-total / --min-ai-per-soc, or set --allow-short."
            )
        half = total_capacity
        print(f"[WARN] allow_short enabled: using max SOC-balanced half={half} => total {2 * half}")

    # Allocate per-SOC quotas q_soc
    quotas = allocate_soc_quotas(
        rng=rng,
        soc_list=soc_ok,
        capacity_by_soc=capacity_by_soc,
        target_per_group=half,
        weight_mode=args.soc_weight_mode,
        cap_per_soc=args.max_per_soc,
    )

    # Precompute per-SOC index arrays
    ai_idx_by_soc: Dict[str, np.ndarray] = {}
    other_idx_by_soc: Dict[str, np.ndarray] = {}
    for soc in soc_ok:
        ai_idx_by_soc[soc] = ai_idx_all[soc_arr[ai_idx_all] == soc]
        other_idx_by_soc[soc] = other_idx_all[soc_arr[other_idx_all] == soc]

    # Sample independently within each SOC according to quotas (NO pairing)
    sel_ai: List[int] = []
    sel_other: List[int] = []
    for soc in soc_ok:
        q = int(quotas.get(soc, 0))
        if q <= 0:
            continue
        ai_pool = ai_idx_by_soc[soc]
        other_pool = other_idx_by_soc[soc]
        if ai_pool.size < q or other_pool.size < q:
            raise RuntimeError(f"Quota infeasible for SOC={soc}: q={q}, ai={ai_pool.size}, other={other_pool.size}")
        sel_ai.extend(rng.choice(ai_pool, size=q, replace=False).tolist())
        sel_other.extend(rng.choice(other_pool, size=q, replace=False).tolist())

    if len(sel_ai) != half or len(sel_other) != half:
        raise RuntimeError(f"Internal error: sampled AI={len(sel_ai)} Other={len(sel_other)} expected={half}")

    selected = np.array(sel_ai + sel_other, dtype=int)
    out = df.loc[selected].copy()
    out["IS_AI"] = False
    out.loc[out.index.isin(sel_ai), "IS_AI"] = True

    # Shuffle output rows
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Sanity summaries
    print(f"Saved: {args.output}")
    print(f"Rows saved: {len(out)} (AI={int(out['IS_AI'].sum())}, Other={int((~out['IS_AI']).sum())})")
    print(f"Unique {args.soc_col} in output: {out[args.soc_col].nunique()}")

    # Show SOC balance check (should match by construction)
    soc_ai = out[out["IS_AI"]].groupby(args.soc_col).size()
    soc_ot = out[~out["IS_AI"]].groupby(args.soc_col).size()
    chk = pd.concat([soc_ai.rename("AI"), soc_ot.rename("Other")], axis=1).fillna(0).astype(int)
    bad = chk[chk["AI"] != chk["Other"]]
    if len(bad) == 0:
        print("SOC balance check: OK (AI == Other within every SOC).")
    else:
        print("[WARN] SOC balance check failed for some SOCs:")
        print(bad.to_string())


if __name__ == "__main__":
    main()
