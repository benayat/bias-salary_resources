#!/usr/bin/env python3
"""
Create a balanced (n_ai/n_non_ai), frequency-matched sample from the ORIGINAL dataset.

Output includes:
- ROW_ID: stable identifier = original row order (0..N-1)
- LOCATION_BIN: top-K locations kept; others -> "OTHER"
- STRATUM: concatenation of match columns (default: LOCATION_BIN + experience_level)

This sample is meant to be reused across all model-output CSVs to keep the comparison identical.

Example:
  python sample_balanced_frequency_matched.py \
    --input salaries_original.csv \
    --out matched_base.csv \
    --is-ai-col is_ai \
    --location-col employee_residence \
    --experience-col experience_level \
    --actual-col salary_in_usd \
    --n-ai 500 --n-non-ai 500 \
    --topk-locations 15 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def normalize_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])


def topk_bin(series: pd.Series, k: int, other_label: str = "OTHER") -> pd.Series:
    vc = series.value_counts(dropna=False)
    top = set(vc.head(k).index.tolist())
    return series.where(series.isin(top), other_label).fillna(other_label)


def build_stratum(df: pd.DataFrame, match_cols: Sequence[str]) -> pd.Series:
    parts = [df[c].astype(str) for c in match_cols]
    return pd.Series(["|".join(row) for row in zip(*parts)], index=df.index, name="STRATUM")


def sample_balanced_frequency_matched(
        df: pd.DataFrame,
        is_ai_col: str,
        match_cols: Sequence[str],
        n_ai: int,
        n_non_ai: int,
        seed: int,
) -> pd.DataFrame:
    """
    Frequency-matched sampling:
    - sample AI rows (n_ai) from strata that have non-AI coverage
    - sample non-AI rows so stratum histogram matches the AI sample
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    df["STRATUM"] = build_stratum(df, match_cols)

    ai = df[df[is_ai_col] == True].copy()
    non = df[df[is_ai_col] == False].copy()

    if len(ai) < n_ai:
        raise ValueError(f"Not enough AI rows: have {len(ai)}, need {n_ai}")
    if len(non) < n_non_ai:
        raise ValueError(f"Not enough non-AI rows: have {len(non)}, need {n_non_ai}")

    non_counts = non["STRATUM"].value_counts()
    ai["NON_POOL_SIZE"] = ai["STRATUM"].map(non_counts).fillna(0).astype(int)

    eligible_ai = ai[ai["NON_POOL_SIZE"] > 0].copy()
    if len(eligible_ai) < n_ai:
        raise ValueError(
            f"Only {len(eligible_ai)} AI rows remain after removing strata with zero non-AI overlap; need {n_ai}."
        )

    # Weighted AI sample: prefer strata with bigger non-AI pools => fewer feasibility issues
    weights = eligible_ai["NON_POOL_SIZE"].clip(lower=1)
    ai_sample = eligible_ai.sample(n=n_ai, random_state=seed, weights=weights).copy()

    # Enforce feasibility per stratum: don't demand more non-AI than exists
    targets = ai_sample["STRATUM"].value_counts().to_dict()
    over = {s: targets[s] - int(non_counts.get(s, 0)) for s in targets if targets[s] > int(non_counts.get(s, 0))}
    if over:
        # Trim excess AI from overloaded strata
        for s, excess in over.items():
            drop_idx = ai_sample[ai_sample["STRATUM"] == s].sample(n=excess, random_state=seed).index
            ai_sample = ai_sample.drop(index=drop_idx)

        # Refill AI sample to n_ai while maintaining feasibility
        while len(ai_sample) < n_ai:
            needs = ai_sample["STRATUM"].value_counts()
            cap_all = (non_counts - needs).fillna(non_counts).clip(lower=0)

            remaining_ai = eligible_ai.drop(index=ai_sample.index, errors="ignore").copy()
            remaining_ai["CAP"] = remaining_ai["STRATUM"].map(cap_all).fillna(0).astype(int)
            remaining_ai = remaining_ai[remaining_ai["CAP"] > 0]
            if remaining_ai.empty:
                raise ValueError("Cannot refill AI sample to desired size while preserving feasibility.")

            k = min(n_ai - len(ai_sample), len(remaining_ai))
            ai_add = remaining_ai.sample(
                n=k,
                random_state=int(rng.integers(1, 1_000_000)),
                weights=remaining_ai["CAP"],
            )
            ai_sample = pd.concat([ai_sample, ai_add], axis=0)

    targets = ai_sample["STRATUM"].value_counts().to_dict()

    # Build matched non-AI sample to match targets
    non_parts = []
    for s, cnt in targets.items():
        pool = non[non["STRATUM"] == s]
        if len(pool) < cnt:
            raise ValueError(f"Feasibility broken: need {cnt} non-AI in stratum {s}, but have {len(pool)}")
        take = pool.sample(n=cnt, random_state=int(rng.integers(1, 1_000_000)))
        non_parts.append(take)

    non_sample = pd.concat(non_parts, axis=0)

    # Optional trim (if user asked for n_non_ai != n_ai)
    if len(non_sample) > n_non_ai:
        non_sample = non_sample.sample(n=n_non_ai, random_state=seed)
    elif len(non_sample) < n_non_ai:
        raise ValueError(f"Could not reach n_non_ai={n_non_ai}; got {len(non_sample)}")

    out = pd.concat([ai_sample, non_sample], axis=0).copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="matched_base.csv")
    ap.add_argument("--out-meta", default="matched_meta.json")

    ap.add_argument("--is-ai-col", default="is_ai")
    ap.add_argument("--actual-col", default="salary_in_usd")
    ap.add_argument("--location-col", default="employee_residence")
    ap.add_argument("--experience-col", default="experience_level")

    ap.add_argument("--topk-locations", type=int, default=15)
    ap.add_argument("--n-ai", type=int, default=500)
    ap.add_argument("--n-non-ai", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--drop-nonpositive-actual", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Stable key for joining later
    if "ROW_ID" not in df.columns:
        df.insert(0, "ROW_ID", np.arange(len(df), dtype=np.int64))

    # Normalize required columns
    if args.is_ai_col not in df.columns:
        raise ValueError(f"Missing is-ai column: {args.is_ai_col}")
    df[args.is_ai_col] = normalize_bool(df[args.is_ai_col])

    for c in [args.actual_col, args.location_col, args.experience_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df[args.actual_col] = pd.to_numeric(df[args.actual_col], errors="coerce")
    df = df.dropna(subset=[args.actual_col, args.location_col, args.experience_col])

    if args.drop_nonpositive_actual:
        df = df[df[args.actual_col] > 0]

    df["LOCATION_BIN"] = topk_bin(df[args.location_col].astype(str), k=args.topk_locations, other_label="OTHER")

    match_cols = ["LOCATION_BIN", args.experience_col]

    matched = sample_balanced_frequency_matched(
        df=df,
        is_ai_col=args.is_ai_col,
        match_cols=match_cols,
        n_ai=args.n_ai,
        n_non_ai=args.n_non_ai,
        seed=args.seed,
    )

    # Save and summarize
    matched.to_csv(args.out, index=False)

    meta: Dict[str, object] = {
        "input": args.input,
        "out": args.out,
        "n_rows_out": int(len(matched)),
        "n_ai_out": int((matched[args.is_ai_col] == True).sum()),
        "n_non_ai_out": int((matched[args.is_ai_col] == False).sum()),
        "match_cols": match_cols,
        "topk_locations": args.topk_locations,
        "seed": args.seed,
        "stratum_counts_ai": matched[matched[args.is_ai_col] == True]["STRATUM"].value_counts().to_dict(),
        "stratum_counts_non_ai": matched[matched[args.is_ai_col] == False]["STRATUM"].value_counts().to_dict(),
    }
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[saved] {args.out} (rows={len(matched)})")
    print(f"[saved] {args.out_meta}")


if __name__ == "__main__":
    main()
