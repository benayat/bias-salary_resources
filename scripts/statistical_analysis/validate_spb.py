#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Your function (as provided)
# -----------------------------
def calculate_statistics(actual: pd.Series, estimated: pd.Series):
    """
    Returns (mae, mpe_percent, spb_percent, n_used).
    Drops NaNs. MPE/SPB ignore rows where actual==0 (avoid divide by zero).
    """
    mask = actual.notna() & estimated.notna()
    a = pd.to_numeric(actual[mask], errors="coerce")
    e = pd.to_numeric(estimated[mask], errors="coerce")
    mask2 = a.notna() & e.notna()
    a = a[mask2].astype(float)
    e = e[mask2].astype(float)

    n = int(len(a))
    if n == 0:
        return np.nan, np.nan, np.nan, 0

    mae = float(np.mean(np.abs(a - e)))

    nonzero = a != 0
    mpe = float(np.mean(((a[nonzero] - e[nonzero]) / a[nonzero]) * 100)) if nonzero.any() else np.nan
    spb = float(np.mean(((e[nonzero] - a[nonzero]) / a[nonzero]) * 100)) if nonzero.any() else np.nan

    return mae, mpe, spb, n


# -----------------------------
# Validator
# -----------------------------
@dataclass
class ValidationConfig:
    atol: float = 1e-9
    rtol: float = 1e-9
    near_zero_abs: float = 1.0          # "near zero" threshold for actual (USD)
    huge_abs_spb_warn: float = 500.0    # warn if |SPB| has outliers beyond this
    strict: bool = False               # if True: raise on failures


def _assert_or_warn(ok: bool, msg: str, strict: bool):
    if ok:
        return
    if strict:
        raise AssertionError(msg)
    print(f"  [WARN] {msg}")


def _finite(x: float) -> bool:
    return x is not None and isinstance(x, (float, int)) and math.isfinite(float(x))


def validate_calculate_statistics(
        actual: pd.Series,
        estimated: pd.Series,
        label: str,
        cfg: ValidationConfig,
) -> None:
    print(f"\n=== VALIDATION: {label} ===")

    mae, mpe, spb, n = calculate_statistics(actual, estimated)

    # Rebuild the exact numeric arrays your function ends up using
    mask = actual.notna() & estimated.notna()
    a = pd.to_numeric(actual[mask], errors="coerce")
    e = pd.to_numeric(estimated[mask], errors="coerce")
    mask2 = a.notna() & e.notna()
    a = a[mask2].astype(float)
    e = e[mask2].astype(float)

    nonzero = a != 0
    a_nz = a[nonzero]
    e_nz = e[nonzero]

    # Basic counts
    print(f"  n_numeric={len(a)}  n_nonzero={int(nonzero.sum())}  n_zero={int((~nonzero).sum())}")
    print(f"  actual_min={float(a.min()) if len(a) else float('nan'):.6f}  actual_p1={float(a.quantile(0.01)) if len(a) else float('nan'):.6f}")
    print(f"  est_min={float(e.min()) if len(e) else float('nan'):.6f}")

    # Denominator sanity
    near0 = (a.abs() < cfg.near_zero_abs)
    if near0.any():
        print(f"  [WARN] {int(near0.sum())} actual values have |actual| < {cfg.near_zero_abs} (near-zero denominators can inflate mean SPB).")

    neg_actual = (a < 0)
    _assert_or_warn(not neg_actual.any(), f"{int(neg_actual.sum())} rows have actual < 0 (unexpected for salary).", cfg.strict)

    # Independent recomputation (vectorized)
    mae_v = float(np.mean(np.abs(a - e))) if len(a) else np.nan
    mpe_v = float(np.mean(((a_nz - e_nz) / a_nz) * 100.0)) if len(a_nz) else np.nan
    spb_v = float(np.mean(((e_nz - a_nz) / a_nz) * 100.0)) if len(a_nz) else np.nan

    # Independent recomputation (loop)
    if len(a_nz):
        spb_loop = float(np.mean([((float(ei) - float(ai)) / float(ai)) * 100.0 for ai, ei in zip(a_nz.to_numpy(), e_nz.to_numpy())]))
    else:
        spb_loop = np.nan

    # Checks: equality to recomputations
    if _finite(mae) and _finite(mae_v):
        _assert_or_warn(np.isclose(mae, mae_v, rtol=cfg.rtol, atol=cfg.atol), f"MAE mismatch: func={mae} vs vec={mae_v}", cfg.strict)
    if _finite(mpe) and _finite(mpe_v):
        _assert_or_warn(np.isclose(mpe, mpe_v, rtol=cfg.rtol, atol=cfg.atol), f"MPE mismatch: func={mpe} vs vec={mpe_v}", cfg.strict)
    if _finite(spb) and _finite(spb_v):
        _assert_or_warn(np.isclose(spb, spb_v, rtol=cfg.rtol, atol=cfg.atol), f"SPB mismatch: func={spb} vs vec={spb_v}", cfg.strict)
    if _finite(spb) and _finite(spb_loop):
        _assert_or_warn(np.isclose(spb, spb_loop, rtol=cfg.rtol, atol=cfg.atol), f"SPB mismatch: func={spb} vs loop={spb_loop}", cfg.strict)

    # Key identity: with symmetric definitions, SPB == -MPE (same rows)
    if _finite(spb) and _finite(mpe):
        _assert_or_warn(np.isclose(spb + mpe, 0.0, rtol=cfg.rtol, atol=1e-7), f"Identity violation: SPB + MPE != 0 (spb={spb}, mpe={mpe})", cfg.strict)

    # Alternate identity: SPB = (mean(est/act) - 1) * 100 on the same row set
    if len(a_nz):
        ratio = (e_nz / a_nz).to_numpy(dtype=float)
        alt = (float(np.mean(ratio)) - 1.0) * 100.0
        if _finite(spb) and _finite(alt):
            _assert_or_warn(np.isclose(spb, alt, rtol=cfg.rtol, atol=1e-7), f"Alt identity mismatch: spb={spb} vs (mean(est/act)-1)*100={alt}", cfg.strict)

        # Outlier scan (not a correctness failure, but catches “silent blowups”)
        spb_row = ((e_nz - a_nz) / a_nz) * 100.0
        max_abs = float(np.max(np.abs(spb_row.to_numpy(dtype=float)))) if len(spb_row) else 0.0
        if max_abs > cfg.huge_abs_spb_warn:
            print(f"  [WARN] max |row_SPB| = {max_abs:.2f}% > {cfg.huge_abs_spb_warn:.2f}% (likely tiny actuals or unit mismatch).")

    # Report summary values
    print(f"  RESULT: MAE={mae:.6f}  MPE={mpe:.6f}%  SPB={spb:.6f}%  n_used={n}")
    print("  OK: core arithmetic checks passed (or warned).")


# -----------------------------
# Main runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Validate SPB computation in calculate_statistics().")
    ap.add_argument("--csv", required=True, help="Path to the estimation CSV.")
    ap.add_argument("--actual-col", default="salary_in_usd")
    ap.add_argument("--estimate-col", default="estimated_salary_in_usd")

    # Optional splits (so you can validate AI vs Other separately)
    ap.add_argument("--ai-col", default="", help="Boolean AI label column. If set, validates AI and Other groups too.")
    ap.add_argument("--strict", action="store_true", help="Raise AssertionError on any check failure.")
    ap.add_argument("--near-zero-abs", type=float, default=1.0, help="Near-zero threshold for actual (USD).")
    ap.add_argument("--huge-abs-spb-warn", type=float, default=500.0, help="Warn if any row has |SPB| above this.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    for c in [args.actual_col, args.estimate_col]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c!r}. Available: {list(df.columns)[:25]} ...")

    cfg = ValidationConfig(
        strict=args.strict,
        near_zero_abs=args.near_zero_abs,
        huge_abs_spb_warn=args.huge_abs_spb_warn,
    )

    # Whole dataset validation
    validate_calculate_statistics(df[args.actual_col], df[args.estimate_col], "ALL ROWS", cfg)

    # Optional AI/Other validation (same arithmetic, different subset)
    if args.ai_col:
        if args.ai_col not in df.columns:
            raise SystemExit(f"--ai-col {args.ai_col!r} not found.")
        ai = df[args.ai_col].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])
        validate_calculate_statistics(df.loc[ai, args.actual_col], df.loc[ai, args.estimate_col], f"AI GROUP ({args.ai_col})", cfg)
        validate_calculate_statistics(df.loc[~ai, args.actual_col], df.loc[~ai, args.estimate_col], f"OTHER GROUP ({args.ai_col})", cfg)


if __name__ == "__main__":
    main()
