#!/usr/bin/env python3
"""
Welch + HC3 analysis for "AI uplift vs comparable non-AI" using SPB:

  SPB_i = (estimated - actual) / actual * 100

For each model CSV in a directory, prints:
1) Group summaries (AI vs Other)
2) Welch's t-test on mean(SPB_AI - SPB_Other)
3) OLS regression with HC3 robust standard errors:
      SPB ~ IS_AI + controls

Controls are AUTO-CHOSEN by dataset:
- H1B-like: SOC_CODE, WORKSITE_STATE, NAICS2(from NAICS_CODE), FULL_TIME_POSITION, log1p(TOTAL_WORKER_POSITIONS)
- Salaries-like: experience_level, employment_type, remote_ratio, company_size, company_location,
                 employee_residence, work_year

You can override any column via CLI or add extra controls.

Examples:

H1B-like:
  uv run scripts/compare_welch_hc3_spb_auto.py \
    --estimates-dir data/.../sampled_inverse \
    --glob "*.csv" \
    --actual-col PREVAILING_WAGE \
    --estimate-col estimated_salary_in_usd \
    --ai-col IS_AI

Salaries-like (usually auto-detects without overrides):
  uv run scripts/compare_welch_hc3_spb_auto.py \
    --estimates-dir data/salaries-for-data-science-jobs/estimations \
    --glob "*.csv" \
    --actual-col salary_in_usd \
    --estimate-col estimated_salary_in_usd \
    --ai-col is_ai_job
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


# -----------------------------
# Helpers
# -----------------------------

def safe_bool_series(s: pd.Series) -> pd.Series:
    """Convert common truthy strings/ints to boolean."""
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["true", "1", "yes", "y", "t"])


def filename_to_model_tag(path: Path) -> tuple[str, str]:
    """
    Best-effort extraction mirroring your folder structure:
    .../<model_name>/llm_estimated_salaries-<persona>.csv
    """
    model_name = path.parent.name
    stem = path.stem
    m = re.match(r"llm_estimated_salaries(?:_debug)?[-_](.+)$", stem)
    persona = m.group(1) if m and m.group(1) else "unknown"
    return model_name, persona


def naics2(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    m = re.search(r"(\d{2})", s)
    return m.group(1) if m else ""


def mean_ci_diff_welch(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """CI for mean difference (x - y) using Welch-Satterthwaite df."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan"), float("nan")

    mx, my = x.mean(), y.mean()
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    se2 = vx / nx + vy / ny
    if not np.isfinite(se2) or se2 <= 0:
        d = mx - my
        return float(d), float(d)

    num = se2**2
    den = (vx * vx) / (nx * nx * (nx - 1)) + (vy * vy) / (ny * ny * (ny - 1))
    df = num / den if den > 0 else min(nx - 1, ny - 1)

    se = math.sqrt(se2)
    tcrit = stats.t.ppf(1 - alpha / 2, df=df)
    d = mx - my
    return float(d - tcrit * se), float(d + tcrit * se)


def compute_spb_series(
        df: pd.DataFrame,
        actual_col: str,
        estimate_col: str,
        min_actual: float,
        spb_cap: float,
) -> pd.Series:
    act = pd.to_numeric(df[actual_col], errors="coerce")
    est = pd.to_numeric(df[estimate_col], errors="coerce")
    spb = ((est - act) / act) * 100.0
    spb[(act < min_actual) | act.isna() | est.isna()] = np.nan
    if spb_cap and spb_cap > 0:
        spb = spb.clip(-spb_cap, spb_cap)
    return spb


def fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "nan"
    return f"{p:.3e}"


def first_present(cols: set[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


# -----------------------------
# Auto-detection
# -----------------------------

@dataclass(frozen=True)
class Profile:
    name: str
    actual_candidates: List[str]
    estimate_candidates: List[str]
    ai_candidates: List[str]
    base_cat_controls: List[str]
    base_num_controls: List[str]
    # H1B-like derived controls:
    naics_source: Optional[str] = None          # e.g., NAICS_CODE -> NAICS2
    twp_source: Optional[str] = None            # e.g., TOTAL_WORKER_POSITIONS -> LOG_TWP


H1B_PROFILE = Profile(
    name="h1b",
    actual_candidates=["PREVAILING_WAGE", "WAGE_RATE_OF_PAY_FROM", "WAGE_RATE_OF_PAY_TO"],
    estimate_candidates=["estimated_salary_in_usd", "ESTIMATED_SALARY_IN_USD"],
    ai_candidates=["IS_AI", "is_ai_job", "is_ai"],
    base_cat_controls=["SOC_CODE", "WORKSITE_STATE", "FULL_TIME_POSITION"],
    base_num_controls=[],
    naics_source="NAICS_CODE",
    twp_source="TOTAL_WORKER_POSITIONS",
)

SALARIES_PROFILE = Profile(
    name="salaries",
    actual_candidates=["salary_in_usd", "SALARY_IN_USD"],
    estimate_candidates=["estimated_salary_in_usd", "ESTIMATED_SALARY_IN_USD"],
    ai_candidates=["is_ai_job", "IS_AI", "is_ai"],
    base_cat_controls=[
        "experience_level",
        "employment_type",
        "remote_ratio",          # treat as categorical (0/50/100)
        "company_size",
        "company_location",
        "employee_residence",
        "work_year",             # treat as categorical fixed-effects by year
    ],
    base_num_controls=[],
)


def infer_profile(cols: set[str], forced: str) -> Profile:
    if forced != "auto":
        if forced == "h1b":
            return H1B_PROFILE
        if forced == "salaries":
            return SALARIES_PROFILE
        # "generic" means: use whichever known cols exist without assumptions.
        return Profile(
            name="generic",
            actual_candidates=["salary_in_usd", "PREVAILING_WAGE"],
            estimate_candidates=["estimated_salary_in_usd"],
            ai_candidates=["is_ai_job", "IS_AI", "is_ai"],
            base_cat_controls=[],
            base_num_controls=[],
        )

    # Heuristic detection
    if "SOC_CODE" in cols or "WORKSITE_STATE" in cols or "PREVAILING_WAGE" in cols:
        return H1B_PROFILE
    if "salary_in_usd" in cols and ("company_size" in cols or "experience_level" in cols or "employment_type" in cols):
        return SALARIES_PROFILE

    # Fallback: generic
    return Profile(
        name="generic",
        actual_candidates=["salary_in_usd", "PREVAILING_WAGE"],
        estimate_candidates=["estimated_salary_in_usd"],
        ai_candidates=["is_ai_job", "IS_AI", "is_ai"],
        base_cat_controls=[],
        base_num_controls=[],
    )


# -----------------------------
# Welch + HC3
# -----------------------------

def welch_test(ai: np.ndarray, other: np.ndarray) -> dict:
    ai = np.asarray(ai, dtype=float)
    other = np.asarray(other, dtype=float)
    ai = ai[np.isfinite(ai)]
    other = other[np.isfinite(other)]

    res = {"n_ai": int(ai.size), "n_other": int(other.size)}
    if ai.size < 2 or other.size < 2:
        res["ok"] = False
        return res

    t2 = stats.ttest_ind(ai, other, equal_var=False, alternative="two-sided")
    tg = stats.ttest_ind(ai, other, equal_var=False, alternative="greater")  # H1: mean(ai) > mean(other)

    diff = float(ai.mean() - other.mean())
    ci_lo, ci_hi = mean_ci_diff_welch(ai, other)

    res.update({
        "ok": True,
        "t_stat": float(t2.statistic),
        "p_two": float(t2.pvalue),
        "p_one_greater": float(tg.pvalue),

        "mean_ai": float(ai.mean()),
        "mean_other": float(other.mean()),
        "median_ai": float(np.median(ai)),
        "median_other": float(np.median(other)),
        "mape_ai": float(np.mean(np.abs(ai))),
        "mape_other": float(np.mean(np.abs(other))),

        "mean_diff": diff,
        "ci95_diff": (ci_lo, ci_hi),
    })
    return res


def hc3_regression(
        dff: pd.DataFrame,
        spb_col: str,
        ai_num_col: str,
        cat_controls: List[str],
        num_controls: List[str],
) -> Optional[dict]:
    """OLS with HC3 robust SE. Controls included only if columns exist."""
    df = dff.copy()

    # Coerce types
    for c in cat_controls:
        if c in df.columns:
            df[c] = df[c].astype(str)

    for c in num_controls:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop NAs for all involved columns that exist
    drop_cols = [spb_col, ai_num_col] + [c for c in (cat_controls + num_controls) if c in df.columns]
    df = df.dropna(subset=drop_cols)
    if len(df) < 10:
        return None

    terms: List[str] = [ai_num_col]
    terms += [f"C({c})" for c in cat_controls if c in df.columns]
    terms += [c for c in num_controls if c in df.columns]

    formula = f"{spb_col} ~ " + " + ".join(terms)

    try:
        fit = smf.ols(formula, data=df).fit(cov_type="HC3")
    except Exception:
        return None

    if ai_num_col not in fit.params.index:
        return None

    ci = fit.conf_int().loc[ai_num_col].to_numpy(dtype=float)
    return {
        "n": int(fit.nobs),
        "formula": formula,
        "ai_coef": float(fit.params[ai_num_col]),
        "ai_se_hc3": float(fit.bse[ai_num_col]),
        "ai_t": float(fit.tvalues[ai_num_col]),
        "ai_p": float(fit.pvalues[ai_num_col]),
        "ai_ci95": (float(ci[0]), float(ci[1])),
        "r2": float(fit.rsquared),
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Welch + HC3 analysis across model CSV outputs (SPB) with auto column selection.")
    ap.add_argument("--estimates-dir", required=True, help="Directory containing per-model CSVs.")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern for model CSV files.")
    ap.add_argument("--dataset", choices=["auto", "h1b", "salaries", "generic"], default="salaries",
                    help="Dataset selector; 'auto' infers from columns.")

    # Optional overrides (auto-detected if omitted)
    ap.add_argument("--actual-col", default="salary_in_usd", help="Actual wage/salary column. If blank, auto-detected.")
    ap.add_argument("--estimate-col", default="estimated_salary_in_usd", help="Estimate column. If blank, auto-detected.")
    ap.add_argument("--ai-col", default="is_ai_job", help="AI label column. If blank, auto-detected.")

    # Extra controls (repeatable flags)
    ap.add_argument("--extra-cat", action="append", default=[], help="Extra categorical control column (repeatable).")
    ap.add_argument("--extra-num", action="append", default=[], help="Extra numeric control column (repeatable).")
    ap.add_argument("--no-controls", action="store_true", help="If set: regression is SPB ~ IS_AI only (no controls).")

    ap.add_argument("--min-actual", type=float, default=1.0, help="Drop rows with actual < min-actual (avoids SPB blowups).")
    ap.add_argument("--spb-cap", type=float, default=0.0, help="If >0, clip SPB to [-cap, +cap] before tests.")

    args = ap.parse_args()

    d = Path(args.estimates_dir)
    if not d.is_dir():
        raise SystemExit(f"--estimates-dir is not a directory: {d}")

    files = sorted(d.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched glob={args.glob!r} under {d}")

    for fp in files:
        model_name, persona = filename_to_model_tag(fp)
        print(f"\n=== Model: {model_name} | Persona: {persona} | File: {fp.name} ===")

        try:
            df0 = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            print(f"  ERROR reading CSV: {e}")
            continue

        cols = set(df0.columns)
        profile = infer_profile(cols, args.dataset)

        actual_col = args.actual_col.strip() or first_present(cols, profile.actual_candidates)
        estimate_col = args.estimate_col.strip() or first_present(cols, profile.estimate_candidates)
        ai_col = args.ai_col.strip() or first_present(cols, profile.ai_candidates)

        missing_primary = [name for name, val in [("actual", actual_col), ("estimate", estimate_col), ("ai", ai_col)] if not val]
        if missing_primary:
            print(f"  Skipped: could not infer required column(s): {missing_primary}")
            print(f"    Available columns sample: {sorted(list(cols))[:25]} ...")
            continue

        df = df0.copy()
        df[ai_col] = safe_bool_series(df[ai_col])

        df["SPB"] = compute_spb_series(
            df,
            actual_col=actual_col,
            estimate_col=estimate_col,
            min_actual=args.min_actual,
            spb_cap=args.spb_cap,
        )
        df = df.dropna(subset=["SPB", ai_col])
        if len(df) == 0:
            print("  No valid rows after SPB filtering.")
            continue

        # Print detected mapping
        print(f"  Detected dataset: {profile.name}")
        print(f"  Using columns: actual={actual_col}  estimate={estimate_col}  ai={ai_col}")

        spb_ai = df.loc[df[ai_col], "SPB"].to_numpy(float)
        spb_ot = df.loc[~df[ai_col], "SPB"].to_numpy(float)

        # 1) Welch
        w = welch_test(spb_ai, spb_ot)
        if not w.get("ok", False):
            print(f"  Welch test: insufficient data (n_ai={w.get('n_ai')}, n_other={w.get('n_other')})")
        else:
            ci = w["ci95_diff"]
            print(f"  Group means (SPB): AI={w['mean_ai']:.4f}%  Other={w['mean_other']:.4f}%  Diff={w['mean_diff']:.4f} pp")
            print(f"    medians:         AI={w['median_ai']:.4f}%  Other={w['median_other']:.4f}%")
            print(f"    MAPE:            AI={w['mape_ai']:.4f}%  Other={w['mape_other']:.4f}%")
            print(f"  Welch t-test H0(mean diff=0): t={w['t_stat']:.4f}  p(two)={fmt_p(w['p_two'])}  p(one, AI>Other)={fmt_p(w['p_one_greater'])}")
            print(f"    95% CI diff (AI-Other): [{ci[0]:.4f}, {ci[1]:.4f}] percentage-points")

        # 2) Regression controls selection
        df["IS_AI_NUM"] = df[ai_col].astype(int)

        if profile.naics_source and profile.naics_source in df.columns:
            df["NAICS2"] = df[profile.naics_source].map(naics2).astype(str)

        if profile.twp_source and profile.twp_source in df.columns:
            twp = pd.to_numeric(df[profile.twp_source], errors="coerce")
            df["LOG_TWP"] = np.log1p(twp)

        cat_controls = [] if args.no_controls else list(profile.base_cat_controls)
        num_controls = [] if args.no_controls else list(profile.base_num_controls)

        # Include derived controls when created
        if not args.no_controls:
            if "NAICS2" in df.columns:
                cat_controls.append("NAICS2")
            if "LOG_TWP" in df.columns:
                num_controls.append("LOG_TWP")

        # Add user extras
        if not args.no_controls:
            cat_controls += list(args.extra_cat)
            num_controls += list(args.extra_num)

        reg = hc3_regression(
            df,
            spb_col="SPB",
            ai_num_col="IS_AI_NUM",
            cat_controls=cat_controls,
            num_controls=num_controls,
        )

        if not reg:
            print("  OLS + HC3 (conditional): skipped (fit failed or insufficient usable rows).")
        else:
            ci = reg["ai_ci95"]
            print(f"  OLS + HC3 (conditional) n={reg['n']}  R²={reg['r2']:.4f}")
            print(f"    formula: {reg['formula']}")
            print(f"    AI uplift coef (ΔSPB, pp): {reg['ai_coef']:.4f}  SE(HC3)={reg['ai_se_hc3']:.4f}  t={reg['ai_t']:.4f}  p={fmt_p(reg['ai_p'])}")
            print(f"    95% CI AI coef: [{ci[0]:.4f}, {ci[1]:.4f}]")


if __name__ == "__main__":
    main()
