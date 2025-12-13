#!/usr/bin/env python3
"""
Welch + HC3 analysis for "AI uplift vs comparable non-AI" using SPB:

  SPB_i = (estimated - actual) / actual * 100

For each model CSV in a directory, prints:
1) Group summaries (AI vs Other)
2) Welch's t-test on mean(SPB_AI - SPB_Other) (no equal-variance assumption)
3) OLS regression with HC3 robust standard errors:
      SPB ~ IS_AI + controls (SOC_CODE, WORKSITE_STATE, NAICS2, FULL_TIME_POSITION, log1p(TOTAL_WORKER_POSITIONS))

This supports the conditional claim: "AI jobs are more overestimated (higher SPB) than comparable non-AI jobs",
where "comparable" is defined by the controls you include.

Usage:
  uv run scripts/h1b_dataset/compare_welch_hc3.py \
    --estimates-dir data/.../sampled_inverse \
    --glob "*.csv" \
    --actual-col PREVAILING_WAGE \
    --estimate-col estimated_salary_in_usd \
    --min-actual 1 \
    --spb-cap 0
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e

try:
    import statsmodels.formula.api as smf
except Exception as e:
    raise SystemExit("statsmodels is required. Install with: pip install statsmodels") from e


# -----------------------------
# Helpers
# -----------------------------

def safe_bool_series(s: pd.Series) -> pd.Series:
    """Convert common truthy strings/ints to boolean."""
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["true", "1", "yes", "y"])


def filename_to_model_tag(path: Path) -> tuple[str, str]:
    """
    Extract model name and persona from path structure:
    sampled_<weight>/model_name/llm_estimated_salaries-<persona>.csv

    Returns: (model_name, persona)
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
    """
    CI for mean difference (x - y) using Welch-Satterthwaite df.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan"), float("nan")

    mx, my = x.mean(), y.mean()
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    se2 = vx / nx + vy / ny
    if se2 <= 0:
        d = mx - my
        return float(d), float(d)

    # Welch-Satterthwaite df
    num = se2 ** 2
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


# -----------------------------
# Main per-file analysis
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

    # SciPy supports alternative; keep a fallback if needed.
    try:
        t2 = stats.ttest_ind(ai, other, equal_var=False, alternative="two-sided")
        tg = stats.ttest_ind(ai, other, equal_var=False, alternative="greater")  # H1: mean(ai) > mean(other)
        res["t_stat"] = float(t2.statistic)
        res["p_two"] = float(t2.pvalue)
        res["p_one_greater"] = float(tg.pvalue)
    except TypeError:
        # Older SciPy: no alternative kwarg
        t2 = stats.ttest_ind(ai, other, equal_var=False)
        # One-sided from two-sided (valid if symmetric under H0)
        p_two = float(t2.pvalue)
        t_stat = float(t2.statistic)
        p_one = p_two / 2.0 if t_stat > 0 else 1.0 - (p_two / 2.0)
        res["t_stat"] = t_stat
        res["p_two"] = p_two
        res["p_one_greater"] = p_one

    diff = float(ai.mean() - other.mean())
    ci_lo, ci_hi = mean_ci_diff_welch(ai, other)
    res.update({
        "ok": True,
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
        ai_col: str,
        soc_col: str,
        state_col: str,
        naics_col: str,
        ft_col: str,
        twp_col: str,
) -> Optional[dict]:
    """
    OLS with HC3 robust standard errors.
    Returns AI coefficient stats (and some model info).
    """
    df = dff.copy()

    # Build derived controls
    if naics_col in df.columns:
        df["NAICS2"] = df[naics_col].map(naics2).astype(str)
    if twp_col in df.columns:
        twp = pd.to_numeric(df[twp_col], errors="coerce")
        df["LOG_TWP"] = np.log1p(twp)

    # Coerce types for categorical controls
    for c in [soc_col, state_col, "NAICS2", ft_col]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Ensure regression columns exist
    needed = [spb_col, ai_col]
    df = df.dropna(subset=needed)
    if len(df) < 10:
        return None

    terms: List[str] = [ai_col]

    # Fixed-effects style categorical controls
    if soc_col in df.columns:
        terms.append(f"C({soc_col})")
    if state_col in df.columns:
        terms.append(f"C({state_col})")
    if "NAICS2" in df.columns:
        terms.append("C(NAICS2)")
    if ft_col in df.columns:
        terms.append(f"C({ft_col})")

    # Numeric control
    if "LOG_TWP" in df.columns:
        terms.append("LOG_TWP")

    formula = f"{spb_col} ~ " + " + ".join(terms)

    try:
        model = smf.ols(formula, data=df)
        fit = model.fit(cov_type="HC3")  # HC3 robust covariance
    except Exception:
        return None

    if ai_col not in fit.params.index:
        return None

    ci = fit.conf_int().loc[ai_col].to_numpy(dtype=float)
    return {
        "n": int(fit.nobs),
        "formula": formula,
        "ai_coef": float(fit.params[ai_col]),
        "ai_se_hc3": float(fit.bse[ai_col]),
        "ai_t": float(fit.tvalues[ai_col]),
        "ai_p": float(fit.pvalues[ai_col]),
        "ai_ci95": (float(ci[0]), float(ci[1])),
        "r2": float(fit.rsquared),
    }

def check_normality(spb_ai: np.ndarray, spb_ot: np.ndarray) -> dict:
    """
    Check normality of SPB for AI and non-AI groups using Shapiro-Wilk test.
    Returns dict with test statistics and p-values for both groups.
    """
    result = {}

    for label, data in [("AI", spb_ai), ("Other", spb_ot)]:
        data = data[np.isfinite(data)]
        if len(data) < 3:
            result[label] = {"ok": False}
            continue

        try:
            stat, p = stats.shapiro(data)
            result[label] = {
                "ok": True,
                "statistic": float(stat),
                "p_value": float(p),
                "is_normal": p > 0.05
            }
        except Exception:
            result[label] = {"ok": False}

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Welch + HC3 analysis across model CSV outputs.")
    ap.add_argument("--estimates-dir", required=True, help="Directory containing per-model CSVs.")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern for model CSV files.")

    ap.add_argument("--actual-col", default="PREVAILING_WAGE", help="Actual wage column (yearly).")
    ap.add_argument("--estimate-col", default="estimated_salary_in_usd", help="Model estimate column (yearly).")
    ap.add_argument("--ai-col", default="IS_AI", help="AI label column (bool-ish).")

    # Controls aligned with your prompt columns
    ap.add_argument("--soc-col", default="SOC_CODE")
    ap.add_argument("--state-col", default="WORKSITE_STATE")
    ap.add_argument("--naics-col", default="NAICS_CODE")
    ap.add_argument("--ft-col", default="FULL_TIME_POSITION")
    ap.add_argument("--twp-col", default="TOTAL_WORKER_POSITIONS")

    ap.add_argument("--min-actual", type=float, default=1.0, help="Drop rows with actual < min-actual (avoids SPB blowups).")
    ap.add_argument("--spb-cap", type=float, default=0.0, help="If >0, clip SPB to [-cap, +cap] before tests.")

    args = ap.parse_args()

    d = Path(args.estimates_dir)
    if not d.exists() or not d.is_dir():
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

        missing = [c for c in (args.actual_col, args.estimate_col, args.ai_col) if c not in df0.columns]
        if missing:
            print(f"  Skipped: missing columns {missing}")
            continue

        df = df0.copy()
        df[args.ai_col] = safe_bool_series(df[args.ai_col])

        df["SPB"] = compute_spb_series(
            df, actual_col=args.actual_col, estimate_col=args.estimate_col,
            min_actual=args.min_actual, spb_cap=args.spb_cap
        )

        # Drop invalid SPB rows
        df = df.dropna(subset=["SPB", args.ai_col])
        if len(df) == 0:
            print("  No valid rows after SPB filtering.")
            continue

        spb_ai = df.loc[df[args.ai_col], "SPB"].to_numpy(float)
        spb_ot = df.loc[~df[args.ai_col], "SPB"].to_numpy(float)

        norm = check_normality(spb_ai, spb_ot)
        if norm.get("AI", {}).get("ok") and norm.get("Other", {}).get("ok"):
            ai_n = norm["AI"]
            ot_n = norm["Other"]
            print(f"  Normality (Shapiro-Wilk):")
            print(f"    AI:    W={ai_n['statistic']:.4f}  p={fmt_p(ai_n['p_value'])}  {'(Normal)' if ai_n['is_normal'] else '(NOT Normal)'}")
            print(f"    Other: W={ot_n['statistic']:.4f}  p={fmt_p(ot_n['p_value'])}  {'(Normal)' if ot_n['is_normal'] else '(NOT Normal)'}")



        # 1) Welch test
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

        # 2) HC3 regression (conditional comparability)
        df["IS_AI_NUM"] = df[args.ai_col].astype(int)

        reg = hc3_regression(
            df, spb_col="SPB", ai_col="IS_AI_NUM",
            soc_col=args.soc_col, state_col=args.state_col,
            naics_col=args.naics_col, ft_col=args.ft_col, twp_col=args.twp_col,
        )

        if not reg:
            print("  HC3 regression: skipped (fit failed or insufficient usable rows).")
        else:
            ci = reg["ai_ci95"]
            print(f"  OLS + HC3 (conditional) n={reg['n']}  R²={reg['r2']:.4f}")
            print(f"    formula: {reg['formula']}")
            print(f"    AI uplift coef (ΔSPB, pp): {reg['ai_coef']:.4f}  SE(HC3)={reg['ai_se_hc3']:.4f}  t={reg['ai_t']:.4f}  p={fmt_p(reg['ai_p'])}")
            print(f"    95% CI AI coef: [{ci[0]:.4f}, {ci[1]:.4f}]")

if __name__ == "__main__":
    main()
