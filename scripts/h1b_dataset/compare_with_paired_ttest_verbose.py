#!/usr/bin/env python3
"""
Paired tests using ONLY percent error (Signed Percent Bias, SPB):

  SPB_i = (estimated - actual) / actual * 100

Per model CSV in a directory, prints:
- Overall SPB test: one-sample t-test and Wilcoxon on SPB (H0: mean/median SPB = 0)
- (Optional) AI vs Other group SPB tests, if IS_AI exists
- (Optional) Paired delta test using your matched PAIR_IDs:
      delta_spb = SPB(AI) - SPB(Other) per PAIR_ID
  then one-sample t-test / Wilcoxon on delta_spb (H0: mean/median delta = 0)

Notes:
- SPB can explode if actual is very small. Use --min-actual and/or --spb-cap if needed.

Usage:
  uv run scripts/h1b_dataset/compare_with_paired_ttest.py \
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
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e


# ----------------------------
# Helpers: parsing / naming
# ----------------------------

def safe_bool_series(s: pd.Series) -> pd.Series:
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
    model_name = path.parent.name  # Get the parent directory name (model name)
    stem = path.stem
    # Extract persona from filename: llm_estimated_salaries(-|_)<persona>
    m = re.match(r"llm_estimated_salaries(?:_debug)?[-_](.+)$", stem)
    if m and m.group(1):
        persona = m.group(1)
    else:
        persona = "unknown"
    return model_name, persona


# ----------------------------
# Stats utilities
# ----------------------------

def mean_ci_t(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    m = x.mean()
    sd = x.std(ddof=1)
    if sd == 0:
        return float(m), float(m)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(m - tcrit * se), float(m + tcrit * se)


def cohen_d_onesample(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def _ttest_1samp(x: np.ndarray, alternative: str) -> Tuple[float, float]:
    # scipy ttest can return nan with pathological inputs; keep robust.
    try:
        r = stats.ttest_1samp(x, popmean=0.0, alternative=alternative)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def _wilcoxon_0(x: np.ndarray, alternative: str) -> Tuple[float, float]:
    # Wilcoxon can fail when all zeros, very small n, etc.
    try:
        r = stats.wilcoxon(x, alternative=alternative, zero_method="wilcox")
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


# ----------------------------
# Heuristics (interpretation)
# ----------------------------

def _p_strength(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return "unavailable"
    if p < 1e-6:
        return "extremely strong"
    if p < 1e-3:
        return "very strong"
    if p < 1e-2:
        return "strong"
    if p < alpha:
        return "moderate"
    if p < 0.1:
        return "weak"
    return "none"


def _effect_size_label(d: float) -> str:
    if not np.isfinite(d):
        return "unavailable"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _bias_magnitude_label(mean_spb: float) -> str:
    if not np.isfinite(mean_spb):
        return "unavailable"
    a = abs(mean_spb)
    if a < 5:
        return "small"
    if a < 10:
        return "modest"
    if a < 20:
        return "large"
    return "very large"


def _mape_label(mape: float) -> str:
    if not np.isfinite(mape):
        return "unavailable"
    if mape < 10:
        return "excellent"
    if mape < 20:
        return "good"
    if mape < 40:
        return "rough"
    return "poor"


def _direction(x: float) -> str:
    if not np.isfinite(x):
        return "unknown"
    if x > 0:
        return "overestimation"
    if x < 0:
        return "underestimation"
    return "neutral"


def _flag_skew(mean_spb: float, median_spb: float) -> Optional[str]:
    if not (np.isfinite(mean_spb) and np.isfinite(median_spb)):
        return None
    # Heuristic: big mean-median gap suggests skew/outliers.
    gap = abs(mean_spb - median_spb)
    if gap >= 10:
        return "mean vs median differ a lot (likely skew/outliers)"
    if gap >= 5:
        return "mean vs median differ (some skew/outliers)"
    return None


def print_global_heuristics(alpha: float) -> None:
    print("\n[INTERPRETATION HEURISTICS]")
    print("  Metric meanings:")
    print("    - SPB = (estimate-actual)/actual*100. Positive => overestimation, negative => underestimation.")
    print("    - MAPE here is mean(|SPB|): typical absolute percent error (not signed).")
    print("    - Paired delta = SPB(AI) - SPB(Other) within the same PAIR_ID.")
    print("      Positive delta => AI titles get higher SPB than matched non-AI (more overestimation / less underestimation).")
    print("  Statistical tests used:")
    print("    - One-sample t-test on mean(SPB) (and mean(delta)) against 0.")
    print("    - Wilcoxon signed-rank on median(SPB) (and median(delta)) against 0 (more robust to outliers/skew).")
    print("  Heuristic decision rules (used ONLY for printing interpretations):")
    print(f"    - Significance threshold alpha={alpha:.3f}.")
    print("    - 'Directionally significant' if one-sided p < alpha AND the 95% CI for the mean does NOT cross 0.")
    print("    - Cohen's d labels: <0.2 negligible, <0.5 small, <0.8 medium, else large.")
    print("    - Mean SPB magnitude labels: <5 small, <10 modest, <20 large, else very large.")
    print("    - MAPE labels (percent): <10 excellent, <20 good, <40 rough, else poor.")
    print("  Caution:")
    print("    - Many models/personas => multiple comparisons. Treat per-model p-values as per-model evidence (adjust if needed).")
    print("    - If mean and median disagree strongly, results are likely skewed/outlier-driven; trust Wilcoxon more there.")


# ----------------------------
# Core computations
# ----------------------------

def compute_spb_with_meta(
        df: pd.DataFrame,
        actual_col: str,
        estimate_col: str,
        min_actual: float,
        spb_cap: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    act = pd.to_numeric(df[actual_col], errors="coerce")
    est = pd.to_numeric(df[estimate_col], errors="coerce")

    n_total = int(len(df))
    notna = act.notna() & est.notna()
    n_notna = int(notna.sum())

    min_ok = act >= min_actual
    n_drop_nan = n_total - n_notna
    n_drop_min_actual = int((notna & ~min_ok).sum())
    ok = notna & min_ok
    n_valid = int(ok.sum())

    actv = act[ok].to_numpy(float)
    estv = est[ok].to_numpy(float)
    if actv.size == 0:
        return np.array([], dtype=float), {
            "n_total": n_total,
            "n_notna": n_notna,
            "n_valid": n_valid,
            "drop_nan": n_drop_nan,
            "drop_min_actual": n_drop_min_actual,
            "spb_cap": float(spb_cap),
            "n_capped": 0,
        }

    spb_raw = ((estv - actv) / actv) * 100.0
    n_capped = 0
    spb = spb_raw
    if spb_cap and spb_cap > 0:
        n_capped = int(np.sum(np.abs(spb_raw) > spb_cap))
        spb = np.clip(spb_raw, -spb_cap, spb_cap)

    meta = {
        "n_total": n_total,
        "n_notna": n_notna,
        "n_valid": n_valid,
        "drop_nan": n_drop_nan,
        "drop_min_actual": n_drop_min_actual,
        "spb_cap": float(spb_cap),
        "n_capped": n_capped,
    }
    return spb, meta


def run_spb_tests(spb: np.ndarray, alpha: float) -> dict:
    spb = np.asarray(spb, dtype=float)
    n = int(spb.size)
    if n == 0:
        return {"n": 0}

    # one-sample t-tests on SPB (H0 mean=0); one-sided both directions
    t_stat_two, t_p_two = _ttest_1samp(spb, "two-sided")
    _, t_p_greater = _ttest_1samp(spb, "greater")
    _, t_p_less = _ttest_1samp(spb, "less")

    # Wilcoxon signed-rank on SPB vs 0; one-sided both directions
    w_stat_two, w_p_two = _wilcoxon_0(spb, "two-sided")
    _, w_p_greater = _wilcoxon_0(spb, "greater")
    _, w_p_less = _wilcoxon_0(spb, "less")

    ci_lo, ci_hi = mean_ci_t(spb, alpha=alpha)
    return {
        "n": n,
        "mean_spb": float(spb.mean()),
        "median_spb": float(np.median(spb)),
        "mape": float(np.mean(np.abs(spb))),  # mean absolute percent error
        "t_stat": float(t_stat_two),
        "t_p_two": float(t_p_two),
        "t_p_one_greater": float(t_p_greater),
        "t_p_one_less": float(t_p_less),
        "ci_mean": (float(ci_lo), float(ci_hi)),
        "cohen_d": cohen_d_onesample(spb),
        "wilcoxon_stat": float(w_stat_two),
        "wilcoxon_p_two": float(w_p_two),
        "wilcoxon_p_one_greater": float(w_p_greater),
        "wilcoxon_p_one_less": float(w_p_less),
    }


def run_pair_delta_spb_tests(
        df: pd.DataFrame,
        actual_col: str,
        estimate_col: str,
        min_actual: float,
        spb_cap: float,
        alpha: float,
        pair_col: str = "PAIR_ID",
        ai_col: str = "IS_AI",
) -> Optional[dict]:
    if pair_col not in df.columns or ai_col not in df.columns:
        return None

    dff = df[[pair_col, ai_col, actual_col, estimate_col]].copy()
    n_rows_in = int(len(dff))

    dff = dff.dropna(subset=[pair_col, ai_col, actual_col, estimate_col])
    n_rows_dropna = n_rows_in - int(len(dff))

    dff[ai_col] = safe_bool_series(dff[ai_col])

    act = pd.to_numeric(dff[actual_col], errors="coerce")
    est = pd.to_numeric(dff[estimate_col], errors="coerce")
    ok = act.notna() & est.notna() & (act >= min_actual)
    dff = dff[ok].copy()
    if len(dff) == 0:
        return None

    actv = pd.to_numeric(dff[actual_col], errors="coerce").to_numpy(float)
    estv = pd.to_numeric(dff[estimate_col], errors="coerce").to_numpy(float)

    spb_raw = ((estv - actv) / actv) * 100.0
    n_capped = 0
    spb = spb_raw
    if spb_cap and spb_cap > 0:
        n_capped = int(np.sum(np.abs(spb_raw) > spb_cap))
        spb = np.clip(spb_raw, -spb_cap, spb_cap)
    dff["_spb"] = spb

    # 1 row per PAIR_ID per ai_col value (mean handles duplicates, if any)
    wide = dff.pivot_table(index=pair_col, columns=ai_col, values="_spb", aggfunc="mean")
    n_pairs_total = int(len(wide))
    if True not in wide.columns or False not in wide.columns:
        return None

    wide2 = wide.dropna(subset=[True, False])
    n_pairs_complete = int(len(wide2))
    n_pairs_incomplete = n_pairs_total - n_pairs_complete

    delta = (wide2[True] - wide2[False]).to_numpy(float)
    if delta.size == 0:
        return None

    res = run_spb_tests(delta, alpha=alpha)
    return {
        "n_pairs": res["n"],
        "mean_delta_spb": res["mean_spb"],
        "median_delta_spb": res["median_spb"],
        "mape_delta": res["mape"],
        "t_stat": res["t_stat"],
        "t_p_two": res["t_p_two"],
        "t_p_one_greater": res["t_p_one_greater"],  # AI>Other
        "t_p_one_less": res["t_p_one_less"],        # AI<Other
        "ci_mean_delta": res["ci_mean"],
        "cohen_d": res["cohen_d"],
        "wilcoxon_stat": res["wilcoxon_stat"],
        "wilcoxon_p_two": res["wilcoxon_p_two"],
        "wilcoxon_p_one_greater": res["wilcoxon_p_one_greater"],
        "wilcoxon_p_one_less": res["wilcoxon_p_one_less"],
        "meta": {
            "rows_in": n_rows_in,
            "rows_dropped_by_na_pair_ai_actual_est": n_rows_dropna,
            "rows_kept_after_min_actual_and_numeric": int(len(dff)),
            "pairs_total_in_pivot": n_pairs_total,
            "pairs_complete": n_pairs_complete,
            "pairs_incomplete": n_pairs_incomplete,
            "spb_cap": float(spb_cap),
            "n_capped_rows": n_capped,
        }
    }


# ----------------------------
# Printing + interpretation
# ----------------------------

def _interpret_block(
        kind: str,
        res: dict,
        meta: Optional[dict],
        alpha: float,
        one_sided_context: str,
) -> None:
    """
    kind: "SPB" or "DELTA"
    one_sided_context:
      - For SPB: "SPB>0 means overestimate"
      - For DELTA: "delta>0 means AI>Other"
    """
    if not res or res.get("n", 0) == 0:
        print("    [Interpretation] No valid data -> nothing to interpret.")
        return

    mean_v = res["mean_spb"]
    med_v = res["median_spb"]
    mape = res["mape"]
    ci_lo, ci_hi = res["ci_mean"]
    d = res["cohen_d"]

    # Directional p-values for mean
    p_g = res["t_p_one_greater"]
    p_l = res["t_p_one_less"]
    # Directional p-values for median (Wilcoxon)
    wp_g = res["wilcoxon_p_one_greater"]
    wp_l = res["wilcoxon_p_one_less"]

    # Direction selection
    if mean_v > 0:
        dir_mean = "positive"
        p_dir_mean = p_g
        strength_mean = _p_strength(p_dir_mean, alpha)
    elif mean_v < 0:
        dir_mean = "negative"
        p_dir_mean = p_l
        strength_mean = _p_strength(p_dir_mean, alpha)
    else:
        dir_mean = "zero"
        p_dir_mean = float("nan")
        strength_mean = "none"

    if med_v > 0:
        dir_med = "positive"
        p_dir_med = wp_g
        strength_med = _p_strength(p_dir_med, alpha)
    elif med_v < 0:
        dir_med = "negative"
        p_dir_med = wp_l
        strength_med = _p_strength(p_dir_med, alpha)
    else:
        dir_med = "zero"
        p_dir_med = float("nan")
        strength_med = "none"

    ci_crosses_0 = np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo <= 0.0 <= ci_hi)

    # Main interpretation line (mean-based)
    direction_word = _direction(mean_v) if kind == "SPB" else ("AI uplift (delta>0)" if mean_v > 0 else "AI discount (delta<0)" if mean_v < 0 else "no mean delta")
    mag_label = _bias_magnitude_label(mean_v)
    mape_label = _mape_label(mape)
    d_label = _effect_size_label(d)

    # “Directionally significant” heuristic (mean): one-sided p < alpha AND CI doesn't cross 0
    directional_sig_mean = (np.isfinite(p_dir_mean) and p_dir_mean < alpha and not ci_crosses_0)

    print("    [Interpretation] What the stats imply (heuristic):")
    print(f"      - Direction: {direction_word} ({one_sided_context}).")
    print(f"      - Magnitude (mean): {mean_v:.2f}% -> {mag_label}. Typical absolute error: MAPE={mape:.2f}% -> {mape_label}.")
    print(f"      - Effect size (Cohen's d): {d:.3f} -> {d_label} (standardized mean / std).")
    print(f"      - Mean-based evidence: one-sided p={p_dir_mean:.3e} -> {strength_mean}; CI crosses 0? {ci_crosses_0}.")
    print(f"        => Mean-based conclusion: {'directionally significant' if directional_sig_mean else 'not directionally significant'} at alpha={alpha:.3f}.")

    # Median-based (Wilcoxon) check (robustness)
    directional_sig_med = (np.isfinite(p_dir_med) and p_dir_med < alpha)
    print(f"      - Median-based (Wilcoxon) robustness check: one-sided p={p_dir_med:.3e} -> {strength_med}.")
    if directional_sig_mean != directional_sig_med:
        print("        => NOTE: mean vs median evidence disagrees (likely skew/outliers). Treat as distribution-sensitive.")
    skew_flag = _flag_skew(mean_v, med_v)
    if skew_flag:
        print(f"        => NOTE: {skew_flag}.")

    # Meta / filtering / capping transparency
    if meta:
        cap = meta.get("spb_cap", 0.0)
        n_capped = int(meta.get("n_capped", 0))
        if cap and cap > 0:
            frac = (n_capped / max(1, int(meta.get("n_valid", 0)))) * 100.0
            print(f"      - Clipping: SPB was capped to ±{cap:g}%. Capped points: {n_capped} ({frac:.1f}% of valid rows).")
        dropped = int(meta.get("drop_nan", 0)) + int(meta.get("drop_min_actual", 0))
        if dropped > 0:
            print(f"      - Filtering: dropped {dropped} rows (NaN est/act: {meta.get('drop_nan', 0)}, actual<{meta.get('min_actual', 'min_actual')}: {meta.get('drop_min_actual', 0)}).")


def print_spb_block(title: str, spb_meta: dict, res: dict, alpha: float) -> None:
    if not res or res.get("n", 0) == 0:
        print(f"  {title}: n=0 (no valid rows)")
        # Also show filtering summary if present
        if spb_meta:
            print(f"    Data used: total={spb_meta.get('n_total', 0)}  valid={spb_meta.get('n_valid', 0)}  "
                  f"dropped(NaN)={spb_meta.get('drop_nan', 0)}  dropped(actual<min)={spb_meta.get('drop_min_actual', 0)}")
        return

    ci_lo, ci_hi = res["ci_mean"]
    print(f"  {title}: n={res['n']}")
    print(f"    mean(SPB)={res['mean_spb']:.4f}%  median(SPB)={res['median_spb']:.4f}%  MAPE={res['mape']:.4f}%")
    print(f"    t-test H0(mean SPB=0): t={res['t_stat']:.4f}  p(two)={res['t_p_two']:.3e}  "
          f"p(one, SPB>0)={res['t_p_one_greater']:.3e}  p(one, SPB<0)={res['t_p_one_less']:.3e}")
    print(f"    95% CI mean(SPB): [{ci_lo:.4f}%, {ci_hi:.4f}%]  Cohen's d={res['cohen_d']:.4f}")
    print(f"    Wilcoxon H0(median SPB=0): stat={res['wilcoxon_stat']:.4f}  p(two)={res['wilcoxon_p_two']:.3e}  "
          f"p(one, SPB>0)={res['wilcoxon_p_one_greater']:.3e}  p(one, SPB<0)={res['wilcoxon_p_one_less']:.3e}")

    # Data filtering transparency
    if spb_meta:
        print(f"    Data used: total={spb_meta.get('n_total', 0)}  notna={spb_meta.get('n_notna', 0)}  valid={spb_meta.get('n_valid', 0)}  "
              f"dropped(NaN)={spb_meta.get('drop_nan', 0)}  dropped(actual<min)={spb_meta.get('drop_min_actual', 0)}")
        if spb_meta.get("spb_cap", 0.0) and spb_meta.get("spb_cap", 0.0) > 0:
            print(f"    SPB cap: ±{spb_meta['spb_cap']:.4f}%  capped points={spb_meta.get('n_capped', 0)}")

    # Interpretation
    meta_for_interp = dict(spb_meta or {})
    # include min_actual in meta message without changing compute signature
    meta_for_interp["min_actual"] = meta_for_interp.get("min_actual", None)
    _interpret_block(
        kind="SPB",
        res=res,
        meta=meta_for_interp,
        alpha=alpha,
        one_sided_context="SPB>0 means overestimation; SPB<0 means underestimation",
    )


def print_delta_block(title: str, res: Optional[dict], alpha: float) -> None:
    if not res:
        print(f"  {title}: (skipped; missing columns or no complete pairs)")
        return

    ci_lo, ci_hi = res["ci_mean_delta"]
    print(f"  {title}: n_pairs={res['n_pairs']}")
    print(f"    mean(delta SPB)={res['mean_delta_spb']:.4f}%  median(delta SPB)={res['median_delta_spb']:.4f}%  |delta| mean={res['mape_delta']:.4f}%")
    print(f"    t-test H0(mean delta=0): t={res['t_stat']:.4f}  p(two)={res['t_p_two']:.3e}  "
          f"p(one, AI>Other)={res['t_p_one_greater']:.3e}  p(one, AI<Other)={res['t_p_one_less']:.3e}")
    print(f"    95% CI mean(delta): [{ci_lo:.4f}%, {ci_hi:.4f}%]  Cohen's d={res['cohen_d']:.4f}")
    print(f"    Wilcoxon H0(median delta=0): stat={res['wilcoxon_stat']:.4f}  p(two)={res['wilcoxon_p_two']:.3e}  "
          f"p(one, AI>Other)={res['wilcoxon_p_one_greater']:.3e}  p(one, AI<Other)={res['wilcoxon_p_one_less']:.3e}")

    # Pair construction transparency
    meta = res.get("meta", {}) or {}
    print("    Pair coverage:")
    print(f"      rows_in={meta.get('rows_in', 'na')}  rows_dropped_by_na_pair_ai_actual_est={meta.get('rows_dropped_by_na_pair_ai_actual_est', 'na')}  "
          f"rows_kept_after_min_actual_and_numeric={meta.get('rows_kept_after_min_actual_and_numeric', 'na')}")
    print(f"      pairs_total_in_pivot={meta.get('pairs_total_in_pivot', 'na')}  pairs_complete={meta.get('pairs_complete', 'na')}  pairs_incomplete={meta.get('pairs_incomplete', 'na')}")
    if meta.get("spb_cap", 0.0) and meta.get("spb_cap", 0.0) > 0:
        print(f"      SPB cap: ±{meta.get('spb_cap')}%  capped rows (pre-pivot)={meta.get('n_capped_rows', 0)}")

    # Interpretation
    # Reuse the same interpreter by mapping field names to expected keys
    res_as_spb = {
        "n": res["n_pairs"],
        "mean_spb": res["mean_delta_spb"],
        "median_spb": res["median_delta_spb"],
        "mape": res["mape_delta"],
        "t_stat": res["t_stat"],
        "t_p_two": res["t_p_two"],
        "t_p_one_greater": res["t_p_one_greater"],
        "t_p_one_less": res["t_p_one_less"],
        "ci_mean": res["ci_mean_delta"],
        "cohen_d": res["cohen_d"],
        "wilcoxon_stat": res["wilcoxon_stat"],
        "wilcoxon_p_two": res["wilcoxon_p_two"],
        "wilcoxon_p_one_greater": res["wilcoxon_p_one_greater"],
        "wilcoxon_p_one_less": res["wilcoxon_p_one_less"],
    }
    _interpret_block(
        kind="DELTA",
        res=res_as_spb,
        meta=None,  # detailed meta already printed above
        alpha=alpha,
        one_sided_context="delta>0 means AI titles have higher SPB than matched non-AI; delta<0 means lower SPB",
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Run paired tests using percent error (SPB) across model CSV outputs.")
    ap.add_argument("--estimates-dir", required=True, help="Directory containing per-model CSVs.")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern for model CSV files.")
    ap.add_argument("--actual-col", default="PREVAILING_WAGE", help="Actual wage column (yearly).")
    ap.add_argument("--estimate-col", default="estimated_salary_in_usd", help="Model estimate column (yearly).")
    ap.add_argument("--ai-col", default="IS_AI", help="AI label column (optional).")
    ap.add_argument("--pair-col", default="PAIR_ID", help="Pair id column (optional).")
    ap.add_argument("--min-actual", type=float, default=1.0, help="Drop rows with actual < min-actual (avoids blowups).")
    ap.add_argument("--spb-cap", type=float, default=0.0, help="If >0, clip SPB to [-cap, +cap] before tests.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level used ONLY for printed interpretation heuristics.")
    args = ap.parse_args()

    d = Path(args.estimates_dir)
    if not d.exists() or not d.is_dir():
        raise SystemExit(f"--estimates-dir is not a directory: {d}")

    files = sorted(d.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched glob={args.glob!r} under {d}")

    print_global_heuristics(alpha=args.alpha)

    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            model_name, persona = filename_to_model_tag(fp)
            print(f"\n=== Model: {model_name} | Persona: {persona} ===")
            print(f"  ERROR reading CSV: {e}")
            continue

        model_name, persona = filename_to_model_tag(fp)
        print(f"\n=== Model: {model_name} | Persona: {persona} | File: {fp.name} ===")

        missing = [c for c in (args.actual_col, args.estimate_col) if c not in df.columns]
        if missing:
            print(f"  Skipped: missing columns {missing}")
            continue

        # Overall
        spb_all, meta_all = compute_spb_with_meta(df, args.actual_col, args.estimate_col, args.min_actual, args.spb_cap)
        meta_all["min_actual"] = args.min_actual
        overall_res = run_spb_tests(spb_all, alpha=args.alpha)
        print_spb_block("Overall SPB (est vs actual)", meta_all, overall_res, alpha=args.alpha)

        # Groups + delta
        if args.ai_col in df.columns:
            is_ai = safe_bool_series(df[args.ai_col])

            spb_ai, meta_ai = compute_spb_with_meta(df[is_ai], args.actual_col, args.estimate_col, args.min_actual, args.spb_cap)
            meta_ai["min_actual"] = args.min_actual
            print_spb_block("AI only SPB", meta_ai, run_spb_tests(spb_ai, alpha=args.alpha), alpha=args.alpha)

            spb_ot, meta_ot = compute_spb_with_meta(df[~is_ai], args.actual_col, args.estimate_col, args.min_actual, args.spb_cap)
            meta_ot["min_actual"] = args.min_actual
            print_spb_block("Other only SPB", meta_ot, run_spb_tests(spb_ot, alpha=args.alpha), alpha=args.alpha)

            delta_res = run_pair_delta_spb_tests(
                df, args.actual_col, args.estimate_col,
                min_actual=args.min_actual, spb_cap=args.spb_cap,
                alpha=args.alpha,
                pair_col=args.pair_col, ai_col=args.ai_col
            )
            print_delta_block("Paired delta: SPB(AI) - SPB(Other) per PAIR_ID", delta_res, alpha=args.alpha)
        else:
            print("  Note: IS_AI not present; skipping group and pair-delta tests.")


if __name__ == "__main__":
    main()
