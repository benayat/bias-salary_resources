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
    --estimates-dir data/.../sampled \
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
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["true", "1", "yes", "y"])


def filename_to_model_tag(path: Path) -> str:
    stem = path.stem
    m = re.match(r"llm_estimated_salaries(?:_debug)?(.*)$", stem)
    if m and m.group(1):
        tag = m.group(1).lstrip("_-")
        return tag or stem
    return stem


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


def compute_spb(
        df: pd.DataFrame,
        actual_col: str,
        estimate_col: str,
        min_actual: float,
        spb_cap: float,
) -> np.ndarray:
    act = pd.to_numeric(df[actual_col], errors="coerce")
    est = pd.to_numeric(df[estimate_col], errors="coerce")
    ok = act.notna() & est.notna() & (act >= min_actual)
    act = act[ok].to_numpy(float)
    est = est[ok].to_numpy(float)
    if act.size == 0:
        return np.array([], dtype=float)

    spb = ((est - act) / act) * 100.0
    if spb_cap and spb_cap > 0:
        spb = np.clip(spb, -spb_cap, spb_cap)
    return spb


def run_spb_tests(spb: np.ndarray) -> dict:
    spb = np.asarray(spb, dtype=float)
    n = int(spb.size)
    if n == 0:
        return {"n": 0}

    # one-sample t-tests on SPB (H0 mean=0); also one-sided (SPB>0)
    t_two = stats.ttest_1samp(spb, popmean=0.0, alternative="two-sided")
    t_g = stats.ttest_1samp(spb, popmean=0.0, alternative="greater")

    # Wilcoxon signed-rank on SPB vs 0 (robust to outliers vs mean-based t-test)
    try:
        w_two = stats.wilcoxon(spb, alternative="two-sided", zero_method="wilcox")
        w_g = stats.wilcoxon(spb, alternative="greater", zero_method="wilcox")
        w_stat, w_p_two, w_p_g = float(w_two.statistic), float(w_two.pvalue), float(w_g.pvalue)
    except Exception:
        w_stat, w_p_two, w_p_g = float("nan"), float("nan"), float("nan")

    ci_lo, ci_hi = mean_ci_t(spb)
    return {
        "n": n,
        "mean_spb": float(spb.mean()),
        "median_spb": float(np.median(spb)),
        "mape": float(np.mean(np.abs(spb))),  # mean absolute percent error
        "t_stat": float(t_two.statistic),
        "t_p_two": float(t_two.pvalue),
        "t_p_one_greater": float(t_g.pvalue),
        "ci95_mean_spb": (ci_lo, ci_hi),
        "cohen_d": cohen_d_onesample(spb),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_two": w_p_two,
        "wilcoxon_p_one_greater": w_p_g,
    }


def run_pair_delta_spb_tests(
        df: pd.DataFrame,
        actual_col: str,
        estimate_col: str,
        min_actual: float,
        spb_cap: float,
        pair_col: str = "PAIR_ID",
        ai_col: str = "IS_AI",
) -> Optional[dict]:
    if pair_col not in df.columns or ai_col not in df.columns:
        return None

    dff = df[[pair_col, ai_col, actual_col, estimate_col]].copy()
    dff = dff.dropna(subset=[pair_col, ai_col, actual_col, estimate_col])
    dff[ai_col] = safe_bool_series(dff[ai_col])

    act = pd.to_numeric(dff[actual_col], errors="coerce")
    est = pd.to_numeric(dff[estimate_col], errors="coerce")
    ok = act.notna() & est.notna() & (act >= min_actual)
    dff = dff[ok].copy()
    act = act[ok].to_numpy(float)
    est = est[ok].to_numpy(float)

    if len(dff) == 0:
        return None

    spb = ((est - act) / act) * 100.0
    if spb_cap and spb_cap > 0:
        spb = np.clip(spb, -spb_cap, spb_cap)
    dff["_spb"] = spb

    wide = dff.pivot_table(index=pair_col, columns=ai_col, values="_spb", aggfunc="mean")
    if True not in wide.columns or False not in wide.columns:
        return None

    wide = wide.dropna(subset=[True, False])
    delta = (wide[True] - wide[False]).to_numpy(float)

    if delta.size == 0:
        return None

    res = run_spb_tests(delta)
    # rename fields for clarity
    return {
        "n_pairs": res["n"],
        "mean_delta_spb": res["mean_spb"],
        "median_delta_spb": res["median_spb"],
        "mape_delta": res["mape"],
        "t_stat": res["t_stat"],
        "t_p_two": res["t_p_two"],
        "t_p_one_greater": res["t_p_one_greater"],  # AI>Other
        "ci95_mean_delta": res["ci95_mean_spb"],
        "cohen_d": res["cohen_d"],
        "wilcoxon_stat": res["wilcoxon_stat"],
        "wilcoxon_p_two": res["wilcoxon_p_two"],
        "wilcoxon_p_one_greater": res["wilcoxon_p_one_greater"],
    }


def print_spb_block(title: str, res: dict) -> None:
    if not res or res.get("n", 0) == 0:
        print(f"  {title}: n=0 (no valid rows)")
        return
    ci = res["ci95_mean_spb"]
    print(f"  {title}: n={res['n']}")
    print(f"    mean(SPB)={res['mean_spb']:.4f}%  median(SPB)={res['median_spb']:.4f}%  MAPE={res['mape']:.4f}%")
    print(f"    t-test H0(mean SPB=0): t={res['t_stat']:.4f}  p(two)={res['t_p_two']:.3e}  p(one, SPB>0)={res['t_p_one_greater']:.3e}")
    print(f"    95% CI mean(SPB): [{ci[0]:.4f}%, {ci[1]:.4f}%]  Cohen's d={res['cohen_d']:.4f}")
    print(f"    Wilcoxon H0(median SPB=0): stat={res['wilcoxon_stat']:.4f}  p(two)={res['wilcoxon_p_two']:.3e}  p(one, SPB>0)={res['wilcoxon_p_one_greater']:.3e}")


def print_delta_block(title: str, res: Optional[dict]) -> None:
    if not res:
        print(f"  {title}: (skipped; missing columns or no complete pairs)")
        return
    ci = res["ci95_mean_delta"]
    print(f"  {title}: n_pairs={res['n_pairs']}")
    print(f"    mean(delta SPB)={res['mean_delta_spb']:.4f}%  median(delta SPB)={res['median_delta_spb']:.4f}%  |delta| mean={res['mape_delta']:.4f}%")
    print(f"    t-test H0(mean delta=0): t={res['t_stat']:.4f}  p(two)={res['t_p_two']:.3e}  p(one, AI>Other)={res['t_p_one_greater']:.3e}")
    print(f"    95% CI mean(delta): [{ci[0]:.4f}%, {ci[1]:.4f}%]  Cohen's d={res['cohen_d']:.4f}")
    print(f"    Wilcoxon H0(median delta=0): stat={res['wilcoxon_stat']:.4f}  p(two)={res['wilcoxon_p_two']:.3e}  p(one, AI>Other)={res['wilcoxon_p_one_greater']:.3e}")


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
    args = ap.parse_args()

    d = Path(args.estimates_dir)
    if not d.exists() or not d.is_dir():
        raise SystemExit(f"--estimates-dir is not a directory: {d}")

    files = sorted(d.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched glob={args.glob!r} under {d}")

    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception as e:
            print(f"\n=== {fp.name} ===")
            print(f"  ERROR reading CSV: {e}")
            continue

        print(f"\n=== Model: {filename_to_model_tag(fp)}  ({fp.name}) ===")

        missing = [c for c in (args.actual_col, args.estimate_col) if c not in df.columns]
        if missing:
            print(f"  Skipped: missing columns {missing}")
            continue

        spb_all = compute_spb(df, args.actual_col, args.estimate_col, args.min_actual, args.spb_cap)
        print_spb_block("Overall SPB (est vs actual)", run_spb_tests(spb_all))

        if args.ai_col in df.columns:
            is_ai = safe_bool_series(df[args.ai_col])
            spb_ai = compute_spb(df[is_ai], args.actual_col, args.estimate_col, args.min_actual, args.spb_cap)
            spb_ot = compute_spb(df[~is_ai], args.actual_col, args.estimate_col, args.min_actual, args.spb_cap)
            print_spb_block("AI only SPB", run_spb_tests(spb_ai))
            print_spb_block("Other only SPB", run_spb_tests(spb_ot))

            delta_res = run_pair_delta_spb_tests(
                df, args.actual_col, args.estimate_col,
                min_actual=args.min_actual, spb_cap=args.spb_cap,
                pair_col=args.pair_col, ai_col=args.ai_col
            )
            print_delta_block("Paired delta: SPB(AI) - SPB(Other) per PAIR_ID", delta_res)
        else:
            print("  Note: IS_AI not present; skipping group and pair-delta tests.")


if __name__ == "__main__":
    main()
