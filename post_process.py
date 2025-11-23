#!/usr/bin/env python3
"""
Compare model compensation estimates vs BLS actuals.

- Reads BLS "table.csv" and extracts US-national median annual wages by SOC.
- Reads each model's outputs under data/bls_evals/*.jsonl
  * Legacy format per line: {"soc_code":"15-xxxx", "estimate_usd": 123456}
  * New format per line: a single number (or 'null'). Requires an order file that
    lists SOC codes line-by-line to align with outputs.

Outputs:
- Prints per-group summaries&archives (Tech 15-* vs Other).
- Welch/Student t-test, normality checks (Shapiro for n<=50), Levene variance test.
- Effect sizes (Cohen's d, Hedges' g).
- Per-model CSV with detailed rows.
- An aggregate models_summary.csv with headline metrics.

Usage (defaults are fine if your files are in-place):
    python compare_bls.py \
      --bls-csv table.csv \
      --evals-dir data/bls_evals \
      --order-file eval_order.txt   # optional; required for single-number outputs

Where eval_order.txt is one SOC code per line in the exact query order, e.g.:
    11-0000
    13-2011
    15-1252
    ...
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def load_actual_bls(bls_csv: str) -> Dict[str, int]:
    """
    Parse BLS table.csv -> {SOC: annual_median_usd}
    Expects a first column like "Computer occupations (15-0000)" and a column
    named 'Annual median wage  (2)' with values like '$104,200'.
    """
    actual = {}
    with open(bls_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            wage_idx = header.index("Annual median wage  (2)")
        except ValueError:
            raise RuntimeError(
                "Could not find 'Annual median wage  (2)' column in BLS CSV header."
            )

        for row in reader:
            if len(row) <= wage_idx:
                continue
            soc_full = (row[0] or "").strip()
            # Extract SOC code from a trailing "(xx-xxxx)"
            m_soc = re.search(r"\(([0-9]{2}-[0-9]{4})\)\s*$", soc_full)
            if not m_soc:
                continue
            soc = m_soc.group(1)

            val = (row[wage_idx] or "").strip()
            # Skip missing footnote placeholders
            if not val or val.startswith("(5)") or val == "(5) -" or val == "-":
                continue
            m = re.search(r"\$([0-9,]+)", val)
            if not m:
                continue
            usd = int(m.group(1).replace(",", ""))
            actual[soc] = usd

    if not actual:
        raise RuntimeError("No BLS rows parsed — check table.csv format.")
    return actual


def load_order_file(order_file: Optional[str]) -> Optional[List[str]]:
    if not order_file:
        return None
    p = Path(order_file)
    if not p.exists():
        raise FileNotFoundError(f"Order file not found: {order_file}")
    socs = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # allow comments
            if line.startswith("#"):
                continue
            # accept "15-1234, some note" -> take first token that looks like SOC
            m = re.search(r"\b([0-9]{2}-[0-9]{4})\b", line)
            socs.append(m.group(1) if m else line)
    return socs


def parse_estimate_from_line(line: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Try legacy JSON first; else try single-number line.
    Returns (soc_code, estimate_usd) where soc_code may be None for single-number lines.
    """
    s = line.strip()
    if not s:
        return None, None

    # Legacy JSONL format
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            soc = obj.get("soc_code")
            est = obj.get("estimate_usd")
            if isinstance(est, (int, float)):
                return soc, float(est)
            # tolerate 'estimate' or 'value' as fallback keys
            for k in ("estimate", "value"):
                if k in obj and isinstance(obj[k], (int, float)):
                    return soc, float(obj[k])
            # tolerate stringified numbers
            for k in ("estimate_usd", "estimate", "value"):
                v = obj.get(k)
                if isinstance(v, str):
                    v = v.strip()
                    if v.lower() == "null":
                        return soc, None
                    m = re.match(r"^[0-9]+(\.[0-9]+)?$", v)
                    if m:
                        return soc, float(v)
        elif isinstance(obj, (int, float)):
            return None, float(obj)
        elif obj is None:
            return None, None
    except json.JSONDecodeError:
        pass

    # Single-number (or 'null') line
    if s.lower() == "null":
        return None, None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$", s)
    if m:
        return None, float(m.group(1))

    # Try to salvage a $ or numeric token from garbage-y lines
    m = re.search(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", s)
    if m:
        return None, float(m.group(1).replace(",", ""))

    return None, None


def load_model_file(
    jsonl_path: str, order_socs: Optional[List[str]]
) -> Dict[str, float]:
    """
    Returns {SOC: estimate_usd}. Supports both formats.
    If lines contain no SOC codes, requires order_socs to map by line number.
    """
    est: Dict[str, float] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            soc, val = parse_estimate_from_line(line)
            if val is None:
                continue
            if soc:
                est[soc] = float(val)
            else:
                if not order_socs:
                    # Without SOC on the line or an order file, we can't map.
                    continue
                if i >= len(order_socs):
                    # Extra lines beyond order file — ignore
                    continue
                est[order_socs[i]] = float(val)
    return est


def group_by_soc_prefix(
    actual: Dict[str, int], estimated: Dict[str, float]
) -> Tuple[List[Tuple[str, int, float, float, float]], List[Tuple[str, int, float, float, float]]]:
    """
    Build (soc, actual, estimated, abs_diff, pct_diff) tuples for:
        - tech_group: SOC starting with '15-'
        - other_group: everything else
    pct_diff = 100 * (estimated - actual) / actual
    """
    tech, other = [], []
    for soc, act in actual.items():
        if soc not in estimated:
            continue
        est = float(estimated[soc])
        abs_diff = est - act
        pct_diff = 100.0 * abs_diff / act if act != 0 else math.nan
        rec = (soc, act, est, abs_diff, pct_diff)
        (tech if soc.startswith("15-") else other).append(rec)
    return tech, other


def summarize_group(group: List[Tuple[str, int, float, float, float]], name: str) -> Dict[str, float]:
    if not group:
        print(f"{name}: No data")
        return {}
    abs_diffs = np.array([g[3] for g in group], dtype=float)
    pct_diffs = np.array([g[4] for g in group], dtype=float)
    act_vals = np.array([g[1] for g in group], dtype=float)
    est_vals = np.array([g[2] for g in group], dtype=float)

    mean_signed_pct = np.nanmean(pct_diffs)
    mape = np.nanmean(np.abs(pct_diffs))
    mae = np.nanmean(np.abs(abs_diffs))
    rmse = math.sqrt(np.nanmean((abs_diffs) ** 2))
    corr = np.corrcoef(act_vals, est_vals)[0, 1] if len(group) >= 2 else math.nan

    print(f"{name}:")
    print(f"  Count: {len(group)}")
    print(f"  Mean signed % error (est-actual): {mean_signed_pct:.2f}%")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  MAE: ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  Pearson r (actual vs est): {corr:.3f}\n")

    return {
        "count": len(group),
        "mean_signed_pct": float(mean_signed_pct),
        "mape": float(mape),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson_r": float(corr) if not math.isnan(corr) else math.nan,
    }


def normality_report(vals: List[float], label: str):
    if len(vals) <= 50 and len(vals) >= 3:
        sh = stats.shapiro(vals)
        print(f"{label} normality (Shapiro-Wilk): W={sh.statistic:.4f}, p={sh.pvalue:.4f}")
        print("  Normally distributed." if sh.pvalue >= 0.05 else "  Not normally distributed.")


def effect_sizes(mean_a, mean_b, var_a, var_b, n_a, n_b) -> Tuple[float, float]:
    """
    Cohen's d (pooled SD) and Hedges' g (small-sample correction).
    """
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    d = (mean_a - mean_b) / math.sqrt(pooled_var) if pooled_var > 0 else math.nan
    # Hedges' g correction factor J
    J = 1 - (3 / (4 * (n_a + n_b) - 9)) if (n_a + n_b) > 2 else 1.0
    g = d * J
    return d, g


def write_model_csv(model_name: str, rows: List[Tuple[str, int, float, float, float]]):
    out = Path(f"comparison_results_{model_name}.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["soc_code", "actual_usd", "estimated_usd", "abs_diff", "pct_diff"])
        for soc, act, est, abs_diff, pct_diff in sorted(rows, key=lambda x: x[0]):
            w.writerow([soc, act, f"{est:.0f}", f"{abs_diff:.0f}", f"{pct_diff:.2f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bls-csv", default="table.csv", help="Path to BLS table.csv")
    ap.add_argument("--evals-dir", default="data/bls_evals", help="Dir with bls_results_*.jsonl")
    ap.add_argument("--order-file", default=None, help="Optional SOC order file for single-number outputs")
    args = ap.parse_args()

    actual_data = load_actual_bls(args.bls_csv)
    order_socs = load_order_file(args.order_file) if args.order_file else None

    evals_dir = Path(args.evals_dir)
    if not evals_dir.exists():
        print(f"Evals dir not found: {evals_dir}")
        return

    summary_rows = []  # for cross-model summary CSV

    for filename in sorted(os.listdir(evals_dir)):
        if not (filename.endswith(".jsonl") and filename.startswith("bls_results_")):
            continue
        model_name = filename[len("bls_results_") : -len(".jsonl")]
        filepath = str(evals_dir / filename)

        estimated_data = load_model_file(filepath, order_socs)

        tech_group, other_group = group_by_soc_prefix(actual_data, estimated_data)

        print(f"\n=== Results for model: {model_name} ===")
        tech_stats = summarize_group(tech_group, "Tech (15-*)")
        other_stats = summarize_group(other_group, "Other")

        # Hypothesis test on signed % errors between groups
        if tech_group and other_group:
            tech_pct = [g[4] for g in tech_group]
            other_pct = [g[4] for g in other_group]

            normality_report(tech_pct, "Tech (pct-diff)")
            normality_report(other_pct, "Other (pct-diff)")

            levene = stats.levene(tech_pct, other_pct, center="median")
            print(f"Levene's test (equal variances): W={levene.statistic:.4f}, p={levene.pvalue:.4f}")
            equal_var = levene.pvalue >= 0.05

            t_stat, p_val = stats.ttest_ind(tech_pct, other_pct, equal_var=equal_var)
            print("T-test on signed % error (Tech vs Other):")
            print(f"  {'Student' if equal_var else 'Welch'} t={t_stat:.4f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  Significant difference (p < 0.05)")
            else:
                print("  No significant difference (p >= 0.05)")

            mean_t, mean_o = np.mean(tech_pct), np.mean(other_pct)
            var_t = np.var(tech_pct, ddof=1) if len(tech_pct) > 1 else float("nan")
            var_o = np.var(other_pct, ddof=1) if len(other_pct) > 1 else float("nan")
            d, g = effect_sizes(mean_t, mean_o, var_t, var_o, len(tech_pct), len(other_pct))
            print(f"  Cohen's d: {d:.4f} | Hedges' g: {g:.4f}\n")
        else:
            print("Not enough data for t-test.\n")

        # Persist detailed CSVs
        write_model_csv(model_name, tech_group + other_group)

        # Summarize per model
        summary_rows.append(
            [
                model_name,
                tech_stats.get("count", 0) if tech_stats else 0,
                tech_stats.get("mean_signed_pct", math.nan) if tech_stats else math.nan,
                tech_stats.get("mape", math.nan) if tech_stats else math.nan,
                other_stats.get("count", 0) if other_stats else 0,
                other_stats.get("mean_signed_pct", math.nan) if other_stats else math.nan,
                other_stats.get("mape", math.nan) if other_stats else math.nan,
            ]
        )

    # Cross-model summary CSV
    if summary_rows:
        with open("data/summaries&archives/models_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "model",
                    "tech_n",
                    "tech_mean_signed_pct",
                    "tech_mape",
                    "other_n",
                    "other_mean_signed_pct",
                    "other_mape",
                ]
            )
            for row in summary_rows:
                w.writerow(row)

    print("\nDone. Wrote per-model CSVs and models_summary.csv (if any models found).")


if __name__ == "__main__":
    main()
