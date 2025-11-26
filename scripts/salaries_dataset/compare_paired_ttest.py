#!/usr/bin/env python3
"""
Compare AI/ML vs Other using SIGNED percent bias:

  SPB_i = (estimated_i - actual_i) / actual_i * 100

Primary question:
  Is AI/ML more positive in signed-% bias than Other?
  Δ_mean = mean(SPB_AI) - mean(SPB_Other)  (AI more positive if Δ_mean > 0)

Statistical tests:
  1) Welch t-test on SPB (mean difference; two-sided + one-sided AI>Other)
  2) Robust companion: Mann–Whitney U test on SPB (distribution shift; two-sided + one-sided AI>Other)
     + Cliff's delta effect size derived from U.

Outputs per file:
  - CSV summary: {base}_analysis_summary.csv
  - Scatter plot (AI/ML actual vs estimated): {base}_ai_ml_comparison.png
Aggregated:
  - salaries_ALL_MODELS_summary.csv
"""

import glob
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------
# IO / discovery helpers
# -------------------------
def load_csv(file_path: str) -> pd.DataFrame | None:
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        return pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        logging.error(f"Failed to read CSV {file_path}: {e}")
        return None


def read_header_cols(file_path: str) -> set[str]:
    try:
        return set(pd.read_csv(file_path, nrows=0).columns)
    except Exception:
        return set()


def discover_estimation_files(dataset_dir: str, input_glob: str, required_cols: set[str]) -> list[str]:
    pattern = os.path.join(dataset_dir, input_glob)
    candidates = sorted(glob.glob(pattern))

    good = []
    for fp in candidates:
        cols = read_header_cols(fp)
        if required_cols.issubset(cols):
            good.append(fp)

    return good


def extract_model_tag_from_filename(file_path: str) -> str:
    stem = Path(file_path).stem
    for prefix in ("llm_estimated_salaries_debug", "llm_estimated_salaries"):
        if stem.startswith(prefix):
            tag = stem[len(prefix):]
            return tag if tag else stem
    return stem


def infer_estimated_col(df: pd.DataFrame, preferred: str = "estimated_salary_in_usd") -> str | None:
    if preferred in df.columns:
        return preferred
    lower_map = {c.lower(): c for c in df.columns}
    for c_lower, orig in lower_map.items():
        if "estimated" in c_lower and "usd" in c_lower:
            return orig
    for c_lower, orig in lower_map.items():
        if "estimated" in c_lower:
            return orig
    return None


def filter_ai_ml_jobs(df: pd.DataFrame, ai_ml_titles: list[str], job_title_column: str):
    mask = df[job_title_column].isin(ai_ml_titles)
    return df[mask].copy(), df[~mask].copy()


# -------------------------
# Numeric helpers / metrics
# -------------------------
def _as_float_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype(float)


def compute_signed_percent_bias(actual: pd.Series, estimated: pd.Series) -> np.ndarray:
    """
    SPB = (estimated - actual) / actual * 100
    Drops NaNs and excludes actual==0.
    Returns a numpy float64 array.
    """
    a = _as_float_series(actual)
    e = _as_float_series(estimated)
    mask = a.notna() & e.notna() & (a != 0)
    spb = (e[mask] - a[mask]) / a[mask] * 100.0
    spb = spb.to_numpy(dtype=np.float64, copy=False)
    # remove infs just in case
    spb = spb[np.isfinite(spb)]
    return spb


def compute_mae(actual: pd.Series, estimated: pd.Series) -> tuple[float, int]:
    a = _as_float_series(actual)
    e = _as_float_series(estimated)
    mask = a.notna() & e.notna()
    a = a[mask]
    e = e[mask]
    n = int(len(a))
    if n == 0:
        return np.nan, 0
    return float(np.mean(np.abs(a - e))), n


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def safe_median(x: np.ndarray) -> float:
    return float(np.median(x)) if len(x) else np.nan


def is_degenerate(x: np.ndarray, atol: float = 0.0) -> bool:
    """
    Degenerate = too small, no variation, or essentially constant.
    This prevents misleading stats (and SciPy warnings) for "all zeros" models.
    """
    if x is None or len(x) < 2:
        return True
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    return abs(xmax - xmin) <= atol


def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    res = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def one_sided_p_ai_greater(t_stat: float, p_two_sided: float) -> float:
    """
    One-sided p-value for H1: mean(AI) > mean(Other), derived from two-sided p under symmetry.
    (Reasonable with large n.)
    """
    if not np.isfinite(t_stat) or not np.isfinite(p_two_sided):
        return np.nan
    if t_stat > 0:
        return p_two_sided / 2.0
    return 1.0 - (p_two_sided / 2.0)


def mwu_test_and_cliffs_delta(ai: np.ndarray, other: np.ndarray) -> tuple[float, float, float, float]:
    """
    Mann–Whitney U (robust companion) + Cliff's delta effect size.

    Returns:
      (U, p_two_sided, p_one_sided_ai_gt_other, cliffs_delta)

    Cliff's delta for AI vs Other can be computed from U:
        delta = 2U/(n1*n2) - 1
    where U is the U statistic for sample1=AI (as returned by SciPy).
    """
    n1 = len(ai)
    n2 = len(other)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Robust nonparametric test (asymptotic is appropriate for large n; handles ties)
    u_two = mannwhitneyu(ai, other, alternative="two-sided", method="asymptotic")
    u_gt = mannwhitneyu(ai, other, alternative="greater", method="asymptotic")

    U = float(u_two.statistic)
    p_two = float(u_two.pvalue)
    p_gt = float(u_gt.pvalue)

    cliffs = (2.0 * U) / (n1 * n2) - 1.0
    return U, p_two, p_gt, float(cliffs)


# -------------------------
# Plotting
# -------------------------
def plot_ai_scatter(actual: pd.Series, estimated: pd.Series, title: str, output_path: str):
    a = _as_float_series(actual)
    e = _as_float_series(estimated)
    mask = a.notna() & e.notna()
    a = a[mask]
    e = e[mask]

    if len(a) == 0:
        logging.warning(f"Skipping plot (no valid rows): {output_path}")
        return

    lo = float(min(a.min(), e.min()))
    hi = float(max(a.max(), e.max()))

    plt.figure(figsize=(10, 6))
    plt.scatter(a, e, alpha=0.25, s=6)
    plt.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Actual Salary (USD)")
    plt.ylabel("Estimated Salary (USD)")
    plt.title(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


# -------------------------
# Per-file analysis
# -------------------------
def run_analysis_for_file(dataset_cfg: dict, file_path: str, ai_ml_titles: list[str]) -> pd.DataFrame | None:
    df = load_csv(file_path)
    if df is None:
        return None

    preprocess = dataset_cfg.get("preprocess")
    if callable(preprocess):
        df = preprocess(df)

    job_title_col = dataset_cfg["job_title_column"]
    actual_col = dataset_cfg["actual_salary_column"]
    est_col = infer_estimated_col(df, preferred=dataset_cfg.get("estimated_salary_column", "estimated_salary_in_usd"))

    if job_title_col not in df.columns:
        logging.error(f"[{dataset_cfg['name']}] Missing '{job_title_col}' in {file_path}")
        return None
    if actual_col not in df.columns:
        logging.error(f"[{dataset_cfg['name']}] Missing '{actual_col}' in {file_path}")
        return None
    if not est_col or est_col not in df.columns:
        logging.error(f"[{dataset_cfg['name']}] Could not find estimated salary column in {file_path}")
        return None

    ai_df, other_df = filter_ai_ml_jobs(df, ai_ml_titles, job_title_col)

    # missing warnings (signal only)
    for label, sub_df in [(f"{dataset_cfg['name']} AI/ML", ai_df), (f"{dataset_cfg['name']} Other", other_df)]:
        missing = int(sub_df[[job_title_col, actual_col, est_col]].isna().sum().sum())
        if missing > 0:
            logging.warning(f"{label} contains {missing} missing values across key columns.")

    # Metrics
    spb_ai = compute_signed_percent_bias(ai_df[actual_col], ai_df[est_col])
    spb_other = compute_signed_percent_bias(other_df[actual_col], other_df[est_col])

    mean_ai = safe_mean(spb_ai)
    mean_other = safe_mean(spb_other)
    median_ai = safe_median(spb_ai)
    median_other = safe_median(spb_other)

    delta_mean = mean_ai - mean_other
    delta_median = median_ai - median_other

    mae_ai, n_mae_ai = compute_mae(ai_df[actual_col], ai_df[est_col])
    mae_other, n_mae_other = compute_mae(other_df[actual_col], other_df[est_col])

    # Primary test: Welch on SPB (mean difference)
    if is_degenerate(spb_ai) and is_degenerate(spb_other):
        t_stat, p_t_two, p_t_one = np.nan, np.nan, np.nan
        logging.warning(f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] SPB appears degenerate; skipping Welch t-test.")
    else:
        t_stat, p_t_two = welch_ttest(spb_ai, spb_other)
        p_t_one = one_sided_p_ai_greater(t_stat, p_t_two)

    # Robust companion: Mann–Whitney U + Cliff's delta
    if is_degenerate(spb_ai) and is_degenerate(spb_other):
        U, p_u_two, p_u_one, cliffs = np.nan, np.nan, np.nan, np.nan
        logging.warning(f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] SPB appears degenerate; skipping MWU.")
    else:
        U, p_u_two, p_u_one, cliffs = mwu_test_and_cliffs_delta(spb_ai, spb_other)

    # Interpretation (your claim) based on Δ_mean > 0
    interpretation = (
        "AI/ML has MORE positive signed-% bias than Other"
        if np.isfinite(delta_mean) and delta_mean > 0
        else "AI/ML does NOT have more positive signed-% bias than Other"
    )

    base = Path(file_path).stem
    model_tag = extract_model_tag_from_filename(file_path)

    logging.info(
        f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] "
        f"SPB_mean_AI={mean_ai:.2f}%, SPB_mean_Other={mean_other:.2f}%, "
        f"Δ_mean={delta_mean:.2f}% -> {interpretation}"
    )
    logging.info(
        f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] "
        f"SPB_median_AI={median_ai:.2f}%, SPB_median_Other={median_other:.2f}%, "
        f"Δ_median={delta_median:.2f}%"
    )
    logging.info(
        f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] "
        f"Welch t-test on SPB: t={t_stat:.4f}, p(two)={p_t_two:.4g}, p(one, AI>Other)={p_t_one:.4g}"
    )
    logging.info(
        f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] "
        f"MWU (robust) on SPB: U={U:.3g}, p(two)={p_u_two:.4g}, p(one, AI>Other)={p_u_one:.4g}, "
        f"Cliff's δ={cliffs:.4g}"
    )

    # Save per-file summary rows (duplicate shared stats for easy aggregation)
    rows = []
    for group_label, n_spb, mean_spb, median_spb, mae_usd, n_mae in [
        ("AI/ML", int(len(spb_ai)), mean_ai, median_ai, mae_ai, n_mae_ai),
        ("Other", int(len(spb_other)), mean_other, median_other, mae_other, n_mae_other),
    ]:
        rows.append(
            {
                "dataset": dataset_cfg["name"],
                "file": os.path.basename(file_path),
                "base_name": base,
                "model_tag": model_tag,
                "group": group_label,
                "n_used_spb": n_spb,
                "n_used_mae": n_mae,
                "mae_usd": mae_usd,
                "mean_spb_percent": mean_spb,
                "median_spb_percent": median_spb,
                "spb_def": "(estimated - actual) / actual * 100",
                "actual_col": actual_col,
                "estimated_col": est_col,
                # main claim quantities
                "delta_mean_spb_ai_minus_other": delta_mean,
                "delta_median_spb_ai_minus_other": delta_median,
                "delta_mean_is_positive": bool(np.isfinite(delta_mean) and delta_mean > 0),
                "interpretation": interpretation,
                # Welch (mean)
                "welch_t_stat": t_stat,
                "welch_p_two_sided": p_t_two,
                "welch_p_one_sided_ai_gt_other": p_t_one,
                # Robust companion (distribution)
                "mwu_U": U,
                "mwu_p_two_sided": p_u_two,
                "mwu_p_one_sided_ai_gt_other": p_u_one,
                "cliffs_delta_ai_vs_other": cliffs,
            }
        )

    results_dir = Path(dataset_cfg["results_dir"])
    plots_dir = Path(dataset_cfg["plots_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = results_dir / f"{base}_analysis_summary.csv"
    plot_png = plots_dir / f"{base}_ai_ml_comparison.png"

    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    plot_ai_scatter(
        ai_df[actual_col],
        ai_df[est_col],
        f"{dataset_cfg['name']} ({model_tag}) AI/ML: Actual vs Estimated",
        str(plot_png),
    )

    logging.info(f"Saved: {summary_csv}")
    logging.info(f"Saved: {plot_png}")

    return pd.DataFrame(rows)


def main():
    ai_ml_titles_separation_file = "data/salaries-for-data-science-jobs/ai_ml_job_titles.csv"

    if not os.path.exists(ai_ml_titles_separation_file):
        logging.error(f"AI/ML job titles file not found: {ai_ml_titles_separation_file}")
        return

    titles_df = pd.read_csv(ai_ml_titles_separation_file)
    if "AI_job" not in titles_df.columns or "job_title" not in titles_df.columns:
        logging.error("ai_ml_job_titles.csv must contain columns: job_title, AI_job")
        return

    ai_ml_titles = titles_df[titles_df["AI_job"] == True]["job_title"].astype(str).tolist()

    datasets = [
        {
            "name": "Salaries",
            "dataset_dir": "data/salaries-for-data-science-jobs",
            "input_glob": "llm_estimated_salaries*.csv",
            "required_cols_for_discovery": {"job_title", "salary_in_usd", "estimated_salary_in_usd"},
            "job_title_column": "job_title",
            "actual_salary_column": "salary_in_usd",
            "estimated_salary_column": "estimated_salary_in_usd",
            "preprocess": None,
            "results_dir": "data/statistical_analysis_results/salaries",
            "plots_dir": "data/statistical_analysis_plots/salaries",
        }
    ]

    for cfg in datasets:
        files = discover_estimation_files(
            dataset_dir=cfg["dataset_dir"],
            input_glob=cfg["input_glob"],
            required_cols=cfg["required_cols_for_discovery"],
        )

        if not files:
            logging.error(
                f"[{cfg['name']}] No matching estimation files found in {cfg['dataset_dir']} "
                f"with pattern '{cfg['input_glob']}' and required columns {sorted(cfg['required_cols_for_discovery'])}."
            )
            continue

        logging.info(f"[{cfg['name']}] Found {len(files)} estimation file(s).")
        all_rows = []

        for fp in files:
            logging.info(f"[{cfg['name']}] Processing: {fp}")
            res = run_analysis_for_file(cfg, fp, ai_ml_titles)
            if res is not None:
                all_rows.append(res)

        if all_rows:
            agg = pd.concat(all_rows, ignore_index=True)
            agg_path = Path(cfg["results_dir"]) / f"{cfg['name'].lower()}_ALL_MODELS_summary.csv"
            agg.to_csv(agg_path, index=False)
            logging.info(f"[{cfg['name']}] Saved aggregated summary: {agg_path}")


if __name__ == "__main__":
    main()
