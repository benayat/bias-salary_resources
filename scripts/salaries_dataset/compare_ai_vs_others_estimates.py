import glob
import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def read_header_cols(file_path: str) -> set[str]:
    """Read only the CSV header to validate required columns quickly."""
    try:
        return set(pd.read_csv(file_path, nrows=0).columns)
    except Exception:
        return set()


def filter_ai_ml_jobs(df, ai_ml_titles, job_title_column):
    return df[df[job_title_column].isin(ai_ml_titles)], df[~df[job_title_column].isin(ai_ml_titles)]

def infer_estimated_col(df: pd.DataFrame, preferred: str = "estimated_salary_in_usd") -> str | None:
    """
    Prefer exact column name; fallback to heuristic search for an "estimated" column.
    """
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


def extract_model_tag_from_filename(file_path: str) -> str:
    """
    Your estimator writes:
      llm_estimated_salaries{MODEL_TAG}.csv
      llm_estimated_salaries_debug{MODEL_TAG}.csv
    This pulls MODEL_TAG out for reporting.
    """
    stem = Path(file_path).stem
    for prefix in ("llm_estimated_salaries_debug", "llm_estimated_salaries"):
        if stem.startswith(prefix):
            tag = stem[len(prefix):]
            return tag if tag else stem
    return stem

def discover_estimation_files(dataset_dir: str, input_glob: str, required_cols: set[str]) -> list[str]:
    """
    Discover estimation output CSVs matching glob pattern and validate required columns exist.
    Example estimator naming: llm_estimated_salaries{MODEL_TAG}.csv, llm_estimated_salaries_debug{MODEL_TAG}.csv
    """
    pattern = os.path.join(dataset_dir, input_glob)
    candidates = sorted(glob.glob(pattern))

    good = []
    for fp in candidates:
        cols = read_header_cols(fp)
        if required_cols.issubset(cols):
            good.append(fp)

    return good


def calculate_statistics(actual: pd.Series, estimated: pd.Series):
    """
    Returns (mae, mpe_percent, n_used).
    Drops NaNs. MPE ignores rows where actual==0 (avoid divide by zero).
    """
    mask = actual.notna() & estimated.notna()
    a = pd.to_numeric(actual[mask], errors="coerce")
    e = pd.to_numeric(estimated[mask], errors="coerce")
    mask2 = a.notna() & e.notna()
    a = a[mask2].astype(float)
    e = e[mask2].astype(float)

    n = int(len(a))
    if n == 0:
        return np.nan, np.nan, 0

    mae = float(np.mean(np.abs(a - e)))

    nonzero = a != 0
    mpe = float(np.mean(((a[nonzero] - e[nonzero]) / a[nonzero]) * 100)) if nonzero.any() else np.nan

    return mae, mpe, n

def perform_ttest(actual, estimated):
    return ttest_ind(actual, estimated, equal_var=False)

def perform_anova(groups):
    return f_oneway(*groups)

def plot_comparison(actual, estimated, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, estimated, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--')
    plt.xlabel('Actual Salary (USD)')
    plt.ylabel('Estimated Salary (USD)')
    plt.title(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

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
        logging.error(f"[{dataset_cfg['name']}] Missing '{actual_col}' in {file_path} (preprocess may have failed)")
        return None
    if not est_col or est_col not in df.columns:
        logging.error(f"[{dataset_cfg['name']}] Could not find estimated salary column in {file_path}")
        return None

    # Split
    ai_ml_df, other_df = filter_ai_ml_jobs(df, ai_ml_titles, job_title_col)

    # Missing data warnings (rough signal)
    for label, sub_df in [(f"{dataset_cfg['name']} AI/ML", ai_ml_df), (f"{dataset_cfg['name']} Other", other_df)]:
        missing = int(sub_df[[job_title_col, actual_col, est_col]].isna().sum().sum())
        if missing > 0:
            logging.warning(f"{label} contains {missing} missing values across key columns.")

    # Stats rows
    base = Path(file_path).stem
    model_tag = extract_model_tag_from_filename(file_path)

    rows = []
    for group_label, sub_df in [("AI/ML", ai_ml_df), ("Other", other_df)]:
        mae, mpe, n = calculate_statistics(sub_df[actual_col], sub_df[est_col])
        rows.append({
            "dataset": dataset_cfg["name"],
            "file": os.path.basename(file_path),
            "base_name": base,
            "model_tag": model_tag,
            "group": group_label,
            "n_used": n,
            "mae": mae,
            "mpe_percent": mpe,
            "actual_col": actual_col,
            "estimated_col": est_col,
        })
        logging.info(f"[{dataset_cfg['name']} | {os.path.basename(file_path)} | {group_label}] "
                     f"n={n}, MAE={mae:.2f}, MPE={mpe:.2f}%")

    # T-test (AI/ML actual vs estimated)
    t = perform_ttest(ai_ml_df[actual_col], ai_ml_df[est_col])
    if t is None:
        t_stat, p_val = np.nan, np.nan
        logging.warning(f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] Not enough data for t-test.")
    else:
        t_stat, p_val = float(t.statistic), float(t.pvalue)
        logging.info(f"[{dataset_cfg['name']} | {os.path.basename(file_path)}] "
                     f"T-test (AI/ML actual vs estimated): t={t_stat:.4f}, p={p_val:.4g}")

    # Attach t-test results to both rows (convenient for aggregation)
    for r in rows:
        r["ttest_t_stat_ai_ml"] = t_stat
        r["ttest_p_value_ai_ml"] = p_val

    # Save outputs: original filename + suffix
    results_dir = Path(dataset_cfg["results_dir"])
    plots_dir = Path(dataset_cfg["plots_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = results_dir / f"{base}_analysis_summary.csv"
    plot_png = plots_dir / f"{base}_ai_ml_comparison.png"

    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    plot_comparison(
        ai_ml_df[actual_col],
        ai_ml_df[est_col],
        f"{dataset_cfg['name']} ({model_tag}) AI/ML: Actual vs Estimated",
        str(plot_png),
    )

    logging.info(f"Saved: {summary_csv}")
    logging.info(f"Saved: {plot_png}")

    return pd.DataFrame(rows)




def main():
    # File paths
    salaries_file = 'data/salaries-for-data-science-jobs/llm_estimated_salaries.csv'
    # h1b_file = 'data/h1b-lca-disclosure-data-2020-2024/llm_estimated_salaries.csv'
    ai_ml_titles_separation_file = 'data/salaries-for-data-science-jobs/ai_ml_job_titles.csv'

    # Load datasets
    salaries_df = load_csv(salaries_file)

    if salaries_df is None:
        logging.error("Salary dataset could not be loaded. Exiting.")
        return

    # Load AI/ML job titles
    if not os.path.exists(ai_ml_titles_separation_file):
        logging.error(f"AI/ML job titles file not found: {ai_ml_titles_separation_file}")
        return

    # with open(ai_ml_titles_separation_file, 'r') as f:
    #     ai_ml_titles = [line.strip() for line in f]
    ai_ml_titles_df = pd.read_csv(ai_ml_titles_separation_file)
    ai_ml_titles = ai_ml_titles_df[ai_ml_titles_df['AI_job'] == True]['job_title'].tolist()
    # Define dataset-specific columns
    datasets = [
        {
            "name": "Salaries",
            "dataset_dir": "data/salaries-for-data-science-jobs",
            # Matches your estimator-style naming; include debug outputs too
            "input_glob": "llm_estimated_salaries*.csv",
            # Here salary_in_usd is expected to already exist in the created estimation CSV
            "required_cols_for_discovery": {"job_title", "salary_in_usd", "estimated_salary_in_usd"},
            "job_title_column": "job_title",
            "actual_salary_column": "salary_in_usd",
            "estimated_salary_column": "estimated_salary_in_usd",
            "preprocess": None,
            "results_dir": "data/statistical_analysis_results/salaries",
            "plots_dir": "data/statistical_analysis_plots/salaries",
        },
    ]

    for cfg in datasets:
        files = discover_estimation_files(
            dataset_dir=cfg["dataset_dir"],
            input_glob=cfg["input_glob"],
            required_cols=cfg["required_cols_for_discovery"],
        )

        if not files:
            logging.error(f"[{cfg['name']}] No matching estimation files found in {cfg['dataset_dir']} "
                          f"with pattern '{cfg['input_glob']}' and required columns {sorted(cfg['required_cols_for_discovery'])}.")
            continue

        logging.info(f"[{cfg['name']}] Found {len(files)} estimation file(s).")
        all_rows = []

        for fp in files:
            logging.info(f"[{cfg['name']}] Processing: {fp}")
            res = run_analysis_for_file(cfg, fp, ai_ml_titles)
            if res is not None:
                all_rows.append(res)

        # Save aggregated summary per dataset
        if all_rows:
            agg = pd.concat(all_rows, ignore_index=True)
            agg_path = Path(cfg["results_dir"]) / f"{cfg['name'].lower()}_ALL_MODELS_summary.csv"
            agg.to_csv(agg_path, index=False)
            logging.info(f"[{cfg['name']}] Saved aggregated summary: {agg_path}")


if __name__ == '__main__':
    main()
