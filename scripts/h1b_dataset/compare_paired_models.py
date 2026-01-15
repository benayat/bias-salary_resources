#!/usr/bin/env python3
"""
Loads predictions from multiple models on the SAME set of jobs (assuming identical
job identifiers across CSVs), then calculates the mean prediction across models
for each job.

This allows testing whether the average model behavior (across N models) shows
significant AI uplift, with proper paired structure.

For each job:
  - Calculate mean estimated salary across all models
  - Compare AI vs non-AI jobs using this averaged estimate
  - Get CI and p-value with N = number of jobs
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def safe_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])

def load_predictions(filepath, args):
    """
    Load a model's predictions.
    Returns DataFrame with estimate column and AI indicator.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return None

    # Check columns
    needed = [args.actual_col, args.estimate_col, args.ai_col]
    if args.job_id_col:
        needed.append(args.job_id_col)
    
    if not all(c in df.columns for c in needed):
        print(f"  Missing columns in {filepath.name}")
        return None

    # Filter valid rows
    act = pd.to_numeric(df[args.actual_col], errors='coerce')
    est = pd.to_numeric(df[args.estimate_col], errors='coerce')
    
    mask = (act >= args.min_actual) & act.notna() & est.notna()
    if not mask.any():
        return None
    
    df = df[mask].copy()
    
    # Keep relevant columns
    keep_cols = [args.actual_col, args.estimate_col, args.ai_col]
    if args.job_id_col:
        keep_cols.append(args.job_id_col)
    
    df = df[keep_cols].copy()
    df['IS_AI'] = safe_bool_series(df[args.ai_col])
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Calculate mean predictions across models per job, then test AI uplift"
    )
    parser.add_argument("--open-dir", required=True, help="Directory with open model predictions")
    parser.add_argument("--closed-dir", required=True, help="Directory with closed model predictions")
    parser.add_argument("--pattern", default="*.csv", help="File pattern")
    parser.add_argument("--actual-col", default="PREVAILING_WAGE")
    parser.add_argument("--estimate-col", default="estimated_salary_in_usd")
    parser.add_argument("--ai-col", default="IS_AI")
    parser.add_argument("--job-id-col", default=None, 
                        help="Column to use as job identifier (if None, uses row index)")
    parser.add_argument("--min-actual", type=float, default=1.0)
    args = parser.parse_args()

    # Process each cohort
    for cohort_name, cohort_dir in [("OPEN", args.open_dir), ("CLOSED", args.closed_dir)]:
        print("\n" + "="*70)
        print(f"{cohort_name} MODELS - MEAN PREDICTIONS ACROSS MODELS")
        print("="*70)
        
        cohort_path = Path(cohort_dir)
        all_dfs = []
        model_names = []
        
        # Load all models
        print(f"\nLoading models from {cohort_dir}...")
        for f in sorted(cohort_path.glob(args.pattern)):
            df = load_predictions(f, args)
            if df is not None:
                all_dfs.append(df)
                model_names.append(f.stem)
                print(f"  ✓ {f.stem}: {len(df)} jobs")
        
        if not all_dfs:
            print(f"No valid models found in {cohort_dir}")
            continue
        
        print(f"\nTotal models loaded: {len(all_dfs)}")
        
        # Check if we can use job IDs for alignment
        if args.job_id_col:
            print(f"\nAligning by job ID column: {args.job_id_col}")
            # Merge on job_id_col
            base = all_dfs[0][[args.job_id_col, args.actual_col, 'IS_AI']].copy()
            
            for i, (df, name) in enumerate(zip(all_dfs, model_names)):
                df_est = df[[args.job_id_col, args.estimate_col]].copy()
                df_est = df_est.rename(columns={args.estimate_col: f'est_{i}'})
                base = base.merge(df_est, on=args.job_id_col, how='inner')
            
            # Calculate mean estimate across models
            est_cols = [f'est_{i}' for i in range(len(all_dfs))]
            base['mean_estimate'] = base[est_cols].mean(axis=1)
            
            aligned_df = base
            print(f"Aligned jobs: {len(aligned_df)}")
            
        else:
            print("\nNo job ID column - using row index alignment")
            print("Warning: This assumes all CSVs have identical row ordering!")
            
            # Check lengths
            lengths = [len(df) for df in all_dfs]
            if len(set(lengths)) > 1:
                print(f"ERROR: CSVs have different lengths: {lengths}")
                print("Cannot align without job IDs. Use --job-id-col.")
                continue
            
            # Assume identical ordering, stack estimates
            estimates_matrix = np.column_stack([df[args.estimate_col].values for df in all_dfs])
            mean_estimates = estimates_matrix.mean(axis=1)
            
            # Use first model's metadata
            aligned_df = all_dfs[0][[args.actual_col, 'IS_AI']].copy()
            aligned_df['mean_estimate'] = mean_estimates
            
            print(f"Aligned jobs: {len(aligned_df)}")
        
        # Now calculate SPB using mean estimates
        aligned_df['SPB'] = ((aligned_df['mean_estimate'] - aligned_df[args.actual_col]) 
                             / aligned_df[args.actual_col]) * 100.0
        
        # Split by AI/non-AI
        spb_ai = aligned_df.loc[aligned_df['IS_AI'], 'SPB']
        spb_other = aligned_df.loc[~aligned_df['IS_AI'], 'SPB']
        
        n_ai = len(spb_ai)
        n_other = len(spb_other)
        
        if n_ai == 0 or n_other == 0:
            print("ERROR: No AI or non-AI jobs found")
            continue
        
        # Calculate statistics
        mean_ai = spb_ai.mean()
        mean_other = spb_other.mean()
        uplift = mean_ai - mean_other
        
        median_ai = spb_ai.median()
        median_other = spb_other.median()
        
        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(spb_ai, spb_other, equal_var=False)
        
        # Calculate CI for uplift
        var_ai = spb_ai.var(ddof=1)
        var_other = spb_other.var(ddof=1)
        se_diff = np.sqrt(var_ai/n_ai + var_other/n_other)
        
        # Welch-Satterthwaite df
        num = (var_ai/n_ai + var_other/n_other)**2
        den = ((var_ai/n_ai)**2 / (n_ai-1)) + ((var_other/n_other)**2 / (n_other-1))
        df = num / den
        
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = uplift - t_crit * se_diff
        ci_upper = uplift + t_crit * se_diff
        
        # Print results
        print("\n" + "-"*70)
        print("RESULTS (using mean estimate across all models)")
        print("-"*70)
        print(f"Number of models averaged: {len(all_dfs)}")
        print(f"Number of jobs: {len(aligned_df):,}")
        print(f"  AI jobs: {n_ai:,}")
        print(f"  Non-AI jobs: {n_other:,}")
        print()
        print(f"AI jobs:")
        print(f"  Mean SPB: {mean_ai:.4f}%")
        print(f"  Median SPB: {median_ai:.4f}%")
        print(f"  Std Dev: {spb_ai.std():.4f}")
        print()
        print(f"Non-AI jobs:")
        print(f"  Mean SPB: {mean_other:.4f}%")
        print(f"  Median SPB: {median_other:.4f}%")
        print(f"  Std Dev: {spb_other.std():.4f}")
        print()
        print(f"AI UPLIFT: {uplift:.4f} pp")
        print(f"Standard Error: {se_diff:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] pp")
        print()
        print(f"Welch's t-test (AI vs Non-AI):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4e}")
        print(f"  degrees of freedom: {df:.2f}")
        print(f"  → {'✓ Significant AI uplift (p<0.05)' if p_val < 0.05 else '✗ Not significant (p≥0.05)'}")
        
        # Additional: Test if mean AI uplift is different from zero (one-sample)
        t_stat_one = uplift / se_diff
        p_val_one = 2 * (1 - stats.t.cdf(abs(t_stat_one), df))
        
        print()
        print(f"One-sample test (H0: AI uplift = 0):")
        print(f"  t-statistic: {t_stat_one:.4f}")
        print(f"  p-value: {p_val_one:.4e}")
        print(f"  → {'✓ Significantly different from zero (p<0.05)' if p_val_one < 0.05 else '✗ Not significant (p≥0.05)'}")

if __name__ == "__main__":
    main()

