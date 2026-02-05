#!/usr/bin/env python3
"""
Calculates "AI Uplift" per model:
    Uplift_i = Mean(SPB_AI) - Mean(SPB_Other)

Then runs a Welch's t-test comparing the Uplift values of Open vs. Closed groups,
including the 95% Confidence Interval for the difference.
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

def get_model_uplift(filepath, args):
    """
    Returns the scalar 'Uplift' (Mean SPB AI - Mean SPB Other) for one model.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception:
        return None

    # Check columns
    needed = [args.actual_col, args.estimate_col, args.ai_col]
    if not all(c in df.columns for c in needed):
        return None

    # 1. Calculate SPB
    act = pd.to_numeric(df[args.actual_col], errors='coerce')
    est = pd.to_numeric(df[args.estimate_col], errors='coerce')

    mask = (act >= args.min_actual) & act.notna() & est.notna()
    if not mask.any(): return None

    df = df[mask].copy()
    df['SPB'] = ((est - act) / act) * 100.0

    if args.spb_cap > 0:
        df['SPB'] = df['SPB'].clip(-args.spb_cap, args.spb_cap)

    # 2. Split by AI
    is_ai = safe_bool_series(df[args.ai_col])

    spb_ai = df.loc[is_ai, 'SPB']
    spb_other = df.loc[~is_ai, 'SPB']

    if len(spb_ai) == 0 or len(spb_other) == 0:
        return None

    # 3. Calculate Uplift (The "AI Premium")
    uplift = spb_ai.mean() - spb_other.mean()

    return {
        "name": filepath.stem,
        "uplift": uplift,
        "mean_ai": spb_ai.mean(),
        "mean_other": spb_other.mean()
    }

def calculate_cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two independent groups.
    Uses pooled standard deviation.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_sd

    return cohens_d

def permutation_test(group1, group2, n_permutations=10000, random_seed=42):
    """
    Non-parametric permutation test for difference in means.

    Args:
        group1: Array-like, first group values
        group2: Array-like, second group values
        n_permutations: Number of permutation iterations
        random_seed: Random seed for reproducibility

    Returns:
        p_value: Two-tailed p-value
        observed_diff: Observed difference (group2 - group1)
        null_distribution: Array of permuted differences
    """
    np.random.seed(random_seed)

    group1 = np.array(group1)
    group2 = np.array(group2)

    # Observed difference
    observed_diff = np.mean(group2) - np.mean(group1)

    # Combined data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n_total = len(combined)

    # Generate permutation distribution
    null_distribution = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle and split
        shuffled = np.random.permutation(combined)
        perm_group1 = shuffled[:n1]
        perm_group2 = shuffled[n1:]

        # Calculate permuted difference
        null_distribution[i] = np.mean(perm_group2) - np.mean(perm_group1)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_diff))

    return p_value, observed_diff, null_distribution

def calculate_welch_ci(group1, group2, alpha=0.05):
    """
    Calculates the Confidence Interval for the difference between two means
    assuming unequal variances (Welch's).
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Standard Error of the difference
    se_diff = np.sqrt(v1/n1 + v2/n2)

    # Degrees of Freedom (Welch-Satterthwaite equation)
    num = (v1/n1 + v2/n2)**2
    den = ( (v1/n1)**2 / (n1-1) ) + ( (v2/n2)**2 / (n2-1) )
    df = num / den

    # Critical t-value
    t_crit = stats.t.ppf(1 - alpha/2, df)

    # Margin of Error
    margin = t_crit * se_diff

    diff = m2 - m1 # Direction: Group 2 - Group 1

    return diff, (diff - margin, diff + margin), df

def main():
    parser = argparse.ArgumentParser(description="Compare AI Uplift Bias (Open vs Closed) with CI")
    parser.add_argument("--open-dir", required=True)
    parser.add_argument("--closed-dir", required=True)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--actual-col", default="PREVAILING_WAGE")
    parser.add_argument("--estimate-col", default="estimated_salary_in_usd")
    parser.add_argument("--ai-col", default="IS_AI")
    parser.add_argument("--min-actual", type=float, default=1.0)
    parser.add_argument("--spb-cap", type=float, default=0.0)
    parser.add_argument("--n-permutations", type=int, default=10000,
                        help="Number of permutations for permutation test (default: 10000)")
    args = parser.parse_args()

    open_uplifts = []
    closed_uplifts = []

    print(f"--- Processing Open Models ({args.open_dir}) ---")
    for f in sorted(Path(args.open_dir).glob(args.pattern)):
        res = get_model_uplift(f, args)
        if res:
            open_uplifts.append(res['uplift'])
            print(f"  {res['name']}: {res['uplift']:.2f} pp (AI: {res['mean_ai']:.1f}% vs Other: {res['mean_other']:.1f}%)")

    print(f"\n--- Processing Closed Models ({args.closed_dir}) ---")
    for f in sorted(Path(args.closed_dir).glob(args.pattern)):
        res = get_model_uplift(f, args)
        if res:
            closed_uplifts.append(res['uplift'])
            print(f"  {res['name']}: {res['uplift']:.2f} pp (AI: {res['mean_ai']:.1f}% vs Other: {res['mean_other']:.1f}%)")

    # Stats
    print("\n" + "="*60)
    print("RESULTS: Welch's t-test on AI UPLIFT (AI Bias - Other Bias)")
    print("="*60)

    if len(open_uplifts) < 2 or len(closed_uplifts) < 2:
        print("Error: Need at least 2 models per group.")
        return

    # Calculate basic stats
    mu_open = np.mean(open_uplifts)
    mu_closed = np.mean(closed_uplifts)
    sd_open = np.std(open_uplifts, ddof=1)
    sd_closed = np.std(closed_uplifts, ddof=1)

    print(f"OPEN Group (N={len(open_uplifts)}):")
    print(f"  Mean Uplift: {mu_open:.4f} pp")
    print(f"  Std Dev:     {sd_open:.4f}")

    print(f"CLOSED Group (N={len(closed_uplifts)}):")
    print(f"  Mean Uplift: {mu_closed:.4f} pp")
    print(f"  Std Dev:     {sd_closed:.4f}")

    # Calculate Welch's t-test and CI
    diff, ci, df = calculate_welch_ci(open_uplifts, closed_uplifts)
    t_stat, p_val = stats.ttest_ind(open_uplifts, closed_uplifts, equal_var=False)
    # Note: ttest_ind direction matches our manual calculation if we did (open - closed),
    # but we usually care about the magnitude or specific direction (Closed - Open).
    # Since we defined diff = closed - open, let's just use the p-value from scipy.

    # Calculate Cohen's d effect size
    cohens_d = calculate_cohens_d(open_uplifts, closed_uplifts)

    print("-" * 30)
    print(f"Difference (Closed - Open): {diff:.4f} pp")
    print(f"95% Confidence Interval:    [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_val:.4f}")
    print(f"Degrees of Freedom: {df:.2f}")
    print(f"Cohen's d: {cohens_d:.4f}")

    # Interpret Cohen's d
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size: {effect_size}")

    if p_val < 0.05:
        print("CONCLUSION: Significant difference in AI Uplift (p < 0.05)")
    else:
        print("CONCLUSION: No significant difference in AI Uplift (p >= 0.05)")

    # Non-parametric permutation test (robust to small n)
    print("\n" + "-" * 30)
    print("NON-PARAMETRIC PERMUTATION TEST")
    print("-" * 30)
    print(f"Permutations: {args.n_permutations:,}")

    perm_p_val, perm_obs_diff, perm_null_dist = permutation_test(
        open_uplifts, closed_uplifts,
        n_permutations=args.n_permutations
    )

    print(f"Observed difference: {perm_obs_diff:.4f} pp")
    print(f"Permutation p-value (two-tailed): {perm_p_val:.4f}")

    # Calculate 95% CI from permutation distribution
    perm_ci_lower = np.percentile(perm_null_dist, 2.5)
    perm_ci_upper = np.percentile(perm_null_dist, 97.5)

    print(f"95% CI (from null distribution): [{perm_ci_lower:.4f}, {perm_ci_upper:.4f}]")

    if perm_p_val < 0.05:
        print("CONCLUSION (Permutation): Significant difference (p < 0.05)")
    else:
        print("CONCLUSION (Permutation): No significant difference (p >= 0.05)")

    # Calculate 95% CI for each cohort mean (treating each as a separate estimate)
    print("\n" + "="*60)
    print("COHORT-LEVEL 95% CONFIDENCE INTERVALS FOR MEAN AI UPLIFT")
    print("="*60)

    # Open models CI and p-value (one-sample t-test against H0: mean = 0)
    open_se = sd_open / np.sqrt(len(open_uplifts))
    open_t_crit = stats.t.ppf(0.975, df=len(open_uplifts) - 1)
    open_ci_lower = mu_open - open_t_crit * open_se
    open_ci_upper = mu_open + open_t_crit * open_se
    open_df = len(open_uplifts) - 1

    # One-sample t-test: H0: mean uplift = 0
    open_t_stat = mu_open / open_se
    open_p_val = 2 * (1 - stats.t.cdf(abs(open_t_stat), df=open_df))

    # One-sample Cohen's d (effect size relative to zero baseline)
    open_cohens_d = mu_open / sd_open

    print(f"OPEN Models (N={len(open_uplifts)}):")
    print(f"  Mean AI Uplift: {mu_open:.4f} pp")
    print(f"  Standard Error: {open_se:.4f}")
    print(f"  95% CI: [{open_ci_lower:.4f}, {open_ci_upper:.4f}] pp")
    print(f"  Degrees of Freedom: {open_df}")
    print(f"  t-statistic (H0: mean=0): {open_t_stat:.4f}")
    print(f"  p-value: {open_p_val:.4f}")
    print(f"  Cohen's d: {open_cohens_d:.4f}")
    print(f"  → {'Significantly different from zero (p<0.05)' if open_p_val < 0.05 else 'Not significantly different from zero (p≥0.05)'}")

    # Closed models CI and p-value
    closed_se = sd_closed / np.sqrt(len(closed_uplifts))
    closed_t_crit = stats.t.ppf(0.975, df=len(closed_uplifts) - 1)
    closed_ci_lower = mu_closed - closed_t_crit * closed_se
    closed_ci_upper = mu_closed + closed_t_crit * closed_se
    closed_df = len(closed_uplifts) - 1

    # One-sample t-test: H0: mean uplift = 0
    closed_t_stat = mu_closed / closed_se
    closed_p_val = 2 * (1 - stats.t.cdf(abs(closed_t_stat), df=closed_df))

    # One-sample Cohen's d (effect size relative to zero baseline)
    closed_cohens_d = mu_closed / sd_closed

    print(f"\nCLOSED Models (N={len(closed_uplifts)}):")
    print(f"  Mean AI Uplift: {mu_closed:.4f} pp")
    print(f"  Standard Error: {closed_se:.4f}")
    print(f"  95% CI: [{closed_ci_lower:.4f}, {closed_ci_upper:.4f}] pp")
    print(f"  Degrees of Freedom: {closed_df}")
    print(f"  t-statistic (H0: mean=0): {closed_t_stat:.4f}")
    print(f"  p-value: {closed_p_val:.4f}")
    print(f"  Cohen's d: {closed_cohens_d:.4f}")
    print(f"  → {'Significantly different from zero (p<0.05)' if closed_p_val < 0.05 else 'Not significantly different from zero (p≥0.05)'}")

if __name__ == "__main__":
    main()