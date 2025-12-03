#!/usr/bin/env python3
"""
Balanced Stratified Sampling for AI vs Non-AI Job Salary Estimation Study

This script implements Option 2 (Balanced Stratified Sampling) from the sampling plan:
- 500 AI jobs + 500 Non-AI jobs
- Stratified by salary quintile (within each group) and experience level
- Filters: USD currency, US or CA employee residence and company location

Based on the sampling plan documented in:
    data/salaries-for-data-science-jobs/SAMPLING_PLAN.md

Usage:
    python scripts/salaries_dataset/sample_dataset_new.py

Output:
    - data/salaries-for-data-science-jobs/sampled_1000_balanced.csv
    - data/salaries-for-data-science-jobs/sampling_report.json
    - data/salaries-for-data-science-jobs/sampling_report.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_and_filter_data(
    salaries_path: str,
    ai_jobs_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load salary data and AI job classifications, then apply filters.
    
    Args:
        salaries_path: Path to salaries.csv
        ai_jobs_path: Path to ai_ml_job_titles.csv
    
    Returns:
        Tuple of (filtered_df, original_df)
    """
    print("="*80)
    print("STEP 1: Loading Data")
    print("="*80)
    
    # Load datasets
    print(f"Loading salaries from: {salaries_path}")
    salaries_df = pd.read_csv(salaries_path)
    print(f"  Loaded {len(salaries_df):,} salary records")
    
    print(f"Loading AI job classifications from: {ai_jobs_path}")
    ai_jobs_df = pd.read_csv(ai_jobs_path)
    print(f"  Loaded {len(ai_jobs_df):,} job title classifications")
    
    # Create AI job mapping
    ai_job_mapping = dict(zip(ai_jobs_df['job_title'], ai_jobs_df['AI_job']))
    salaries_df['is_ai_job'] = salaries_df['job_title'].map(ai_job_mapping)
    
    # Handle missing mappings
    missing_mappings = salaries_df['is_ai_job'].isna().sum()
    if missing_mappings > 0:
        print(f"  WARNING: {missing_mappings} job titles not found in AI classification")
        print(f"  Setting unmapped jobs as Non-AI (is_ai_job=False)")
        salaries_df['is_ai_job'] = salaries_df['is_ai_job'].fillna(False)
    
    print(f"\nOriginal dataset statistics:")
    print(f"  Total records: {len(salaries_df):,}")
    print(f"  AI jobs: {salaries_df['is_ai_job'].sum():,} ({salaries_df['is_ai_job'].mean()*100:.2f}%)")
    print(f"  Non-AI jobs: {(~salaries_df['is_ai_job']).sum():,} ({(~salaries_df['is_ai_job']).mean()*100:.2f}%)")
    
    # Apply filters
    print("\n" + "="*80)
    print("STEP 2: Applying Filters")
    print("="*80)
    print("Filters:")
    print("  - Currency: USD only")
    print("  - Employee residence: US or CA")
    print("  - Company location: US or CA")
    
    original_df = salaries_df.copy()
    
    filtered_df = salaries_df[
        (salaries_df['salary_currency'] == 'USD') &
        (salaries_df['employee_residence'].isin(['US', 'CA'])) &
        (salaries_df['company_location'].isin(['US', 'CA']))
    ].copy()
    
    print(f"\nFiltered dataset statistics:")
    print(f"  Retained: {len(filtered_df):,} / {len(salaries_df):,} ({len(filtered_df)/len(salaries_df)*100:.2f}%)")
    print(f"  AI jobs: {filtered_df['is_ai_job'].sum():,} ({filtered_df['is_ai_job'].mean()*100:.2f}%)")
    print(f"  Non-AI jobs: {(~filtered_df['is_ai_job']).sum():,} ({(~filtered_df['is_ai_job']).mean()*100:.2f}%)")
    
    # Verify required columns exist
    required_cols = ['job_title', 'salary_in_usd', 'experience_level', 'is_ai_job']
    missing_cols = [col for col in required_cols if col not in filtered_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return filtered_df, original_df


def balanced_stratified_sample(
    df: pd.DataFrame,
    n_per_group: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample n_per_group from AI and Non-AI jobs,
    stratified by salary quintile and experience level.
    
    Implementation of Option 2: Balanced Stratified Sampling
    
    Args:
        df: Filtered salary dataframe with 'is_ai_job' column
        n_per_group: Number of samples per AI/Non-AI group (default: 500)
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled dataframe with exactly 2*n_per_group rows
    """
    print("\n" + "="*80)
    print("STEP 3: Balanced Stratified Sampling")
    print("="*80)
    print(f"Target: {n_per_group} AI jobs + {n_per_group} Non-AI jobs")
    print(f"Stratification: Salary quintile (within group) × Experience level")
    print(f"Random seed: {random_state}")
    
    np.random.seed(random_state)
    sampled_indices = []
    sampling_details = {}
    
    for is_ai in [True, False]:
        job_type = "AI" if is_ai else "Non-AI"
        print(f"\n--- Sampling {job_type} Jobs ---")
        
        # Subset to AI or Non-AI jobs
        subset = df[df['is_ai_job'] == is_ai].copy()
        print(f"Population size: {len(subset):,}")
        
        # Create salary quintiles WITHIN this group
        # This ensures we get comparable salary ranges for both AI and Non-AI
        try:
            subset['salary_quintile_within_group'] = pd.qcut(
                subset['salary_in_usd'], 
                q=5, 
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                duplicates='drop'
            )
        except ValueError as e:
            print(f"  WARNING: Could not create 5 quintiles (likely duplicate edges)")
            print(f"  Falling back to fewer bins or equal-width bins")
            # Fallback: try fewer bins
            subset['salary_quintile_within_group'] = pd.qcut(
                subset['salary_in_usd'], 
                q=4, 
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                duplicates='drop'
            )
        
        # Count samples in each stratum (quintile × experience level)
        strata_counts = subset.groupby([
            'salary_quintile_within_group', 
            'experience_level'
        ]).size()
        
        print(f"Number of strata: {len(strata_counts)}")
        
        # Calculate proportional allocation
        total_in_group = len(subset)
        stratum_proportions = strata_counts / total_in_group
        stratum_samples = (stratum_proportions * n_per_group).round().astype(int)
        
        # Adjust to ensure exactly n_per_group samples
        current_total = stratum_samples.sum()
        adjustment_needed = n_per_group - current_total
        
        if adjustment_needed != 0:
            # Add/subtract from largest stratum
            largest_stratum = stratum_samples.idxmax()
            stratum_samples[largest_stratum] += adjustment_needed
            print(f"Adjusted largest stratum by {adjustment_needed} to reach exactly {n_per_group} samples")
        
        # Store stratum information
        stratum_info = []
        
        # Sample from each stratum
        for (quintile, exp_level), n_samples in stratum_samples.items():
            stratum_mask = (
                (subset['salary_quintile_within_group'] == quintile) &
                (subset['experience_level'] == exp_level)
            )
            stratum_data = subset[stratum_mask]
            stratum_size = len(stratum_data)
            
            if stratum_size >= n_samples:
                # Sample without replacement
                sampled = stratum_data.sample(
                    n=n_samples, 
                    replace=False, 
                    random_state=random_state + len(sampled_indices)  # Vary seed per stratum
                )
                replacement_used = False
            else:
                # Sample with replacement if stratum is too small
                print(f"  WARNING: Stratum (Q={quintile}, Exp={exp_level}) "
                      f"has only {stratum_size} records, sampling {n_samples} with replacement")
                sampled = stratum_data.sample(
                    n=n_samples, 
                    replace=True, 
                    random_state=random_state + len(sampled_indices)
                )
                replacement_used = True
            
            sampled_indices.extend(sampled.index.tolist())
            
            # Record stratum details
            stratum_info.append({
                'quintile': str(quintile),
                'experience_level': exp_level,
                'population_size': int(stratum_size),
                'sample_size': int(n_samples),
                'proportion': float(n_samples / n_per_group),
                'replacement_used': bool(replacement_used)
            })
        
        sampling_details[job_type] = {
            'target_sample_size': int(n_per_group),
            'actual_sample_size': int(len([idx for idx in sampled_indices if idx in subset.index])),
            'population_size': int(len(subset)),
            'num_strata': int(len(strata_counts)),
            'strata': stratum_info
        }
        
        print(f"Successfully sampled {len(stratum_info)} strata")
        print(f"Total {job_type} samples: {len([idx for idx in sampled_indices if idx in subset.index])}")
    
    # Return sampled dataframe
    sampled_df = df.loc[sampled_indices].copy()
    
    # Verify no duplicates (unless replacement was used)
    if len(sampled_df) != len(set(sampled_indices)):
        print(f"\n  NOTE: Sample contains {len(sampled_indices) - len(set(sampled_indices))} duplicates due to replacement")
    
    return sampled_df, sampling_details


def generate_quality_checks(sample_df: pd.DataFrame) -> Dict:
    """
    Perform quality checks on the sampled data.
    
    Args:
        sample_df: Sampled dataframe
    
    Returns:
        Dictionary of quality check results
    """
    print("\n" + "="*80)
    print("STEP 4: Quality Checks")
    print("="*80)
    
    checks = {}
    
    # Check 1: Sample size
    print("\n1. Sample Size Verification:")
    ai_count = sample_df['is_ai_job'].sum()
    non_ai_count = (~sample_df['is_ai_job']).sum()
    total_count = len(sample_df)
    
    print(f"   AI jobs: {ai_count}")
    print(f"   Non-AI jobs: {non_ai_count}")
    print(f"   Total: {total_count}")
    
    checks['sample_size'] = {
        'ai_count': int(ai_count),
        'non_ai_count': int(non_ai_count),
        'total_count': int(total_count),
        'balanced': bool(ai_count == non_ai_count)
    }
    
    if ai_count == 500 and non_ai_count == 500:
        print("   ✓ PASS: Exactly 500 samples per group")
    else:
        print(f"   ✗ WARNING: Expected 500 per group, got AI={ai_count}, Non-AI={non_ai_count}")
    
    # Check 2: Missing values
    print("\n2. Missing Value Check:")
    key_cols = ['salary_in_usd', 'is_ai_job', 'experience_level', 'job_title']
    missing_counts = {col: sample_df[col].isna().sum() for col in key_cols}
    
    checks['missing_values'] = {col: int(count) for col, count in missing_counts.items()}
    
    all_complete = all(count == 0 for count in missing_counts.values())
    if all_complete:
        print("   ✓ PASS: No missing values in key columns")
    else:
        print("   ✗ WARNING: Missing values detected:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"      {col}: {count} missing")
    
    # Check 3: Salary range coverage
    print("\n3. Salary Range Coverage:")
    salary_stats = {}
    for is_ai, label in [(True, "AI"), (False, "Non-AI")]:
        subset = sample_df[sample_df['is_ai_job'] == is_ai]['salary_in_usd']
        stats = {
            'min': float(subset.min()),
            'max': float(subset.max()),
            'mean': float(subset.mean()),
            'median': float(subset.median()),
            'std': float(subset.std())
        }
        salary_stats[label] = stats
        print(f"   {label}:")
        print(f"      Range: ${stats['min']:,.0f} - ${stats['max']:,.0f}")
        print(f"      Mean: ${stats['mean']:,.0f} (SD: ${stats['std']:,.0f})")
        print(f"      Median: ${stats['median']:,.0f}")
    
    checks['salary_stats'] = salary_stats
    
    # Check 4: Experience level distribution
    print("\n4. Experience Level Distribution:")
    exp_dist = sample_df.groupby(['is_ai_job', 'experience_level']).size().unstack(fill_value=0)
    print(exp_dist)
    
    # Convert to JSON-serializable format
    exp_dist_dict = {}
    for idx in exp_dist.index:
        exp_dist_dict[str(idx)] = {col: int(val) for col, val in exp_dist.loc[idx].items()}
    checks['experience_distribution'] = exp_dist_dict

    # Check 5: Top job titles
    print("\n5. Top Job Titles in Sample:")
    for is_ai, label in [(True, "AI"), (False, "Non-AI")]:
        print(f"   {label} Jobs:")
        top_titles = sample_df[sample_df['is_ai_job'] == is_ai]['job_title'].value_counts().head(5)
        for title, count in top_titles.items():
            print(f"      {title}: {count}")
    
    # Check 6: Year distribution
    if 'work_year' in sample_df.columns:
        print("\n6. Work Year Distribution:")
        year_dist = sample_df['work_year'].value_counts().sort_index()
        print(year_dist)
        checks['year_distribution'] = {int(k): int(v) for k, v in year_dist.to_dict().items()}

    print("\n" + "="*80)
    print("Quality checks complete!")
    print("="*80)
    
    return checks


def save_outputs(
    sample_df: pd.DataFrame,
    sampling_details: Dict,
    quality_checks: Dict,
    output_csv: str,
    output_json: str,
    output_txt: str
) -> None:
    """
    Save sampled data and reports.
    
    Args:
        sample_df: Sampled dataframe
        sampling_details: Details about sampling process
        quality_checks: Results of quality checks
        output_csv: Path to save CSV file
        output_json: Path to save JSON report
        output_txt: Path to save text report
    """
    print("\n" + "="*80)
    print("STEP 5: Saving Outputs")
    print("="*80)
    
    # Save CSV
    print(f"Saving sample to: {output_csv}")
    sample_df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved {len(sample_df)} records")
    
    # Prepare full report
    full_report = {
        'metadata': {
            'sampling_method': 'Balanced Stratified Sampling (Option 2)',
            'target_sample_size': 1000,
            'stratification_variables': ['salary_quintile_within_group', 'experience_level'],
            'filters_applied': {
                'currency': 'USD',
                'employee_residence': ['US', 'CA'],
                'company_location': ['US', 'CA']
            },
            'random_seed': 42
        },
        'sampling_details': sampling_details,
        'quality_checks': quality_checks
    }
    
    # Save JSON report
    print(f"Saving JSON report to: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"  ✓ Saved JSON report")
    
    # Save text report
    print(f"Saving text report to: {output_txt}")
    with open(output_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAMPLING REPORT: Balanced Stratified Sampling\n")
        f.write("="*80 + "\n\n")
        
        f.write("SAMPLING METHOD\n")
        f.write("-" * 80 + "\n")
        f.write("Option 2: Balanced Stratified Sampling\n")
        f.write("Target: 500 AI jobs + 500 Non-AI jobs = 1,000 total\n")
        f.write("Stratification: Salary quintile (within group) × Experience level\n\n")
        
        f.write("FILTERS APPLIED\n")
        f.write("-" * 80 + "\n")
        f.write("- Currency: USD only\n")
        f.write("- Employee residence: US or CA\n")
        f.write("- Company location: US or CA\n\n")
        
        f.write("SAMPLE COMPOSITION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {quality_checks['sample_size']['total_count']}\n")
        f.write(f"AI jobs: {quality_checks['sample_size']['ai_count']}\n")
        f.write(f"Non-AI jobs: {quality_checks['sample_size']['non_ai_count']}\n")
        f.write(f"Balanced: {'Yes' if quality_checks['sample_size']['balanced'] else 'No'}\n\n")
        
        f.write("SALARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        for job_type, stats in quality_checks['salary_stats'].items():
            f.write(f"{job_type} Jobs:\n")
            f.write(f"  Range: ${stats['min']:,.0f} - ${stats['max']:,.0f}\n")
            f.write(f"  Mean: ${stats['mean']:,.0f} (SD: ${stats['std']:,.0f})\n")
            f.write(f"  Median: ${stats['median']:,.0f}\n\n")
        
        f.write("EXPERIENCE LEVEL DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        exp_dist = pd.DataFrame(quality_checks['experience_distribution'])
        f.write(exp_dist.to_string())
        f.write("\n\n")
        
        f.write("SAMPLING DETAILS BY GROUP\n")
        f.write("-" * 80 + "\n")
        for job_type, details in sampling_details.items():
            f.write(f"\n{job_type} Jobs:\n")
            f.write(f"  Population size: {details['population_size']:,}\n")
            f.write(f"  Sample size: {details['actual_sample_size']}\n")
            f.write(f"  Number of strata: {details['num_strata']}\n")
            f.write(f"\n  Stratum Details:\n")
            for stratum in details['strata']:
                f.write(f"    {stratum['quintile']} × {stratum['experience_level']}: ")
                f.write(f"{stratum['sample_size']} samples ")
                f.write(f"(pop: {stratum['population_size']}, ")
                f.write(f"prop: {stratum['proportion']*100:.1f}%")
                if stratum['replacement_used']:
                    f.write(", WITH REPLACEMENT")
                f.write(")\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("="*80 + "\n")
        f.write("1. Review this sampling report\n")
        f.write("2. Use sampled_1000_balanced.csv for LLM salary estimation\n")
        f.write("3. Calculate signed_diff = estimate - actual\n")
        f.write("4. Perform Welch's t-test comparing AI vs Non-AI signed differences\n")
        f.write("5. Run robustness checks with HC3 standard errors\n")
        f.write("6. Conduct secondary analyses by quintile and experience level\n")
        
    print(f"  ✓ Saved text report")
    
    print("\n" + "="*80)
    print("All outputs saved successfully!")
    print("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Balanced Stratified Sampling for AI vs Non-AI Job Salary Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default paths
  python scripts/salaries_dataset/sample_dataset_new.py
  
  # Run with custom sample size per group
  python scripts/salaries_dataset/sample_dataset_new.py --n-per-group 600
  
  # Run with different random seed
  python scripts/salaries_dataset/sample_dataset_new.py --seed 123
        """
    )
    
    parser.add_argument(
        '--salaries-path',
        type=str,
        default='data/salaries-for-data-science-jobs/salaries.csv',
        help='Path to salaries CSV file (default: data/salaries-for-data-science-jobs/salaries.csv)'
    )
    
    parser.add_argument(
        '--ai-jobs-path',
        type=str,
        default='data/salaries-for-data-science-jobs/ai_ml_job_titles.csv',
        help='Path to AI job classifications CSV (default: data/salaries-for-data-science-jobs/ai_ml_job_titles.csv)'
    )
    
    parser.add_argument(
        '--output-csv',
        type=str,
        default='data/salaries-for-data-science-jobs/sampled_1000_balanced.csv',
        help='Path to save sampled CSV (default: data/salaries-for-data-science-jobs/sampled_1000_balanced.csv)'
    )
    
    parser.add_argument(
        '--output-json',
        type=str,
        default='data/salaries-for-data-science-jobs/sampling_report.json',
        help='Path to save JSON report (default: data/salaries-for-data-science-jobs/sampling_report.json)'
    )
    
    parser.add_argument(
        '--output-txt',
        type=str,
        default='data/salaries-for-data-science-jobs/sampling_report.txt',
        help='Path to save text report (default: data/salaries-for-data-science-jobs/sampling_report.txt)'
    )
    
    parser.add_argument(
        '--n-per-group',
        type=int,
        default=500,
        help='Number of samples per group (AI and Non-AI) (default: 500)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Display header
    print("\n" + "="*80)
    print("BALANCED STRATIFIED SAMPLING FOR AI vs NON-AI SALARY STUDY")
    print("="*80)
    print(f"Implementation: Option 2 from SAMPLING_PLAN.md")
    print(f"Target: {args.n_per_group} AI + {args.n_per_group} Non-AI = {args.n_per_group * 2} total")
    print("="*80 + "\n")
    
    # Execute sampling pipeline
    try:
        # Step 1-2: Load and filter data
        filtered_df, original_df = load_and_filter_data(
            args.salaries_path,
            args.ai_jobs_path
        )
        
        # Step 3: Perform sampling
        sample_df, sampling_details = balanced_stratified_sample(
            filtered_df,
            n_per_group=args.n_per_group,
            random_state=args.seed
        )
        
        # Step 4: Quality checks
        quality_checks = generate_quality_checks(sample_df)
        
        # Step 5: Save outputs
        save_outputs(
            sample_df,
            sampling_details,
            quality_checks,
            args.output_csv,
            args.output_json,
            args.output_txt
        )
        
        # Final summary
        print("\n" + "="*80)
        print("SAMPLING COMPLETE!")
        print("="*80)
        print(f"\nOutput files:")
        print(f"  1. Sample data: {args.output_csv}")
        print(f"  2. JSON report: {args.output_json}")
        print(f"  3. Text report: {args.output_txt}")
        print(f"\nNext steps:")
        print(f"  1. Review the sampling reports")
        print(f"  2. Use the sampled data for LLM salary estimation")
        print(f"  3. Perform statistical analysis (Welch's t-test)")
        print("\nFor more information, see:")
        print("  data/salaries-for-data-science-jobs/SAMPLING_PLAN.md")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        raise


if __name__ == "__main__":
    main()

