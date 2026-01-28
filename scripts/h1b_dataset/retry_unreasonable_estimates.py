"""
Script to identify and retry unreasonable salary estimations in LLM output files.
Finds rows with estimated_salary_in_usd <= threshold and re-runs the LLM to get proper estimates.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from llm import LLMClient, SamplingConfig
from constants import (
    HOME_CONFIG, HOME_4GPU_CONFIG, HOME_CONFIG_SMALL, 
    HPC_CONFIG, HPC_2H200_CONFIG, PERSONAS, SALARY_SYSTEM_PROMPT
)


SALARY_USER_PROMPT = """Estimate the yearly salary in USD for this H1B job application based on the following features:
- job_title: {job_title}
- soc_code: {soc_code}
- soc_title: {soc_title}
- full_time_position: {full_time_position}
- total_worker_positions: {total_worker_positions}
- worksite_city: {worksite_city}
- worksite_state: {worksite_state}
- naics_code: {naics_code}
- pw_wage_level: {pw_wage_level}

Return only the integer amount, nothing else."""


def find_unreasonable_estimates(df, low_threshold=1000, high_threshold=10_000_000, pct_error_threshold=500):
    """
    Identify unreasonable estimates using multiple criteria:
    1. Too low (< low_threshold)
    2. Too high (> high_threshold) - likely data corruption
    3. Extreme percentage error vs prevailing wage (> pct_error_threshold%)
    """
    unreasonable_mask = (
            (df["estimated_salary_in_usd"] < low_threshold) |
            (df["estimated_salary_in_usd"] > high_threshold) |
            (
                    (abs((df["estimated_salary_in_usd"] - df["PREVAILING_WAGE"]) / df["PREVAILING_WAGE"]) * 100 > pct_error_threshold)
                    & (df["PREVAILING_WAGE"] > 0)
            )
    )
    return unreasonable_mask

def main():
    parser = argparse.ArgumentParser(
        description="Retry unreasonable salary estimations in LLM output files"
    )
    parser.add_argument(
        "--input-csv", 
        required=True,
        help="Path to CSV file with estimations to check"
    )
    parser.add_argument(
        "--model", 
        required=True,
        help="LLM model name to use for retrying"
    )
    parser.add_argument(
        "--threshold", 
        type=int, 
        default=10,
        help="Threshold for unreasonable estimates (default: 10)"
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=10_000_000,
        help="Upper threshold for unreasonable estimates (default: 10,000,000)"
    )
    parser.add_argument(
        "--pct-error-threshold",
        type=int,
        default=500,
        help="Percentage error threshold vs prevailing wage (default: 500)"
    )
    parser.add_argument(
        "--client-type", 
        choices=["vllm", "openai"], 
        default="vllm",
        help="Type of LLM client to use"
    )
    parser.add_argument(
        "--openai-api-key", 
        type=str, 
        default="crap",
        help="OpenAI API key (if using OpenAI client)"
    )
    parser.add_argument(
        "--openai-base-url", 
        type=str, 
        default="https://api.openai.com/v1",
        help="OpenAI API base URL"
    )
    parser.add_argument(
        "--llm-config", 
        choices=["home", "home_4gpu", "hpc", "hpc2h200", "home_small"],
        default="home_small",
        help="Choose LLM configuration"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=None,
        help="Tensor parallel size for distributed inference"
    )
    parser.add_argument(
        "--persona", 
        type=str, 
        default="salary_estimator",
        help="Persona to use for estimation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only identify unreasonable estimates without retrying"
    )
    
    args = parser.parse_args()

    # Load the CSV file
    print(f"Loading CSV file: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    
    # Check if required columns exist
    required_cols = ["estimated_salary_in_usd", "JOB_TITLE", "SOC_CODE", "SOC_TITLE", 
                     "FULL_TIME_POSITION", "TOTAL_WORKER_POSITIONS", "WORKSITE_STATE", 
                     "NAICS_CODE"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Identify unreasonable estimates
    # unreasonable_mask = df["estimated_salary_in_usd"] <= args.threshold
    unreasonable_mask = find_unreasonable_estimates(
        df,
        low_threshold=args.threshold,
        high_threshold=args.high_threshold if hasattr(args, 'high_threshold') else 10_000_000,
        pct_error_threshold=args.pct_error_threshold if hasattr(args, 'pct_error_threshold') else 500
    )
    unreasonable_df = df[unreasonable_mask].copy()
    
    print(f"\n{'='*60}")
    print(f"Found {len(unreasonable_df)} unreasonable estimates (<= {args.threshold})")
    print(f"{'='*60}\n")
    
    if len(unreasonable_df) == 0:
        print("No unreasonable estimates found. Exiting.")
        return
    
    # Show some examples
    print("Examples of unreasonable estimates:")
    print(unreasonable_df[["JOB_TITLE", "SOC_TITLE", "WORKSITE_STATE", 
                           "estimated_salary_in_usd", "PREVAILING_WAGE"]].head(10))
    print()
    
    if args.dry_run:
        print("Dry run mode - not retrying. Exiting.")
        return
    
    # Set up LLM configuration
    if args.llm_config == "home_4gpu":
        llm_config = HOME_4GPU_CONFIG
    elif args.llm_config == "hpc":
        llm_config = HPC_CONFIG
    elif args.llm_config == "home_small":
        llm_config = HOME_CONFIG_SMALL
    elif args.llm_config == "hpc2h200":
        llm_config = HPC_2H200_CONFIG
    else:
        llm_config = HOME_CONFIG
    
    # Initialize LLM client
    if args.client_type == "vllm":
        # Extract model size for configuration scaling
        model_size_match = re.search(r'(\d+)x(\d+)[Bb]|(\d+(?:\.\d+)?)[Bb]', args.model)
        if model_size_match:
            if model_size_match.group(1) and model_size_match.group(2):
                model_size_b = float(model_size_match.group(1)) * float(model_size_match.group(2))
            else:
                model_size_b = float(model_size_match.group(3))
            print(f"Model size: {model_size_b}B")
        
        if args.tensor_parallel_size is not None:
            llm_config.tensor_parallel_size = args.tensor_parallel_size
            print(f"Using tensor_parallel_size={args.tensor_parallel_size}")
        
        llm = LLMClient(model_name=args.model, config=llm_config)
        sampling_params = SamplingConfig(
            temperature=0.0, top_p=1.0, 
            max_tokens=512 if "deepseek" in args.model else 16
        )
    else:
        from openai_llm.openai_client import (
            OpenAIConfig, LLMClient as OpenAILLMClient, 
            SamplingConfig as OpenAISamplingConfig
        )
        llm_config = OpenAIConfig(
            api_key=args.openai_api_key, 
            base_url=args.openai_base_url
        )
        llm = OpenAILLMClient(model_name=args.model, config=llm_config)
        sampling_params = OpenAISamplingConfig(
            temperature=0.0, top_p=1.0,
            max_tokens=6500 if "gpt-oss" in args.model else 16
        )
    
    # Prepare system prompt
    if args.persona:
        system_prompt = PERSONAS.get(args.persona, "") + SALARY_SYSTEM_PROMPT
    else:
        system_prompt = SALARY_SYSTEM_PROMPT
    
    # Prepare prompts for unreasonable estimates
    print(f"\nPreparing prompts for {len(unreasonable_df)} rows...")
    prompts = []
    row_indices = []
    
    for idx, row in unreasonable_df.iterrows():
        user_content = SALARY_USER_PROMPT.format(
            job_title=row.get("JOB_TITLE", ""),
            soc_code=row.get("SOC_CODE", ""),
            soc_title=row.get("SOC_TITLE", ""),
            full_time_position=row.get("FULL_TIME_POSITION", ""),
            total_worker_positions=row.get("TOTAL_WORKER_POSITIONS", ""),
            worksite_city=row.get("WORKSITE_CITY", ""),
            worksite_state=row.get("WORKSITE_STATE", ""),
            naics_code=row.get("NAICS_CODE", ""),
            pw_wage_level=row.get("PW_WAGE_LEVEL", ""),
        )
        prompts.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        })
        row_indices.append(idx)
    
    # Run batch inference
    print(f"Running LLM inference for {len(prompts)} prompts...")
    results = llm.run_batch(prompts, sampling_params, output_field="output")
    
    # Extract salary estimates
    corrections = {}
    for idx, result in zip(row_indices, results):
        output = str(result.get("output", "")).strip()
        digits = "".join(c for c in output if c.isdigit())
        try:
            est = int(digits)
        except ValueError:
            est = 0
        corrections[idx] = est
        old_val = int(df.at[idx, 'estimated_salary_in_usd'])
        print(f"Row {idx}: {df.at[idx, 'JOB_TITLE'][:40]:40s} | "
              f"Old: ${old_val:6d} → New: ${est:6d}")

    # Apply corrections to dataframe
    print(f"\nApplying corrections to {len(corrections)} rows...")
    for idx, new_estimate in corrections.items():
        df.at[idx, "estimated_salary_in_usd"] = new_estimate
    
    # Create backup of original file
    backup_path = args.input_csv.replace(".csv", "_backup_before_retry.csv")
    print(f"Creating backup: {backup_path}")
    pd.read_csv(args.input_csv).to_csv(backup_path, index=False)
    
    # Save corrected file
    print(f"Saving corrected file: {args.input_csv}")
    df.to_csv(args.input_csv, index=False)
    
    # Summary statistics
    old_values = [df.at[idx, "estimated_salary_in_usd"] for idx in row_indices]
    new_values = [corrections[idx] for idx in row_indices]
    still_unreasonable = sum(1 for v in new_values if v <= args.threshold)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows processed: {len(corrections)}")
    print(f"Still unreasonable (<= {args.threshold}): {still_unreasonable}")
    print(f"Successfully corrected: {len(corrections) - still_unreasonable}")
    print(f"Backup saved to: {backup_path}")
    print(f"Corrected file saved to: {args.input_csv}")
    print(f"{'='*60}\n")
    
    # Clean up
    llm.delete_client()


if __name__ == "__main__":
    main()

