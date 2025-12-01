#!/bin/bash

# Sample the dataset
uv run scripts/h1b_dataset/sample_dataset.py --input data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv --ai-titles data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv --output data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled_uniform.csv --seed 42 --target-total 200 --seed 42 --max-pairs-per-soc 5 --ai-soc-weight-mode uniform

# Model to use from first arg.
model = $1
  personas = (
    "ai_minimalistic_no_behavior_mod"
    "ai_minimalistic_behavior_mod"
    "ai_extended_no_behavior_mod"
    "ai_extended_behavior_mod"
    "human_minimalistic_no_behavior_mod"
    "human_minimalistic_behavior_mod"
    "human_extended_no_behavior_mod"
    "human_extended_behavior_mod"
    "neutral_minimalistic_no_behavior_mod"
    "neutral_extended_no_behavior_mod"
    "salary_estimator"

# Run estimation for each model

for persona in "${personas[@]}"; do
  uv run scripts/h1b_dataset/estimate_h1b_salaries.py --model "$model" --input-csv-file data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled_uniform.csv --persona-to-use "$persona"
  done

# Compare with paired t-test
uv run scripts/h1b_dataset/compare_with_paired_ttest.py --estimates-dir data/h1b-lca-disclosure-data-2020-2024/sampled_uniform --glob "*.csv" --actual-col PREVAILING_WAGE --estimate-col estimated_salary_in_usd
