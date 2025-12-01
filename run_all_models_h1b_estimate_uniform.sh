#!/bin/bash

# Sample the dataset
uv run scripts/h1b_dataset/sample_dataset.py --input data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv --ai-titles data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv --output data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled_uniform.csv --seed 42 --target-total 200 --seed 42 --max-pairs-per-soc 5 --ai-soc-weight-mode uniform

# List of models
models=(
    "tiiuae/Falcon-H1-7B-Instruct"
    "ibm-granite/granite-4.0-h-tiny"
    "ibm-granite/granite-4.0-micro"
#    "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
#    "meta-llama/Llama-3.2-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen3-4B-Instruct-2507"
#    "HuggingFaceTB/SmolLM3-3B"
)

# Run estimation for each model
for model in "${models[@]}"; do
    uv run scripts/h1b_dataset/estimate_h1b_salaries.py --model "$model" --input-csv-file data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled_uniform.csv
done

# Compare with paired t-test
uv run scripts/h1b_dataset/compare_with_paired_ttest.py --estimates-dir data/h1b-lca-disclosure-data-2020-2024/sampled_uniform --glob "*.csv" --actual-col PREVAILING_WAGE --estimate-col estimated_salary_in_usd
