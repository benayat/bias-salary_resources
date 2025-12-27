#!/bin/bash

# Sample the dataset
#uv run scripts/h1b_dataset/sample_dataset.py --input data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv --ai-titles data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv --output data/h1b-lca-disclosure-data-2020-2024/uniform.csv --seed 42 --target-total 200 --seed 42 --max-pairs-per-soc 5 --ai-soc-weight-mode sqrt

# List of models
models=(
#    "dphn/dolphin-2.9.1-yi-1.5-34b"
#    "01-ai/Yi-1.5-34B-Chat"
#    "Qwen/Qwen3-32B"
#    "Qwen/Qwen3-Next-80B-A3B-Instruct"
#    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
#    "meta-llama/Llama-3.3-70B-Instruct"
    "google/gemma-3-27b-it"
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
)
served_models=(
"openai/gpt-oss-20b"
"openai/gpt-oss-120b"
"mistralai/Ministral-3-14B-Instruct-2512"
)

# Run estimation for each model
for model in "${models[@]}"; do
    uv run scripts/h1b_dataset/estimate_h1b_salaries.py --model "$model" --input-csv-file data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled-1000_block_balanced.csv
done
#for model in "${served_models[@]}"; do
#    uv run scripts/h1b_dataset/estimate_h1b_salaries.py --model "$model" --client-type openai --openai-api-key crap --openai-base-url http://localhost:8000/v1 --input-csv-file data/h1b-lca-disclosure-data-2020-2024/uniform.csv
#done
# Compare with paired t-test
uv run scripts/h1b_dataset/compare_welch_hc3.py --estimates-dir data/h1b-lca-disclosure-data-2020-2024/sampled_uniform --glob "*.csv" --actual-col PREVAILING_WAGE --estimate-col estimated_salary_in_usd
uv run scripts/h1b_dataset/compare_open_vs_closed.py --estimates-dir data/h1b-lca-disclosure-data-2020-2024/sampled_uniform --glob "*.csv" --actual-col PREVAILING_WAGE --estimate-col estimated_salary_in_usd
