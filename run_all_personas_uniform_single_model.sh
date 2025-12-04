#!/bin/bash

# Model to use from first arg, SOC weight mode from second arg
model=$1
soc_weight=${2:-uniform}  # Default to 'uniform' if not provided

# Extract model tag (last part after /)
model_tag="${model##*/}"

echo "Running pipeline with:"
echo "  Model: $model"
echo "  Model tag: $model_tag"
echo "  SOC weight mode: $soc_weight"
echo ""

# Sample the dataset
#uv run scripts/h1b_dataset/sample_dataset.py \
#    --input data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv \
#    --ai-titles data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv \
#    --output data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled-${soc_weight}.csv \
#    --seed 42 \
#    --target-total 1000 \
#    --max-pairs-per-soc 9 \
#    --ai-soc-weight-mode "$soc_weight"

uv run scripts/h1b_dataset/sample_dataset_new.py \
    --input data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv \
    --ai-titles data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv \
    --output data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled-1000.csv \
    --soc-weight-mode "$soc_weight" \
    --seed 42 \
    --target-total 1000


# Run estimation for all personas in a single command
#uv run scripts/h1b_dataset/estimate_h1b_salaries.py \
#    --model "$model" \
#    --llm-config hpc2h200 \
#    --input-csv-file data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled-${soc_weight}.csv \
#    --personas-to-use ai_minimalistic_no_behavior_mod ai_minimalistic_behavior_mod ai_extended_no_behavior_mod ai_extended_behavior_mod human_minimalistic_no_behavior_mod human_minimalistic_behavior_mod human_extended_no_behavior_mod human_extended_behavior_mod neutral_minimalistic_no_behavior_mod neutral_extended_no_behavior_mod salary_estimator


uv run scripts/h1b_dataset/estimate_h1b_salaries.py \
    --model "$model" \
    --input-csv-file data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled-1000.csv \
    --llm-config hpc2h200 \
    --personas-to-use salary_estimator



#uv run scripts/h1b_dataset/compare_welch_hc3.py \
#     --estimates-dir data/h1b-lca-disclosure-data-2020-2024/sampled-${soc_weight}/${model_tag} \
#     --glob "*.csv" \
#     --actual-col PREVAILING_WAGE \
#     --estimate-col estimated_salary_in_usd \
#     --min-actual 1 \
#     --spb-cap 0

uv run scripts/h1b_dataset/compare_welch_hc3.py \
     --estimates-dir data/h1b-lca-disclosure-data-2020-2024/sampled-1000/${model_tag} \
     --glob "*.csv" \
     --actual-col PREVAILING_WAGE \
     --estimate-col estimated_salary_in_usd \
     --min-actual 1 \
     --spb-cap 0
