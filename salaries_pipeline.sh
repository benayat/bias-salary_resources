#!/bin/bash

# Model to use from first arg, SOC weight mode from second arg
model=$1
soc_weight=${2:-uniform}  # Default to 'uniform' if not provided
llm_config=${3:-hpc2h200}  # Default to 'hpc2h200' if not provided

# Extract model tag (last part after /)
model_tag="${model##*/}"

echo "Running pipeline with:"
echo "  Model: $model"
echo "  Model tag: $model_tag"
echo "  SOC weight mode: $soc_weight"
echo "  LLM config: $llm_config"
echo ""

uv run scripts/salaries_dataset/sample_salaries_block_balanced.py --input data/salaries-for-data-science-jobs/salaries.csv --ai-titles data/salaries-for-data-science-jobs/ai_ml_job_titles.csv --output data/salaries-for-data-science-jobs/sampled_1000_"$soc_weight".csv --target-total 1000 --max-per-block 20 --weight-mode "$soc_weight" --seed 0
uv run scripts/salaries_dataset/estimate_salaries.py --model "$model" --input-csv data/salaries-for-data-science-jobs/sampled_1000_"$soc_weight".csv --llm-config "$llm_config"
#uv run scripts/statistical_analysis/compare_estimates_all.py --estimates-dir data/salaries-for-data-science-jobs/estimations/llm_estimated_salaries-"$model_tag" --output-dir data/salaries-for-data-science-jobs/statistical_results
uv run scripts/salaries_dataset/calc_statistics.py --estimates-dir data/salaries-for-data-science-jobs/estimations