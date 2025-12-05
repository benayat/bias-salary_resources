#!/bin/bash
#SBATCH --job-name=salaries_estimate_granite_falcon_llama
#SBATCH --output=/home/fast/trabelb1/projects/bias-salary_resources/salary_eval%j.out
#SBATCH --error=/home/fast/trabelb1/projects/bias-salary_resources/salary_eval_err%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
source ~/.bash_profile
REPO_LOCATION="/home/fast/trabelb1/projects/bias-salary_resources"
nvidia-smi
cd $REPO_LOCATION
export PYTHONPATH=$(pwd)
#uv run scripts/h1b_dataset/estimate_h1b_salaries.py --model Qwen/Qwen3-4B-Instruct-2507 --chunk-size 600000
#uv run scripts/h1b_dataset/estimate_h1b_salaries.py --model HuggingFaceTB/SmolLM3-3B --chunk-size 600000
#uv run scripts/salaries_dataset/estimate_salaries.py --model Qwen/Qwen3-4B-Instruct-2507 --chunk-size 600000
#uv run scripts/salaries_dataset/estimate_salaries.py --model HuggingFaceTB/SmolLM3-3B --chunk-size 600000
uv run scripts/salaries_dataset/estimate_salaries.py --model ibm-granite/granite-4.0-h-tiny --chunk-size 600000
uv run scripts/salaries_dataset/estimate_salaries.py --model tiiuae/Falcon-H1-7B-Instruct --chunk-size 600000
uv run scripts/salaries_dataset/estimate_salaries.py --model meta-llama/Llama-3.1-8B-Instruct --chunk-size 600000
uv run scripts/salaries_dataset/estimate_salaries.py --model ibm-granite/granite-4.0-micro --chunk-size 600000
uv run scripts/salaries_dataset/estimate_salaries.py --model nvidia/Llama-3.1-Nemotron-Nano-8B-v1 --chunk-size 600000
uv run scripts/salaries_dataset/estimate_salaries.py --model google/gemma-3-27b-it --chunk-size 600000
send_telegram "job ${SLURM_JOB_ID} is done"
echo "Done"
