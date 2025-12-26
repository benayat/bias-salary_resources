#!/bin/bash
#SBATCH --job-name=salaries_pipeline_h1b_medium_models
#SBATCH --output=/home/fast/trabelb1/projects/bias-salary_resources/h1b_pipeline%j.out
#SBATCH --error=/home/fast/trabelb1/projects/bias-salary_resources/h1b_pipeline%j.err
#SBATCH --partition=p_b200_kraus
#SBATCH --account=ug_kraus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
source ~/.bash_profile
REPO_LOCATION="/home/fast/trabelb1/projects/bias-salary_resources"
nvidia-smi
cd $REPO_LOCATION
export PYTHONPATH=$(pwd)
./run_all_personas_uniform_single_model.sh mistralai/Ministral-3-14B-Instruct-2512 sqrt home
./run_all_personas_uniform_single_model.sh microsoft/phi-4 sqrt home
./run_all_personas_uniform_single_model.sh Qwen/Qwen3-30B-A3B sqrt home
./run_all_personas_uniform_single_model.sh Qwen/Qwen3-32B sqrt home
./run_all_personas_uniform_single_model.sh nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4 sqrt home

send_telegram "job ${SLURM_JOB_ID} is done"
echo "Done"
