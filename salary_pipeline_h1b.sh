#!/bin/bash
#SBATCH --job-name=salary_pipeline_big_models
#SBATCH --output=/home/fast/trabelb1/projects/bias-salary_resources/salary_pipeline%j.out
#SBATCH --error=/home/fast/trabelb1/projects/bias-salary_resources/salary_pipeline_err%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
source ~/.bash_profile
REPO_LOCATION="/home/fast/trabelb1/projects/bias-salary_resources"
nvidia-smi
cd $REPO_LOCATION
export PYTHONPATH=$(pwd)
./run_all_personas_uniform_single_model.sh openai/gpt-oss-20b sqrt
#./run_all_personas_uniform_single_model.sh RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4 sqrt
send_telegram "job ${SLURM_JOB_ID} is done"
echo "Done"
