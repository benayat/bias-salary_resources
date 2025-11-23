import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from llm.llm_client import LLMClient, SamplingConfig
from constants import HOME_CONFIG, HOME_4GPU_CONFIG

import argparse

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LLM_CONFIG = HOME_CONFIG

# AI/ML classification prompts
AI_ML_SYSTEM_PROMPT = "You are a job title classifier. Determine if the job title directly mentions AI or Machine Learning."
AI_ML_USER_PROMPT = "Does '{job_title}' directly mention to AI or Machine Learning? Answer only 'yes' or 'no'."

median_salary_df = pd.read_csv('data/salaries-for-data-science-jobs/salaries_median_by_job_title.csv')
unique_job_titles = median_salary_df['job_title'].dropna().unique().tolist()
llm = LLMClient(model_name=MODEL_NAME, config=LLM_CONFIG)
sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=4)

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Classify AI/ML job titles using LLM")
parser.add_argument('--model', default=MODEL_NAME, help='LLM model name')
parser.add_argument("--debug", action="store_true", help="Enable debug mode (process first 10 rows)")
parser.add_argument("--llm-config", choices=['home', 'home_4gpu', 'hpc'], default='home', help="Choose LLM configuration")
# Add CLI argument for chunk size
parser.add_argument('--chunk-size', type=int, default=30000, help='Chunk size for processing prompts')
args = parser.parse_args()
is_debug_mode = args.debug
chunk_size = args.chunk_size

# Update LLM configuration based on CLI arguments
if args.llm_config == 'home_4gpu':
    LLM_CONFIG = HOME_4GPU_CONFIG
elif args.llm_config == 'hpc':
    LLM_CONFIG = HPC_CONFIG
else:
    LLM_CONFIG = HOME_CONFIG

# Update model name
MODEL_NAME = args.model

# Apply debug mode if enabled
if is_debug_mode:
    unique_job_titles = unique_job_titles[:10]

# Chunk processing
prompts = []
for title in unique_job_titles:
    user_content = AI_ML_USER_PROMPT.format(job_title=title)
    messages = [
        {"role": "system", "content": AI_ML_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    prompt = {"messages": messages}
    prompts.append(prompt)

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

ai_ml_related_titles = []
for chunk in chunk_list(prompts, chunk_size):
    results = llm.run_batch(chunk, sampling_params)
    for title, result in zip(unique_job_titles, results):
        output = result['output'].lower().strip()
        if output == 'yes':
            ai_ml_related_titles.append(title)

llm.delete_client()

print(f"AI/ML related titles: {ai_ml_related_titles}")
with open('data/salaries-for-data-science-jobs/salaries_ai_ml_job_titles.txt', 'w') as f:
    f.write('\n'.join(ai_ml_related_titles))

print("Classification complete.")
