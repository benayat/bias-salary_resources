import pandas as pd
from llm.llm_client import LLMClient, SamplingConfig
from constants import HOME_CONFIG, HOME_4GPU_CONFIG
import argparse

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# AI/ML classification prompts
AI_ML_SYSTEM_PROMPT = "You are a job title classifier. Determine if the job title directly mentions AI or Machine Learning."
AI_ML_USER_PROMPT = "Does '{job_title}' directly mention to AI or Machine Learning in any way? Answer only 'yes' or 'no'."

median_wage_df = pd.read_csv('data/h1b_median_prevailing_wages_by_job_title.csv')
unique_job_titles = median_wage_df['JOB_TITLE'].dropna().unique().tolist()

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Classify AI/ML job titles using LLM")
parser.add_argument('--model', default=MODEL_NAME, help='LLM model name')
parser.add_argument("--debug", action="store_true", help="Enable debug mode (process first 10 rows)")
# Add CLI argument for 4-GPU setup
parser.add_argument("--use-4gpu", action="store_true", help="Enable 4-GPU setup for LLM configuration")
# Add CLI argument for chunk size
parser.add_argument('--chunk-size', type=int, default=30000, help='Chunk size for processing prompts')
args = parser.parse_args()
is_debug_mode = args.debug

# Update model name and LLM config based on CLI arguments
MODEL_NAME = args.model
llm_config = HOME_4GPU_CONFIG if args.use_4gpu else HOME_CONFIG
llm = LLMClient(model_name=MODEL_NAME, config=llm_config)

# Apply debug mode if enabled
if is_debug_mode:
    unique_job_titles = unique_job_titles[:10]

sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=4)


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

ai_ml_related_titles = []

for chunk in chunk_list(unique_job_titles, args.chunk_size):
    prompts = []
    for title in chunk:
        user_content = AI_ML_USER_PROMPT.format(job_title=title)
        messages = [
            {"role": "system", "content": AI_ML_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompts.append({"messages": messages})

    results = llm.run_batch(prompts, sampling_params, output_field='output')
    for title, result in zip(chunk, results):
        output = result['output'].lower().strip()
        if output == 'yes':
            ai_ml_related_titles.append(title)

llm.delete_client()

print(f"AI/ML related titles: {ai_ml_related_titles}")
with open('data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.txt', 'w') as f:
    f.write('\n'.join(ai_ml_related_titles))

print("Classification complete.")
