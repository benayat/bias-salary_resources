import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from llm.llm_client import LLMClient, SamplingConfig
from constants import HOME_CONFIG

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LLM_CONFIG = HOME_CONFIG

# AI/ML classification prompts
AI_ML_SYSTEM_PROMPT = "You are a job title classifier. Determine if the job title directly mentions AI or Machine Learning."
AI_ML_USER_PROMPT = "Is '{job_title}' directly related to AI or Machine Learning? Answer only 'yes' or 'no'."

median_salary_df = pd.read_csv('../../data/salaries-for-data-science-jobs/salaries_median_by_job_title.csv')
unique_job_titles = median_salary_df['job_title'].dropna().unique().tolist()
llm = LLMClient(model_name=MODEL_NAME, config=LLM_CONFIG)
sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=4)

prompts = []
for title in unique_job_titles:
    user_content = AI_ML_USER_PROMPT.format(job_title=title)
    messages = [
        {"role": "system", "content": AI_ML_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    prompt = {"messages": messages}
    prompts.append(prompt)

results = llm.run_batch(prompts, sampling_params)

ai_ml_related_titles = []
for title, result in zip(unique_job_titles, results):
    output = result['output'].lower().strip()
    if output == 'yes':
        ai_ml_related_titles.append(title)

llm.delete_client()

print(f"AI/ML related titles: {ai_ml_related_titles}")
with open('../data/salaries_ai_ml_job_titles.txt', 'w') as f:
    f.write('\n'.join(ai_ml_related_titles))

print("Classification complete.")
