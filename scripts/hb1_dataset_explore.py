import pandas as pd
from llm.llm_client import LLMClient, SamplingConfig
from constants import HOME_CONFIG
import argparse

# Script constants
KAGGLE_DATASET_ZIP_PATH = '../data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv'
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LLM_CONFIG = HOME_CONFIG

# AI/ML classification prompts
AI_ML_SYSTEM_PROMPT = "You are a job title classifier. Determine if the job title directly mentions AI or Machine Learning."
AI_ML_USER_PROMPT = "Is '{job_title}' directly related to AI or Machine Learning? Answer only 'yes' or 'no'."

# Add CLI argument for chunk size
parser = argparse.ArgumentParser()
parser.add_argument('--chunk-size', type=int, default=30000, help='Chunk size for processing prompts')
args = parser.parse_args()

df = pd.read_csv(KAGGLE_DATASET_ZIP_PATH, low_memory=False)
print(df.head())
print(df.columns.tolist())
# print(f"Total records: {len(df)}")
# print(f"Unique job titles: {df['JOB_TITLE'].nunique()}")
# print(f"Unique SOC codes: {df['SOC_CODE'].nunique()}")
# print(f"job titles containint AI or ML: {df[df['JOB_TITLE'].str.contains('AI|ML', case=False, na=False)]['JOB_TITLE'].unique()}")
## retrieve all unique job titles and for each validate with llm if needed later
unique_job_titles = df['JOB_TITLE'].dropna().unique().tolist()
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

# Process in chunks
ai_ml_related_titles = []
for i in range(0, len(prompts), args.chunk_size):
    chunk = prompts[i:i + args.chunk_size]
    results = llm.run_batch(chunk, sampling_params)

    for title, result in zip(unique_job_titles[i:i + args.chunk_size], results):
        output = result['output'].lower().strip()
        if output == 'yes':
            ai_ml_related_titles.append(title)

llm.delete_client()

# Normalize prevailing wage to numeric and convert to annual where unit is provided
df['prevailing_wage_num'] = pd.to_numeric(df['PREVAILING_WAGE'], errors='coerce')

def _to_annual(row):
    v = row['prevailing_wage_num']
    if pd.isna(v):
        return None
    unit = str(row.get('PW_UNIT_OF_PAY', '')).lower()
    if 'hour' in unit:
        return v * 2080  # 40 hrs/week * 52 weeks
    if 'week' in unit:
        return v * 52
    if 'month' in unit:
        return v * 12
    # assume yearly or unspecified -> treat as annual
    return v

df['prevailing_wage_annual'] = df.apply(_to_annual, axis=1)

# Create dataframe with median prevailing wage per job title
median_wage_df = (
    df.dropna(subset=['JOB_TITLE', 'prevailing_wage_annual'])
      .groupby('JOB_TITLE', as_index=False)
      .agg(median_prevailing_wage=('prevailing_wage_annual', 'median'),
           observations=('prevailing_wage_annual', 'count'))
      .sort_values('median_prevailing_wage', ascending=False)
)

print(median_wage_df.head())
# save to csv
median_wage_df.to_csv('../data/h1b_median_prevailing_wages_by_job_title.csv', index=False)

print(f"AI/ML related titles: {ai_ml_related_titles}")
with open('../data/ai_ml_job_titles.txt', 'w') as f:
    f.write('\n'.join(ai_ml_related_titles))
