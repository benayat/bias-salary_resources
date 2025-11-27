import re

from constants import HOME_CONFIG
from constants.llm_configs import HOME_CONFIG_SMALL
from llm import LLMClient, SamplingConfig
import pandas as pd
def main():
    with open('data/salaries-for-data-science-jobs/unique_job_titles.txt' , 'r', encoding='utf-8') as f:
        unique_job_titles = [line.strip() for line in f.readlines()]

    # prompt to identify AI/ML directly-related related job titles, by the following definition: Either the job title explicitly mentions AI or ML, or it includes roles that are fundamentally centered around AI/ML tasks such as "chatbot developer", machine learning engineering, AI research, etc. Not included are roles that AI/ML indirectly supports, like data engineering or data analysis.
    AI_ML_SYSTEM_PROMPT = (
        "You are an expert at classifying job titles. "
        "Determine if the given job title is directly related to AI/ML based on the following definition: "
        "The job title explicitly mentions AI or ML in it's various paraphrases"
        "Not included are roles that AI/ML indirectly supports, like data scientist, data engineering or data analysis. "
        "Respond with 'Yes' or 'No' only."
    )
    AI_ML_USER_PROMPT = "Is the job title '{job_title}' directly related to AI/ML as per the above definition? Answer only 'Yes' or 'No'."

    model = 'Qwen/Qwen3-4B-Instruct-2507'
    model_size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model)
    if model_size_match:
        model_size_b = float(model_size_match.group(1))
        print("model size in B:", model_size_b)
        HOME_CONFIG_SMALL.scale_for_model_size(model_size_b)

    llm = LLMClient(model_name=model, config=HOME_CONFIG_SMALL)

    sampling_params = SamplingConfig( temperature=0.0, top_p=1.0, max_tokens=4)

    prompts = []
    for title in unique_job_titles:
        user_content = AI_ML_USER_PROMPT.format(job_title=title)
        messages = [
            {"role": "system", "content": AI_ML_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompt = {"messages": messages}
        prompts.append(prompt)
    results = llm.run_batch(prompts, sampling_params, output_field='output')
    # save results to 'data/salaries-for-data-science-jobs/ai_ml_job_titles_only.txt'
    data = []
    for title, result in zip(unique_job_titles, results):
        output = result['output'].strip()
        is_ai_job = output == 'Yes'
        data.append({'job_title': title, 'AI_job': is_ai_job})
    df = pd.DataFrame(data)
    df.to_csv('data/salaries-for-data-science-jobs/ai_ml_job_titles.csv', index=False)

if __name__ == '__main__':
    main()