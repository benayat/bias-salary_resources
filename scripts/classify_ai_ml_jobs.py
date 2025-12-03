import re
import argparse

from constants import HOME_CONFIG, HOME_4GPU_CONFIG, HPC_CONFIG
from constants.llm_configs import HOME_CONFIG_SMALL
from llm import LLMClient, SamplingConfig
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Classify job titles as AI/ML-specific using LLM")
    parser.add_argument("--input", default="data/h1b-lca-disclosure-data-2020-2024/unique_job_titles.txt",
                        help="Input file containing job titles (one per line)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="LLM model name")
    parser.add_argument("--llm-config", choices=["home", "home_small", "home_4gpu", "hpc"],
                        default="home_small", help="Choose LLM configuration")
    parser.add_argument("--output", default="data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        unique_job_titles = [line.strip() for line in f.readlines()]

    AI_ML_SYSTEM_PROMPT = (
        "You classify job titles as AI/ML-specific. Reply 'Yes' iff the title *intentionally* indicates AI/ML/LLM work.\n"
        "Say 'Yes' when the title contains clear AI/ML terms as whole words/phrases (not substrings): "
        "AI, A.I., Artificial Intelligence, ML, Machine Learning, Deep Learning/DL, Neural Network(s), "
        "LLM/Large Language Model(s), Generative AI/GenAI, Foundation Model(s), MLOps, AIOps; "
        "or role names like Prompt Engineer/Prompt Engineering, Chatbot Developer/Engineer, Conversational AI Engineer.\n"
        "Say 'No' for accidental matches (e.g., 'first aid' != 'ai') and for non-AI meanings of 'intelligence' (e.g., BI/Business Intelligence). "
        "'Data Engineer/Scientist/Analyst' alone is 'No', but becomes 'Yes' if it explicitly includes AI/ML terms (e.g., 'Data AI Engineer', 'BI AI Manager')."
    )
    AI_ML_USER_PROMPT = "Is the job title '{job_title}' directly related to AI/ML as per the above definition? Answer only 'Yes' or 'No'."

    # Select LLM configuration
    if args.llm_config == "home_4gpu":
        llm_config = HOME_4GPU_CONFIG
    elif args.llm_config == "hpc":
        llm_config = HPC_CONFIG
    elif args.llm_config == "home":
        llm_config = HOME_CONFIG
    else:  # home_small
        llm_config = HOME_CONFIG_SMALL

    # Scale configuration for model size
    model_size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', args.model)
    if model_size_match:
        model_size_b = float(model_size_match.group(1))
        print(f"Model size: {model_size_b}B")
        llm_config.scale_for_model_size(model_size_b)

    llm = LLMClient(model_name=args.model, config=llm_config)

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

    # Process results
    data = []
    for title, result in zip(unique_job_titles, results):
        output = result['output'].strip()
        is_ai_job = output == 'Yes'
        data.append({'job_title': title, 'AI_job': is_ai_job})

    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Classification complete. Saved results to: {args.output}")

    llm.delete_client()

if __name__ == '__main__':
    main()