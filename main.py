import argparse
import os
import json
import csv
import re
from llm import LLMClient, SamplingConfig
from constants import BLS_PROMPT, SYSTEM_MINIMAL, HOME_CONFIG, DEFAULT_SAMPLING_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Run BLS compensation estimates for jobs in jobs.txt")
    parser.add_argument('--model', default="meta-llama/Llama-3.2-3B-Instruct", help='LLM model name (e.g., meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    is_debug_mode = args.debug
    model_size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', args.model)
    if model_size_match:
        model_size_b = float(model_size_match.group(1))
        HOME_CONFIG.scale_for_model_size(model_size_b)

    client = LLMClient(model_name=args.model, config=HOME_CONFIG)

    # Read jobs from table.csv (first column)
    with open('data/summaries&archives/table.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        # if in debug mode take only first 5 jobs
        jobs = [row[0] for row in reader] if not is_debug_mode else [row[0] for i, row in enumerate(reader) if i < 5]

    # Ensure output directory exists
    os.makedirs('data/bls_evals', exist_ok=True)

    # Prepare all prompts
    prompts = []
    job_data = []
    for job in jobs:
        # Parse job line: "Title (SOC code)"
        if '(' in job and job.endswith(')'):
            title = job.rsplit('(', 1)[0].strip()
            soc = job.rsplit('(', 1)[1].rstrip(')').strip()
        else:
            print(f"Skipping invalid job line: {job}")
            continue

        # Format the BLS prompt
        user_content = BLS_PROMPT.format(role_description=title, as_of_date="2024-05-01")
        messages = [
            {"role": "system", "content": SYSTEM_MINIMAL},
            {"role": "user", "content": user_content}
        ]
        prompt = {"messages": messages}
        prompts.append(prompt)
        job_data.append((soc, title))

    # Run the initial batch with all prompts
    sampling_params = DEFAULT_SAMPLING_CONFIG
    results = client.run_batch(prompts, sampling_params)

    data = []
    # Process initial results
    for (soc, title), result in zip(job_data, results):
        output = ''.join(c for c in result['output'] if c.isdigit())

        if output.lower() == 'null':
            data_temp = {"error": "Null output", "raw_output": output, "soc_code": soc}
        else:
            try:
                estimate_usd = float(output)
                data_temp = {"estimate_usd": estimate_usd, "soc_code": soc}
            except ValueError:
                data_temp = {"error": "Invalid number output", "raw_output": output, "soc_code": soc}

        data.append(data_temp)


    with open(f'data/bls_evals/bls_results_{args.model.split("/")[-1]}.jsonl', 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    client.delete_client()
    print("All jobs processed. Results saved in data/bls_evals/")

if __name__ == '__main__':
    main()
