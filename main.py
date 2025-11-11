import argparse
import os
import json
import csv
from llm import LLMClient, SamplingConfig
from constants import BLS_PROMPT, SYSTEM_MINIMAL, HOME_CONFIG, DEFAULT_SAMPLING_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Run BLS compensation estimates for jobs in jobs.txt")
    parser.add_argument('--model', required=True, help='LLM model name (e.g., meta-llama/Llama-2-7b-chat-hf)')
    args = parser.parse_args()

    client = LLMClient(model_name=args.model, config=HOME_CONFIG)

    # Read jobs from table.csv (first column)
    with open('table.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        jobs = [row[0] for row in reader]

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

    # Run the batch with all prompts
    sampling_params = DEFAULT_SAMPLING_CONFIG
    results = client.run_batch(prompts, sampling_params)

    # Process results
    for (soc, title), result in zip(job_data, results):
        output = result['output']

        # Parse the JSON output
        try:
            # Check if output is wrapped in code blocks
            import re
            json_match = re.search(r'```\s*\n(.*?)\n```', output, re.DOTALL)
            if json_match:
                output = json_match.group(1)
            data = json.loads(output)
            data['soc_code'] = soc  # Include original SOC code as identifier
        except json.JSONDecodeError:
            data = {"error": "Invalid JSON output", "raw_output": output, "soc_code": soc}

        # Write to JSONL file (one per job)
        filename = f"data/bls_evals/{soc}.jsonl"
        with open(filename, 'w') as f:
            f.write(json.dumps(data) + '\n')

        print(f"Processed {soc}: {title}")

    client.delete_client()
    print("All jobs processed. Results saved in data/bls_evals/")

if __name__ == '__main__':
    main()