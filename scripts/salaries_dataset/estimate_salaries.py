import pandas as pd
from llm import LLMClient, SamplingConfig
from constants import HOME_CONFIG, HOME_4GPU_CONFIG, HPC_CONFIG
import argparse

# set env var VLLM_ENABLE_V1_MULTIPROCESSING to 0 to avoid multiprocessing issues on some systems
# os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

# Correct the scope of MODEL_NAME by ensuring it is passed correctly
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
CHUNK_SIZE=30000
LLM_CONFIG=HOME_CONFIG
# LLM_CONFIG=HOME_4GPU_CONFIG

# Salary estimation prompts
SALARY_SYSTEM_PROMPT = """You are a salary estimation assistant for data science jobs.

You must return ONLY a single integer (the annual USD salary).
- No dollar signs, commas, decimals, or explanations
- Round to the nearest whole number
"""

SALARY_USER_PROMPT = """Estimate the yearly salary in USD for this data science job based on the following features:
- work_year: {work_year}
- experience_level: {experience_level}
- employment_type: {employment_type}
- job_title: {job_title}
- salary_currency: {salary_currency}
- employee_residence: {employee_residence}
- remote_ratio: {remote_ratio}
- company_location: {company_location}
- company_size: {company_size}

Return only the integer amount, nothing else."""

def normalize_codes(df: pd.DataFrame) -> pd.DataFrame:
    exp_map = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior-level", "EX": "Executive"}
    emp_map = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
    remote_map = {0: "On-site", 50: "Hybrid", 100: "Fully remote"}
    size_map = {"S": "Small", "M": "Medium", "L": "Large"}

    if "experience_level" in df.columns:
        df["experience_level"] = df["experience_level"].replace(exp_map)
    if "employment_type" in df.columns:
        df["employment_type"] = df["employment_type"].replace(emp_map)
    if "remote_ratio" in df.columns:
        df["remote_ratio"] = df["remote_ratio"].replace(remote_map)
    if "company_size" in df.columns:
        df["company_size"] = df["company_size"].replace(size_map)

    return df

def main():
    global MODEL_NAME
    parser = argparse.ArgumentParser(description="Estimate salaries for data science jobs using LLM")
    # Explicitly pass MODEL_NAME as a default value in the argument parser
    parser.add_argument('--model', default="meta-llama/Llama-3.2-3B-Instruct", help='LLM model name')
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (process first 10 rows)")
    parser.add_argument("--llm-config", choices=['home', 'home_4gpu', 'hpc'], default='home', help="Choose LLM configuration")
    # Add CLI argument for chunk size
    parser.add_argument('--chunk-size', type=int, default=30000, help='Chunk size for processing prompts')
    args = parser.parse_args()
    is_debug_mode = args.debug

    # Update LLM configuration based on CLI arguments
    if args.llm_config == 'home_4gpu':
        LLM_CONFIG = HOME_4GPU_CONFIG
    elif args.llm_config == 'hpc':
        LLM_CONFIG = HPC_CONFIG
    else:
        LLM_CONFIG = HOME_CONFIG

    # Update model name
    MODEL_NAME = args.model
    CHUNK_SIZE=args.chunk_size
    salaries_df = pd.read_csv('data/salaries-for-data-science-jobs/salaries.csv')
    salaries_df = normalize_codes(salaries_df)
    # Apply debug mode if enabled
    if is_debug_mode:
        salaries_df = salaries_df.head(10)

    llm = LLMClient(model_name=args.model, config=LLM_CONFIG)
    sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=10)
    
    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    # Prepare prompts
    prompts = []
    row_indices = []
    for idx, row in salaries_df.iterrows():
        user_content = SALARY_USER_PROMPT.format(
            work_year=row['work_year'],
            experience_level=row['experience_level'],
            employment_type=row['employment_type'],
            job_title=row['job_title'],
            salary_currency=row['salary_currency'],
            employee_residence=row['employee_residence'],
            remote_ratio=row['remote_ratio'],
            company_location=row['company_location'],
            company_size=row['company_size']
        )
        messages = [
            {"role": "system", "content": SALARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompts.append({"messages": messages})
        row_indices.append(idx)

    estimated_salaries = []
    for chunk_prompts, chunk_indices in zip(chunk_list(prompts, CHUNK_SIZE), chunk_list(row_indices, CHUNK_SIZE)):
        results = llm.run_batch(chunk_prompts, sampling_params, output_field='output')
        for idx, result in zip(chunk_indices, results):
            output = result['output'].strip()
            # Extract digits
            digits = ''.join(c for c in output if c.isdigit())
            try:
                estimated_salary = int(digits)
            except ValueError:
                estimated_salary = 0  # or some default
            estimated_salaries.append((idx, estimated_salary))

    # Update the dataframe
    for idx, est in estimated_salaries:
        salaries_df.at[idx, 'estimated_salary_in_usd'] = est

    # Save to CSV
    output_path = f'data/salaries-for-data-science-jobs/llm_estimated_salaries{args.model.split("/")[-1]}.csv'
    if is_debug_mode:
        output_path = f'data/salaries-for-data-science-jobs/llm_estimated_salaries_debug{args.model.split("/")[-1]}.csv'
    salaries_df.to_csv(output_path, index=False)

    llm.delete_client()

    print("Salary estimation complete.")

if __name__ == '__main__':
    main()
