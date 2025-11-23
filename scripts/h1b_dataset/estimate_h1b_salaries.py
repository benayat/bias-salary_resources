import pandas as pd
from llm import LLMClient, SamplingConfig
from constants import HOME_CONFIG
import argparse

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CHUNK_SIZE = 30000

# Salary estimation prompts
SALARY_SYSTEM_PROMPT = """You are a salary estimation assistant for H1B job applications.

You must return ONLY a single integer (the annual USD salary).
- No dollar signs, commas, decimals, or explanations
- Round to the nearest whole number
"""

SALARY_USER_PROMPT = """Estimate the yearly salary in USD for this H1B job application based on the following features:
- Job Title: {job_title}
- SOC Code: {soc_code}
- SOC Title: {soc_title}
- Wage Rate of Pay From: {wage_rate_of_pay_from}
- Wage Rate of Pay To: {wage_rate_of_pay_to}
- Wage Unit of Pay: {wage_unit_of_pay}
- Prevailing Wage: {prevailing_wage}
- Prevailing Wage Unit of Pay: {pw_unit_of_pay}
- Worksite City: {worksite_city}
- Worksite State: {worksite_state}

Return only the integer amount, nothing else."""

def main():
    parser = argparse.ArgumentParser(description="Estimate salaries for H1B job applications using LLM")
    parser.add_argument('--model', default=MODEL_NAME, help='LLM model name')
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (process first 10 rows)")
    args = parser.parse_args()
    is_debug_mode = args.debug

    # Load the dataset
    h1b_df = pd.read_csv('data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv', low_memory=False)
    if is_debug_mode:
        h1b_df = h1b_df.head(10)

    llm = LLMClient(model_name=args.model, config=HOME_CONFIG)
    sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=10)

    def chunk_list(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    # Prepare prompts
    prompts = []
    row_indices = []
    for idx, row in h1b_df.iterrows():
        user_content = SALARY_USER_PROMPT.format(
            job_title=row['JOB_TITLE'],
            soc_code=row['SOC_CODE'],
            soc_title=row['SOC_TITLE'],
            wage_rate_of_pay_from=row['WAGE_RATE_OF_PAY_FROM'],
            wage_rate_of_pay_to=row['WAGE_RATE_OF_PAY_TO'],
            wage_unit_of_pay=row['WAGE_UNIT_OF_PAY'],
            prevailing_wage=row['PREVAILING_WAGE'],
            pw_unit_of_pay=row['PW_UNIT_OF_PAY'],
            worksite_city=row['WORKSITE_CITY'],
            worksite_state=row['WORKSITE_STATE']
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
        h1b_df.at[idx, 'estimated_salary_in_usd'] = est

    # Save to CSV
    output_path = 'data/h1b-lca-disclosure-data-2020-2024/llm_estimated_salaries.csv'
    if is_debug_mode:
        output_path = 'data/h1b-lca-disclosure-data-2020-2024/llm_estimated_salaries_debug.csv'
    h1b_df.to_csv(output_path, index=False)

    llm.delete_client()

    print("Salary estimation complete.")

if __name__ == '__main__':
    main()
