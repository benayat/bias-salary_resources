import os
import re

import pandas as pd
import numpy as np
from llm import LLMClient, SamplingConfig
from constants import HOME_CONFIG, HOME_4GPU_CONFIG, HOME_CONFIG_SMALL, HPC_CONFIG, HPC_2H200_CONFIG, PERSONAS, PERSONA_COMPLETION_SUFFIX, SALARY_SYSTEM_PROMPT
import argparse


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CHUNK_SIZE = 30000

# Salary estimation prompts
# SALARY_SYSTEM_PROMPT = """You are a salary estimation assistant for H1B job applications.
#
# You must return ONLY a single integer (the annual USD salary).
# - No dollar signs, commas, decimals, or explanations
# - Round to the nearest whole number
# """

SALARY_USER_PROMPT = """Estimate the yearly salary in USD for this H1B job application based on the following features:
- job_title: {job_title}
- soc_code: {soc_code}
- soc_title: {soc_title}
- full_time_position: {full_time_position}
- total_worker_positions: {total_worker_positions}
- worksite_city: {worksite_city}
- worksite_state: {worksite_state}
- naics_code: {naics_code}

Return only the integer amount, nothing else."""


def main():
    parser = argparse.ArgumentParser(description="Estimate salaries for H1B job applications using LLM")
    parser.add_argument("--model", default=MODEL_NAME, help="LLM model name")
    parser.add_argument("--client-type", choices=["vllm", "openai"], default="vllm", help="Type of LLM client to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (process first 10 rows)")
    parser.add_argument("--openai-api-key", type=str, default="", help="OpenAI API key (if using OpenAI client)")
    parser.add_argument("--llm-config", choices=["home", "home_4gpu", "hpc", "hpc2h200", "home_small"], default="home", help="Choose LLM configuration")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for processing prompts")
    parser.add_argument("--scale-model-size", action="store_true", help="Scale LLM configuration based on model size")
    parser.add_argument("--input-csv-file", type=str, default="data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled.csv", help="Path to input CSV file")
    parser.add_argument("--personas-to-use", nargs="*", default=["salary_estimator"], help="List of personas to use for estimation")
    args = parser.parse_args()

    is_debug_mode = args.debug
    chunk_size = args.chunk_size

    # Update LLM configuration based on CLI arguments
    if args.llm_config == "home_4gpu":
        llm_config = HOME_4GPU_CONFIG
    elif args.llm_config == "hpc":
        llm_config = HPC_CONFIG
    elif args.llm_config == "home_small":
        llm_config = HOME_CONFIG_SMALL
    elif args.llm_config == "hpc2h200":
        llm_config = HPC_2H200_CONFIG
    else:
        llm_config = HOME_CONFIG

    # Load the dataset
    input_csv = args.input_csv_file
    derived_soc_weight_mode = input_csv.split("-")[-1].replace(".csv", "")
    print(f"Using derived_soc_weight_mode: {derived_soc_weight_mode}")
    h1b_df = pd.read_csv(input_csv, low_memory=False)

    # Keep only columns we need + keep everything else (you can drop if you want)
    required_cols = [
        "JOB_TITLE",
        "SOC_CODE",
        "SOC_TITLE",
        "FULL_TIME_POSITION",
        "TOTAL_WORKER_POSITIONS",
        "WORKSITE_CITY",
        "WORKSITE_STATE",
        "NAICS_CODE",
    ]
    missing = [c for c in required_cols if c not in h1b_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {input_csv}: {missing}")

    if is_debug_mode:
        h1b_df = h1b_df.head(10).copy()
    if args.scale_model_size:
        model_size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', args.model)
        if model_size_match:
            model_size_b = float(model_size_match.group(1))
            print("model size in B:", model_size_b)
            llm_config.scale_for_model_size(model_size_b)

    if args.client_type == "openai":
        from openai_llm.openai_client import OpenAIConfig, LLMClient as OpenAILLMClient
        llm_config = OpenAIConfig(api_key=args.openai_api_key)
        # llm_config = OpenAIConfig(api_key=os.getenv("OPENAI_API_KEY"))
        llm = OpenAILLMClient(model_name=args.model, config=llm_config)
        sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=10)
    else:
        llm = LLMClient(model_name=args.model, config=llm_config)
        sampling_params = SamplingConfig(temperature=0.0, top_p=1.0, max_tokens=10)

    def chunk_list(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]

    model_tag = args.model.split("/")[-1]
    out_dir = f"data/h1b-lca-disclosure-data-2020-2024/sampled-{derived_soc_weight_mode}/{model_tag}"
    os.makedirs(out_dir, exist_ok=True)

    # Loop through each persona
    for persona_name in args.personas_to_use:
        print(f"\n{'='*60}")
        print(f"Processing persona: {persona_name}")
        print(f"{'='*60}")

        if persona_name:
            system_prompt = PERSONAS.get(persona_name, "") + SALARY_SYSTEM_PROMPT
        else:
            system_prompt = SALARY_SYSTEM_PROMPT

        # Prepare prompts
        prompts = []
        row_indices = []
        for idx, row in h1b_df.iterrows():
            user_content = SALARY_USER_PROMPT.format(
                job_title=row.get("JOB_TITLE", ""),
                soc_code=row.get("SOC_CODE", ""),
                soc_title=row.get("SOC_TITLE", ""),
                full_time_position=row.get("FULL_TIME_POSITION", ""),
                total_worker_positions=row.get("TOTAL_WORKER_POSITIONS", ""),
                worksite_city=row.get("WORKSITE_CITY", ""),
                worksite_state=row.get("WORKSITE_STATE", ""),
                naics_code=row.get("NAICS_CODE", ""),
            )
            prompts.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ]
                }
            )
            row_indices.append(idx)

        # Run in chunks
        estimated_salaries = []
        for chunk_prompts, chunk_indices in zip(chunk_list(prompts, chunk_size), chunk_list(row_indices, chunk_size)):
            results = llm.run_batch(chunk_prompts, sampling_params, output_field="output")
            for row_idx, result in zip(chunk_indices, results):
                output = str(result.get("output", "")).strip()
                digits = "".join(c for c in output if c.isdigit())
                try:
                    est = int(digits)
                except ValueError:
                    est = 0  # default for invalid outputs
                estimated_salaries.append((row_idx, est))

        # Write predictions back
        h1b_df_copy = h1b_df.copy()
        h1b_df_copy["estimated_salary_in_usd"] = np.nan
        for row_idx, est in estimated_salaries:
            h1b_df_copy.at[row_idx, "estimated_salary_in_usd"] = est

        output_cols = [
            "ROW_ID",
            "JOB_TITLE",
            "SOC_CODE",
            "SOC_TITLE",
            "WORKSITE_STATE",
            "NAICS_CODE",
            "FULL_TIME_POSITION",
            "TOTAL_WORKER_POSITIONS",
            "PREVAILING_WAGE",
            "estimated_salary_in_usd",
            "IS_AI",
            "PAIR_ID",
            "MATCH_LEVEL",
        ]
        h1b_df_copy = h1b_df_copy[output_cols]

        # Save
        if is_debug_mode:
            output_path = f"{out_dir}/llm_estimated_salaries_debug-{persona_name}.csv"
        else:
            output_path = f"{out_dir}/llm_estimated_salaries-{persona_name}.csv"

        h1b_df_copy.to_csv(output_path, index=False)
        print(f"Salary estimation complete for persona '{persona_name}'. Saved: {output_path}")

    llm.delete_client()
    print(f"\n{'='*60}")
    print(f"All personas processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
