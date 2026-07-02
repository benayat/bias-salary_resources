import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from llm import LLMClient, SamplingConfig
from constants import (
    HOME_CONFIG,
    HOME_4GPU_CONFIG,
    HOME_CONFIG_SMALL,
    HPC_CONFIG,
    HPC_2H200_CONFIG,
    PERSONAS,
    SALARY_SYSTEM_PROMPT,
)


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


SALARY_USER_PROMPT = """Estimate the yearly salary in USD for this H1B job application based on the following features:
- job_title: {job_title}
- soc_code: {soc_code}
- soc_title: {soc_title}
- full_time_position: {full_time_position}
- total_worker_positions: {total_worker_positions}
- worksite_city: {worksite_city}
- worksite_state: {worksite_state}
- naics_code: {naics_code}
- pw_wage_level: {pw_wage_level}

Return only the integer amount, nothing else."""


REQUIRED_PROMPT_COLS = [
    "JOB_TITLE",
    "SOC_CODE",
    "SOC_TITLE",
    "FULL_TIME_POSITION",
    "TOTAL_WORKER_POSITIONS",
    "WORKSITE_CITY",
    "WORKSITE_STATE",
    "NAICS_CODE",
    "PW_WAGE_LEVEL",
]


OUTPUT_COLS = [
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
    "salary_parse_status",
    "salary_raw_output",
    "IS_AI",
]


COMPLETED_PARSE_STATUSES = {
    "ok",
    "empty",
    "no_number",
    "multiple_numbers_or_range",
    "non_integer_number",
    "out_of_plausible_range",
    "malformed",
}


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def get_llm_config(config_name: str):
    if config_name == "home_4gpu":
        return HOME_4GPU_CONFIG
    if config_name == "hpc":
        return HPC_CONFIG
    if config_name == "home_small":
        return HOME_CONFIG_SMALL
    if config_name == "hpc2h200":
        return HPC_2H200_CONFIG
    return HOME_CONFIG


def extract_model_size_b(model_name: str) -> float:
    """
    Match patterns like:
      3B
      70B
      8x7B
      8x22B
      235B-A22B

    Used only when --scale-model-size is enabled.
    """
    model_size_match = re.search(r"(\d+)x(\d+)[Bb]|(\d+(?:\.\d+)?)[Bb]", model_name)

    if not model_size_match:
        raise ValueError(
            f"Could not extract model size from model name: {model_name}. "
            "Expected format like '3B', '70B', '8x7B', '235B-A22B', etc."
        )

    if model_size_match.group(1) and model_size_match.group(2):
        return float(model_size_match.group(1)) * float(model_size_match.group(2))

    return float(model_size_match.group(3))


def parse_salary_output(output: object) -> tuple[float, str]:
    """
    Strict parser for the main experiment.

    Policy:
      - Accept only one annual salary number.
      - Do not concatenate digits from malformed outputs.
      - Do not impute invalid outputs as 0.
      - Do not take midpoint/mean of ranges.
      - Invalid outputs become NaN with an auditable parse status.

    Accepted examples:
      120000
      120,000
      $120000
      $120,000
      120000.00
      120000 USD
      120000 dollars

    Rejected examples:
      salary: 120000
      about 120000
      120000-140000
      120000 or 130000
      120000 130000
      I estimate 120000
    """
    text = "" if output is None else str(output).strip()

    if not text:
        return np.nan, "empty"

    # Obvious range patterns before numeric extraction.
    range_patterns = [
        r"\d[\d,]*(?:\.\d+)?\s*[-–—]\s*\$?\s*\d",
        r"\d[\d,]*(?:\.\d+)?\s+(?:to|or|and)\s+\$?\s*\d",
        r"between\s+\$?\s*\d",
        r"range",
    ]
    for pat in range_patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return np.nan, "multiple_numbers_or_range"

    # Numeric-looking salary tokens.
    numeric_tokens = re.findall(r"\$?\s*\d[\d,]*(?:\.\d+)?", text)

    if len(numeric_tokens) == 0:
        return np.nan, "no_number"

    if len(numeric_tokens) > 1:
        return np.nan, "multiple_numbers_or_range"

    # Strict full-output acceptance.
    allowed = re.fullmatch(
        r"\s*\$?\s*\d[\d,]*(?:\.0+)?\s*(?:USD|usd|dollars|Dollars)?\s*",
        text,
    )
    if not allowed:
        return np.nan, "malformed"

    token = numeric_tokens[0]
    normalized = token.replace("$", "").replace(",", "").strip()

    try:
        value = float(normalized)
    except ValueError:
        return np.nan, "malformed"

    if not value.is_integer():
        return np.nan, "non_integer_number"

    salary = int(value)

    # Plausibility guard for annual H-1B salary estimates.
    # This should catch parse artifacts, not censor valid model behavior.
    if salary < 20_000 or salary > 1_000_000:
        return np.nan, "out_of_plausible_range"

    return salary, "ok"


def build_output_frame(h1b_df: pd.DataFrame) -> pd.DataFrame:
    missing_output_cols = [c for c in OUTPUT_COLS if c not in h1b_df.columns]
    if missing_output_cols:
        raise KeyError(f"Missing required output columns: {missing_output_cols}")

    return h1b_df[OUTPUT_COLS].copy()


def init_output_state(h1b_df: pd.DataFrame) -> None:
    """
    Ensure output/resume columns exist.
    """
    if "estimated_salary_in_usd" not in h1b_df.columns:
        h1b_df["estimated_salary_in_usd"] = np.nan
    else:
        h1b_df["estimated_salary_in_usd"] = pd.to_numeric(
            h1b_df["estimated_salary_in_usd"],
            errors="coerce",
        )

    if "salary_parse_status" not in h1b_df.columns:
        h1b_df["salary_parse_status"] = pd.NA
    else:
        h1b_df["salary_parse_status"] = h1b_df["salary_parse_status"].astype("string")

    if "salary_raw_output" not in h1b_df.columns:
        h1b_df["salary_raw_output"] = pd.NA
    else:
        h1b_df["salary_raw_output"] = h1b_df["salary_raw_output"].astype("string")


def load_existing_progress(output_path: str, h1b_df: pd.DataFrame) -> set[int]:
    """
    Safe resume:
      - Reads existing salary output.
      - Copies completed estimates/status/raw output back into h1b_df.
      - Returns row indices that should not be rerun.

    A row is considered completed if salary_parse_status exists and is one of
    COMPLETED_PARSE_STATUSES. This includes parse failures, because they are
    valid model-compliance outcomes and should not be silently rerun.

    Backward compatibility:
      - If an older output lacks salary_parse_status, rows with non-null
        estimated_salary_in_usd are considered completed.
    """
    processed_indices: set[int] = set()

    if not os.path.exists(output_path):
        return processed_indices

    existing_df = pd.read_csv(output_path, low_memory=False)

    if "salary_parse_status" in existing_df.columns:
        status = existing_df["salary_parse_status"].astype("string")
        completed_mask = status.isin(COMPLETED_PARSE_STATUSES)
    elif "estimated_salary_in_usd" in existing_df.columns:
        existing_salary = pd.to_numeric(
            existing_df["estimated_salary_in_usd"],
            errors="coerce",
        )
        completed_mask = existing_salary.notna()
    else:
        return processed_indices

    processed_indices = set(existing_df.index[completed_mask].tolist())

    for idx in processed_indices:
        if idx >= len(h1b_df):
            continue

        if "estimated_salary_in_usd" in existing_df.columns:
            salary = pd.to_numeric(
                pd.Series([existing_df.at[idx, "estimated_salary_in_usd"]]),
                errors="coerce",
            ).iloc[0]
            h1b_df.at[idx, "estimated_salary_in_usd"] = salary

        if "salary_parse_status" in existing_df.columns:
            h1b_df.at[idx, "salary_parse_status"] = existing_df.at[idx, "salary_parse_status"]

        if "salary_raw_output" in existing_df.columns:
            h1b_df.at[idx, "salary_raw_output"] = existing_df.at[idx, "salary_raw_output"]

    print(f"Found existing output with {len(processed_indices)} completed rows. Resuming...")
    return processed_indices


def summarize_parse_status(h1b_df: pd.DataFrame) -> str:
    if "salary_parse_status" not in h1b_df.columns:
        return ""

    status_counts = (
        h1b_df["salary_parse_status"]
        .fillna("pending")
        .astype(str)
        .value_counts(dropna=False)
    )

    return "\n".join(f"{k}: {v}" for k, v in status_counts.items())


def main():
    parser = argparse.ArgumentParser(
        description="Estimate salaries for H1B job applications using an LLM."
    )

    parser.add_argument("--model", default=MODEL_NAME, help="LLM model name")
    parser.add_argument(
        "--client-type",
        choices=["vllm", "openai"],
        default="vllm",
        help="Type of LLM client to use",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: process first 10 rows only.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default="crap",
        help="OpenAI API key, if using OpenAI client.",
    )
    parser.add_argument(
        "--openai-base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI API base URL.",
    )
    parser.add_argument(
        "--llm-config",
        choices=["home", "home_4gpu", "hpc", "hpc2h200", "home_small"],
        default="home_small",
        help="Choose LLM configuration.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size for processing prompts.",
    )
    parser.add_argument(
        "--scale-model-size",
        action="store_true",
        help="Scale LLM configuration based on model size.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for distributed inference; overrides config.",
    )
    parser.add_argument(
        "--input-csv-file",
        type=str,
        default="data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--personas-to-use",
        nargs="*",
        default=["salary_estimator"],
        help="List of personas to use for estimation.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/h1b-lca-disclosure-data-2020-2024",
        help="Root directory for salary-estimation outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name for output directory. Defaults to input CSV stem.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save progress every N newly processed prompts.",
    )

    args = parser.parse_args()

    is_debug_mode = args.debug
    chunk_size = args.chunk_size
    save_interval = args.save_interval

    llm_config = get_llm_config(args.llm_config)

    input_csv = args.input_csv_file
    run_name = args.run_name or Path(input_csv).stem
    print(f"Using run_name: {run_name}")

    h1b_df = pd.read_csv(input_csv, low_memory=False)

    missing_prompt_cols = [c for c in REQUIRED_PROMPT_COLS if c not in h1b_df.columns]
    if missing_prompt_cols:
        raise KeyError(f"Missing required prompt columns in {input_csv}: {missing_prompt_cols}")

    # estimated_salary_in_usd / parse metadata are produced by this script.
    generated_output_cols = {
        "estimated_salary_in_usd",
        "salary_parse_status",
        "salary_raw_output",
    }
    base_output_cols = [c for c in OUTPUT_COLS if c not in generated_output_cols]
    missing_output_cols = [c for c in base_output_cols if c not in h1b_df.columns]
    if missing_output_cols:
        raise KeyError(f"Missing required output columns in {input_csv}: {missing_output_cols}")

    init_output_state(h1b_df)

    if is_debug_mode:
        h1b_df = h1b_df.head(10).copy()
        init_output_state(h1b_df)

    if args.client_type == "vllm":
        model_size_b = extract_model_size_b(args.model)

        if args.scale_model_size:
            print("model size in B:", model_size_b)
            llm_config.scale_for_model_size(model_size_b)

        if args.tensor_parallel_size is not None:
            llm_config.tensor_parallel_size = args.tensor_parallel_size
            print(f"Using tensor_parallel_size={args.tensor_parallel_size} from CLI argument")

    if args.client_type == "openai":
        from openai_llm.openai_client import (
            OpenAIConfig,
            LLMClient as OpenAILLMClient,
            SamplingConfig as OpenAISamplingConfig,
        )

        llm_config = OpenAIConfig(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
        )
        llm = OpenAILLMClient(model_name=args.model, config=llm_config)

        if "gpt-oss" in args.model:
            sampling_params = OpenAISamplingConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=6500,
            )
        else:
            sampling_params = OpenAISamplingConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=16,
            )
    else:
        llm = LLMClient(model_name=args.model, config=llm_config)

        if "deepseek" in args.model.lower():
            sampling_params = SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
            )
        else:
            sampling_params = SamplingConfig(
                temperature=0.0,
                top_p=1.0,
                max_tokens=16,
            )

    model_tag = args.model.split("/")[-1] if args.client_type == "vllm" else args.model.split("-")[0]

    out_dir = f"{args.output_root}/sampled-{run_name}/{model_tag}"
    os.makedirs(out_dir, exist_ok=True)

    for persona_name in args.personas_to_use:
        print(f"\n{'=' * 60}")
        print(f"Processing persona: {persona_name}")
        print(f"{'=' * 60}")

        if persona_name:
            system_prompt = PERSONAS.get(persona_name, "") + SALARY_SYSTEM_PROMPT
        else:
            system_prompt = SALARY_SYSTEM_PROMPT

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
                pw_wage_level=row.get("PW_WAGE_LEVEL", ""),
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

        if is_debug_mode:
            output_path = f"{out_dir}/llm_estimated_salaries_debug-{persona_name}.csv"
        else:
            output_path = f"{out_dir}/llm_estimated_salaries-{persona_name}.csv"

        processed_indices = load_existing_progress(output_path, h1b_df)

        total_processed = 0

        for chunk_prompts, chunk_indices in zip(
            chunk_list(prompts, chunk_size),
            chunk_list(row_indices, chunk_size),
        ):
            indices_to_process = [
                (i, idx)
                for i, idx in enumerate(chunk_indices)
                if idx not in processed_indices
            ]

            if not indices_to_process:
                print(f"Skipping chunk - all {len(chunk_indices)} rows already completed")
                continue

            filtered_prompts = [chunk_prompts[i] for i, _ in indices_to_process]
            filtered_indices = [idx for _, idx in indices_to_process]

            results = llm.run_batch(
                filtered_prompts,
                sampling_params,
                output_field="output",
            )

            if is_debug_mode:
                print(f"Chunk results: {results}")

            for row_idx, result in zip(filtered_indices, results):
                output = str(result.get("output", "")).strip()
                print(f"output: {output}")

                est, parse_status = parse_salary_output(output)

                h1b_df.at[row_idx, "estimated_salary_in_usd"] = est
                h1b_df.at[row_idx, "salary_parse_status"] = parse_status
                h1b_df.at[row_idx, "salary_raw_output"] = output

                total_processed += 1

                if total_processed % save_interval == 0:
                    h1b_df_copy = build_output_frame(h1b_df)
                    h1b_df_copy.to_csv(output_path, index=False)
                    print(f"Progress saved: {total_processed} new prompts processed")

        h1b_df_copy = build_output_frame(h1b_df)
        h1b_df_copy.to_csv(output_path, index=False)

        completed_total = (
            h1b_df_copy["salary_parse_status"]
            .fillna("pending")
            .astype(str)
            .isin(COMPLETED_PARSE_STATUSES)
            .sum()
        )
        ok_total = h1b_df_copy["salary_parse_status"].fillna("").eq("ok").sum()

        print(
            f"Salary estimation complete for persona '{persona_name}'. "
            f"Newly processed: {total_processed}. "
            f"Completed total: {completed_total}/{len(h1b_df_copy)}. "
            f"Parse-ok total: {ok_total}/{len(h1b_df_copy)}. "
            f"Saved: {output_path}"
        )

        print("\nParse-status summary:")
        print(summarize_parse_status(h1b_df_copy))

    llm.delete_client()

    print(f"\n{'=' * 60}")
    print("All personas processed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
