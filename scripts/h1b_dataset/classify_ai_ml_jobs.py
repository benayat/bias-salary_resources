#!/usr/bin/env python3
import argparse
import os
from typing import List

import pandas as pd

from llm.llm_client import LLMClient, SamplingConfig
from constants import HOME_CONFIG, HOME_4GPU_CONFIG, HPC_CONFIG


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# AI/ML classification prompts
AI_ML_SYSTEM_PROMPT = (
    "You are a job title classifier. Determine if the job title directly "
    "mentions AI or Machine Learning."
)
AI_ML_USER_PROMPT = (
    "Does '{job_title}' directly mention to AI or Machine Learning in any way? "
    "Answer only 'yes' or 'no'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify AI/ML job titles using LLMClient with pre-tokenization"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="LLM model name (HF/vLLM-compatible)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process first 10 titles)",
    )
    parser.add_argument(
        "--llm-config",
        choices=["home", "home_4gpu", "hpc"],
        default="home",
        help="Choose LLM configuration",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30_000,
        help="Chunk size for processing prompts",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.txt",
        help="Output path for AI/ML-related job titles (one per line)",
    )
    parser.add_argument(
        "--disable-pre-tokenization",
        action="store_true",
        help="Let vLLM handle tokenization (slower, simpler path)",
    )
    return parser.parse_args()


def chunk_list(lst, chunk_size: int):
    """Yield consecutive chunks from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def main():
    args = parse_args()

    # -------------------------
    # Load job titles
    # -------------------------
    df = pd.read_csv("data/h1b_median_prevailing_wages_by_job_title.csv")
    job_titles = df["JOB_TITLE"].dropna().unique().tolist()

    if args.debug:
        job_titles = job_titles[:10]
        print(f"Debug mode enabled: using first {len(job_titles)} titles.")

    print(f"Total unique job titles: {len(job_titles)}")

    # -------------------------
    # Choose LLM config
    # -------------------------
    if args.llm_config == "home_4gpu":
        llm_config = HOME_4GPU_CONFIG
    elif args.llm_config == "hpc":
        llm_config = HPC_CONFIG
    else:
        llm_config = HOME_CONFIG

    model_name = args.model

    # -------------------------
    # Init LLM client
    # -------------------------
    llm_client = LLMClient(model_name=model_name, config=llm_config)

    sampling_cfg = SamplingConfig(
        temperature=0.0,
        top_p=1.0,
        max_tokens=4,
    )

    ai_ml_related_titles: List[str] = []

    # -------------------------
    # Main loop: build messages → LLMClient.run_batch
    # -------------------------
    for titles_chunk in chunk_list(job_titles, args.chunk_size):
        print(f"Processing chunk of size {len(titles_chunk)}...")

        # Build prompts in the format expected by LLMClient.run_batch
        prompts = []
        for title in titles_chunk:
            user_content = AI_ML_USER_PROMPT.format(job_title=title)
            messages = [
                {"role": "system", "content": AI_ML_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            prompts.append(
                {
                    "messages": messages,
                    "metadata": {"job_title": title},
                }
            )

        results = llm_client.run_batch(
            prompts=prompts,
            sampling_params=sampling_cfg,
            output_field="output",
            disable_pre_tokenization=args.disable_pre_tokenization,
        )

        # Interpret outputs as yes/no classification
        for res in results:
            output_text = res["output"].strip().lower()
            title = res["job_title"]
            if output_text == "yes":
                ai_ml_related_titles.append(title)

    # -------------------------
    # Cleanup + save results
    # -------------------------
    llm_client.delete_client()

    out_path = args.out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ai_ml_related_titles))

    print(f"\nAI/ML related titles: {len(ai_ml_related_titles)}")
    print(f"Written to: {out_path}")
    print("Classification complete.")


if __name__ == "__main__":
    main()
