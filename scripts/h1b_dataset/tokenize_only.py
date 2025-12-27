#!/usr/bin/env python3
import argparse
import os
import time
from typing import List, Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from constants import HOME_CONFIG, HOME_4GPU_CONFIG, HPC_CONFIG


MODEL_NAME = "meta-maverick-llama3.3-70b-instruct-api-llama/Llama-3.2-3B-Instruct"

AI_ML_SYSTEM_PROMPT = (
    "You are a job title classifier. Determine if the job title directly "
    "mentions AI or Machine Learning."
)
AI_ML_USER_PROMPT = (
    "Does '{job_title}' directly mention to AI or Machine Learning in any way? "
    "Answer only 'yes' or 'no'."
)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize AI/ML job title prompts (no vLLM, tokenization only)"
    )
    parser.add_argument("--model", default=MODEL_NAME, help="HF model name")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process first 10 titles)",
    )
    parser.add_argument(
        "--llm-config",
        choices=["home", "home_4gpu", "hpc"],
        default="home",
        help="Choose LLM configuration (for trust_remote_code etc.)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size for processing prompts",
    )
    parser.add_argument(
        "--out-stats",
        type=str,
        default="data/h1b_ai_ml_job_titles_token_lengths.csv",
        help="Path to save token length stats as CSV",
    )
    parser.add_argument(
        "--arrow-path",
        type=str,
        default="data/h1b_tokenized/job_title_tokens.arrow",
        help="Path for the single Arrow file with all tokenized prompts",
    )
    return parser.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def chunk_list(lst, chunk_size):
    """Yield consecutive chunks from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def build_messages(job_title: str) -> List[Dict[str, str]]:
    """Build HF-style chat messages for a single job title."""
    user_content = AI_ML_USER_PROMPT.format(job_title=job_title)
    return [
        {"role": "system", "content": AI_ML_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def messages_to_text(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Convert messages to a single text using the chat template, without tokenizing.

    This matches the pipeline we want:
      - tokenize=False
      - add_generation_prompt=True
      - chat_template_kwargs={"enable_thinking": False}
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )


def tokenize_batch_text(
        tokenizer: AutoTokenizer,
        prompts: List[Dict[str, Any]],
) -> List[List[int]]:
    """
    Tokenize via:
      - messages -> text via chat template (tokenize=False)
      - batch call tokenizer(texts, add_special_tokens=False)

    This is the fast path:
      text = apply_chat_template(..., tokenize=False)
      ids  = tokenizer(text, add_special_tokens=False)
    """
    messages_list = [p["messages"] for p in prompts]

    # Build chat-formatted texts (with tqdm for visibility)
    texts = [
        messages_to_text(tokenizer, msgs)
        for msgs in tqdm(messages_list, desc="Building chat texts")
    ]

    # Batch tokenization: HF fast tokenizer will parallelize internally
    enc = tokenizer(
        texts,
        padding=False,
        truncation=False,
        add_special_tokens=False,  # chat template already includes special tokens
        return_attention_mask=False,
    )
    return enc["input_ids"]


# -----------------------------
# Arrow saving (single file)
# -----------------------------
def save_all_to_arrow(
        job_titles: List[str],
        token_ids_list: List[List[int]],
        arrow_path: str,
):
    """
    Save the entire dataset to a single Arrow IPC file,
    with two columns: job_title (string) and token_ids (list<int64>).
    """
    table = pa.table(
        {
            "job_title": job_titles,
            "token_ids": token_ids_list,
        }
    )
    dir_name = os.path.dirname(arrow_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with pa.OSFile(arrow_path, "wb") as sink:
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    print(f"Saved single Arrow file: {arrow_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # Load data
    median_wage_df = pd.read_csv(
        "data/h1b_median_prevailing_wages_by_job_title.csv"
    )
    unique_job_titles = (
        median_wage_df["JOB_TITLE"].dropna().unique().tolist()
    )

    if args.debug:
        unique_job_titles = unique_job_titles[:10]

    # LLM config (just for trust_remote_code flag)
    if args.llm_config == "home_4gpu":
        llm_config = HOME_4GPU_CONFIG
    elif args.llm_config == "hpc":
        llm_config = HPC_CONFIG
    else:
        llm_config = HOME_CONFIG

    model_name = args.model

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=getattr(llm_config, "trust_remote_code", True),
    )

    print(
        f"Loaded tokenizer for {model_name} | "
        f"num_titles={len(unique_job_titles)} | "
        f"chunk_size={args.chunk_size}"
    )

    all_titles: List[str] = []
    all_token_ids: List[List[int]] = []
    all_stats = []

    start_time = time.perf_counter()

    chunk_iter = enumerate(
        chunk_list(unique_job_titles, args.chunk_size), start=1
    )

    for chunk_idx, chunk in chunk_iter:
        print(
            f"\nProcessing chunk {chunk_idx} "
            f"({len(chunk)} titles) with batch_text method..."
        )

        prompts = [
            {"job_title": title, "messages": build_messages(title)}
            for title in chunk
        ]

        tokenized = tokenize_batch_text(tokenizer, prompts)

        # Accumulate in memory for final single Arrow file
        all_titles.extend(chunk)
        all_token_ids.extend(tokenized)

        # Collect stats
        for title, ids in zip(chunk, tokenized):
            all_stats.append(
                {
                    "job_title": title,
                    "token_len": len(ids),
                }
            )

    total_time = time.perf_counter() - start_time

    # Save single Arrow file
    save_all_to_arrow(all_titles, all_token_ids, args.arrow_path)

    # Aggregate + save stats
    stats_df = pd.DataFrame(all_stats)
    print("\nTokenization summary:")
    print(f"Total titles tokenized: {len(stats_df)}")
    print(f"Mean tokens per prompt: {stats_df['token_len'].mean():.2f}")
    print(f"Median tokens per prompt: {stats_df['token_len'].median():.2f}")
    print(f"Max tokens: {stats_df['token_len'].max()}")
    print(f"95th percentile: {stats_df['token_len'].quantile(0.95):.2f}")
    print(f"Total time: {total_time:.2f}s | {len(stats_df) / total_time:.1f} prompts/s")

    out_stats = args.out_stats
    stats_dir = os.path.dirname(out_stats)
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)
    stats_df.to_csv(out_stats, index=False)
    print(f"\nSaved token length stats to: {out_stats}")


if __name__ == "__main__":
    main()
