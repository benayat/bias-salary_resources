#!/usr/bin/env python3
"""
Explore SOC codes and titles NOT starting with '15' that have AI-related job titles,
and print the first 100 AI-related job titles found.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Set, List

import pandas as pd


def norm_title(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_ai_titles(path: str) -> Set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AI titles file not found: {path}")

    if p.suffix.lower() == ".txt":
        titles: List[str] = []
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                titles.append(line)
        return {norm_title(t) for t in titles if norm_title(t)}

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "AI_job" not in df.columns:
            raise KeyError("AI titles CSV must contain column 'AI_job' (boolean).")
        if "job_title" not in df.columns:
            raise KeyError("AI titles CSV must contain column 'job_title'.")

        only_ai_df = df[df.get("AI_job", False) == True]
        print(f"AI titles: {len(only_ai_df)}")
        return {norm_title(t) for t in only_ai_df.get("job_title") if norm_title(t)}

    raise ValueError(f"Unsupported AI titles extension: {p.suffix} (use .txt or .csv)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv")
    ap.add_argument("--ai-titles", default="data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv")
    ap.add_argument("--job-title-col", default="JOB_TITLE")
    ap.add_argument("--soc-col", default="SOC_CODE")
    ap.add_argument("--soc-title-col", default="SOC_TITLE")

    args = ap.parse_args()

    ai_titles = load_ai_titles(args.ai_titles)
    if not ai_titles:
        raise SystemExit("AI titles list is empty after loading/normalization.")

    df = pd.read_csv(args.input, low_memory=False)

    # Filter SOC codes NOT starting with '15'
    soc_not_15 = df[~df[args.soc_col].astype(str).str.startswith('15')]

    # Normalize job titles and check if AI
    title_norm = soc_not_15[args.job_title_col].astype(str).map(norm_title)
    is_ai = title_norm.isin(ai_titles)

    ai_rows = soc_not_15[is_ai]

    # Unique SOC codes and titles with AI titles
    unique_socs = ai_rows[[args.soc_col, args.soc_title_col]].drop_duplicates()

    print("SOC codes and titles NOT starting with '15' that have AI-related job titles:")
    for _, row in unique_socs.iterrows():
        print(f"{row[args.soc_col]}: {row[args.soc_title_col]}")

    # First 100 AI job titles
    ai_titles_list = ai_rows[args.job_title_col].head(100).tolist()

    print("\nFirst 100 AI-related job titles:")
    for i, title in enumerate(ai_titles_list, 1):
        print(f"{i}: {title}")


if __name__ == "__main__":
    main()
