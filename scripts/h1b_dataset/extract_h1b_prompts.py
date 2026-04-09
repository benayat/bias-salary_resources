"""Extract the exact chat prompts used by estimate_h1b_salaries.py.

This utility mirrors the prompt construction logic so users can inspect the
full messages payload without invoking an LLM.
"""

import argparse
import csv
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
PERSONAS = runpy.run_path(str(ROOT / "constants" / "personas.py"))["PERSONAS"]
SALARY_SYSTEM_PROMPT = runpy.run_path(str(ROOT / "constants" / "salary_prompts.py"))["SALARY_SYSTEM_PROMPT"]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all prompts sent to the LLM")
    parser.add_argument("--input-csv-file", required=True, help="Path to input CSV")
    parser.add_argument(
        "--persona",
        default="salary_estimator",
        help="Persona key in constants.PERSONAS (default: salary_estimator)",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/sampled_2000_sqrt_prompts.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    system_prompt = PERSONAS.get(args.persona, "") + SALARY_SYSTEM_PROMPT
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_csv_file, newline="", encoding="utf-8") as infile, open(
        output_path, "w", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        count = 0
        for idx, row in enumerate(reader):
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

            prompt = {
                "row_index": idx,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
            outfile.write(json.dumps(prompt, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} prompts to {output_path}")


if __name__ == "__main__":
    main()
