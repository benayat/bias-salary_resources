## Plan: Separate Salaries Dataset Processing into Two Scripts

Adapt H1B script logic to process salaries-for-data-science-jobs dataset: create one script for data loading, median salary calculation by job_title, and CSV export; another for LLM-based AI/ML classification of job titles, saving results to text file.

### Steps
1. Create `scripts/load_salaries_and_process.py` to load `data/salaries-for-data-science-jobs/salaries.csv`, extract `salary_in_usd`, group by `job_title` for median calculation, save to `data/salaries_median_by_job_title.csv`.
2. Create `scripts/classify_salaries_ai_ml.py` to read `data/salaries_median_by_job_title.csv`, extract unique `job_title`s, query LLM using prompts from `constants/prompts.py`, save AI/ML-related titles to `data/salaries_ai_ml_job_titles.txt`.
3. Update `scripts/salaries-for-data-science-jobs-explore.py` as a wrapper to run both scripts sequentially.
4. Test scripts with sample data to ensure median calculation and LLM queries work correctly.

### Further Considerations
1. Confirm column names in `salaries.csv` (e.g., `salary_in_usd`, `job_title`) via initial data inspection.
2. Reuse LLM client and sampling config from `llm/llm_client.py` and `constants/llm_configs.py` for consistency.
3. Handle potential data issues like missing values or large file sizes during loading.
