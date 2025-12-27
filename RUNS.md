1. Sampling:
(salary-resources) benaya-trabelsi@desktop:/mnt/storage-ssd/projects/salary_resources$ uv run scripts/h1b_dataset/sample_block_balanced.py \
--input data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv \
--ai-titles data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv \
--output data/h1b_2024_sampled-1000_block_balanced.csv \
--target-total 1000 \
--max-per-block 10 \
--weight-mode uniform \
--seed 0
AI titles: 2116
Rows after PW_UNIT_OF_PAY == 'Year' (case-insensitive): 761,558 (dropped 48,284)
Rows after WAGE_UNIT_OF_PAY == 'Year' (case-insensitive): 760,479 (dropped 1,079)
Input rows: 890,368
Rows after CASE_STATUS == 'Certified': 809,842
Rows after yearly-unit filters: 760,479
block_ok count: 950
Saved: data/h1b_2024_sampled-1000_block_balanced.csv
Rows saved: 1000 (AI=500, Other=500)
Unique BLOCK_ID in output: 420
Block balance check: OK (AI == Other within every BLOCK_ID).