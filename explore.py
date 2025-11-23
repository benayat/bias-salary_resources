import pandas as pd
median_wage_df = pd.read_csv('data/h1b_median_prevailing_wages_by_job_title.csv')
unique_job_titles = median_wage_df['JOB_TITLE'].dropna().unique().tolist()
print(f"Total unique job titles: {len(unique_job_titles)}")