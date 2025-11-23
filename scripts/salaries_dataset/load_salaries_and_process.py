import pandas as pd

DATA_PATH = 'data/salaries-for-data-science-jobs/salaries.csv'

df = pd.read_csv(DATA_PATH)

# Create dataframe with median salary per job title
median_salary_df = (
    df.dropna(subset=['job_title', 'salary_in_usd'])
      .groupby('job_title', as_index=False)
      .agg(median_salary_usd=('salary_in_usd', 'median'),
           observations=('salary_in_usd', 'count'))
      .sort_values('median_salary_usd', ascending=False)
)
print(f"number of unique job titles: {median_salary_df.shape[0]}")
# save to csv
median_salary_df.to_csv('data/salaries-for-data-science-jobs/salaries_median_by_job_title.csv', index=False)

print("Salary processing complete.")
