import pandas as pd

KAGGLE_DATASET_CSV_PATH = '../../data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv'
# KAGGLE_DATASET_CSV_PATH = 'data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv'

df = pd.read_csv(KAGGLE_DATASET_CSV_PATH, low_memory=False)
unique_titles = df['JOB_TITLE'].dropna().unique().tolist()
with open('../../data/h1b-lca-disclosure-data-2020-2024/unique_job_titles.txt', 'w', encoding='utf-8') as f:
    for title in unique_titles:
        f.write(f"{title}\n")

df = df[df['PW_UNIT_OF_PAY'].str.upper()=='YEAR']
# Normalize prevailing wage to numeric and convert to annual where unit is provided
df['prevailing_wage_num'] = pd.to_numeric(df['PREVAILING_WAGE'], errors='coerce')



df['prevailing_wage_annual'] = df['prevailing_wage_num']

# Create dataframe with median prevailing wage per job title
# median_wage_df = (
#     df.dropna(subset=['JOB_TITLE', 'prevailing_wage_annual'])
#       .groupby('JOB_TITLE', as_index=False)
#       .agg(median_prevailing_wage=('prevailing_wage_annual', 'median'),
#            observations=('prevailing_wage_annual', 'count'))
#       .sort_values('median_prevailing_wage', ascending=False)
# )
#
# # save to csv
# median_wage_df.to_csv('data/h1b_median_prevailing_wages_by_job_title.csv', index=False)
#
# print("Wage processing complete.")
