import pandas as pd

KAGGLE_DATASET_CSV_PATH = 'data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv'
# KAGGLE_DATASET_CSV_PATH = 'data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2020_to_FY2024.csv'

df = pd.read_csv(KAGGLE_DATASET_CSV_PATH, low_memory=False)
unique_titles = df['JOB_TITLE'].dropna().unique().tolist()
with open('data/h1b-lca-disclosure-data-2020-2024/unique_job_titles.txt', 'w', encoding='utf-8') as f:
    for title in unique_titles:
        f.write(f"{title}\n")

value_counts_for_locations = df['EMPLOYER_COUNTRY'].value_counts()
# print(f"location counts: {value_counts_for_locations}")
# percentage of entries in united states out of total
us_entries = df[df['EMPLOYER_COUNTRY']=='UNITED STATES OF AMERICA'].shape[0]
total_entries = df.shape[0]
print(f"US entries: {us_entries}, total entries: {total_entries}, percent: {us_entries/total_entries:.2%}")

# filter to only united states entries
df = df[df['EMPLOYER_COUNTRY']=='UNITED STATES OF AMERICA']


num_yearly_wages = df[df['PW_UNIT_OF_PAY'].str.upper()=='YEAR'].shape[0]
print(f"Number of yearly wages: {num_yearly_wages}")
percent_yearly_wages = num_yearly_wages / len(df)
print(f"Percent of yearly wages: {percent_yearly_wages:.2%}")

df = df[df['PW_UNIT_OF_PAY'].str.upper()=='YEAR']
# Normalize prevailing wage to numeric and convert to annual where unit is provided
df['prevailing_wage_num'] = pd.to_numeric(df['PREVAILING_WAGE'], errors='coerce')



df['prevailing_wage_annual'] = df['prevailing_wage_num']

# soc_code value counts, where there are more than 0.01% of total entries
soc_code_counts = df['SOC_CODE'].value_counts()
threshold = total_entries / 10000  # 0.01% of total entries
filtered_soc_codes = soc_code_counts[soc_code_counts >= threshold]
print(f"SOC codes with more than 0.01% of total entries ({threshold}):")
for soc_code, count in filtered_soc_codes.items():
    print(f"- {soc_code}: {count}")


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
