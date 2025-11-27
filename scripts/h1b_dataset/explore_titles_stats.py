import pandas as pd

JOBS_BY_TYPE_FILE = 'data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv'
H1B_DATA_FILE = 'data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv'

def main():
    df = pd.read_csv(JOBS_BY_TYPE_FILE)
    total_jobs = len(df)
    ai_ml_jobs = df['AI_job'].sum()
    non_ai_ml_jobs = total_jobs - ai_ml_jobs

    print(f"Total job titles: {total_jobs}")
    print(f"AI/ML related job titles: {ai_ml_jobs} ({(ai_ml_jobs / total_jobs) * 100:.2f}%)")
    print(f"Non-AI/ML related job titles: {non_ai_ml_jobs} ({(non_ai_ml_jobs / total_jobs) * 100:.2f}%)")

    h1b_df = pd.read_csv(H1B_DATA_FILE, low_memory=False)
    total_h1b_jobs = len(h1b_df)
    ai_ml_h1b_jobs = h1b_df[h1b_df['JOB_TITLE'].isin(
        df[df['AI_job'] == True]['job_title'].tolist()
    )]
    ai_ml_h1b_count = len(ai_ml_h1b_jobs)
    non_ai_ml_h1b_count = total_h1b_jobs - ai_ml_h1b_count
    print(f"\nTotal H1B job entries: {total_h1b_jobs}")
    print(f"AI/ML H1B job entries: {ai_ml_h1b_count} ({(ai_ml_h1b_count / total_h1b_jobs) * 100:.2f}%)")
    print(f"Non-AI/ML H1B job entries: {non_ai_ml_h1b_count} ({(non_ai_ml_h1b_count / total_h1b_jobs) * 100:.2f}%)")


if __name__ == '__main__':
    main()