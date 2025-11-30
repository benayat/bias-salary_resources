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
    counts = h1b_df['JOB_TITLE'].value_counts()
    counts_gt1 = counts[counts >= (total_h1b_jobs/10000)]
    print("Job titles with more than 1 H1B entry:")
    print(f"number of filtered jobs: {len(counts_gt1)}")

    threshold = total_h1b_jobs / 10000  # 1/100th of a percent
    ai_title_counts = ai_ml_h1b_jobs['JOB_TITLE'].value_counts()
    ai_titles_ge_thresh = ai_title_counts[ai_title_counts >= threshold]
    print(f"AI/ML job titles with count >= {threshold:.0f} (1/100th of a percent): {len(ai_titles_ge_thresh)}")

    # for title, count in counts_gt1.items():
    #     print(f"- {title}: {count}")
if __name__ == '__main__':
    main()