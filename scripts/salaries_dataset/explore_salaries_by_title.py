import pandas as pd

JOBS_BY_TYPE_FILE = 'data/salaries-for-data-science-jobs/ai_ml_job_titles.csv'
SALARIES_FILE = 'data/salaries-for-data-science-jobs/salaries.csv'

def main():
    df = pd.read_csv(JOBS_BY_TYPE_FILE)
    total_jobs = len(df)
    ai_ml_jobs = df['AI_job'].sum()
    non_ai_ml_jobs = total_jobs - ai_ml_jobs

    print(f"Total job titles: {total_jobs}")
    print(f"AI/ML related job titles: {ai_ml_jobs} ({(ai_ml_jobs / total_jobs) * 100:.2f}%)")
    print(f"Non-AI/ML related job titles: {non_ai_ml_jobs} ({(non_ai_ml_jobs / total_jobs) * 100:.2f}%)")

    ai_ml_titles = df[df['AI_job'] == True]['job_title'].tolist()
    print("AI/ML job titles:")
    for title in ai_ml_titles:
        print(f"- {title}")

    # Load salaries dataset
    salaries_df = pd.read_csv(SALARIES_FILE)
    total_salaries = len(salaries_df)
    ai_ml_salaries = salaries_df[salaries_df['job_title'].isin(ai_ml_titles)]
    ai_ml_count = len(ai_ml_salaries)
    non_ai_ml_count = total_salaries - ai_ml_count

    print(f"\nTotal salary entries: {total_salaries}")
    print(f"Actual AI/ML job entries: {ai_ml_count} ({(ai_ml_count / total_salaries) * 100:.2f}%)")
    print(f"Actual non-AI/ML job entries: {non_ai_ml_count} ({(non_ai_ml_count / total_salaries) * 100:.2f}%)")

if __name__ == '__main__':
    main()
