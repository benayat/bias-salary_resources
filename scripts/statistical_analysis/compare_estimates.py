import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path)

def filter_ai_ml_jobs(df, ai_ml_titles, job_title_column):
    return df[df[job_title_column].isin(ai_ml_titles)], df[~df[job_title_column].isin(ai_ml_titles)]

def calculate_statistics(actual, estimated):
    mae = np.mean(np.abs(actual - estimated))
    mpe = np.mean((actual - estimated) / actual) * 100
    return mae, mpe

def perform_ttest(actual, estimated):
    return ttest_ind(actual, estimated, equal_var=False)

def perform_anova(groups):
    return f_oneway(*groups)

def plot_comparison(actual, estimated, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, estimated, alpha=0.5)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--')
    plt.xlabel('Actual Salary (USD)')
    plt.ylabel('Estimated Salary (USD)')
    plt.title(title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def to_annual(row):
    salary = row['salary_in_usd']
    if pd.isna(salary):
        return None
    unit = str(row.get('WAGE_UNIT_OF_PAY', '')).lower()
    if 'hour' in unit:
        return salary * 2080
    if 'week' in unit:
        return salary * 52
    if 'month' in unit:
        return salary * 12
    return salary


def process_h1b_salary(df):
    
    #This function will convert various pay units (hourly, weekly, monthly) into a consistent annual salary.
    
    df['salary_in_usd'] = pd.to_numeric(df['WAGE_RATE_OF_PAY_FROM'], errors='coerce')
    df['salary_in_usd'] = df.apply(to_annual, axis=1)
    return df



def main():
    # File paths
    salaries_file = 'data/salaries-for-data-science-jobs/llm_estimated_salaries.csv'
    h1b_file = 'data/h1b-lca-disclosure-data-2020-2024/llm_estimated_salaries.csv'
    ai_ml_titles_file = 'data/salaries-for-data-science-jobs/salaries_ai_ml_job_titles.txt'

    # Load datasets
    salaries_df = load_data(salaries_file)
    h1b_df = load_data(h1b_file)

    if salaries_df is None or h1b_df is None:
        logging.error("One or more datasets could not be loaded. Exiting.")
        return

    # Process H1B salary data
    h1b_df = process_h1b_salary(h1b_df)

    # Load AI/ML job titles
    if not os.path.exists(ai_ml_titles_file):
        logging.error(f"AI/ML job titles file not found: {ai_ml_titles_file}")
        return

    with open(ai_ml_titles_file, 'r') as f:
        ai_ml_titles = [line.strip() for line in f]

    # Define dataset-specific columns
    datasets = [
        {
            'name': 'Salaries',
            'df': salaries_df,
            'job_title_column': 'job_title',
            'actual_salary_column': 'salary_in_usd',
            'estimated_salary_column': 'estimated_salary_in_usd'
        },
        {
            'name': 'H1B',
            'df': h1b_df,
            'job_title_column': 'JOB_TITLE',
            'actual_salary_column': 'salary_in_usd',
            'estimated_salary_column': 'estimated_salary_in_usd'
        }
    ]

    for dataset in datasets:
        name = dataset['name']
        df = dataset['df']
        job_title_column = dataset['job_title_column']
        actual_salary_column = dataset['actual_salary_column']
        estimated_salary_column = dataset['estimated_salary_column']

        # Filter AI/ML and other jobs
        ai_ml_df, other_df = filter_ai_ml_jobs(df, ai_ml_titles, job_title_column)

        # Notify user of missing data
        for df_name, sub_df in [(f"{name} AI/ML", ai_ml_df), (f"{name} Other", other_df)]:
            missing_data = sub_df.isnull().sum().sum()
            if missing_data > 0:
                logging.warning(f"{df_name} contains {missing_data} missing values.")

        # Calculate statistics
        for sub_name, sub_df in [(f"{name} AI/ML", ai_ml_df), (f"{name} Other", other_df)]:
            actual = sub_df[actual_salary_column]
            estimated = sub_df[estimated_salary_column]
            mae, mpe = calculate_statistics(actual, estimated)
            logging.info(f"{sub_name} - MAE: {mae:.2f}, MPE: {mpe:.2f}%")

        # Perform t-tests
        ttest_results = perform_ttest(ai_ml_df[actual_salary_column], ai_ml_df[estimated_salary_column])
        logging.info(f"T-Test for {name} AI/ML: {ttest_results}")

        # Generate visualizations
        plot_comparison(
            ai_ml_df[actual_salary_column],
            ai_ml_df[estimated_salary_column],
            f'{name} AI/ML: Actual vs Estimated',
            f'data/statistical_analysis_plots/{name.lower()}_ai_ml_comparison.png'
        )

if __name__ == '__main__':
    main()
