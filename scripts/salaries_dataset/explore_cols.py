import pandas as pd
SALARIES_DATA_FILE = 'data/salaries-for-data-science-jobs/salaries.csv'
def main():
    salaries_df = pd.read_csv(SALARIES_DATA_FILE, low_memory=False)
    all_cols = salaries_df.columns.tolist()
    print("Columns in H1B dataset:")
    for col in all_cols:
        print(f"- {col}")
    print(salaries_df.head())
if __name__ == '__main__':
    main()