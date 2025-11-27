import pandas as pd
H1B_DATA_FILE = 'data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv'
def main():
    h1b_df = pd.read_csv(H1B_DATA_FILE, low_memory=False)
    all_cols = h1b_df.columns.tolist()
    print("Columns in H1B dataset:")
    for col in all_cols:
        print(f"- {col}")
if __name__ == '__main__':
    main()