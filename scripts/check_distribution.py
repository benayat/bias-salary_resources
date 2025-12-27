import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

df = pd.read_csv('data/h1b-lca-disclosure-data-2020-2024/h1b_2024_sampled-1000_block_balanced.csv')

# Check distribution of a continuous variable separately for AI/Other
for group in [True, False]:
    data = df[df['IS_AI'] == group]['SOME_NUMERIC_COLUMN']

    print(f"\n{'AI' if group else 'Other'} Jobs:")
    stat, p = stats.shapiro(data)
    print(f"  Shapiro-Wilk: p={p:.4e} {'(Normal)' if p > 0.05 else '(NOT Normal)'}")

    # Q-Q plot
    plt.figure(figsize=(6, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {'AI' if group else 'Other'} Jobs")
    plt.show()