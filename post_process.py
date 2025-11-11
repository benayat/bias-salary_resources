import csv
import json
import os
from collections import defaultdict
from scipy import stats

def main():
    # Read actual data from table.csv
    actual_data = {}
    with open('table.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        annual_median_idx = header.index('Annual median wage  (2)')
        for row in reader:
            if len(row) > annual_median_idx:
                soc_full = row[0]
                if '(' in soc_full and soc_full.endswith(')'):
                    soc = soc_full.rsplit('(', 1)[1].rstrip(')').strip()
                    actual_str = row[annual_median_idx].strip()
                    if actual_str and actual_str != '(5) -':
                        import re
                        match = re.search(r'\$([0-9,]+)', actual_str)
                        if match:
                            actual = int(match.group(1).replace(',', ''))
                            actual_data[soc] = actual

    # Read estimated data from data/bls_evals/
    estimated_data = {}
    evals_dir = 'data/bls_evals'
    if os.path.exists(evals_dir):
        for filename in os.listdir(evals_dir):
            if filename.endswith('.jsonl'):
                soc = filename[:-6]  # Remove .jsonl
                filepath = os.path.join(evals_dir, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'estimate_usd' in data and isinstance(data['estimate_usd'], (int, float)):
                                estimated_data[soc] = data['estimate_usd']
                        except json.JSONDecodeError:
                            pass

    # Compare and group
    tech_group = []
    other_group = []
    for soc, actual in actual_data.items():
        if soc in estimated_data:
            estimated = estimated_data[soc]
            diff = estimated - actual
            if soc.startswith('15-'):
                tech_group.append((soc, actual, estimated, diff))
            else:
                other_group.append((soc, actual, estimated, diff))

    # Compute summaries
    def summarize(group, name):
        if not group:
            print(f"{name} group: No data")
            return
        diffs = [item[3] for item in group]
        avg_diff = sum(diffs) / len(diffs)
        abs_diffs = [abs(d) for d in diffs]
        avg_abs_diff = sum(abs_diffs) / len(abs_diffs)
        print(f"{name} group:")
        print(f"  Count: {len(group)}")
        print(f"  Average difference (estimated - actual): ${avg_diff:.2f}")
        print(f"  Average absolute difference: ${avg_abs_diff:.2f}")
        print()

    summarize(tech_group, "Tech (15-*)")
    summarize(other_group, "Other")

    # T-test for statistical significance
    if tech_group and other_group:
        tech_diffs = [item[3] for item in tech_group]
        other_diffs = [item[3] for item in other_group]

        # Check normality (Shapiro-Wilk for small samples)
        if len(tech_diffs) <= 50:
            tech_shapiro = stats.shapiro(tech_diffs)
            print(f"Tech group normality (Shapiro-Wilk): statistic={tech_shapiro.statistic:.4f}, p={tech_shapiro.pvalue:.4f}")
            if tech_shapiro.pvalue < 0.05:
                print("  Tech group not normally distributed.")
            else:
                print("  Tech group normally distributed.")

        if len(other_diffs) <= 50:
            other_shapiro = stats.shapiro(other_diffs)
            print(f"Other group normality (Shapiro-Wilk): statistic={other_shapiro.statistic:.4f}, p={other_shapiro.pvalue:.4f}")
            if other_shapiro.pvalue < 0.05:
                print("  Other group not normally distributed.")
            else:
                print("  Other group normally distributed.")

        # Check equal variances (Levene's test)
        levene_test = stats.levene(tech_diffs, other_diffs)
        print(f"Levene's test for equal variances: statistic={levene_test.statistic:.4f}, p={levene_test.pvalue:.4f}")
        equal_var = levene_test.pvalue >= 0.05
        if equal_var:
            print("  Variances are equal.")
        else:
            print("  Variances are not equal (will use unequal variance t-test).")

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(tech_diffs, other_diffs, equal_var=equal_var)
        print("T-test for difference in mean differences between Tech and Other groups:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  Significant difference (p < 0.05)")
        else:
            print("  No significant difference (p >= 0.05)")

        # Effect size (Cohen's d)
        mean_tech = sum(tech_diffs) / len(tech_diffs)
        mean_other = sum(other_diffs) / len(other_diffs)
        var_tech = sum((x - mean_tech)**2 for x in tech_diffs) / (len(tech_diffs) - 1)
        var_other = sum((x - mean_other)**2 for x in other_diffs) / (len(other_diffs) - 1)
        pooled_std = ((len(tech_diffs) - 1) * var_tech + (len(other_diffs) - 1) * var_other) / (len(tech_diffs) + len(other_diffs) - 2)
        cohens_d = (mean_tech - mean_other) / (pooled_std ** 0.5)
        print(f"  Cohen's d effect size: {cohens_d:.4f}")
    else:
        print("Not enough data for t-test.")

    # Optionally, write detailed to file
    with open('comparison_results.txt', 'w') as f:
        f.write("Tech Group:\n")
        for soc, actual, estimated, diff in tech_group:
            f.write(f"{soc}: Actual ${actual}, Estimated ${estimated}, Diff ${diff}\n")
        f.write("\nOther Group:\n")
        for soc, actual, estimated, diff in other_group:
            f.write(f"{soc}: Actual ${actual}, Estimated ${estimated}, Diff ${diff}\n")

if __name__ == '__main__':
    main()
