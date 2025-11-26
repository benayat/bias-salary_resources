## External resources - summary

- Bls - for general and tech jobs, no ai mention.
- [Hb1-lca-disclosure](https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024) - hb1 visa holders
  salary data - need to account for their specialty.
- [Salaries for data science jobs](https://www.kaggle.com/datasets/adilshamim8/salaries-for-data-science-jobs): a 2025
  dataset for comprehensive look into global salary trends for roles in Data Science, Machine Learning, and Artificial
  Intelligence.

### Data Analysis

#### BLS data summary:

- No ai-mention in job titles.
- No bias, except for Llama.

#### Hb1-lca-disclosure summary:

#### Relevant tech jobs from bls

Mentioned in tech_jobs_bls.txt file.

##### Their respective salary data:

| Job Title                 | Median Total Annual Compensation (USD) |
|---------------------------|----------------------------------------|
| AI Engineer               | $155,000                               |
| Machine Learning Engineer | $255,000                               |
| AI Researcher             | $165,000                               |
| Data Scientist            | $138,000                               |

#### citing levels.fyi:

- Zita, W.; Abou El Faouz, S.; Alayedi, M.; Elsayed, E.E. A Hybrid Bayesian Machine Learning Framework for Simultaneous
  Job Title Classification and Salary Estimation. Symmetry 2025, 17, 1261. https://doi.org/10.3390/sym17081261

# Salary Estimation Bias Analysis (AI/ML Titles vs Other Titles)

This project evaluates whether LLM-based salary estimators exhibit **systematic signed percent bias** for job titles
associated with AI/ML compared to other roles.

The core research question is:

> Do AI/ML job titles receive **more positive signed-percent estimation bias** than non-AI/ML titles?

“More positive” covers all relevant cases:

- both groups underestimate, but AI/ML underestimates less
- AI/ML overestimates while other titles underestimate
- both overestimate, but AI/ML overestimates more

---

## Data & Inputs

### Dataset

We use the “Salaries for Data Science Jobs” dataset, stored under:

`data/salaries-for-data-science-jobs/`

### Estimation outputs (one CSV per model)

Each LLM produces an estimation file with at least these columns:

- `job_title`
- `salary_in_usd` (actual)
- `estimated_salary_in_usd` (estimated)

Expected naming pattern:

- `llm_estimated_salaries{MODEL_TAG}.csv`
- optionally `llm_estimated_salaries_debug{MODEL_TAG}.csv`

The analysis script automatically discovers all matching files:

- `data/salaries-for-data-science-jobs/llm_estimated_salaries*.csv`

### AI/ML title labeling file

Group membership is controlled by:

`data/salaries-for-data-science-jobs/ai_ml_job_titles.csv`

Required columns:

- `job_title` (string; must match titles in the estimation CSVs)
- `AI_job` (boolean; `True` => AI/ML group)

This file is the *only* determinant of which rows are considered AI/ML vs Other.

---

## Metric: Signed Percent Bias (SPB)

For each row *i* we compute:

\[
SPB_i = \frac{\hat{y}_i - y_i}{y_i} \cdot 100
\]

Where:

- \(y_i\) is the actual salary (`salary_in_usd`)
- \(\hat{y}_i\) is the model estimate (`estimated_salary_in_usd`)

Interpretation:

- SPB > 0: overestimation (estimate larger than actual)
- SPB < 0: underestimation (estimate smaller than actual)

Rows with `actual == 0` are excluded from SPB computation to avoid division by zero.

---

## Primary Effect: AI/ML vs Other SPB Difference

For each model output file:

- \( \mu_{AI} = \mathrm{mean}(SPB \mid AI/ML) \)
- \( \mu_{Other} = \mathrm{mean}(SPB \mid Other) \)

Define the main quantity of interest:

\[
\Delta_{\text{mean}} = \mu_{AI} - \mu_{Other}
\]

Decision rule (aligned with the research question):

- **Δ_mean > 0** ⇒ AI/ML titles receive **more positive signed-% bias**

We also report the **median** SPB per group and the **median difference**:

\[
\Delta_{\text{median}} = \mathrm{median}(SPB_{AI}) - \mathrm{median}(SPB_{Other})
\]

---

## Statistical Tests

Because SPB distributions may be heavy-tailed and contain outliers, we report both a mean-based test and a robust
companion test.

### 1) Welch t-test (mean difference)

We apply a two-sample Welch t-test to the SPB values:

- H0: mean(SPB_AI) = mean(SPB_Other)
- H1 (directional / one-sided for the paper): mean(SPB_AI) > mean(SPB_Other)

We record both two-sided and one-sided p-values.

### 2) Robust companion: Mann–Whitney U + Cliff’s delta

We run a Mann–Whitney U test (nonparametric distribution shift test) on SPB:

- H0: SPB distributions are equal
- H1 (one-sided): AI/ML has larger SPB values than Other

We also report **Cliff’s delta (δ)** as an interpretable nonparametric effect size. Cliff’s delta is derived from the U
statistic:

\[
\delta = \frac{2U}{n_{AI}n_{Other}} - 1
\]

Interpretation:

- δ > 0: AI/ML tends to have larger SPB values (more positive bias)
- δ < 0: Other tends to have larger SPB values

---

## Outputs

For each estimation file (per model), the script writes:

1) A per-file summary CSV:

- `data/statistical_analysis_results/salaries/{base}_analysis_summary.csv`

This includes:

- mean and median SPB for AI/ML and Other
- Δ_mean and Δ_median
- Welch t-test statistics and p-values
- Mann–Whitney U statistics and p-values
- Cliff’s delta effect size
- MAE (in USD) for context (not the primary bias metric)

2) A scatter plot for AI/ML only:

- `data/statistical_analysis_plots/salaries/{base}_ai_ml_comparison.png`

3) An aggregated summary over all model files:

- `data/statistical_analysis_results/salaries/salaries_ALL_MODELS_summary.csv`

---

## Running the analysis

From the repository root:

```bash
uv run scripts/salaries_dataset/compare_signed_pct_bias.py
