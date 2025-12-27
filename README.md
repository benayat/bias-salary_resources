# Large Language Models Systematically Inflate the Value of AI Labor

## Abstract

As Large Language Models (LLMs) are increasingly integrated into human resources and labor market analytics, their internal representations of value become critical economic signals. This study investigates whether LLMs exhibit a systematic "AI Premium" bias—hallucinating higher wages for AI-labeled job titles compared to comparable non-AI roles within identical job contexts. Using a block-structured audit design on administrative labor data, we find that frontier proprietary models (e.g., GPT-5.1, Claude-Sonnet-4.5) significantly overestimate salaries for AI roles, assigning an average inflationary premium of **9.64 percentage points**. Furthermore, we identify a crucial distinction in model provenance: this bias is **significantly attenuated in open-weights models** (3.84 pp average uplift), suggesting that commercial alignment pipelines may inadvertently amplify narratives of AI exceptionalism.

## Key Findings

### 1. Systematic AI Wage Inflation Across All Models

Every model tested—both proprietary and open-source—assigned higher estimated salaries to AI-labeled positions compared to identical non-AI positions within the same occupation, geography, industry, and employment type context.

**Proprietary Models (N=4):**
- **Mean AI Uplift:** 9.64 pp (SD=3.50)
- **Range:** 4.87 pp (Grok-4.1-Fast) to 13.01 pp (Claude-Sonnet-4.5)
- **Statistical Significance:** All models showed p ≤ 0.05

**Open-Weight Models (N=10):**
- **Mean AI Uplift:** 3.84 pp (SD=1.78)
- **Range:** 1.78 pp (Gemma-3-27B) to 7.07 pp (Qwen3-32B)
- **Statistical Significance:** 5 of 10 models reached p < 0.05

### 2. Proprietary vs. Open Models: A Significant Divide

A Welch's t-test comparing model-level AI uplifts revealed a **statistically significant difference** between the two cohorts:

- **Closed Models Mean:** 9.64 pp (SD=3.50)
- **Open Models Mean:** 3.84 pp (SD=1.78)
- **Difference:** 5.80 pp (95% CI: [0.49, 11.12])
- **t-statistic:** -3.16, **p = 0.039**

This indicates that proprietary models amplify AI wage inflation by approximately **2.5× compared to open-source alternatives**.

### 3. General Overestimation vs. AI-Specific Bias

The study reveals two distinct layers of bias:

**Proprietary Models:**
- General mean bias (across all roles): 32.61%
- AI-specific uplift: +9.64 pp

**Open-Weight Models:**
- General mean bias: 17.71%
- AI-specific uplift: +3.84 pp

This suggests proprietary models are both **less grounded in absolute monetary values** and **more susceptible to AI hype**.

## Detailed Results by Model

### Proprietary Models

| Model | AI Mean SPB | Other Mean SPB | General Bias | AI Uplift | p-value |
|-------|-------------|----------------|--------------|-----------|---------|
| Claude-Sonnet-4.5 | 28.91% | 15.90% | 22.41% | **13.01 pp** | < 0.001 |
| GPT-5.1 | 37.61% | 26.34% | 31.98% | **11.26 pp** | < 0.001 |
| Gemini-2.5-Flash | 33.55% | 24.14% | 28.85% | **9.41 pp** | < 0.001 |
| Grok-4.1-Fast | 45.64% | 40.77% | 43.21% | **4.87 pp** | 0.050 |

### Open-Weight Models

| Model | AI Mean SPB | Other Mean SPB | General Bias | AI Uplift | p-value |
|-------|-------------|----------------|--------------|-----------|---------|
| Qwen3-32B | 28.21% | 21.14% | 24.68% | **7.07 pp** | 0.002 ✓ |
| GPT-OSS-120B | 21.58% | 15.92% | 18.75% | **5.66 pp** | 0.007 ✓ |
| GPT-OSS-20B | 20.95% | 15.30% | 18.13% | **5.65 pp** | 0.005 ✓ |
| DeepSeek-V3.2 | 26.10% | 21.88% | 23.99% | **4.22 pp** | 0.042 ✓ |
| Qwen3-Next-80B | 14.58% | 11.13% | 12.86% | **3.45 pp** | 0.029 ✓ |
| Mixtral-8x22B | 7.66% | 4.38% | 6.02% | **3.28 pp** | 0.052 |
| Qwen3-235B | 16.07% | 13.15% | 14.61% | **2.92 pp** | 0.114 |
| Llama-3.3-70B | 21.71% | 19.18% | 20.45% | **2.52 pp** | 0.206 |
| Mixtral-8x7B | 7.04% | 5.23% | 6.14% | **1.81 pp** | 0.340 |
| Gemma-3-27B | 36.96% | 35.17% | 36.07% | **1.78 pp** | 0.426 |

*✓ indicates statistical significance at p < 0.05*

## Methodology

### Design and Estimand

We employed a **block-structured audit design** to isolate AI labeling effects from confounding job characteristics:

1. **Block Construction:** Each block is defined by pre-label covariates:
    - Standard Occupational Classification (SOC) code
    - Geographic location (worksite state)
    - Industry sector (NAICS-2 digit)
    - Full-time vs. part-time status

2. **Overlap Restriction:** Analysis restricted to blocks containing both AI and non-AI titles

3. **Balanced Sampling:** Equal numbers of AI and non-AI positions sampled within each block (n=1,000 per model)

4. **Primary Outcome:** Signed Percent Bias (SPB)
   ```
   SPB = (LLM_Estimate - Ground_Truth) / Ground_Truth × 100
   ```

5. **Estimand:** AI Uplift = Mean SPB\_AI - Mean SPB\_Other

### Statistical Analysis

- **Within-model comparisons:** Welch's unequal-variance t-test
- **Between-cohort comparison:** Welch's t-test on model-level AI uplifts
- **Conditional analysis:** OLS regression with heteroscedasticity-robust standard errors (HC3)
- **Regression formula:**
  ```
  SPB ~ IS_AI + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + log(PREVAILING_WAGE)
  ```

## Implications

### 1. LLMs as Active Market Participants

The results demonstrate that LLMs are **not neutral observers** of the labor market. They actively participate in inflating AI-related valuations, potentially creating feedback loops that:
- Reinforce market hype through hallucinatory valuation
- Amplify wage inequality between AI and non-AI workers with comparable skills
- Distort compensation benchmarking and workforce planning decisions

### 2. The "Commercial Hype" Factor

The **2.5× amplification** of AI bias in proprietary models (p = 0.039) suggests that commercial alignment pipelines—specifically Reinforcement Learning from Human Feedback (RLHF)—may inadvertently encode narratives of AI exceptionalism. This raises critical questions about:
- The neutrality of "helpful" model behavior
- The economic consequences of optimizing for user satisfaction
- The role of training data selection in perpetuating hype cycles

### 3. Open Models as a Partial Solution

While open-weight models still exhibit AI wage inflation, their significantly lower bias suggests that:
- Transparency in training processes may reduce systemic distortions
- Community oversight can help ground model outputs in empirical reality

## Data & Code Availability

All analysis code, model configurations, and detailed results are available in this repository. The study used vLLM for inference with deterministic sampling (temperature=0.0) to ensure reproducibility.

## Citation

If you use these findings in your research, please cite:

```bibtex
@article{llm_ai_wage_bias_2025,
  title={Socioeconomic Hallucinations: Large Language Models Systematically Inflate the Value of AI Labor},
  author={[Your Name]},
  year={2025},
  note={GitHub repository: [Your Repo URL]}
}
```

---
