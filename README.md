# LLM Salary Estimation Bias: AI Jobs vs. Non-AI Jobs

## Key Findings

### Systematic AI Wage Overestimation

When asked to estimate salaries, LLMs systematically overestimate AI-labeled jobs **more** than they overestimate comparable non-AI jobs within tightly matched job contexts (same occupation, geography, industry, employment type).

**Key Metric: AI Uplift (ΔSPB)**
- **Definition**: The difference in Signed Percent Bias (SPB) between AI and non-AI jobs
- **SPB Formula**: `(LLM_Estimate - Ground_Truth) / Ground_Truth × 100`
- **AI Uplift**: `Mean(SPB_AI) - Mean(SPB_Other)`

### Results by Model Family

**Proprietary Models (N=4):**
- **Mean AI Uplift:** 10.29 percentage points (pp)
- **All models significant:** p < 0.001
- **Range:** 6.09 pp (Grok-4.1-Fast) to 13.45 pp (Claude-Sonnet-4.5)

**Open-Weight Models (N=10):**
- **Mean AI Uplift:** 4.24 pp
- **9 of 10 models significant:** p < 0.05
- **Range:** 1.56 pp (Mixtral-8x7B, n.s.) to 7.06 pp (Qwen3-32B)

### Proprietary vs. Open-Weight Gap

**Statistical Comparison:**
- **Difference:** +6.05 pp (proprietary higher)
- **Effect Size:** Cohen's d = 0.33 (small-to-medium)
- **Significance:** t(1979) = 7.47, **p < 0.001**
- **Interpretation:** Proprietary models exhibit approximately **2.4× larger** AI salary inflation than open-weight models

## Detailed Results by Model

### Proprietary Models

| Model | AI SPB (%) | Other SPB (%) | AI Uplift (pp) | Cohen's d | p-value | 95% CI (pp) |
|-------|------------|---------------|----------------|-----------|---------|-------------|
| Claude-Sonnet-4.5 | 27.29 | 13.84 | **13.45** | 0.43 | < 0.001 | [10.70, 16.21] |
| GPT-5.1 | 35.36 | 24.41 | **10.95** | 0.35 | < 0.001 | [8.20, 13.69] |
| Gemini-2.5-Flash | 32.88 | 22.22 | **10.67** | 0.35 | < 0.001 | [7.96, 13.37] |
| Grok-4.1-Fast | 44.13 | 38.05 | **6.09** | 0.17 | < 0.001 | [2.91, 9.27] |
| **Cohort Mean** | **34.92** | **24.63** | **10.29** | **0.32** | **< 0.001** | — |

### Open-Weight Models

| Model | AI SPB (%) | Other SPB (%) | AI Uplift (pp) | Cohen's d | p-value | 95% CI (pp) |
|-------|------------|---------------|----------------|-----------|---------|-------------|
| Qwen3-32B | 26.39 | 19.34 | **7.06** | 0.20 | < 0.001 ✓ | [4.02, 10.10] |
| GPT-OSS-120B | 20.75 | 13.80 | **6.94** | 0.23 | < 0.001 ✓ | [4.31, 9.58] |
| GPT-OSS-20B | 19.80 | 14.48 | **5.32** | 0.17 | < 0.001 ✓ | [2.65, 7.99] |
| DeepSeek-3.2 | 24.31 | 20.25 | **4.06** | 0.13 | 0.001 ✓ | [1.41, 6.70] |
| Mixtral-8x22B | 6.96 | 2.94 | **4.01** | 0.16 | < 0.001 ✓ | [1.78, 6.25] |
| Qwen3-Next-80B | 13.57 | 9.68 | **3.88** | 0.14 | 0.001 ✓ | [1.49, 6.28] |
| Qwen3-235B | 14.84 | 11.18 | **3.66** | 0.13 | 0.002 ✓ | [1.24, 6.08] |
| Gemma-3-27B | 36.00 | 32.87 | **3.12** | 0.09 | 0.019 ✓ | [0.17, 6.07] |
| Llama-3.3-70B | 20.26 | 17.50 | **2.77** | 0.09 | 0.019 ✓ | [0.17, 5.37] |
| Mixtral-8x7B | 5.58 | 4.03 | **1.56** | 0.05 | 0.112 | [−0.95, 4.06] |
| **Cohort Mean** | **18.85** | **14.61** | **4.24** | **0.14** | **< 0.001** | — |

*✓ indicates statistical significance at p < 0.05*  
*All tests: Welch's t-test, n=2000 jobs (1000 AI, 1000 non-AI) per model*

## Methodology

### Design and Estimand

We employed a **block-structured audit design** to isolate AI labeling effects from confounding job characteristics:

1. **Block Construction:** Each block is defined by pre-label covariates:
    - Standard Occupational Classification (SOC) code
    - Geographic location (worksite state)
    - Industry sector (NAICS-2 digit)
    - Full-time vs. part-time status

2. **Overlap Restriction:** Analysis restricted to blocks containing at least one AI-labeled and one non-AI title

3. **Balanced Sampling:** Equal numbers of AI and non-AI positions sampled within each block using power allocation with exponent 0.5, yielding n=2,000 titles per model (1,000 AI, 1,000 non-AI)

4. **Primary Outcome:** Signed Percent Bias (SPB)
   ```
   SPB = (LLM_Estimate - Ground_Truth) / Ground_Truth × 100
   ```

5. **Estimand:** AI Uplift (ΔSPB) = Mean(SPB_AI) - Mean(SPB_Other)

### Statistical Analysis

- **Within-model comparisons:** Welch's unequal-variance t-test with job title as unit of analysis
- **Effect size:** Cohen's d with degrees of freedom via Welch-Satterthwaite approximation
- **Significance threshold:** p < 0.05
- **Sample size:** n=2,000 jobs per model across 385 distinct overlap blocks

## Implications

### 1. LLMs as Active Market Participants

The results demonstrate that LLMs are **not neutral observers** of the labor market. They actively participate in inflating AI-related valuations, potentially creating feedback loops that:
- Reinforce market hype through hallucinatory valuation
- Distort compensation benchmarking and workforce planning decisions
- Bias salary negotiations as both candidates and employers anchor on model estimates

### 2. The Open vs. Proprietary Gap

The **2.4× amplification** of AI bias in proprietary models (t(1979) = 7.47, p < 0.001, d = 0.33) suggests that commercial alignment pipelines—specifically Reinforcement Learning from Human Feedback (RLHF)—may inadvertently encode narratives of AI exceptionalism. This raises critical questions about:
- Whether post-training alignment systematically rewards AI-favorable responses
- The role of training data selection and human feedback in perpetuating hype cycles
- The neutrality of "helpful" model behavior in advisory contexts

### 3. Methodological Insight: Matched Context Design

The block-structured design isolates AI labeling effects from confounding factors. The observed AI uplift represents **excess overestimation** beyond baseline model bias, demonstrating that the effect is attributable to the AI label itself rather than genuine salary differences or job characteristics.

## Data & Code Availability

**Source Data:** H1B LCA Disclosure Data (FY 2024) publicly available at [Kaggle](https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024)

**This Repository:** Contains the salary estimation experiment code, sampled dataset (2,000 job titles with block identifiers and model predictions), and analysis scripts.

**Main Project:** This experiment is part of a larger study on Pro-AI Bias in LLMs. See the [main repository](https://github.com/benayat/Pro-AI-bias-in-LLMs) for the complete project including recommendations and representation experiments.

**Technical Details:**
- All evaluations used greedy decoding (temperature=0.0) for deterministic outputs
- Inference powered by vLLM
- 14 models evaluated: 4 proprietary, 10 open-weight

## Citation

```bibtex
@article{trabelsi2026proai,
  title={Pro-AI Bias in Large Language Models},
  author={Trabelsi, Benaya and Shaki, Jonathan and Kraus, Sarit},
  year={2026},
  note={GitHub repository: https://github.com/benayat/Pro-AI-bias-in-LLMs}
}
```

---
