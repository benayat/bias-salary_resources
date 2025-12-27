# Multi-Factor SOC Matching Decision Graph

## Overview

This document explains the decision graph approach for matching salaries dataset entries to H1B SOC codes using multiple features beyond just job title matching.

## Problem Statement

**Goal**: Assign SOC codes from H1B dataset to salaries dataset entries to enable:
1. Better occupation-level matching for statistical tests (Welch t-test)
2. Reduced ambiguity compared to simple title matching
3. Control for confounders in salary comparisons

**Challenge**: The two datasets have different feature sets and only partial overlap.

---

## Feature Mapping & Weights

### Available Features for Matching

| Feature | Salaries Dataset | H1B Dataset | Weight | Notes |
|---------|------------------|-------------|--------|-------|
| **Job Title** | `job_title` | `JOB_TITLE` | 0.40 | Primary matching key |
| **Experience Level** | `experience_level` (EN/MI/SE/EX) | `PW_WAGE_LEVEL` (I/II/III/IV) | 0.15 | Indirect mapping via wage levels |
| **Employment Type** | `employment_type` (FT/PT/CT/FL) | `FULL_TIME_POSITION` (Y/N) | 0.10 | Binary match |
| **Salary** | `salary_in_usd` | `PREVAILING_WAGE` | 0.20 | Compatibility scoring |
| **Year** | `work_year` | `RECEIVED_DATE` / `DECISION_DATE` | 0.10 | Temporal proximity |
| **Location** | `company_location` | `WORKSITE_STATE` | 0.05 | US only, partial |

**Total Weight**: 1.00

---

## Decision Graph Algorithm

### Step 1: Title-Based Candidate Generation

```
For each salaries record:
  1. Normalize job title (lowercase, trim, collapse whitespace)
  2. Find H1B records with overlapping tokens (fast filter)
  3. Compute Jaccard similarity on title tokens
  4. Keep candidates with similarity ≥ 0.2
```

**Purpose**: Narrow search space from ~500K H1B records to ~100-1000 candidates per salaries record.

### Step 2: Multi-Factor Scoring

For each candidate H1B record, compute feature scores:

#### A. Title Similarity (Weight: 0.40)
```
Jaccard(salaries_title, h1b_title) = |tokens_A ∩ tokens_B| / |tokens_A ∪ tokens_B|

Example:
  "Machine Learning Engineer" vs "ML Engineer"
  → Jaccard = 2/4 = 0.50
```

#### B. Experience Level Match (Weight: 0.15)
```
Map experience levels:
  EN → Level I, II (Entry to Qualified)
  MI → Level II, III (Qualified to Experienced)
  SE → Level III, IV (Experienced to Fully Competent)
  EX → Level IV (Fully Competent)

Score:
  - 1.0 if H1B wage level in expected range
  - 0.3 if H1B wage level outside range but present
  - 0.5 if missing (neutral)
```

#### C. Employment Type Match (Weight: 0.10)
```
Map employment types:
  FT → Y (Full-time)
  PT → N (Part-time)
  CT → Y (Contract treated as FT)
  FL → N (Freelance)

Score:
  - 1.0 if match
  - 0.3 if mismatch
  - 0.5 if missing
```

#### D. Salary Compatibility (Weight: 0.20)
```
ratio = |salaries_salary - h1b_wage| / max(salaries_salary, h1b_wage)

Score:
  - 1.0 if ratio ≤ 0.3 (within 30%)
  - 0.7 if ratio ≤ 0.5 (within 50%)
  - 0.4 if ratio ≤ 1.0 (within 100%)
  - 0.2 if ratio > 1.0 (more than 100% difference)
  - 0.5 if missing
```

#### E. Year Proximity (Weight: 0.10)
```
diff = |salaries_year - h1b_year|

Score:
  - 1.0 if diff = 0 (same year)
  - 0.8 if diff = 1 (adjacent years)
  - 0.6 if diff ≤ 2 (within 2 years)
  - 0.3 if diff > 2
  - 0.5 if missing
```

#### F. Location Match (Weight: 0.05)
```
Only for company_location = 'US':
  - 0.7 if H1B worksite in US (partial credit, don't know exact state)
  - 0.5 otherwise
```

### Step 3: Final Score & Selection

```
total_score = Σ (feature_score × feature_weight)

For each salaries record:
  1. Compute scores for all candidates
  2. Filter candidates with score < min_threshold (default: 0.4)
  3. Rank by total_score (descending)
  4. Select top-k matches (default: k=3)
  5. Assign best match as primary SOC code
```

---

## Confidence Levels

Based on final score:

| Score Range | Confidence | Interpretation |
|-------------|------------|----------------|
| ≥ 0.7 | **HIGH** | Strong match across multiple features |
| 0.5 - 0.7 | **MEDIUM** | Good title match, some feature disagreement |
| 0.3 - 0.5 | **LOW** | Weak match, use with caution |
| < 0.3 | **VERY_LOW** | Poor match, likely incorrect |
| No match | **NO_MATCH** | No suitable candidate found |

---

## Example Decision Flow

### Example Record
```
Salaries Dataset Entry:
  job_title: "Senior Data Scientist"
  experience_level: "SE"
  employment_type: "FT"
  salary_in_usd: 150000
  work_year: 2024
  company_location: "US"
```

### Candidate Evaluation

#### Candidate 1: H1B Record A
```
JOB_TITLE: "Data Scientist"
SOC_CODE: "15-2051.00"
SOC_TITLE: "Data Scientists"
PW_WAGE_LEVEL: "Level III"
FULL_TIME_POSITION: "Y"
PREVAILING_WAGE: 145000
RECEIVED_DATE: "2024-03-15"
WORKSITE_STATE: "CA"

Feature Scores:
  title:      0.67 (2/3 tokens match: "data scientist")
  experience: 1.00 (Level III matches SE)
  employment: 1.00 (FT = Y)
  salary:     1.00 (|150k - 145k| / 150k = 3.3% < 30%)
  year:       1.00 (2024 = 2024)
  location:   0.70 (US match)

Total: 0.67×0.40 + 1.00×0.15 + 1.00×0.10 + 1.00×0.20 + 1.00×0.10 + 0.70×0.05
     = 0.268 + 0.15 + 0.10 + 0.20 + 0.10 + 0.035
     = 0.853

Confidence: HIGH ✅
```

#### Candidate 2: H1B Record B
```
JOB_TITLE: "Senior Data Analyst"
SOC_CODE: "15-2051.01"
SOC_TITLE: "Business Intelligence Analysts"
PW_WAGE_LEVEL: "Level II"
FULL_TIME_POSITION: "Y"
PREVAILING_WAGE: 95000
RECEIVED_DATE: "2023-11-20"
WORKSITE_STATE: "NY"

Feature Scores:
  title:      0.40 (1/4 tokens match: "data")
  experience: 0.30 (Level II doesn't match SE)
  employment: 1.00 (FT = Y)
  salary:     0.40 (|150k - 95k| / 150k = 36.7% > 30%)
  year:       0.80 (|2024 - 2023| = 1)
  location:   0.70 (US match)

Total: 0.40×0.40 + 0.30×0.15 + 1.00×0.10 + 0.40×0.20 + 0.80×0.10 + 0.70×0.05
     = 0.16 + 0.045 + 0.10 + 0.08 + 0.08 + 0.035
     = 0.50

Confidence: MEDIUM ⚠️
```

**Decision**: Select Candidate 1 (15-2051.00) as primary match.

---

## Output Format

### Primary Output: `salaries_with_matched_soc.csv`

Augmented salaries dataset with matched SOC codes:

```csv
work_year,experience_level,employment_type,job_title,salary_in_usd,...,MATCHED_SOC_CODE,MATCHED_SOC_TITLE,match_score,match_confidence,num_alternatives,alt1_soc,alt1_score,alt2_soc,alt2_score,alt3_soc,alt3_score
2024,SE,FT,Senior Data Scientist,150000,...,15-2051.00,Data Scientists,0.853,HIGH,3,15-2051.00,0.853,15-2051.01,0.500,15-1252.00,0.456
```

### Statistics Output: `salaries_soc_matching_stats.csv`

Matching quality metrics:

```csv
category,metric,value
Overall,Total Records,15000
Overall,Matched Records,13500
Overall,Match Rate (%),90.0
Confidence,HIGH,8000
Confidence,MEDIUM,4500
Confidence,LOW,1000
Confidence,NO_MATCH,1500
Experience Level Match Rate,EN,85.2%
Experience Level Match Rate,MI,92.1%
Experience Level Match Rate,SE,94.3%
Experience Level Match Rate,EX,88.7%
Top SOC Codes,15-2051.00 - Data Scientists,4500
Top SOC Codes,15-1252.00 - Software Developers,3200
...
```

---

## Advantages Over Simple Title Matching

1. **Reduces Ambiguity**: 
   - "Engineer" titles can map to different SOCs based on salary/experience
   - Example: "AI Engineer" with $80K (entry) → different SOC than $200K (senior)

2. **Handles Variations**:
   - "Data Scientist" vs "Data Science" vs "Scientist, Data" → same SOC
   - Typos and formatting differences normalized

3. **Experience-Aware**:
   - Junior vs Senior roles correctly distinguished
   - Wage level provides additional signal

4. **Temporal Consistency**:
   - Matches records from similar time periods
   - Accounts for salary inflation over years

5. **Multi-Modal Validation**:
   - Low confidence matches flagged for manual review
   - Alternative matches provided for ambiguous cases

---

## Usage

### Basic Usage
```bash
python scripts/match_salaries_to_h1b_soc_multifactor.py
```

### With Custom Parameters
```bash
python scripts/match_salaries_to_h1b_soc_multifactor.py \
    --salaries-csv data/salaries-for-data-science-jobs/salaries.csv \
    --h1b-csv data/h1b-lca-disclosure-data-2020-2024/Combined_LCA_Disclosure_Data_FY2024.csv \
    --output data/salaries_with_soc.csv \
    --output-stats data/soc_match_stats.csv \
    --top-k 5 \
    --min-score 0.35
```

### Testing Mode (Limited Records)
```bash
python scripts/match_salaries_to_h1b_soc_multifactor.py \
    --limit 1000 \
    --min-score 0.3
```

---

## Integration with Sampling & Statistical Tests

### Step 1: Match SOC Codes (Deterministic)
```bash
python scripts/match_salaries_to_h1b_soc_deterministic.py \
    --salaries-csv data/salaries-for-data-science-jobs/salaries.csv \
    --output data/salaries_with_soc_deterministic.csv \
    --output-stats data/salaries_soc_matching_stats_deterministic.csv
```

### Step 2: Sample with SOC-Based Matching
```bash
python scripts/salaries_dataset/sample_dataset.py \
    --input data/salaries_with_soc_deterministic.csv \
    --out data/salaries_sampled.csv \
    --match-cols "MATCHED_SOC_CODE,LOCATION_BIN,experience_level" \
    --n-ai 500 --n-non-ai 500
```

### Step 3: Run Statistical Tests
```bash
python scripts/salaries_dataset/compare_welch_hc3.py \
    --estimates-dir data/results \
    --actual-col salary_in_usd \
    --estimate-col estimated_salary_in_usd
```

**Key Benefit**: The matching in Step 1 is fully deterministic, so results are 100% reproducible across runs.

---

## Future Improvements

1. **Embedding-Based Title Matching**:
   - Replace Jaccard with sentence-transformers embeddings
   - Use cosine similarity for better semantic matching
   - Example: "ML Engineer" ≈ "Machine Learning Specialist"

2. **Industry/Domain Context**:
   - Add NAICS code matching if available
   - Weight by industry-specific salary norms

3. **Learned Weights**:
   - Train logistic regression to optimize feature weights
   - Use labeled validation set of correct matches

4. **Hierarchical SOC Matching**:
   - Match at SOC group level (2-digit) first, then refine
   - Example: 15-XXXX (Computer & Mathematical) → 15-2051 (Data Scientists)

5. **Active Learning**:
   - Flag low-confidence matches for manual review
   - Use feedback to improve scoring model

---

## Validation & Quality Control

### Recommended Validation Steps

1. **Manual Review Sample**:
   ```python
   # Review 50 random matches
   df = pd.read_csv('data/salaries_with_soc.csv')
   sample = df.sample(50)
   for _, row in sample.iterrows():
       print(f"Job: {row['job_title']}")
       print(f"→ SOC: {row['MATCHED_SOC_CODE']} ({row['MATCHED_SOC_TITLE']})")
       print(f"  Score: {row['match_score']:.3f} | Confidence: {row['match_confidence']}")
       print()
   ```

2. **Check Low-Confidence Matches**:
   ```python
   low_conf = df[df['match_confidence'].isin(['LOW', 'VERY_LOW'])]
   print(f"{len(low_conf)} low-confidence matches require review")
   ```

3. **Compare with Simple Matching**:
   ```python
   # Compare results with simple title matching
   simple_matches = pd.read_csv('data/salaries_to_h1b_soc_matches.csv')
   # Compute agreement rate
   ```

4. **SOC Distribution Validation**:
   ```python
   # Ensure SOC distribution makes sense for data science/AI domain
   soc_dist = df['MATCHED_SOC_CODE'].value_counts()
   print("Top SOCs should be tech-related (15-XXXX codes)")
   ```

