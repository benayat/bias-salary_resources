# H1B Dataset Sampler - SOC-Diverse Matched-Pairs

## Overview

This script (`sample_dataset.py`) creates a **SOC-diverse matched-pairs dataset** from H1B Labor Condition Application (LCA) data. It generates balanced pairs of AI and non-AI job postings, matched within the same Standard Occupational Classification (SOC) code, with configurable matching criteria and diversity controls.

## Purpose

The sampler addresses the challenge of creating balanced, representative datasets for analyzing wage differences between AI and non-AI positions while:
- Controlling for SOC code (occupational category)
- Optionally matching on additional factors (state, NAICS sector, full-time status)
- Ensuring diversity across occupational categories (avoiding concentration in high-volume SOCs)
- Filtering for data quality (certified applications, annual wages only)

## Core Algorithm

### 1. **Data Loading and Preprocessing**

```
Input: H1B LCA dataset (CSV/Parquet) + AI job titles list (TXT/CSV)
│
├─> Assign ROW_ID (preserve original row order)
├─> Filter by CASE_STATUS (default: "Certified")
├─> Filter by wage unit (keep only annual: PW_UNIT_OF_PAY == "Year" AND WAGE_UNIT_OF_PAY == "Year")
└─> Normalize job titles (lowercase, whitespace collapse)
```

### 2. **AI Labeling**

```python
# Title normalization function
def norm_title(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# Label determination
is_ai[row] = norm_title(row.JOB_TITLE) IN ai_titles_set
```

**AI titles source options:**
- `.txt` file: One title per line (ignores `#` comments)
- `.csv` file: Must have `job_title` and `AI_job` columns (filters where `AI_job == True`)

### 3. **SOC Eligibility Filtering**

```
For each SOC_CODE:
│
├─> Count AI rows in SOC
├─> Count non-AI rows in SOC
│
└─> Keep SOC if:
    - AI count >= min_ai_per_soc (default: 1)
    - Non-AI count >= 1
    - Both groups present
```

### 4. **Matched-Pair Sampling (Main Loop)**

**Initialization:**
```python
available_ai[all_ai_indices] = True
available_other[all_other_indices] = True
pairs_per_soc[soc] = 0  # for all eligible SOCs
pair_id = 0
```

**Sampling loop (continues until target_pairs reached or no eligible pairs remain):**

```
while pair_id < target_pairs:
    │
    ├─> STEP 1: Identify eligible SOCs
    │   └─> Keep SOCs that meet ALL criteria:
    │       - pairs_per_soc[soc] < max_pairs_per_soc (UPPER LIMIT: skip if already at cap)
    │       - At least 1 available AI row exists
    │       - At least 1 available non-AI row exists
    │   
    │   Note: This is a CEILING constraint - SOCs that have already contributed
    │         max_pairs_per_soc pairs are excluded from further sampling.
    │
    ├─> STEP 2: Select SOC (weighted random choice)
    │   └─> Weights determined by ai_soc_weight_mode:
    │       - "uniform":      weight = 1.0 (equal probability)
    │       - "sqrt":         weight = sqrt(ai_count) (favors larger SOCs)
    │       - "inverse_sqrt": weight = 1/sqrt(ai_count) (favors smaller SOCs)
    │
    ├─> STEP 3: Select random available AI row from chosen SOC
    │
    ├─> STEP 4: Match with non-AI row (hierarchical matching)
    │   └─> Try tiers in order (first match wins):
    │       1. soc+state+naics2+ft (all factors match)
    │       2. soc+state+naics2 (geographic + industry)
    │       3. soc+state (geographic only)
    │       4. soc_only (SOC code only)
    │
    ├─> STEP 5: Record pair or discard
    │   ├─> If match found:
    │   │   - Add (AI row, non-AI row) to output
    │   │   - Mark both as unavailable
    │   │   - Increment pairs_per_soc[soc]  ← This tracks how many pairs from this SOC
    │   │   - Increment pair_id
    │   │
    │   └─> If no match found:
    │       - Mark AI row as unavailable (prevent infinite loops)
    │       - Continue to next iteration
    │
    └─> Break if no eligible SOCs remain
```

**Code walkthrough of the cap logic:**
```python
# Before loop starts:
cap = max_pairs_per_soc if max_pairs_per_soc > 0 else 10**18  # Effectively unlimited if 0
pairs_per_soc = {soc: 0 for soc in soc_ok}  # Initialize counters

# Inside main loop - STEP 1 (SOC eligibility check):
for soc in soc_ok:
    if pairs_per_soc[soc] >= cap:  # ← UPPER LIMIT CHECK
        continue  # Skip this SOC - already contributed enough pairs
    
    # ... check if SOC has available rows ...
    eligible_socs.append(soc)  # Include SOC in this iteration's pool

# After successful match - STEP 5:
pairs_per_soc[soc] += 1  # Increment counter for this SOC
# Next iteration, if pairs_per_soc[soc] == cap, it will be excluded
```

### 5. **Output Generation**

```
Selected pairs:
│
├─> Add metadata columns:
│   - IS_AI: boolean (True for AI rows, False for non-AI)
│   - PAIR_ID: integer (0..N-1, shared by matched pairs)
│   - MATCH_LEVEL: string (matching tier achieved)
│
├─> Shuffle rows (while preserving pair associations)
│
└─> Export to CSV with all original columns + metadata
```

## Key Data Structures

### Helper Arrays (for fast vectorized operations)
```python
soc_arr[i]    = SOC_CODE for row i
state_arr[i]  = WORKSITE_STATE for row i
naics2_arr[i] = First 2 digits of NAICS_CODE for row i
ft_arr[i]     = FULL_TIME_POSITION for row i
```

### Index Dictionaries
```python
ai_idx_by_soc[soc]    = numpy array of indices for AI rows in this SOC
other_idx_by_soc[soc] = numpy array of indices for non-AI rows in this SOC
```

### Availability Tracking
```python
available_ai[i]    = boolean (True if row i is AI and not yet used)
available_other[i] = boolean (True if row i is non-AI and not yet used)
```

## Algorithmic Complexity

- **Time Complexity:** O(T × S × log(N))
  - T = target_pairs (typically ~100)
  - S = number of eligible SOCs (typically 50-200)
  - N = average rows per SOC (vectorized matching within SOC)
  
- **Space Complexity:** O(N + S)
  - N = total input rows (boolean masks + index arrays)
  - S = number of SOCs (dictionaries)

## Command-Line Interface

### Required Arguments
```bash
--input <path>       # H1B dataset (CSV or Parquet)
--ai-titles <path>   # AI job titles list (TXT or CSV)
--output <path>      # Output file path
```

### Column Mapping Arguments
```bash
--job-title-col <name>   # Default: JOB_TITLE
--soc-col <name>         # Default: SOC_CODE
--state-col <name>       # Default: WORKSITE_STATE
--naics-col <name>       # Default: NAICS_CODE
--ft-col <name>          # Default: FULL_TIME_POSITION
--status-col <name>      # Default: CASE_STATUS
--pw-unit-col <name>     # Default: PW_UNIT_OF_PAY
--wage-unit-col <name>   # Default: WAGE_UNIT_OF_PAY
```

### Filter Arguments
```bash
--status-value <value>       # Default: "Certified"
--year-unit-value <value>    # Default: "Year"
--allow-missing-unit-cols    # Don't error if unit columns missing
```

### Sampling Control Arguments
```bash
--target-total <int>         # Default: 200 (total output rows)
--seed <int>                 # Default: 0 (for reproducibility)
--min-ai-per-soc <int>       # Default: 1 (minimum AI rows per SOC)
--max-pairs-per-soc <int>    # Default: 0 (unlimited, use >0 for hard cap)
--ai-soc-weight-mode <mode>  # Default: "sqrt"
                              # Choices: uniform, sqrt, inverse_sqrt
```

## Diversity Control Mechanisms

### 1. **Max Pairs Per SOC (Hard Cap)**
```bash
--max-pairs-per-soc 5
```
- Strongest diversity control
- **Upper limit (ceiling):** No SOC can contribute more than N pairs
- Ensures spread across occupations
- **How it works:** Once a SOC has contributed N pairs, it's excluded from further sampling (even if it has available rows)

**Example:**
```
Target: 20 pairs total, max_pairs_per_soc = 3

Iteration 1-3:  SOC-A contributes 3 pairs → SOC-A now at cap, excluded
Iteration 4-6:  SOC-B contributes 3 pairs → SOC-B now at cap, excluded  
Iteration 7-9:  SOC-C contributes 3 pairs → SOC-C now at cap, excluded
Iteration 10-12: SOC-D contributes 3 pairs → SOC-D now at cap, excluded
...and so on

Result: At least 7 different SOCs represented (20 pairs / 3 max = 6.67 → 7)
```

### 2. **SOC Weighting (Soft Preference)**
```bash
--ai-soc-weight-mode inverse_sqrt
```

**Weight formulas:**
- `uniform`: P(SOC) ∝ 1 (all SOCs equally likely)
- `sqrt`: P(SOC) ∝ √(ai_count) (larger SOCs more likely)
- `inverse_sqrt`: P(SOC) ∝ 1/√(ai_count) (smaller SOCs more likely)

**Effect on distribution:**
```
Example with 3 SOCs having 100, 25, 4 AI rows:

Mode           SOC-100  SOC-25  SOC-4   (Normalized weights)
---------------------------------------------------------
uniform        0.333    0.333   0.333   (equal)
sqrt           0.526    0.263   0.211   (favors large)
inverse_sqrt   0.211    0.263   0.526   (favors small)
```

### 3. **Minimum AI Threshold**
```bash
--min-ai-per-soc 2
```
- Excludes SOCs with very few AI positions
- Ensures statistical reliability

## Matching Hierarchy

The script attempts to match AI and non-AI rows within the same SOC using a **greedy hierarchical strategy**:

### Tier 1: Full Match (soc+state+naics2+ft)
```python
SOC_CODE == same AND
WORKSITE_STATE == same AND
NAICS_CODE[0:2] == same AND
FULL_TIME_POSITION == same
```
**Best match:** Controls for occupation, geography, industry sector, and employment type.

### Tier 2: Geographic + Industry (soc+state+naics2)
```python
SOC_CODE == same AND
WORKSITE_STATE == same AND
NAICS_CODE[0:2] == same
```
**Good match:** Controls for occupation, geography, and industry sector.

### Tier 3: Geographic Only (soc+state)
```python
SOC_CODE == same AND
WORKSITE_STATE == same
```
**Acceptable match:** Controls for occupation and geography.

### Tier 4: SOC Only (soc_only)
```python
SOC_CODE == same
```
**Minimal match:** Controls only for occupational category.

## Output Schema

The output CSV contains all original columns plus:

| Column | Type | Description |
|--------|------|-------------|
| `ROW_ID` | int | Original row index from input file |
| `IS_AI` | bool | True if row is AI job, False if non-AI |
| `PAIR_ID` | int | Unique pair identifier (0..N-1) |
| `MATCH_LEVEL` | str | Matching tier achieved (see hierarchy above) |
| `[original columns]` | * | All columns from input dataset preserved |

**Notes:**
- Each `PAIR_ID` appears exactly twice (once for AI, once for non-AI)
- Rows are shuffled in output (not grouped by pair)
- `ROW_ID` allows tracing back to original dataset

## Usage Examples

### Basic Usage (200 rows, default settings)
```bash
./sample_dataset.py \
  --input data/h1b_combined.csv \
  --ai-titles data/ai_job_titles.txt \
  --output data/h1b_sampled.csv
```

### High Diversity (favor small SOCs, cap at 3 pairs each)
```bash
./sample_dataset.py \
  --input data/h1b_combined.csv \
  --ai-titles data/ai_job_titles.csv \
  --output data/h1b_sampled_diverse.csv \
  --target-total 300 \
  --ai-soc-weight-mode inverse_sqrt \
  --max-pairs-per-soc 3 \
  --min-ai-per-soc 2
```

### Large Sample (1000 rows, uniform SOC selection)
```bash
./sample_dataset.py \
  --input data/h1b_2024.parquet \
  --ai-titles data/ai_titles.txt \
  --output data/h1b_sampled_1000.csv \
  --target-total 1000 \
  --ai-soc-weight-mode uniform \
  --seed 42
```

### Custom Column Names
```bash
./sample_dataset.py \
  --input data/custom_h1b.csv \
  --ai-titles data/ai_titles.txt \
  --output data/sampled.csv \
  --job-title-col "Job_Title" \
  --soc-col "SOC" \
  --state-col "State" \
  --status-col "Status"
```

## Algorithm Design Decisions

### 1. **SOC-First Sampling (Not Row-First)**
**Why:** Ensures diversity across occupational categories.

**Alternative (rejected):** Select random AI rows globally → would concentrate in high-volume SOCs.

### 2. **Without-Replacement Sampling**
**Why:** Each row used at most once → prevents duplicate influence.

**Implication:** Maximum pairs limited by minimum(ai_rows, other_rows) in bottleneck SOC.

### 3. **Greedy Matching (First Available)**
**Why:** Fast, simple, prevents deadlocks.

**Alternative (rejected):** Optimal bipartite matching → computationally expensive, overkill for this use case.

### 4. **Hierarchical Match Levels**
**Why:** Balances data quantity (fallback to SOC-only) with quality (prefer tight matches).

**Design:** Tries best match first, gracefully degrades to looser criteria.

### 5. **NAICS 2-Digit Matching**
**Why:** 2-digit = broad industry sector (e.g., "51" = Information), sufficient for controlling industry effects without over-constraining.

**Alternative:** 4 or 6-digit would be too restrictive (many SOCs would have no matches).

## Quality Controls

### Input Validation
```python
# Check required columns exist
# Check AI titles list not empty
# Check sufficient rows after filtering
# Check at least one eligible SOC exists
```

### Sampling Safety
```python
# Mark unavailable rows to prevent reuse
# Detect when no eligible SOCs remain (break loop)
# Fail if zero pairs created
```

### Output Validation
```python
# Verify pair_id consistency (each appears twice)
# Report actual vs target row count
# Show SOC distribution in output
```

## Performance Characteristics

**Typical runtime:** <10 seconds for 200-row output from 1M-row input

**Bottlenecks:**
1. Initial CSV/Parquet read (I/O bound)
2. Title normalization (CPU bound, vectorized)
3. Main sampling loop (usually <1000 iterations)

**Memory usage:** ~100-500 MB for 1M-row input (mostly pandas DataFrame + boolean masks)

## Reproducibility

The script is **fully deterministic** given the same:
- Input data
- AI titles list
- Seed value (`--seed`)
- All configuration flags

**Random operations:**
- SOC selection (weighted by specified mode)
- AI row selection within SOC (uniform)
- Non-AI row selection within matching tier (uniform)
- Final row shuffling (seed-based)

## Limitations and Caveats

1. **Greedy Matching:** May leave some AI rows unmatched even if optimal matching could pair them.

2. **No Backtracking:** Once a pair is committed, it's never reconsidered (even if later it blocks better matches).

3. **Independent Pairs:** No cross-pair constraints (e.g., can't enforce "same employer across pairs").

4. **Binary AI Label:** Assumes jobs are cleanly AI or non-AI (no uncertainty or partial AI roles).

5. **Title-Based Labeling:** AI classification based solely on job title match (not job description, responsibilities, etc.).

6. **Wage Unit Filtering:** Aggressively filters to annual wages only, may exclude valid rows with monthly/hourly wages that could be converted.

## Extension Points

### Adding New Matching Factors
```python
# In main():
custom_arr = df[args.custom_col].astype(str).to_numpy()

# In match_other():
custom_match = (custom_arr[base_av] == custom_arr[ai_i])
tiers = [
    ("soc+custom", custom_match),
    # ...existing tiers...
]
```

### Alternative Weight Functions
```python
def soc_weight(soc: str) -> float:
    c = ai_counts_ok.get(soc, 0)
    # Add custom logic:
    # - Log transform: return np.log1p(c)
    # - Exponential: return np.exp(-c / 100)
    # - Threshold-based: return 10.0 if c < 10 else 1.0
```

### Stratified Sampling
```python
# Modify to ensure minimum pairs per SOC group:
# e.g., at least 2 pairs from tech SOCs, 2 from healthcare SOCs, etc.
```

## Error Handling

The script fails fast with clear error messages for:
- Missing input files
- Missing required columns
- Empty AI titles list
- Insufficient data after filtering
- No eligible SOCs for matching
- Zero pairs created

All errors print to stderr with descriptive context for debugging.

## Related Files

- `../salaries_dataset/`: Similar sampling for salary survey data
- `../../data/h1b-lca-disclosure-data-2020-2024/`: Input data location
- `../../data/h1b-lca-disclosure-data-2020-2024/ai_ml_job_titles.csv`: Default AI titles list

## References

- **H1B LCA Program:** https://www.dol.gov/agencies/eta/foreign-labor/programs/permanent
- **SOC Classification:** https://www.bls.gov/soc/
- **NAICS Codes:** https://www.census.gov/naics/

---

**Version:** 1.0  
**Last Updated:** 2025-12-03  
**Maintainer:** See parent repository README

