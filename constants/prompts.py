SYSTEM_PROMPT = """You are a compensation estimation assistant.

You must return ONLY a single integer (the annual USD salary).
- No dollar signs, commas, decimals, or explanations
- Round to the nearest whole number
"""

BLS_PROMPT = """Estimate the US national median BASE SALARY for this role using BLS OEWS May 2024 data.

Role: {role_description}

Instructions:
- Find the closest matching occupation in BLS data
- Use the May 2024 median annual wage (base salary only, no bonuses or equity)
- This should be the US national median

Return only the integer amount, nothing else."""

LEVELS_FYI_PROMPT = """Estimate the US national median TOTAL COMPENSATION for this role using levels.fyi data.

Role: {role_description}
Date: {as_of_date}

Instructions:
- Find the matching levels.fyi title and level
- Use median total compensation (base + bonus + annualized equity)
- This should be the US national median

Return only the integer amount, nothing else."""

GENERAL_PROMPT = """Estimate the US national median compensation for this role.

Role: {role_description}
Time frame: {time_frame}
Source: {source}
Date: {as_of_date}

Instructions:
- If source is "bls": Return median BASE SALARY from BLS OEWS May 2024
- If source is "levels_fyi": Return median TOTAL COMPENSATION (base + bonus + equity) from levels.fyi

Return only the integer amount, nothing else."""

# Backwards compatibility
SYSTEM_MINIMAL = SYSTEM_PROMPT