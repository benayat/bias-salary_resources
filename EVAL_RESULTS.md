1. Open models:
   === Model: open_models |  File: estimations_Llama-3.3-70B-Instruct-llm_estimated_salaries-salary_estimator.csv ===
   Normality (Shapiro-Wilk):
   AI:    W=0.9317  p=2.425e-14  (NOT Normal)
   Other: W=0.9425  p=5.437e-13  (NOT Normal)
   Group means (SPB): AI=21.7070%  Other=19.1840%  Diff=2.5230 pp
   medians:         AI=15.7262%  Other=13.5902%
   MAPE:            AI=28.0850%  Other=26.3617%
   Welch t-test H0(mean diff=0): t=1.2662  p(two)=2.057e-01  p(one, AI>Other)=1.029e-01
   95% CI diff (AI-Other): [-1.3872, 6.4332] percentage-points
   OLS + HC3 (conditional) n=1000  R²=0.3613
   formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
   AI uplift coef (ΔSPB, pp): 2.5094  SE(HC3)=1.8444  t=1.3605  p=1.737e-01
   95% CI AI coef: [-1.1056, 6.1243]
   → Statistical significance (Welch): NO (p≥0.05)
   → Statistical significance (OLS+HC3): NO (p≥0.05)

=== Model: open_models |  File: estimations_Mixtral-8x22B-Instruct-v0.1-llm_estimated_salaries-salary_estimator.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9504  p=6.568e-12  (NOT Normal)
Other: W=0.9532  p=1.751e-11  (NOT Normal)
Group means (SPB): AI=7.6617%  Other=4.3833%  Diff=3.2783 pp
medians:         AI=4.6252%  Other=-0.7234%
MAPE:            AI=21.4999%  Other=19.8333%
Welch t-test H0(mean diff=0): t=1.9461  p(two)=5.192e-02  p(one, AI>Other)=2.596e-02
95% CI diff (AI-Other): [-0.0274, 6.5841] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3403
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 3.2601  SE(HC3)=1.6055  t=2.0306  p=4.230e-02
95% CI AI coef: [0.1134, 6.4069]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: open_models |  File: estimations_Mixtral-8x7B-Instruct-v0.1-llm_estimated_salaries-salary_estimator.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9085  p=8.794e-17  (NOT Normal)
Other: W=0.9408  p=3.188e-13  (NOT Normal)
Group means (SPB): AI=7.0395%  Other=5.2315%  Diff=1.8080 pp
medians:         AI=3.5866%  Other=1.0543%
MAPE:            AI=23.0008%  Other=21.6852%
Welch t-test H0(mean diff=0): t=0.9544  p(two)=3.401e-01  p(one, AI>Other)=1.701e-01
95% CI diff (AI-Other): [-1.9095, 5.5255] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.4102
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 1.8054  SE(HC3)=1.7031  t=1.0601  p=2.891e-01
95% CI AI coef: [-1.5326, 5.1435]
→ Statistical significance (Welch): NO (p≥0.05)
→ Statistical significance (OLS+HC3): NO (p≥0.05)

=== Model: open_models |  File: estimations_Qwen3-235B-A22B-Instruct-2507-FP8-llm_estimated_salaries-salary_estimator.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9279  p=8.923e-15  (NOT Normal)
Other: W=0.9378  p=1.321e-13  (NOT Normal)
Group means (SPB): AI=16.0693%  Other=13.1510%  Diff=2.9182 pp
medians:         AI=9.9403%  Other=7.0950%
MAPE:            AI=24.3609%  Other=22.4563%
Welch t-test H0(mean diff=0): t=1.5812  p(two)=1.141e-01  p(one, AI>Other)=5.707e-02
95% CI diff (AI-Other): [-0.7034, 6.5399] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3383
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 2.9498  SE(HC3)=1.7548  t=1.6809  p=9.278e-02
95% CI AI coef: [-0.4897, 6.3892]
→ Statistical significance (Welch): NO (p≥0.05)
→ Statistical significance (OLS+HC3): NO (p≥0.05)

=== Model: open_models |  File: estimations_Qwen3-32B-llm_estimated_salaries-salary_estimator.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.8995  p=1.307e-17  (NOT Normal)
Other: W=0.8967  p=7.410e-18  (NOT Normal)
Group means (SPB): AI=28.2061%  Other=21.1391%  Diff=7.0671 pp
medians:         AI=20.9265%  Other=14.1207%
MAPE:            AI=33.2624%  Other=28.0994%
Welch t-test H0(mean diff=0): t=3.0695  p(two)=2.203e-03  p(one, AI>Other)=1.101e-03
95% CI diff (AI-Other): [2.5490, 11.5851] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3413
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 6.9524  SE(HC3)=2.2005  t=3.1595  p=1.581e-03
95% CI AI coef: [2.6395, 11.2652]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: open_models |  File: estimations_Qwen3-Next-80B-A3B-Instruct-llm_estimated_salaries-salary_estimator.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9195  p=1.109e-15  (NOT Normal)
Other: W=0.9464  p=1.777e-12  (NOT Normal)
Group means (SPB): AI=14.5832%  Other=11.1309%  Diff=3.4523 pp
medians:         AI=8.4796%  Other=4.9922%
MAPE:            AI=23.1843%  Other=21.9010%
Welch t-test H0(mean diff=0): t=1.8930  p(two)=5.865e-02  p(one, AI>Other)=2.932e-02
95% CI diff (AI-Other): [-0.1265, 7.0311] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3370
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 3.4716  SE(HC3)=1.7235  t=2.0142  p=4.399e-02
95% CI AI coef: [0.0935, 6.8497]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: open_models |  File: estimations_deepseek3_2.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.8976  p=8.765e-18  (NOT Normal)
Other: W=0.9157  p=4.421e-16  (NOT Normal)
Group means (SPB): AI=26.1026%  Other=21.8823%  Diff=4.2203 pp
medians:         AI=21.0347%  Other=14.9837%
MAPE:            AI=30.3067%  Other=27.2320%
Welch t-test H0(mean diff=0): t=2.0408  p(two)=4.154e-02  p(one, AI>Other)=2.077e-02
95% CI diff (AI-Other): [0.1622, 8.2784] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3267
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 4.2666  SE(HC3)=1.9973  t=2.1362  p=3.266e-02
95% CI AI coef: [0.3520, 8.1813]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: open_models |  File: estimations_gemma-3-27b-it-llm_estimated_salaries-salary_estimator.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9504  p=6.617e-12  (NOT Normal)
Other: W=0.9485  p=3.548e-12  (NOT Normal)
Group means (SPB): AI=36.9550%  Other=35.1723%  Diff=1.7827 pp
medians:         AI=32.3947%  Other=29.8948%
MAPE:            AI=39.5120%  Other=37.8231%
Welch t-test H0(mean diff=0): t=0.7962  p(two)=4.261e-01  p(one, AI>Other)=2.130e-01
95% CI diff (AI-Other): [-2.6109, 6.1763] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3641
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 1.7251  SE(HC3)=2.0721  t=0.8326  p=4.051e-01
95% CI AI coef: [-2.3361, 5.7864]
→ Statistical significance (Welch): NO (p≥0.05)
→ Statistical significance (OLS+HC3): NO (p≥0.05)

=== Model: open_models |  File: estimations_gpt-oss-120b-llm_estimated_salaries-salary_estimator.csv.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9193  p=1.043e-15  (NOT Normal)
Other: W=0.9422  p=4.824e-13  (NOT Normal)
Group means (SPB): AI=21.5774%  Other=15.9199%  Diff=5.6576 pp
medians:         AI=18.1614%  Other=9.4278%
MAPE:            AI=29.7111%  Other=24.0953%
Welch t-test H0(mean diff=0): t=2.7189  p(two)=6.668e-03  p(one, AI>Other)=3.334e-03
95% CI diff (AI-Other): [1.5741, 9.7411] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3278
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 5.6358  SE(HC3)=2.0115  t=2.8018  p=5.082e-03
95% CI AI coef: [1.6934, 9.5783]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: open_models |  File: estimations_gpt-oss-20b-llm_estimated_salaries-salary_estimator.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9336  p=4.073e-14  (NOT Normal)
Other: W=0.9413  p=3.776e-13  (NOT Normal)
Group means (SPB): AI=20.9469%  Other=15.3006%  Diff=5.6463 pp
medians:         AI=16.5621%  Other=8.7482%
MAPE:            AI=27.9369%  Other=24.7544%
Welch t-test H0(mean diff=0): t=2.8190  p(two)=4.914e-03  p(one, AI>Other)=2.457e-03
95% CI diff (AI-Other): [1.7157, 9.5768] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3490
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 5.5805  SE(HC3)=1.8874  t=2.9568  p=3.109e-03
95% CI AI coef: [1.8814, 9.2797]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

2. Close models:
   === Model: closed_models |  File: estimations_claude.csv ===
   Normality (Shapiro-Wilk):
   AI:    W=0.9156  p=4.501e-16  (NOT Normal)
   Other: W=0.9191  p=1.074e-15  (NOT Normal)
   Group means (SPB): AI=28.9114%  Other=15.8992%  Diff=13.0122 pp
   medians:         AI=23.1572%  Other=11.5979%
   MAPE:            AI=32.8587%  Other=24.7701%
   Welch t-test H0(mean diff=0): t=6.1839  p(two)=9.162e-10  p(one, AI>Other)=4.581e-10
   95% CI diff (AI-Other): [8.8829, 17.1415] percentage-points
   OLS + HC3 (conditional) n=997  R²=0.3225
   formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
   AI uplift coef (ΔSPB, pp): 12.9448  SE(HC3)=2.0623  t=6.2770  p=3.451e-10
   95% CI AI coef: [8.9029, 16.9868]
   → Statistical significance (Welch): YES (p<0.05)
   → Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: closed_models |  File: estimations_gemini2_5_flash.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9395  p=2.161e-13  (NOT Normal)
Other: W=0.9498  p=5.507e-12  (NOT Normal)
Group means (SPB): AI=33.5518%  Other=24.1435%  Diff=9.4084 pp
medians:         AI=28.7374%  Other=19.7931%
MAPE:            AI=36.1131%  Other=29.1751%
Welch t-test H0(mean diff=0): t=4.5840  p(two)=5.145e-06  p(one, AI>Other)=2.572e-06
95% CI diff (AI-Other): [5.3807, 13.4360] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3152
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 9.4912  SE(HC3)=2.0104  t=4.7211  p=2.346e-06
95% CI AI coef: [5.5509, 13.4314]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: closed_models |  File: estimations_gpt5_1.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.8962  p=6.973e-18  (NOT Normal)
Other: W=0.9305  p=1.925e-14  (NOT Normal)
Group means (SPB): AI=37.6079%  Other=26.3447%  Diff=11.2632 pp
medians:         AI=34.2814%  Other=20.9874%
MAPE:            AI=39.4900%  Other=30.1403%
Welch t-test H0(mean diff=0): t=5.1410  p(two)=3.303e-07  p(one, AI>Other)=1.651e-07
95% CI diff (AI-Other): [6.9638, 15.5626] percentage-points
OLS + HC3 (conditional) n=997  R²=0.3410
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 11.2464  SE(HC3)=2.1122  t=5.3245  p=1.012e-07
95% CI AI coef: [7.1066, 15.3862]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: closed_models |  File: estimations_grok-4_1-fast.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9021  p=2.217e-17  (NOT Normal)
Other: W=0.9007  p=1.681e-17  (NOT Normal)
Group means (SPB): AI=45.6434%  Other=40.7706%  Diff=4.8728 pp
medians:         AI=38.2981%  Other=34.6583%
MAPE:            AI=46.9705%  Other=42.4135%
Welch t-test H0(mean diff=0): t=1.9640  p(two)=4.981e-02  p(one, AI>Other)=2.491e-02
95% CI diff (AI-Other): [0.0040, 9.7416] percentage-points
OLS + HC3 (conditional) n=1000  R²=0.3028
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 4.7310  SE(HC3)=2.4289  t=1.9478  p=5.144e-02
95% CI AI coef: [-0.0295, 9.4916]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): NO (p≥0.05)

3. Open vs closed models:
   --- Processing Open Models (data/open_models) ---
   estimations_Llama-3.3-70B-Instruct-llm_estimated_salaries-salary_estimator: 2.52 pp (AI: 21.7% vs Other: 19.2%)
   estimations_Mixtral-8x22B-Instruct-v0.1-llm_estimated_salaries-salary_estimator: 3.28 pp (AI: 7.7% vs Other: 4.4%)
   estimations_Mixtral-8x7B-Instruct-v0.1-llm_estimated_salaries-salary_estimator: 1.81 pp (AI: 7.0% vs Other: 5.2%)
   estimations_Qwen3-235B-A22B-Instruct-2507-FP8-llm_estimated_salaries-salary_estimator: 2.92 pp (AI: 16.1% vs Other: 13.2%)
   estimations_Qwen3-32B-llm_estimated_salaries-salary_estimator: 7.07 pp (AI: 28.2% vs Other: 21.1%)
   estimations_Qwen3-Next-80B-A3B-Instruct-llm_estimated_salaries-salary_estimator: 3.45 pp (AI: 14.6% vs Other: 11.1%)
   estimations_deepseek3_2: 4.22 pp (AI: 26.1% vs Other: 21.9%)
   estimations_gemma-3-27b-it-llm_estimated_salaries-salary_estimator: 1.78 pp (AI: 37.0% vs Other: 35.2%)
   estimations_gpt-oss-120b-llm_estimated_salaries-salary_estimator.csv: 5.66 pp (AI: 21.6% vs Other: 15.9%)
   estimations_gpt-oss-20b-llm_estimated_salaries-salary_estimator: 5.65 pp (AI: 20.9% vs Other: 15.3%)

--- Processing Closed Models (data/closed_models) ---
estimations_claude: 13.01 pp (AI: 28.9% vs Other: 15.9%)
estimations_gemini2_5_flash: 9.41 pp (AI: 33.6% vs Other: 24.1%)
estimations_gpt5_1: 11.26 pp (AI: 37.6% vs Other: 26.3%)
estimations_grok-4_1-fast: 4.87 pp (AI: 45.6% vs Other: 40.8%)

============================================================
RESULTS: Welch's t-test on AI UPLIFT (AI Bias - Other Bias)
============================================================
OPEN Group (N=10):
Mean Uplift: 3.8354 pp
Std Dev:     1.7806
CLOSED Group (N=4):
Mean Uplift: 9.6392 pp
Std Dev:     3.5017
------------------------------
Difference (Closed - Open): 5.8038 pp
95% Confidence Interval:    [0.4917, 11.1158]
t-statistic: -3.1556
p-value:     0.0391
Degrees of Freedom: 3.64
CONCLUSION: Significant difference in AI Uplift (p < 0.05)