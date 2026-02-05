1. Open models:
   === Model: Llama-3.3-70B-Instruct |  File: estimations_Llama-3.3-70B-Instruct.csv ===
   Normality (Shapiro-Wilk):
   AI:    W=0.9508  p=8.923e-18  (NOT Normal)
   Other: W=0.9489  p=4.053e-18  (NOT Normal)
   Group means (SPB): AI=20.2644%  Other=17.4958%  Diff=2.7686 pp
   medians:         AI=15.6759%  Other=12.7184%
   MAPE:            AI=26.9020%  Other=24.5089%
   Welch t-test H0(mean diff=0): t=2.0860  p(two)=3.711e-02  p(one, AI>Other)=1.855e-02
   95% CI diff (AI-Other): [0.1656, 5.3715] percentage-points
   OLS + HC3 (conditional) n=2000  R²=0.3139
   formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
   AI uplift coef (ΔSPB, pp): 2.7653  SE(HC3)=1.1929  t=2.3182  p=2.044e-02
   95% CI AI coef: [0.4274, 5.1033]
   → Statistical significance (Welch): YES (p<0.05)
   → Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: Mixtral-8x22B-Instruct-v0.1 |  File: estimations_Mixtral-8x22B-Instruct-v0.1.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9614  p=1.369e-15  (NOT Normal)
Other: W=0.9485  p=3.324e-18  (NOT Normal)
Group means (SPB): AI=6.9591%  Other=2.9449%  Diff=4.0141 pp
medians:         AI=3.9740%  Other=-1.2239%
MAPE:            AI=20.6149%  Other=18.6544%
Welch t-test H0(mean diff=0): t=3.5246  p(two)=4.337e-04  p(one, AI>Other)=2.169e-04
95% CI diff (AI-Other): [1.7806, 6.2477] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2777
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 3.8827  SE(HC3)=1.0556  t=3.6783  p=2.348e-04
95% CI AI coef: [1.8138, 5.9516]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: Mixtral-8x7B-Instruct-v0.1 |  File: estimations_Mixtral-8x7B-Instruct-v0.1.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9390  p=7.948e-20  (NOT Normal)
Other: W=0.9277  p=1.537e-21  (NOT Normal)
Group means (SPB): AI=5.5837%  Other=4.0278%  Diff=1.5559 pp
medians:         AI=1.5915%  Other=-0.7793%
MAPE:            AI=21.7727%  Other=21.0072%
Welch t-test H0(mean diff=0): t=1.2177  p(two)=2.235e-01  p(one, AI>Other)=1.117e-01
95% CI diff (AI-Other): [-0.9499, 4.0616] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.3531
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 1.5258  SE(HC3)=1.1109  t=1.3735  p=1.696e-01
95% CI AI coef: [-0.6515, 3.7031]
→ Statistical significance (Welch): NO (p≥0.05)
→ Statistical significance (OLS+HC3): NO (p≥0.05)

=== Model: Qwen3-235B-A22B-Instruct-2507-FP8 |  File: estimations_Qwen3-235B-A22B-Instruct-2507-FP8.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9504  p=7.638e-18  (NOT Normal)
Other: W=0.9374  p=4.352e-20  (NOT Normal)
Group means (SPB): AI=14.8366%  Other=11.1795%  Diff=3.6570 pp
medians:         AI=9.5641%  Other=6.6847%
MAPE:            AI=23.5209%  Other=20.9880%
Welch t-test H0(mean diff=0): t=2.9633  p(two)=3.080e-03  p(one, AI>Other)=1.540e-03
95% CI diff (AI-Other): [1.2368, 6.0773] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2713
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 3.6182  SE(HC3)=1.1536  t=3.1364  p=1.710e-03
95% CI AI coef: [1.3572, 5.8792]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: Qwen3-32B |  File: estimations_Qwen3-32B.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9130  p=1.732e-23  (NOT Normal)
Other: W=0.9065  p=2.861e-24  (NOT Normal)
Group means (SPB): AI=26.3946%  Other=19.3355%  Diff=7.0591 pp
medians:         AI=19.0685%  Other=15.0985%
MAPE:            AI=32.0203%  Other=26.6039%
Welch t-test H0(mean diff=0): t=4.5501  p(two)=5.689e-06  p(one, AI>Other)=2.844e-06
95% CI diff (AI-Other): [4.0165, 10.1017] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2912
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 6.8467  SE(HC3)=1.4161  t=4.8347  p=1.333e-06
95% CI AI coef: [4.0711, 9.6223]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: Qwen3-Next-80B-A3B-Instruct |  File: estimations_Qwen3-Next-80B-A3B-Instruct.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9403  p=1.273e-19  (NOT Normal)
Other: W=0.9503  p=7.359e-18  (NOT Normal)
Group means (SPB): AI=13.5659%  Other=9.6819%  Diff=3.8839 pp
medians:         AI=8.1058%  Other=4.5838%
MAPE:            AI=22.6165%  Other=20.3739%
Welch t-test H0(mean diff=0): t=3.1841  p(two)=1.475e-03  p(one, AI>Other)=7.373e-04
95% CI diff (AI-Other): [1.4917, 6.2761] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2744
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 3.8678  SE(HC3)=1.1250  t=3.4382  p=5.857e-04
95% CI AI coef: [1.6629, 6.0727]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: deepseek3_2(aka 'deepseek-chat') |  File: estimations_deepseek3_2.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9293  p=2.652e-21  (NOT Normal)
Other: W=0.9437  p=4.761e-19  (NOT Normal)
Group means (SPB): AI=24.3079%  Other=20.2516%  Diff=4.0563 pp
medians:         AI=19.4860%  Other=15.9390%
MAPE:            AI=29.0188%  Other=25.6190%
Welch t-test H0(mean diff=0): t=3.0099  p(two)=2.646e-03  p(one, AI>Other)=1.323e-03
95% CI diff (AI-Other): [1.4133, 6.6993] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2594
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 4.1355  SE(HC3)=1.2658  t=3.2671  p=1.086e-03
95% CI AI coef: [1.6546, 6.6164]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: gemma-3-27b-it |  File: estimations_gemma-3-27b-it.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9620  p=1.792e-15  (NOT Normal)
Other: W=0.9367  p=3.404e-20  (NOT Normal)
Group means (SPB): AI=35.9947%  Other=32.8748%  Diff=3.1199 pp
medians:         AI=31.8191%  Other=27.7774%
MAPE:            AI=38.7041%  Other=35.4495%
Welch t-test H0(mean diff=0): t=2.0708  p(two)=3.851e-02  p(one, AI>Other)=1.925e-02
95% CI diff (AI-Other): [0.1651, 6.0746] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.3109
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 3.0807  SE(HC3)=1.3664  t=2.2547  p=2.415e-02
95% CI AI coef: [0.4027, 5.7587]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: gpt-oss-120b |  File: estimations_gpt-oss-120b.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9471  p=1.877e-18  (NOT Normal)
Other: W=0.9463  p=1.337e-18  (NOT Normal)
Group means (SPB): AI=20.7484%  Other=13.8038%  Diff=6.9446 pp
medians:         AI=17.7910%  Other=9.6147%
MAPE:            AI=27.5249%  Other=22.9944%
Welch t-test H0(mean diff=0): t=5.1622  p(two)=2.684e-07  p(one, AI>Other)=1.342e-07
95% CI diff (AI-Other): [4.3063, 9.5829] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2459
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 6.8665  SE(HC3)=1.2768  t=5.3778  p=7.541e-08
95% CI AI coef: [4.3640, 9.3690]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: gpt-oss-20b |  File: estimations_gpt-oss-20b.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9491  p=4.343e-18  (NOT Normal)
Other: W=0.9297  p=3.011e-21  (NOT Normal)
Group means (SPB): AI=19.7951%  Other=14.4751%  Diff=5.3200 pp
medians:         AI=15.8637%  Other=9.3297%
MAPE:            AI=26.9931%  Other=23.7574%
Welch t-test H0(mean diff=0): t=3.9019  p(two)=9.859e-05  p(one, AI>Other)=4.929e-05
95% CI diff (AI-Other): [2.6461, 7.9939] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2720
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 5.2432  SE(HC3)=1.2618  t=4.1554  p=3.247e-05
95% CI AI coef: [2.7702, 7.7162]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

2. Close models:
   === Model: claude-sonnet-4.5 |  File: estimations_claude_sonnet_4.5.csv ===
   Normality (Shapiro-Wilk):
   AI:    W=0.9360  p=2.606e-20  (NOT Normal)
   Other: W=0.9218  p=2.344e-22  (NOT Normal)
   Group means (SPB): AI=27.2943%  Other=13.8424%  Diff=13.4519 pp
   medians:         AI=21.8605%  Other=8.8750%
   MAPE:            AI=31.5086%  Other=22.8781%
   Welch t-test H0(mean diff=0): t=9.5774  p(two)=2.868e-21  p(one, AI>Other)=1.434e-21
   95% CI diff (AI-Other): [10.6973, 16.2064] percentage-points
   OLS + HC3 (conditional) n=2000  R²=0.2719
   formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
   AI uplift coef (ΔSPB, pp): 13.2416  SE(HC3)=1.3283  t=9.9690  p=2.084e-23
   95% CI AI coef: [10.6382, 15.8450]
   → Statistical significance (Welch): YES (p<0.05)
   → Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: gemini-2.5-flash |  File: estimations_gemini_2.5_flash.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9596  p=5.435e-16  (NOT Normal)
Other: W=0.9483  p=3.105e-18  (NOT Normal)
Group means (SPB): AI=32.8826%  Other=22.2158%  Diff=10.6668 pp
medians:         AI=28.1117%  Other=17.9747%
MAPE:            AI=35.5581%  Other=27.2921%
Welch t-test H0(mean diff=0): t=7.7345  p(two)=1.638e-14  p(one, AI>Other)=8.192e-15
95% CI diff (AI-Other): [7.9622, 13.3715] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2429
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 10.6515  SE(HC3)=1.3246  t=8.0415  p=8.875e-16
95% CI AI coef: [8.0554, 13.2476]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: gpt-5.1 |  File: estimations_gpt-5.1.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9525  p=1.875e-17  (NOT Normal)
Other: W=0.9436  p=4.575e-19  (NOT Normal)
Group means (SPB): AI=35.3557%  Other=24.4077%  Diff=10.9480 pp
medians:         AI=32.5669%  Other=20.3637%
MAPE:            AI=37.7696%  Other=28.2310%
Welch t-test H0(mean diff=0): t=7.8215  p(two)=8.460e-15  p(one, AI>Other)=4.230e-15
95% CI diff (AI-Other): [8.2029, 13.6931] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2713
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 10.8170  SE(HC3)=1.3110  t=8.2508  p=1.573e-16
95% CI AI coef: [8.2474, 13.3865]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

=== Model: grok-4.1-fast |  File: estimations_grok-4.1-fast.csv ===
Normality (Shapiro-Wilk):
AI:    W=0.9445  p=6.615e-19  (NOT Normal)
Other: W=0.9156  p=3.725e-23  (NOT Normal)
Group means (SPB): AI=44.1334%  Other=38.0463%  Diff=6.0871 pp
medians:         AI=36.6424%  Other=31.7601%
MAPE:            AI=45.4934%  Other=39.7272%
Welch t-test H0(mean diff=0): t=3.7543  p(two)=1.788e-04  p(one, AI>Other)=8.940e-05
95% CI diff (AI-Other): [2.9073, 9.2668] percentage-points
OLS + HC3 (conditional) n=2000  R²=0.2693
formula: SPB ~ IS_AI_NUM + C(SOC_CODE) + C(WORKSITE_STATE) + C(NAICS2) + C(FULL_TIME_POSITION) + LOG_TWP
AI uplift coef (ΔSPB, pp): 5.9514  SE(HC3)=1.5083  t=3.9459  p=7.952e-05
95% CI AI coef: [2.9953, 8.9076]
→ Statistical significance (Welch): YES (p<0.05)
→ Statistical significance (OLS+HC3): YES (p<0.05)

3. Open vs closed models:
   --- Processing Open Models (data/sqrt-2000/open-models/) ---
   estimations_Llama-3.3-70B-Instruct: 2.77 pp (AI: 20.3% vs Other: 17.5%)
   estimations_Mixtral-8x22B-Instruct-v0.1: 4.01 pp (AI: 7.0% vs Other: 2.9%)
   estimations_Mixtral-8x7B-Instruct-v0.1: 1.56 pp (AI: 5.6% vs Other: 4.0%)
   estimations_Qwen3-235B-A22B-Instruct-2507-FP8: 3.66 pp (AI: 14.8% vs Other: 11.2%)
   estimations_Qwen3-32B: 7.06 pp (AI: 26.4% vs Other: 19.3%)
   estimations_Qwen3-Next-80B-A3B-Instruct: 3.88 pp (AI: 13.6% vs Other: 9.7%)
   estimations_deepseek3_2: 4.06 pp (AI: 24.3% vs Other: 20.3%)
   estimations_gemma-3-27b-it: 3.12 pp (AI: 36.0% vs Other: 32.9%)
   estimations_gpt-oss-120b: 6.94 pp (AI: 20.7% vs Other: 13.8%)
   estimations_gpt-oss-20b: 5.32 pp (AI: 19.8% vs Other: 14.5%)

--- Processing Closed Models (data/sqrt-2000/closed-models/) ---
estimations_claude_sonnet_4.5: 13.45 pp (AI: 27.3% vs Other: 13.8%)
estimations_gemini_2.5_flash: 10.67 pp (AI: 32.9% vs Other: 22.2%)
estimations_gpt-5.1: 10.95 pp (AI: 35.4% vs Other: 24.4%)
estimations_grok-4.1-fast: 6.09 pp (AI: 44.1% vs Other: 38.0%)

============================================================
RESULTS: Welch's t-test on AI UPLIFT (AI Bias - Other Bias)
============================================================
OPEN Group (N=10):
Mean Uplift: 4.2379 pp
Std Dev:     1.7513
CLOSED Group (N=4):
Mean Uplift: 10.2884 pp
Std Dev:     3.0680
------------------------------
Difference (Closed - Open): 6.0505 pp
95% Confidence Interval:    [1.4327, 10.6683]
t-statistic: -3.7099
p-value:     0.0225
Degrees of Freedom: 3.81
CONCLUSION: Significant difference in AI Uplift (p < 0.05)

