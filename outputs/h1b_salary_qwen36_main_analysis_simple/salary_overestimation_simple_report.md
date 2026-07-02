# H-1B Salary Overestimation Analysis

## Question
Do LLM salary estimators overestimate AI-related H-1B jobs more than matched non-AI H-1B jobs?

## Primary outcome
`overestimation_pct_points = 100 × (estimated_salary_in_usd - PREVAILING_WAGE) / PREVAILING_WAGE`.
The reported AI effect is therefore measured in percentage points of actual prevailing wage.

## Primary test
The primary test first averages predictions across models for each H-1B row, then computes AI-minus-Other overestimation contrasts within each sampler block, and finally runs a one-sample t-test over block-level contrasts.

## Secondary test
The secondary test runs Welch's t-test on row-level model-averaged percentage-point overestimation, comparing AI rows to Other rows.

## Data
Model-row records used: 21,164
Unique H-1B rows used: 2,000
Prediction models used: 12
Row-average table rows: 2,000
Blocks in row-average table: 530
AI rows: 1,000
Other rows: 1,000
AI threshold: >= 0.80; Other threshold: <= 0.30

## Main result
Primary block-level contrast: mean AI-minus-Other difference = 11.5361 percentage points, t=10.2327, p=1.52e-22, across 530 common-support blocks.

## Secondary result
Secondary row-level Welch test on model-averaged predictions: AI mean = 5.97351 percentage points, Other mean = -6.0656 percentage points, difference = 12.0391 percentage points, t=10.626, p=1.10e-25.

## Output files
- `model_row_salary_overestimation_long.csv/parquet`: model × row data with percentage-point overestimation.
- `row_average_salary_overestimation.csv`: one row per H-1B case after averaging across models.
- `primary_block_level_contrasts.csv`: within-block AI-minus-Other contrasts.
- `primary_block_level_contrast_summary.csv`: primary t-test over block contrasts.
- `secondary_row_average_welch.csv`: row-average Welch test.
- `per_model_welch_tests.csv`: per-model Welch tests.
- `per_model_block_level_contrasts.csv`: per-model within-block contrasts.
- `per_model_block_contrast_summary.csv`: per-model t-tests over block contrasts.
- `model_salary_overestimation_summary.csv`: descriptive model summary.
- `ai_group_salary_overestimation_summary.csv`: row-average group summary.
- `continuous_ai_confidence_summary.csv`: correlation between AI confidence and overestimation.
- `parse_status_summary.csv`: parsing/compliance summary.
- `prediction_vector_qc.csv` and `identical_prediction_vectors.csv`: output-integrity checks.

## Prediction-output QC

No identical prediction vectors were detected.

## Parse-status summary

| model_slug                          | salary_parse_status       |   n_rows |
|:------------------------------------|:--------------------------|---------:|
| Llama-3.1-70B-Instruct              | no_number                 |       10 |
| Llama-3.1-70B-Instruct              | ok                        |     1990 |
| Llama-3.1-8B-Instruct               | malformed                 |     1332 |
| Llama-3.1-8B-Instruct               | no_number                 |      108 |
| Llama-3.1-8B-Instruct               | ok                        |      560 |
| Llama-3.2-3B-Instruct               | ok                        |     2000 |
| Llama-3.3-70B-Instruct              | ok                        |     2000 |
| Mistral-Small-3.1-24B-Instruct-2503 | no_number                 |     2000 |
| Olmo-3-7B-Instruct                  | malformed                 |        7 |
| Olmo-3-7B-Instruct                  | multiple_numbers_or_range |      230 |
| Olmo-3-7B-Instruct                  | no_number                 |      330 |
| Olmo-3-7B-Instruct                  | ok                        |     1433 |
| Qwen2.5-32B-Instruct                | ok                        |     2000 |
| Qwen2.5-72B-Instruct                | ok                        |     2000 |
| Qwen3-14B                           | ok                        |     2000 |
| Qwen3-32B                           | ok                        |     2000 |
| Qwen3-8B                            | ok                        |     1998 |
| Qwen3-8B                            | out_of_plausible_range    |        2 |
| granite-4.1-30b                     | ok                        |     2000 |
| granite-4.1-8b                      | malformed                 |      439 |
| granite-4.1-8b                      | multiple_numbers_or_range |      198 |
| granite-4.1-8b                      | no_number                 |      180 |
| granite-4.1-8b                      | ok                        |     1183 |

## Raw model-output QC

| model_slug                          | model_family   |   n_rows |   n_unique_rows |   parse_ok_rows |   parse_ok_rate |   missing_prediction_rows |
|:------------------------------------|:---------------|---------:|----------------:|----------------:|----------------:|--------------------------:|
| Llama-3.1-70B-Instruct              | Llama          |     2000 |            2000 |            1990 |          0.995  |                        10 |
| Llama-3.1-8B-Instruct               | Llama          |     2000 |            2000 |             560 |          0.28   |                      1440 |
| Llama-3.2-3B-Instruct               | Llama          |     2000 |            2000 |            2000 |          1      |                         0 |
| Llama-3.3-70B-Instruct              | Llama          |     2000 |            2000 |            2000 |          1      |                         0 |
| Mistral-Small-3.1-24B-Instruct-2503 | Mistral        |     2000 |            2000 |               0 |          0      |                      2000 |
| Olmo-3-7B-Instruct                  | Olmo           |     2000 |            2000 |            1433 |          0.7165 |                       567 |
| Qwen2.5-32B-Instruct                | Qwen           |     2000 |            2000 |            2000 |          1      |                         0 |
| Qwen2.5-72B-Instruct                | Qwen           |     2000 |            2000 |            2000 |          1      |                         0 |
| Qwen3-14B                           | Qwen           |     2000 |            2000 |            2000 |          1      |                         0 |
| Qwen3-32B                           | Qwen           |     2000 |            2000 |            2000 |          1      |                         0 |
| Qwen3-8B                            | Qwen           |     2000 |            2000 |            1998 |          0.999  |                         2 |
| granite-4.1-30b                     | Granite        |     2000 |            2000 |            2000 |          1      |                         0 |
| granite-4.1-8b                      | Granite        |     2000 |            2000 |            1183 |          0.5915 |                       817 |

## Primary block-level contrast summary

| analysis                           | outcome                         | unit     |   mean_delta_pct_points_ai_minus_other |   median_delta_pct_points_ai_minus_other |   std_delta_pct_points_ai_minus_other |   n_common_support_blocks |   positive_blocks |   negative_blocks |   zero_blocks |   t_value |     p_value |   sign_test_p_value | status   |
|:-----------------------------------|:--------------------------------|:---------|---------------------------------------:|-----------------------------------------:|--------------------------------------:|--------------------------:|------------------:|------------------:|--------------:|----------:|------------:|--------------------:|:---------|
| primary_block_level_contrast_ttest | delta_pct_points_ai_minus_other | BLOCK_ID |                                11.5361 |                                  8.57778 |                                25.954 |                       530 |               377 |               153 |             0 |   10.2327 | 1.51727e-22 |         7.73837e-23 | ok       |

## Secondary row-average Welch test

| analysis                                | outcome                        | unit                           |   mean_ai_pct_points |   mean_other_pct_points |   difference_ai_minus_other_pct_points |   median_ai_pct_points |   median_other_pct_points |   n_ai_rows |   n_other_rows |   t_value |     p_value | status   |
|:----------------------------------------|:-------------------------------|:-------------------------------|---------------------:|------------------------:|---------------------------------------:|-----------------------:|--------------------------:|------------:|---------------:|----------:|------------:|:---------|
| secondary_row_average_welch_ai_vs_other | mean_overestimation_pct_points | H1B row averaged across models |              5.97351 |                 -6.0656 |                                12.0391 |                1.40944 |                  -10.3246 |        1000 |           1000 |    10.626 | 1.10238e-25 | ok       |

## Continuous AI-confidence summary

| analysis                                         | outcome                        |   n_rows |   pearson_r |   pearson_p_value |   spearman_r |   spearman_p_value | status   |
|:-------------------------------------------------|:-------------------------------|---------:|------------:|------------------:|-------------:|-------------------:|:---------|
| row_average_continuous_ai_confidence_correlation | mean_overestimation_pct_points |     2000 |    0.234578 |       2.09736e-26 |     0.248958 |        1.23637e-29 | ok       |

## AI group descriptive summary

| group   |   n_rows |   mean_ai_confidence |   mean_actual_usd |   mean_predicted_salary_usd |   mean_overestimation_usd |   median_overestimation_usd |   mean_overestimation_pct_points |   median_overestimation_pct_points |
|:--------|---------:|---------------------:|------------------:|----------------------------:|--------------------------:|----------------------------:|---------------------------------:|-----------------------------------:|
| Other   |     1000 |              0.11175 |            147311 |                      133732 |                 -13578.9  |                    -14020.8 |                         -6.0656  |                          -10.3246  |
| AI      |     1000 |              0.90696 |            150321 |                      154305 |                   3983.88 |                      1713.9 |                          5.97351 |                            1.40944 |

## Model summary

| model_slug             | model_family   |   n_rows |   n_ai_rows |   n_other_rows |   mean_actual_usd |   mean_predicted_usd |   mean_overestimation_usd |   median_overestimation_usd |   mean_overestimation_pct_points |   median_overestimation_pct_points |   mean_ai_overestimation_pct_points |   mean_other_overestimation_pct_points |   delta_ai_minus_other_pct_points |   mean_ai_overestimation_usd |   mean_other_overestimation_usd |   delta_ai_minus_other_usd |   corr_ai_confidence_overestimation_pct_points |
|:-----------------------|:---------------|---------:|------------:|---------------:|------------------:|---------------------:|--------------------------:|----------------------------:|---------------------------------:|-----------------------------------:|------------------------------------:|---------------------------------------:|----------------------------------:|-----------------------------:|--------------------------------:|---------------------------:|-----------------------------------------------:|
| Llama-3.1-70B-Instruct | Llama          |     1990 |         996 |            994 |            149017 |             146329   |                  -2687.77 |                      1227   |                         3.85857  |                           0.843248 |                             6.58801 |                                1.12365 |                           5.46436 |                      544.671 |                        -5926.72 |                    6471.39 |                                      0.106546  |
| Llama-3.1-8B-Instruct  | Llama          |      560 |         208 |            352 |            125476 |             123001   |                  -2475.6  |                      1195   |                         7.72627  |                           0.981079 |                             9.0802  |                                6.92622 |                           2.15397 |                    -1566.09  |                        -3013.04 |                    1446.95 |                                      0.0339668 |
| Llama-3.2-3B-Instruct  | Llama          |     2000 |        1000 |           1000 |            148816 |             288995   |                 140178    |                    -26266   |                        75.691    |                         -23.4589   |                           109.858   |                               41.5242  |                          68.3335  |                   198130     |                        82227    |                  115903    |                                      0.201605  |
| Llama-3.3-70B-Instruct | Llama          |     2000 |        1000 |           1000 |            148816 |             145068   |                  -3748.28 |                       825   |                         3.63593  |                           0.628244 |                             6.62204 |                                0.64982 |                           5.97222 |                     -209.128 |                        -7287.44 |                    7078.31 |                                      0.113266  |
| Olmo-3-7B-Instruct     | Olmo           |     1433 |         679 |            754 |            155342 |             106964   |                 -48378.4  |                    -43550   |                       -26.8825   |                         -32.3495   |                           -22.8364  |                              -30.5261  |                           7.68967 |                   -43868.4   |                       -52439.7  |                    8571.35 |                                      0.150039  |
| Qwen2.5-32B-Instruct   | Qwen           |     2000 |        1000 |           1000 |            148816 |             129905   |                 -18910.9  |                    -13868   |                        -7.9327   |                         -10.1661   |                            -5.5701  |                              -10.2953  |                           4.72521 |                   -16569     |                       -21252.8  |                    4683.75 |                                      0.103256  |
| Qwen2.5-72B-Instruct   | Qwen           |     2000 |        1000 |           1000 |            148816 |             105195   |                 -43621.6  |                    -38187   |                       -26.3035   |                         -27.8729   |                           -24.9529  |                              -27.654   |                           2.70114 |                   -42458.1   |                       -44785.1  |                    2326.96 |                                      0.0731438 |
| Qwen3-14B              | Qwen           |     2000 |        1000 |           1000 |            148816 |             130095   |                 -18721.3  |                    -14506   |                        -6.33926  |                         -10.3981   |                            -2.45668 |                              -10.2218  |                           7.76516 |                   -14321.5   |                       -23121.2  |                    8799.73 |                                      0.137018  |
| Qwen3-32B              | Qwen           |     2000 |        1000 |           1000 |            148816 |             139530   |                  -9286.83 |                     -8786.5 |                        -1.14734  |                          -6.58122  |                             3.33955 |                               -5.63423 |                           8.97378 |                    -2698.46  |                       -15875.2  |                   13176.7  |                                      0.154298  |
| Qwen3-8B               | Qwen           |     1998 |         999 |            999 |            148803 |             142265   |                  -6538.73 |                     -1757   |                         2.14464  |                          -1.09551  |                             4.80319 |                               -0.51391 |                           5.3171  |                    -3868.7   |                        -9208.77 |                    5340.07 |                                      0.0890171 |
| granite-4.1-30b        | Granite        |     2000 |        1000 |           1000 |            148816 |             140958   |                  -7858.83 |                     -7548.5 |                        -0.831016 |                          -5.25298  |                             3.0444  |                               -4.70644 |                           7.75084 |                    -3165.46  |                       -12552.2  |                    9386.73 |                                      0.153725  |
| granite-4.1-8b         | Granite        |     1183 |         578 |            605 |            156633 |              72645.4 |                 -83987.7  |                    -80822   |                       -49.4856   |                         -53.0957   |                           -47.2291  |                              -51.6414  |                           4.4123  |                   -82168.9   |                       -85725.4  |                    3556.52 |                                      0.126548  |

## Per-model Welch tests

| analysis                    | model_slug             | model_family   | outcome                   |   mean_ai_pct_points |   mean_other_pct_points |   difference_ai_minus_other_pct_points |   n_ai_rows |   n_other_rows |   t_value |     p_value | status   |
|:----------------------------|:-----------------------|:---------------|:--------------------------|---------------------:|------------------------:|---------------------------------------:|------------:|---------------:|----------:|------------:|:---------|
| per_model_welch_ai_vs_other | Llama-3.1-70B-Instruct | Llama          | overestimation_pct_points |              6.58801 |                 1.12365 |                                5.46436 |         996 |            994 |  5.29737  | 1.3071e-07  | ok       |
| per_model_welch_ai_vs_other | Llama-3.1-8B-Instruct  | Llama          | overestimation_pct_points |              9.0802  |                 6.92622 |                                2.15397 |         208 |            352 |  0.694762 | 0.48759     | ok       |
| per_model_welch_ai_vs_other | Llama-3.2-3B-Instruct  | Llama          | overestimation_pct_points |            109.858   |                41.5242  |                               68.3335  |        1000 |           1000 |  8.30104  | 1.93645e-16 | ok       |
| per_model_welch_ai_vs_other | Llama-3.3-70B-Instruct | Llama          | overestimation_pct_points |              6.62204 |                 0.64982 |                                5.97222 |        1000 |           1000 |  5.49508  | 4.41238e-08 | ok       |
| per_model_welch_ai_vs_other | Olmo-3-7B-Instruct     | Olmo           | overestimation_pct_points |            -22.8364  |               -30.5261  |                                7.68967 |         679 |            754 |  5.71482  | 1.34746e-08 | ok       |
| per_model_welch_ai_vs_other | Qwen2.5-32B-Instruct   | Qwen           | overestimation_pct_points |             -5.5701  |               -10.2953  |                                4.72521 |        1000 |           1000 |  5.26956  | 1.51821e-07 | ok       |
| per_model_welch_ai_vs_other | Qwen2.5-72B-Instruct   | Qwen           | overestimation_pct_points |            -24.9529  |               -27.654   |                                2.70114 |        1000 |           1000 |  3.89964  | 9.95399e-05 | ok       |
| per_model_welch_ai_vs_other | Qwen3-14B              | Qwen           | overestimation_pct_points |             -2.45668 |               -10.2218  |                                7.76516 |        1000 |           1000 |  6.73962  | 2.08866e-11 | ok       |
| per_model_welch_ai_vs_other | Qwen3-32B              | Qwen           | overestimation_pct_points |              3.33955 |                -5.63423 |                                8.97378 |        1000 |           1000 |  7.31006  | 3.85912e-13 | ok       |
| per_model_welch_ai_vs_other | Qwen3-8B               | Qwen           | overestimation_pct_points |              4.80319 |                -0.51391 |                                5.3171  |         999 |            999 |  4.44469  | 9.29749e-06 | ok       |
| per_model_welch_ai_vs_other | granite-4.1-30b        | Granite        | overestimation_pct_points |              3.0444  |                -4.70644 |                                7.75084 |        1000 |           1000 |  7.43947  | 1.49808e-13 | ok       |
| per_model_welch_ai_vs_other | granite-4.1-8b         | Granite        | overestimation_pct_points |            -47.2291  |               -51.6414  |                                4.4123  |         578 |            605 |  4.62655  | 4.13564e-06 | ok       |

## Per-model block-contrast summary

| analysis                       | model_slug             | model_family   | outcome                         |   mean_delta_pct_points_ai_minus_other |   median_delta_pct_points_ai_minus_other |   n_common_support_blocks |   positive_blocks |   negative_blocks |   t_value |     p_value | status   |
|:-------------------------------|:-----------------------|:---------------|:--------------------------------|---------------------------------------:|-----------------------------------------:|--------------------------:|------------------:|------------------:|----------:|------------:|:---------|
| per_model_block_contrast_ttest | Llama-3.1-70B-Instruct | Llama          | delta_pct_points_ai_minus_other |                                7.25048 |                                  5.88453 |                       524 |               317 |               152 |  9.18918  | 9.28641e-19 | ok       |
| per_model_block_contrast_ttest | Llama-3.1-8B-Instruct  | Llama          | delta_pct_points_ai_minus_other |                               -1.28032 |                                  0       |                       119 |                30 |                46 | -0.845372 | 0.399614    | ok       |
| per_model_block_contrast_ttest | Llama-3.2-3B-Instruct  | Llama          | delta_pct_points_ai_minus_other |                               48.8769  |                                  0       |                       530 |               199 |               202 |  6.12353  | 1.78595e-09 | ok       |
| per_model_block_contrast_ttest | Llama-3.3-70B-Instruct | Llama          | delta_pct_points_ai_minus_other |                                7.41428 |                                  6.38478 |                       530 |               349 |               151 |  9.37135  | 2.0663e-19  | ok       |
| per_model_block_contrast_ttest | Olmo-3-7B-Instruct     | Olmo           | delta_pct_points_ai_minus_other |                                9.07787 |                                  4.95866 |                       281 |               164 |                67 |  6.92827  | 2.92649e-11 | ok       |
| per_model_block_contrast_ttest | Qwen2.5-32B-Instruct   | Qwen           | delta_pct_points_ai_minus_other |                                6.5889  |                                  5.69776 |                       530 |               330 |               165 |  9.06575  | 2.40658e-18 | ok       |
| per_model_block_contrast_ttest | Qwen2.5-72B-Instruct   | Qwen           | delta_pct_points_ai_minus_other |                                3.48105 |                                  1.50918 |                       530 |               286 |               184 |  5.76035  | 1.42546e-08 | ok       |
| per_model_block_contrast_ttest | Qwen3-14B              | Qwen           | delta_pct_points_ai_minus_other |                                9.55533 |                                  6.61551 |                       530 |               343 |               143 |  9.30264  | 3.60643e-19 | ok       |
| per_model_block_contrast_ttest | Qwen3-32B              | Qwen           | delta_pct_points_ai_minus_other |                               10.0053  |                                  7.36195 |                       530 |               344 |               142 |  9.27105  | 4.65422e-19 | ok       |
| per_model_block_contrast_ttest | Qwen3-8B               | Qwen           | delta_pct_points_ai_minus_other |                                6.95611 |                                  3.49214 |                       530 |               296 |               149 |  7.60765  | 1.28613e-13 | ok       |
| per_model_block_contrast_ttest | granite-4.1-30b        | Granite        | delta_pct_points_ai_minus_other |                                9.64637 |                                  7.59702 |                       530 |               339 |               143 |  9.33826  | 2.70289e-19 | ok       |
| per_model_block_contrast_ttest | granite-4.1-8b         | Granite        | delta_pct_points_ai_minus_other |                                5.51793 |                                  3.23749 |                       200 |               124 |                55 |  6.58718  | 3.91409e-10 | ok       |
