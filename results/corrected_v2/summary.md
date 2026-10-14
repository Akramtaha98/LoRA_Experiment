runs=228 failed=0 margin=+-0.01

| contrast   | scope                         | metric          |   n_seeds |    mean |   ci95_lo |   ci95_hi |        p |   ci90_lo |   ci90_hi |   equivalent_within_margin | cells_positive   |   p_holm |
|:-----------|:------------------------------|:----------------|----------:|--------:|----------:|----------:|---------:|----------:|----------:|---------------------------:|:-----------------|---------:|
| C-B        | pooled(6 cells)               | mean_f_faith    |         8 |  0.3043 |    0.2868 |    0.3217 |   0      |    0.2903 |    0.3183 |                          0 | 8/8 seeds        |      nan |
| C-B        | adapter=dora                  | mean_f_faith    |         8 |  0.3115 |    0.2827 |    0.3403 |   0      |  nan      |  nan      |                        nan | nan              |        0 |
| C-B        | adapter=lora                  | mean_f_faith    |         8 |  0.2866 |    0.2422 |    0.3309 |   0      |  nan      |  nan      |                        nan | nan              |        0 |
| C-B        | adapter=qlora                 | mean_f_faith    |         8 |  0.3148 |    0.2828 |    0.3467 |   0      |  nan      |  nan      |                        nan | nan              |        0 |
| C-B        | language=arabic               | mean_f_faith    |         8 |  0.6008 |    0.5607 |    0.6409 |   0      |  nan      |  nan      |                        nan | nan              |      nan |
| C-B        | language=malay                | mean_f_faith    |         8 |  0.0078 |   -0.0008 |    0.0164 |   0.0688 |  nan      |  nan      |                        nan | nan              |      nan |
| C-B        | pooled(6 cells)               | mean_em         |         8 | -0.043  |   -0.0492 |   -0.0368 |   0      |   -0.0479 |   -0.038  |                          0 | 0/8 seeds        |      nan |
| C-B        | pooled(6 cells)               | mean_f1         |         8 | -0.0243 |   -0.0292 |   -0.0193 |   0      |   -0.0282 |   -0.0203 |                          0 | 0/8 seeds        |      nan |
| C-B        | pooled(6 cells)               | option_acc      |         8 | -0.0121 |   -0.0242 |    0      |   0.0501 |   -0.0218 |   -0.0024 |                          0 | 2/8 seeds        |      nan |
| C-B        | pooled(6 cells)               | in_context_rate |         8 | -0.3765 |   -0.4048 |   -0.3482 |   0      |   -0.3992 |   -0.3539 |                          0 | 0/8 seeds        |      nan |
| C-C_l0     | pooled(6 cells)               | mean_f_faith    |         8 |  0.3091 |    0.2901 |    0.3281 |   0      |    0.2939 |    0.3243 |                          0 | 8/8 seeds        |      nan |
| C-C_l0     | pooled(6 cells)               | mean_em         |         8 | -0.0393 |   -0.0498 |   -0.0288 |   0      |   -0.0477 |   -0.0309 |                          0 | 0/8 seeds        |      nan |
| C-C_l0     | pooled(6 cells)               | mean_f1         |         8 | -0.0258 |   -0.0348 |   -0.0167 |   0.0003 |   -0.033  |   -0.0185 |                          0 | 0/8 seeds        |      nan |
| C-C_l0     | pooled(6 cells)               | option_acc      |         8 | -0.0066 |   -0.0195 |    0.0064 |   0.2676 |   -0.017  |    0.0038 |                          0 | 4/8 seeds        |      nan |
| C-C_l0     | pooled(6 cells)               | in_context_rate |         8 | -0.3677 |   -0.3911 |   -0.3443 |   0      |   -0.3864 |   -0.349  |                          0 | 0/8 seeds        |      nan |
| C-C_shuf   | pooled(6 cells)               | mean_f_faith    |         8 |  0.3087 |    0.2897 |    0.3278 |   0      |    0.2935 |    0.324  |                          0 | 8/8 seeds        |      nan |
| C-C_shuf   | pooled(6 cells)               | mean_em         |         8 | -0.0427 |   -0.0502 |   -0.0352 |   0      |   -0.0487 |   -0.0367 |                          0 | 0/8 seeds        |      nan |
| C-C_shuf   | pooled(6 cells)               | mean_f1         |         8 | -0.0257 |   -0.0334 |   -0.018  |   0.0001 |   -0.0319 |   -0.0195 |                          0 | 0/8 seeds        |      nan |
| C-C_shuf   | pooled(6 cells)               | option_acc      |         8 | -0.0167 |   -0.0249 |   -0.0084 |   0.002  |   -0.0233 |   -0.01   |                          0 | 0/8 seeds        |      nan |
| C-C_shuf   | pooled(6 cells)               | in_context_rate |         8 | -0.363  |   -0.3898 |   -0.3362 |   0      |   -0.3845 |   -0.3415 |                          0 | 0/8 seeds        |      nan |
| C_l0-B     | pooled(6 cells)               | mean_f_faith    |         8 | -0.0048 |   -0.0081 |   -0.0015 |   0.0103 |   -0.0074 |   -0.0022 |                          1 | 1/8 seeds        |      nan |
| C_l0-B     | pooled(6 cells)               | mean_em         |         8 | -0.0037 |   -0.0148 |    0.0075 |   0.4625 |   -0.0126 |    0.0053 |                          0 | 3/8 seeds        |      nan |
| C_l0-B     | pooled(6 cells)               | mean_f1         |         8 |  0.0015 |   -0.0084 |    0.0113 |   0.7298 |   -0.0064 |    0.0094 |                          1 | 4/8 seeds        |      nan |
| C_l0-B     | pooled(6 cells)               | option_acc      |         8 | -0.0055 |   -0.0192 |    0.0082 |   0.3717 |   -0.0165 |    0.0054 |                          0 | 2/8 seeds        |      nan |
| C_l0-B     | pooled(6 cells)               | in_context_rate |         8 | -0.0088 |   -0.016  |   -0.0016 |   0.0231 |   -0.0146 |   -0.0031 |                          0 | 1/8 seeds        |      nan |
| C_shuf-B   | pooled(6 cells)               | mean_f_faith    |         8 | -0.0045 |   -0.0079 |   -0.001  |   0.0185 |   -0.0072 |   -0.0017 |                          1 | 1/8 seeds        |      nan |
| C_shuf-B   | pooled(6 cells)               | mean_em         |         8 | -0.0003 |   -0.008  |    0.0074 |   0.9331 |   -0.0064 |    0.0059 |                          1 | 4/8 seeds        |      nan |
| C_shuf-B   | pooled(6 cells)               | mean_f1         |         8 |  0.0015 |   -0.0069 |    0.0098 |   0.6947 |   -0.0053 |    0.0082 |                          1 | 4/8 seeds        |      nan |
| C_shuf-B   | pooled(6 cells)               | option_acc      |         8 |  0.0046 |   -0.0052 |    0.0143 |   0.3055 |   -0.0033 |    0.0124 |                          0 | 3/8 seeds        |      nan |
| C_shuf-B   | pooled(6 cells)               | in_context_rate |         8 | -0.0135 |   -0.0203 |   -0.0067 |   0.0022 |   -0.019  |   -0.0081 |                          0 | 0/8 seeds        |      nan |
| B_ckpt-B   | pooled(6 cells)               | mean_f_faith    |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| B_ckpt-B   | pooled(6 cells)               | mean_em         |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| B_ckpt-B   | pooled(6 cells)               | mean_f1         |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| B_ckpt-B   | pooled(6 cells)               | option_acc      |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| B_ckpt-B   | pooled(6 cells)               | in_context_rate |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| C_ckpt-C   | pooled(6 cells)               | mean_f_faith    |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| C_ckpt-C   | pooled(6 cells)               | mean_em         |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| C_ckpt-C   | pooled(6 cells)               | mean_f1         |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| C_ckpt-C   | pooled(6 cells)               | option_acc      |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| C_ckpt-C   | pooled(6 cells)               | in_context_rate |         3 |  0      |    0      |    0      | nan      |    0      |    0      |                          1 | 0/3 seeds        |      nan |
| C-B        | language=arabic (exploratory) | mean_em         |         8 | -0.0704 |   -0.0763 |   -0.0645 |   0      |  nan      |  nan      |                        nan | nan              |      nan |
| C-B        | language=malay (exploratory)  | mean_em         |         8 | -0.0156 |   -0.0294 |   -0.0018 |   0.0321 |  nan      |  nan      |                        nan | nan              |      nan |
| C-B        | language=arabic (exploratory) | mean_f1         |         8 | -0.0397 |   -0.051  |   -0.0283 |   0.0001 |  nan      |  nan      |                        nan | nan              |      nan |
| C-B        | language=malay (exploratory)  | mean_f1         |         8 | -0.0089 |   -0.0187 |    0.0009 |   0.0694 |  nan      |  nan      |                        nan | nan              |      nan |

PRIMARY:
| contrast   | scope           | metric       |   n_seeds |   mean |   ci95_lo |   ci95_hi |   p |   ci90_lo |   ci90_hi | equivalent_within_margin   | cells_positive   |   p_holm |
|:-----------|:----------------|:-------------|----------:|-------:|----------:|----------:|----:|----------:|----------:|:---------------------------|:-----------------|---------:|
| C-B        | pooled(6 cells) | mean_f_faith |         8 | 0.3043 |    0.2868 |    0.3217 |   0 |    0.2903 |    0.3183 | False                      | 8/8 seeds        |      nan |

ITEM BOOTSTRAP:
{"contrast": "C-B", "metric": "mean_f_faith", "point": 0.30430397710417484, "ci95": [0.2955997080268015, 0.3127934595215669], "unit": "source item (seeds averaged within item)", "n_cells": 6, "note": "resamples items independently within each cell; superseded for reporting by joint_bootstrap.py (items resampled jointly across adapters, and by passage)"}

DIAGNOSTICS:
{
 "sanitize_total": {
  "gen_calls": 5776478,
  "gen_nan": 0,
  "gen_posinf": 0,
  "score_calls": 229392,
  "score_nan": 0,
  "score_posinf": 0
 },
 "runs_with_nonfinite_logits": 0,
 "frac_eval_prompt_truncated": {
  "mean": 0.022899999999999997,
  "max": 0.0458
 },
 "frac_train_prompt_truncated": {
  "mean": 0.022350000000000002,
  "max": 0.0447
 },
 "n_passage_overlap_excluded": {
  "mean": 2.0,
  "max": 4.0
 },
 "legacy_eval_state_minus_headline_by_label": {
  "B": -0.0195,
  "B_ckpt": -0.1811,
  "C": -0.302,
  "C_ckpt": -0.492,
  "C_l0": -0.0156,
  "C_shuf": -0.0186
 },
 "adalora_rank_first_last": [],
 "n_failed_runs": 0,
 "n_runs": 228
}
