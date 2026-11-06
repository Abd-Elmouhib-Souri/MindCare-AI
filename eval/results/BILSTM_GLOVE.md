# Bi-LSTM + GloVe (frozen) - train -> validation, challenge set

95 % bootstrap CIs (1000 resamples). Test split untouched.

## Validation
| metric | threshold 0.5 | chosen threshold |
|---|---|---|
| threshold | 0.5 | 0.8690 |
| accuracy | 0.942 [0.939, 0.944] | 0.937 [0.934, 0.939] |
| macro_f1 | 0.942 [0.939, 0.944] | 0.937 [0.934, 0.939] |
| precision_1 | 0.924 [0.920, 0.928] | 0.971 [0.969, 0.974] |
| recall_1 | 0.963 [0.960, 0.966] | 0.901 [0.897, 0.905] |
| roc_auc | 0.987 [0.986, 0.988] | 0.987 [0.986, 0.988] |
| pr_auc | 0.987 [0.986, 0.988] | 0.987 [0.986, 0.988] |
| brier | 0.0445 | (same) |
| ECE (10 bins) | 0.0309 | (same) |
| confusion (tp, fp, fn, tn) | (16764, 1384, 643, 15946) | (15678, 468, 1729, 16862) |

## Challenge set v1 (threshold 0.8690)
| group | correct / n | accuracy [Wilson 95 %] | mean score |
|---|---|---|---|
| a_implicit_distress (label 1) | 3/20 | 0.15 [0.05, 0.36] | 0.597 |
| b_keyword_no_distress (label 0) | 17/20 | 0.85 [0.64, 0.95] | 0.284 |
| c_teen_topic_distress (label 1) | 1/20 | 0.05 [0.01, 0.24] | 0.148 |
| d_negation (label 0) | 10/10 | 1.00 [0.72, 1.00] | 0.412 |
| d_negation (label 1) | 2/10 | 0.20 [0.06, 0.51] | 0.653 |

Overall challenge accuracy: 0.412, macro-F1: 0.369 (n = 80)
