# Final test evaluation (frozen test split, scored once)

n = 34,635 posts, class-1 share 50.1%. Thresholds chosen on validation. 95 % bootstrap CIs (1000 resamples).

| metric | TF-IDF + LR | Bi-LSTM + GloVe | difference (Bi-LSTM - TF-IDF) [paired 95 % CI] |
|---|---|---|---|
| threshold (from validation) | 0.6150 | 0.8690 | |
| macro_f1 | 0.936 [0.933, 0.938] | 0.938 [0.936, 0.941] | +0.0024 [-0.0000, +0.0049] (not significant) |
| accuracy | 0.936 [0.933, 0.938] | 0.938 [0.936, 0.941] | +0.0024 [-0.0000, +0.0049] (not significant) |
| precision_1 | 0.966 [0.963, 0.969] | 0.969 [0.966, 0.971] | +0.0023 [-0.0001, +0.0048] (not significant) |
| recall_1 | 0.904 [0.899, 0.908] | 0.906 [0.902, 0.911] | +0.0027 [-0.0012, +0.0070] (not significant) |
| roc_auc | 0.985 [0.984, 0.986] | 0.987 [0.986, 0.988] | +0.0022 [+0.0015, +0.0030] (significant) |
| pr_auc | 0.984 | 0.987 | |
| ECE (10 bins) | 0.0344 | 0.0315 | |
| confusion (tp, fp, fn, tn) | (15680, 546, 1671, 16738) | (15727, 509, 1624, 16775) | |
