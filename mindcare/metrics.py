"""
One evaluation module for every model (TF-IDF baseline, Bi-LSTM, transformer...).
Same code -> fair comparison. Covered by tests/test_metrics.py.

Pre-registered decision rule (fixed 5 Nov 2026, BEFORE any model's validation
scores were used with it):
    threshold = the one with the best precision among thresholds that reach
    distress (class 1) recall >= TARGET_RECALL on the VALIDATION set.
Reason: missing a distress post costs more than a false alert, but 0.99 recall
would flood the system with false alerts.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

TARGET_RECALL = 0.90   # pre-registered; do not tune after seeing results
N_BOOT = 1000          # bootstrap resamples for 95 % confidence intervals
SEED = 42


def _counts(y: np.ndarray, pred: np.ndarray):
    """Confusion-matrix counts (tp, fp, fn, tn) with class 1 = distress."""
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def threshold_metrics(y, score, threshold: float) -> dict:
    """Metrics that depend on a decision threshold."""
    y, score = np.asarray(y), np.asarray(score)
    pred = (score >= threshold).astype(int)
    tp, fp, fn, tn = _counts(y, pred)
    p1, r1 = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)   # distress class
    p0, r0 = _safe_div(tn, tn + fn), _safe_div(tn, tn + fp)   # non-distress class
    f1_1 = _safe_div(2 * p1 * r1, p1 + r1)
    f1_0 = _safe_div(2 * p0 * r0, p0 + r0)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": _safe_div(tp + tn, len(y)),
        "macro_f1": (f1_1 + f1_0) / 2,      # both classes count equally, unlike accuracy
        "precision_1": p1, "recall_1": r1, "f1_1": f1_1,
        "precision_0": p0, "recall_0": r0, "f1_0": f1_0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def expected_calibration_error(y, score, n_bins: int = 10) -> dict:
    """ECE: average gap between predicted probability and observed frequency.
    0 = perfectly calibrated ("70 % confident" is right 70 % of the time)."""
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(score, edges[1:-1], right=False), 0, n_bins - 1)
    ece, bins = 0.0, []
    for b in range(n_bins):
        mask = bin_id == b
        if not mask.any():
            continue
        conf, freq, n = float(score[mask].mean()), float(y[mask].mean()), int(mask.sum())
        ece += abs(freq - conf) * n / len(y)
        bins.append({"lo": round(edges[b], 2), "hi": round(edges[b + 1], 2),
                     "n": n, "mean_score": round(conf, 4), "frac_positive": round(freq, 4)})
    return {"ece": ece, "bins": bins}


def ranking_metrics(y, score) -> dict:
    """Threshold-free metrics: ROC-AUC, PR-AUC (average precision), Brier score."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    out = {"brier": float(np.mean((score - y) ** 2))}
    if len(np.unique(y)) == 2:   # AUCs are undefined with a single class
        out["roc_auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    return out


def choose_threshold(y_val, score_val, target_recall: float = TARGET_RECALL) -> float:
    """Pre-registered rule: best precision among thresholds with recall_1 >= target (VALIDATION only)."""
    precision, recall, thresholds = precision_recall_curve(y_val, score_val)
    # precision/recall have one more element than thresholds (the end point); drop it
    ok = recall[:-1] >= target_recall
    if not ok.any():
        return float(thresholds.min())
    candidates = np.where(ok)[0]
    best = candidates[np.argmax(precision[:-1][candidates])]
    return float(thresholds[best])


def bootstrap_ci(y, score, threshold: float, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95 % percentile bootstrap intervals: resample the evaluation set with replacement
    and recompute each metric; the 2.5th and 97.5th percentiles give the interval."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    keys = ("accuracy", "macro_f1", "precision_1", "recall_1")
    samples = {k: [] for k in keys + ("roc_auc", "pr_auc")}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ys, ss = y[idx], score[idx]
        m = threshold_metrics(ys, ss, threshold)
        for k in keys:
            samples[k].append(m[k])
        if len(np.unique(ys)) == 2:
            samples["roc_auc"].append(roc_auc_score(ys, ss))
            samples["pr_auc"].append(average_precision_score(ys, ss))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in samples.items() if v}


def wilson_interval(k: int, n: int, z: float = 1.96) -> list:
    """95 % Wilson interval for a proportion k/n. Better than the normal
    approximation for small n (e.g. 20 challenge items per group)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    # clip to [0, 1]: floating-point rounding can give -1e-17 when k = 0
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def evaluate(y, score, threshold: float, with_ci: bool = True) -> dict:
    """Everything for one (model, dataset, threshold): point metrics, AUCs, calibration, CIs."""
    out = threshold_metrics(y, score, threshold)
    out.update(ranking_metrics(y, score))
    out["calibration"] = expected_calibration_error(y, score)
    out["n"] = int(len(y))
    if with_ci:
        out["ci95"] = bootstrap_ci(y, score, threshold)
    return out
