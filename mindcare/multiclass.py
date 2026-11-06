"""
Multi-class metrics for the face and voice emotion models (7 emotions).
Covered by tests/test_multiclass.py.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABELS = list(range(len(EMOTIONS)))


def multiclass_metrics(y, pred) -> dict:
    """Accuracy, macro-F1 (each emotion counts equally), per-class P/R/F1, confusion matrix."""
    y, pred = np.asarray(y), np.asarray(pred)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {EMOTIONS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                    "support": int(s[i])} for i in LABELS},
        "confusion": confusion_matrix(y, pred, labels=LABELS).tolist(),   # rows = true, columns = predicted
    }


def multiclass_bootstrap(y, pred, n_boot: int = 1000, seed: int = 42) -> dict:
    """95 % percentile bootstrap intervals for accuracy and macro-F1."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    acc, mf1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        acc.append(np.mean(y[idx] == pred[idx]))
        mf1.append(f1_score(y[idx], pred[idx], labels=LABELS, average="macro", zero_division=0))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]   # noqa: E731
    return {"accuracy": ci(acc), "macro_f1": ci(mf1)}


def top_label_ece(y, probs, n_bins: int = 10) -> float:
    """Calibration of the top prediction: is a '70 % confident' emotion right 70 % of the time?"""
    from mindcare.metrics import expected_calibration_error
    probs = np.asarray(probs)
    correct = (probs.argmax(axis=1) == np.asarray(y)).astype(float)
    return expected_calibration_error(correct, probs.max(axis=1), n_bins)["ece"]


def markdown_table(name_to_result: dict) -> list:
    """Rows: protocol / set -> accuracy and macro-F1 with CIs."""
    rows = ["| evaluation | n | accuracy [95 % CI] | macro-F1 [95 % CI] |", "|---|---|---|---|"]
    for name, r in name_to_result.items():
        a, f = r["ci95"]["accuracy"], r["ci95"]["macro_f1"]
        rows.append(f"| {name} | {r['n']:,} | {r['accuracy']:.3f} [{a[0]:.3f}, {a[1]:.3f}] | "
                    f"{r['macro_f1']:.3f} [{f[0]:.3f}, {f[1]:.3f}] |")
    return rows


def per_class_table(result: dict) -> list:
    rows = ["| emotion | support | precision | recall | F1 |", "|---|---|---|---|---|"]
    for e, m in result["per_class"].items():
        rows.append(f"| {e} | {m['support']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    return rows
