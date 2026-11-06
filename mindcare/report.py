"""
Shared reporting: the same validation + challenge-set tables for every model,
so the TF-IDF baseline, the Bi-LSTM and later models are compared on equal terms.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mindcare.metrics import TARGET_RECALL, choose_threshold, evaluate, wilson_interval

CHALLENGE = Path("eval/challenge/challenge_v1.csv")


def _fmt(m: dict, key: str) -> str:
    """'0.941 [0.938, 0.944]' : point estimate with its 95 % bootstrap interval."""
    lo, hi = m.get("ci95", {}).get(key, [None, None])
    return f"{m[key]:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else "")


def build_report(model_name: str, y_val, val_score, challenge: pd.DataFrame, challenge_score, extra: dict) -> dict:
    """Choose the threshold on validation (pre-registered rule), then score val + challenge."""
    y_val, val_score = np.asarray(y_val), np.asarray(val_score, dtype=float)
    thr = choose_threshold(y_val, val_score)
    res = {"model": model_name, "target_recall": TARGET_RECALL, "threshold": thr, **extra,
           "val_at_0.5": evaluate(y_val, val_score, 0.5),
           "val_at_chosen": evaluate(y_val, val_score, thr)}
    ch = challenge.copy()
    ch["score"] = np.asarray(challenge_score, dtype=float)
    ch["pred"] = (ch["score"] >= thr).astype(int)
    ch["correct"] = (ch["pred"] == ch["label"]).astype(int)
    groups = {}
    for (g, lab), sub in ch.groupby(["group", "label"]):
        k, n = int(sub.correct.sum()), len(sub)
        groups[f"{g} (label {lab})"] = {"correct": k, "n": n, "accuracy": k / n,
                                        "wilson95": wilson_interval(k, n),
                                        "mean_score": float(sub.score.mean())}
    res["challenge_overall"] = evaluate(ch.label, ch.score, thr, with_ci=False)
    res["challenge_groups"] = groups
    res["challenge_items"] = ch[["id", "label", "score", "pred"]].round(4).to_dict("records")
    return res


def write_report(res: dict, out_dir: Path, stem: str, title: str) -> str:
    """Write <stem>.json and <STEM>.md; return the markdown text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    v5, vc, thr = res["val_at_0.5"], res["val_at_chosen"], res["threshold"]
    md = [f"# {title}", "", "95 % bootstrap CIs (1000 resamples). Test split untouched.", "",
          "## Validation", "| metric | threshold 0.5 | chosen threshold |", "|---|---|---|",
          f"| threshold | 0.5 | {thr:.4f} |"]
    for k in ("accuracy", "macro_f1", "precision_1", "recall_1", "roc_auc", "pr_auc"):
        md.append(f"| {k} | {_fmt(v5, k)} | {_fmt(vc, k)} |")
    md += [f"| brier | {v5['brier']:.4f} | (same) |",
           f"| ECE (10 bins) | {v5['calibration']['ece']:.4f} | (same) |",
           f"| confusion (tp, fp, fn, tn) | {tuple(v5['confusion'].values())} | {tuple(vc['confusion'].values())} |",
           "", f"## Challenge set v1 (threshold {thr:.4f})",
           "| group | correct / n | accuracy [Wilson 95 %] | mean score |", "|---|---|---|---|"]
    for g, s in res["challenge_groups"].items():
        lo, hi = s["wilson95"]
        md.append(f"| {g} | {s['correct']}/{s['n']} | {s['accuracy']:.2f} [{lo:.2f}, {hi:.2f}] | {s['mean_score']:.3f} |")
    co = res["challenge_overall"]
    md.append(f"\nOverall challenge accuracy: {co['accuracy']:.3f}, macro-F1: {co['macro_f1']:.3f} (n = {co['n']})")
    text = "\n".join(md) + "\n"
    (out_dir / f"{stem.upper()}.md").write_text(text, encoding="utf-8")
    return text
