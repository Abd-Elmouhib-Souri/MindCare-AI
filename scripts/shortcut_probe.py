"""
CORE 1.3 - Measure the shortcuts BEFORE training the real model.

Question: how much of the 'suicide vs non-suicide' signal can be captured by
features that have nothing to do with distress (length, style, one word)?
If these 'dumb' probes already score high, a high score from the Bi-LSTM does
NOT prove it understands distress.

Rules fixed in advance:
  - Probes are trained on TRAIN and scored on VAL only. TEST is never touched.
  - Data cleaning: drop posts whose normalized text is empty, drop normalized
    duplicates (keep first), drop texts that appear with both labels.

Writes: eval/audit/shortcut_probe.json and eval/audit/SHORTCUT_PROBE.md

Usage (repo root, .venv active):
  python scripts/shortcut_probe.py --csv ..\\mindcare_data\\dataset\\text\\Suicide_Detection.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Make the repo root importable, so "mindcare.splits" works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.splits import assign_split, normalize  # noqa: E402

NEGATION_RE = re.compile(r"\b(not|no|never|nothing|nobody|nowhere|neither|nor|cannot|none)\b|n't\b", re.I)
SECOND_PERSON_RE = re.compile(r"\b(you|your|you're|ur|u)\b", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
HTML_ENTITY_RE = re.compile(r"&(gt|lt|amp|quot|#\d+);", re.I)


def scores(y_true, y_pred, y_score) -> dict:
    """Accuracy, macro-F1 (both classes count equally) and ROC-AUC (ranking quality)."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4) if y_score is not None else None,
    }


def style_features(texts: pd.Series) -> np.ndarray:
    """Features with NO distress vocabulary: length, negation rate, 'you' rate, links, HTML entities."""
    n_words = texts.str.split().str.len().clip(lower=1)
    return np.column_stack([
        np.log1p(n_words),                                             # length (log)
        texts.map(lambda t: len(NEGATION_RE.findall(t))) / n_words,   # negations per word
        texts.map(lambda t: len(SECOND_PERSON_RE.findall(t))) / n_words,  # 'you' per word
        texts.map(lambda t: bool(URL_RE.search(t))).astype(float),    # has a link
        texts.map(lambda t: len(HTML_ENTITY_RE.findall(t))).astype(float),  # &gt; &amp; ...
    ])


def fit_lr(X_train, y_train, X_val):
    """Plain logistic regression; returns hard predictions and probabilities on val."""
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    return clf, clf.predict(X_val), clf.predict_proba(X_val)[:, 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--positive", default="suicide")
    p.add_argument("--out", default="eval/audit")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # ---- 1. Load, then explain the 'conflicting label' group from the audit ----------
    df = pd.read_csv(args.csv).dropna(subset=["text", "class"])
    df["text"] = df["text"].astype(str)
    df["y"] = (df["class"] == args.positive).astype(int)
    df["norm"] = df["text"].map(normalize)

    empty = df["norm"] == ""
    report["empty_after_normalization"] = {
        "rows": int(empty.sum()),
        "by_class": {str(k): int(v) for k, v in df.loc[empty, "class"].value_counts().items()},
        # first characters of a few such posts, to see WHAT they are (emoji? other script?)
        "examples_repr": [repr(t[:40]) for t in df.loc[empty, "text"].head(5)],
    }
    conflict = df[~empty].groupby("norm")["y"].nunique()
    report["conflicting_nonempty_texts"] = int((conflict > 1).sum())

    # ---- 2. Clean according to the fixed rules --------------------------------------
    n0 = len(df)
    df = df[~empty]
    df = df[~df["norm"].isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates("norm", keep="first").reset_index(drop=True)
    report["rows_before_cleaning"] = n0
    report["rows_after_cleaning"] = int(len(df))

    # ---- 3. Frozen split (hash of the normalized text) ------------------------------
    df["split"] = df["text"].map(assign_split)
    report["split_sizes"] = {
        s: {"rows": int((df["split"] == s).sum()),
            "class1_share": round(float(df.loc[df["split"] == s, "y"].mean()), 4)}
        for s in ("train", "val", "test")
    }
    # Sanity check: no normalized text appears in two splits
    report["norm_texts_in_more_than_one_split"] = int((df.groupby("norm")["split"].nunique() > 1).sum())

    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]   # TEST is not used below
    ytr, yva = tr["y"].to_numpy(), va["y"].to_numpy()
    probes = {}

    # P0 - majority class: the floor every model must beat
    majority = int(round(ytr.mean()))
    probes["P0_majority_class"] = scores(yva, np.full(len(yva), majority), None)

    # P1 - length only (one number per post)
    Ltr = np.log1p(tr["text"].str.split().str.len().to_numpy()).reshape(-1, 1)
    Lva = np.log1p(va["text"].str.split().str.len().to_numpy()).reshape(-1, 1)
    _, pred, prob = fit_lr(Ltr, ytr, Lva)
    probes["P1_length_only"] = scores(yva, pred, prob)

    # P2 - the single word "filler"
    has_filler = va["text"].str.contains(r"\bfiller\b", case=False, regex=True).to_numpy()
    probes["P2_word_filler"] = {
        "share_of_class0_val_posts_containing_it": round(float(has_filler[yva == 0].mean()), 4),
        "share_of_class1_val_posts_containing_it": round(float(has_filler[yva == 1].mean()), 4),
    }

    # P3 - style only: length, negation rate, 'you' rate, links, HTML entities (no topic words)
    scaler = StandardScaler()
    Str = scaler.fit_transform(style_features(tr["text"]))   # scaler fitted on TRAIN only
    Sva = scaler.transform(style_features(va["text"]))
    _, pred, prob = fit_lr(Str, ytr, Sva)
    probes["P3_style_only"] = scores(yva, pred, prob)

    # P4 - full bag of words: TF-IDF (1-2 grams) + logistic regression = baseline (b)
    print("Fitting TF-IDF + logistic regression (a few minutes)...")
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=200_000,
                          sublinear_tf=True, dtype=np.float32)   # NO stopword removal: keeps negations
    Ttr = vec.fit_transform(tr["text"])                         # vocabulary from TRAIN only
    Tva = vec.transform(va["text"])
    clf, pred, prob = fit_lr(Ttr, ytr, Tva)
    probes["P4_tfidf_logreg"] = scores(yva, pred, prob)
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    report["P4_top_weights"] = {
        "toward_class1": [[str(vocab[i]), round(float(clf.coef_[0][i]), 2)] for i in order[::-1][:20]],
        "toward_class0": [[str(vocab[i]), round(float(clf.coef_[0][i]), 2)] for i in order[:20]],
    }
    report["probes_on_val"] = probes

    # ---- 4. Write outputs ----------------------------------------------------------
    (out / "shortcut_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    e = report["empty_after_normalization"]
    md = [
        "# Shortcut probes (train -> validation; test untouched)", "",
        "Generated by `scripts/shortcut_probe.py`. Do not edit by hand.", "",
        "## Cleaning",
        f"- Empty after normalization: {e['rows']} rows {e['by_class']}; examples: {e['examples_repr']}",
        f"- Non-empty texts with conflicting labels: {report['conflicting_nonempty_texts']}",
        f"- Rows: {report['rows_before_cleaning']:,} -> {report['rows_after_cleaning']:,} after cleaning", "",
        "## Frozen split (hash of normalized text)",
        "| split | rows | class-1 share |", "|---|---|---|",
    ]
    for s, v in report["split_sizes"].items():
        md.append(f"| {s} | {v['rows']:,} | {v['class1_share']:.1%} |")
    md += [f"- Normalized texts found in more than one split: {report['norm_texts_in_more_than_one_split']}", "",
           "## Probes (validation set)", "| probe | accuracy | macro-F1 | ROC-AUC |", "|---|---|---|---|"]
    for name, s in probes.items():
        if "accuracy" in s:
            md.append(f"| {name} | {s['accuracy']} | {s['macro_f1']} | {s['roc_auc']} |")
    f2 = probes["P2_word_filler"]
    md += ["", f"- 'filler' appears in {f2['share_of_class0_val_posts_containing_it']:.1%} of class-0 val posts "
               f"and {f2['share_of_class1_val_posts_containing_it']:.2%} of class-1 val posts.", "",
           "## TF-IDF + LR: strongest weights",
           "| toward class 1 | weight | toward class 0 | weight |", "|---|---|---|---|"]
    for (w1, c1), (w0, c0) in zip(report["P4_top_weights"]["toward_class1"], report["P4_top_weights"]["toward_class0"]):
        md.append(f"| {w1} | {c1} | {w0} | {c0} |")
    (out / "SHORTCUT_PROBE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
