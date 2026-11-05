"""
CORE 1.2 - Audit of the text dataset BEFORE any training.

Why: a model is only as honest as its labels. Before we train anything, we
measure what is really inside the CSV: size, label balance, duplicates, text
length, negations, and the words that separate the two classes (a first sign
of a topic shortcut). Every number below is computed by this script.

What it writes (from the repo root):
  eval/audit/text_audit.json        -> all numbers, machine-readable (commit it)
  eval/audit/TEXT_AUDIT.md          -> the same numbers as readable tables (commit it)
  eval/audit/examples_PRIVATE.md    -> random raw posts for MANUAL reading
                                       (sensitive content: git-ignored, never commit)

Usage (repo root, .venv active):
  python scripts/audit_data.py --csv ..\\mindcare_data\\dataset\\text\\Suicide_Detection.csv
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

SEED = 42                      # fixed seed: the same examples are drawn on every run
MODEL_WINDOW = 150             # max_sequence_len used by src/data/text_pipeline.py
N_EXAMPLES = 8                 # raw examples per class to read by hand
TOP_WORDS = 25                 # words shown per class in the log-odds table

# Negation words. NLTK's English stopword list deletes "not", "no", "nor", and
# every "n't" contraction ("don't" -> "don", "t"), so we count how often they occur.
NEGATION_RE = re.compile(
    r"\b(not|no|never|nothing|nobody|nowhere|neither|nor|cannot|none)\b|n't\b",
    flags=re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Lowercase, keep letters and digits only, collapse spaces.
    Two posts that differ only by punctuation, case or spacing become identical,
    which lets us catch 'near-exact' duplicates cheaply. Digits are KEPT:
    posts that differ by a number (an age, a date) are different posts."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)   # anything that is not a letter/digit -> space
    return re.sub(r"\s+", " ", text).strip()  # collapse repeated spaces


def describe_lengths(word_counts: pd.Series) -> dict:
    """Length statistics (in whitespace-separated words) for one class."""
    return {
        "mean": round(float(word_counts.mean()), 1),
        "median": int(word_counts.median()),
        "p95": int(word_counts.quantile(0.95)),
        "max": int(word_counts.max()),
        # Share of posts longer than the model window: those get truncated,
        # so the model never sees part of the text.
        f"share_over_{MODEL_WINDOW}_words": round(float((word_counts > MODEL_WINDOW).mean()), 4),
    }


def log_odds_dirichlet(texts: pd.Series, y: np.ndarray, top_k: int) -> dict:
    """Words that most separate class 1 from class 0.

    Method: log-odds ratio with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn, 2008, 'Fightin' Words'). The prior is the word's
    count in the whole corpus. The z-score tells us how strongly a word is
    tied to one class, while not over-rewarding rare words (unlike raw ratios).
    """
    vec = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z']+\b",   # words of 2+ letters, keeps "don't"
        min_df=20,                               # ignore very rare words (noise, names)
        max_features=50000,
        dtype=np.int32,
    )
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    y1 = np.asarray(X[y == 1].sum(axis=0)).ravel().astype(np.float64)  # counts in class 1
    y0 = np.asarray(X[y == 0].sum(axis=0)).ravel().astype(np.float64)  # counts in class 0
    prior = y1 + y0                                                      # background counts
    a0 = prior.sum()
    n1, n0 = y1.sum(), y0.sum()

    # log-odds of each word in each class, smoothed by the prior
    l1 = np.log((y1 + prior) / (n1 + a0 - y1 - prior))
    l0 = np.log((y0 + prior) / (n0 + a0 - y0 - prior))
    delta = l1 - l0
    variance = 1.0 / (y1 + prior) + 1.0 / (y0 + prior)
    z = delta / np.sqrt(variance)

    def rows(indices):
        # per-10k-token rates make the two classes comparable despite different sizes
        return [
            {
                "word": str(vocab[i]),
                "z": round(float(z[i]), 1),
                "per10k_class1": round(float(y1[i] / n1 * 1e4), 2),
                "per10k_class0": round(float(y0[i] / n0 * 1e4), 2),
            }
            for i in indices
        ]

    order = np.argsort(z)
    return {"class1": rows(order[::-1][:top_k]), "class0": rows(order[:top_k])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the text dataset before training.")
    parser.add_argument("--csv", required=True, help="Path to Suicide_Detection.csv")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="class")
    parser.add_argument("--positive", default="suicide", help="Label value treated as class 1")
    parser.add_argument("--out", default="eval/audit", help="Output folder")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load and check columns --------------------------------------------
    df = pd.read_csv(args.csv)
    report: dict = {"source_file": Path(args.csv).name, "columns": list(df.columns)}
    for col in (args.text_col, args.label_col):
        if col not in df.columns:
            raise SystemExit(f"Column '{col}' not found. Columns are: {list(df.columns)}")

    report["rows_raw"] = int(len(df))
    report["missing_text"] = int(df[args.text_col].isna().sum())
    df = df.dropna(subset=[args.text_col, args.label_col]).copy()
    df["text"] = df[args.text_col].astype(str)
    report["empty_text"] = int((df["text"].str.strip() == "").sum())
    # No author or thread id in the file -> a group-aware split (by author)
    # is impossible; deduplication is the best protection we have.
    report["has_author_or_thread_id"] = any(
        c.lower() in {"author", "user", "username", "thread_id", "link_id", "id"} for c in df.columns
    )

    # ---- 2. Labels --------------------------------------------------------------
    report["label_values"] = {str(k): int(v) for k, v in df[args.label_col].value_counts().items()}
    df["y"] = (df[args.label_col].astype(str) == args.positive).astype(int)
    report["class1_share"] = round(float(df["y"].mean()), 4)

    # ---- 3. Duplicates -------------------------------------------------------------
    # A duplicate that lands in train AND test inflates the test score:
    # the model is asked about a post it has already seen.
    df["norm"] = df["text"].map(normalize)
    exact_dup_rows = int(df.duplicated("text").sum())
    norm_dup_rows = int(df.duplicated("norm").sum())
    labels_per_text = df.groupby("norm")["y"].nunique()
    conflict_texts = labels_per_text[labels_per_text > 1].index
    report["duplicates"] = {
        "exact_duplicate_rows": exact_dup_rows,
        "normalized_duplicate_rows": norm_dup_rows,
        "rows_after_dedup": int(len(df) - norm_dup_rows),
        # Same text, different labels: the label is necessarily wrong for some of them
        "texts_with_conflicting_labels": int(len(conflict_texts)),
        "rows_with_conflicting_labels": int(df["norm"].isin(conflict_texts).sum()),
    }

    # ---- 4. Length per class ----------------------------------------------------------
    df["n_words"] = df["text"].str.split().str.len()
    report["length_words"] = {
        str(name): describe_lengths(group["n_words"])
        for name, group in df.groupby(args.label_col)
    }

    # ---- 5. Negations per class -------------------------------------------------------
    df["n_neg"] = df["text"].map(lambda t: len(NEGATION_RE.findall(t)))
    report["negation"] = {
        str(name): {
            "share_posts_with_negation": round(float((g["n_neg"] > 0).mean()), 4),
            "negations_per_100_words": round(float(g["n_neg"].sum() / max(g["n_words"].sum(), 1) * 100), 2),
        }
        for name, g in df.groupby(args.label_col)
    }

    # ---- 6. Most class-specific words (topic-shortcut hint) ------------------------------
    print("Computing class-specific words (1-2 minutes on the full file)...")
    report["log_odds_top_words"] = log_odds_dirichlet(df["text"], df["y"].to_numpy(), TOP_WORDS)

    # ---- 7. Raw examples for manual reading (PRIVATE, not committed) ---------------------
    rng = np.random.default_rng(SEED)
    lines = [
        "# PRIVATE - raw examples for manual label checking",
        "",
        "Content warning: posts about suicide and self-harm. Do not commit this file.",
        "",
    ]
    for name, g in df.groupby(args.label_col):
        idx = rng.choice(len(g), size=min(N_EXAMPLES, len(g)), replace=False)
        lines.append(f"## class = {name}\n")
        for j, i in enumerate(idx, 1):
            snippet = g["text"].iloc[i].replace("\n", " ")[:400]
            lines.append(f"{j}. {snippet}{'...' if len(g['text'].iloc[i]) > 400 else ''}\n")
    (out / "examples_PRIVATE.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- Write JSON + Markdown ---------------------------------------------------------
    (out / "text_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    d = report["duplicates"]
    md = [
        f"# Text dataset audit - `{report['source_file']}`",
        "",
        "Generated by `scripts/audit_data.py`. Do not edit by hand.",
        "",
        "## Size and labels",
        f"- Rows: {report['rows_raw']:,} (missing text: {report['missing_text']}, empty: {report['empty_text']})",
        f"- Labels: {report['label_values']} -> class-1 share {report['class1_share']:.1%}",
        f"- Author / thread id available: {report['has_author_or_thread_id']}",
        "",
        "## Duplicates",
        f"- Exact duplicate rows: {d['exact_duplicate_rows']:,}",
        f"- Normalized duplicate rows (case/punctuation ignored): {d['normalized_duplicate_rows']:,}",
        f"- Rows after deduplication: {d['rows_after_dedup']:,}",
        f"- Texts with conflicting labels: {d['texts_with_conflicting_labels']:,} "
        f"({d['rows_with_conflicting_labels']:,} rows)",
        "",
        f"## Length (words) - model window = {MODEL_WINDOW} tokens",
        "| class | mean | median | p95 | max | share > window |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in report["length_words"].items():
        md.append(f"| {name} | {s['mean']} | {s['median']} | {s['p95']} | {s['max']} | "
                  f"{s[f'share_over_{MODEL_WINDOW}_words']:.1%} |")
    md += ["", "## Negations", "| class | posts with >=1 negation | negations / 100 words |", "|---|---|---|"]
    for name, s in report["negation"].items():
        md.append(f"| {name} | {s['share_posts_with_negation']:.1%} | {s['negations_per_100_words']} |")
    for side, title in (("class1", f"'{args.positive}'"), ("class0", f"not '{args.positive}'")):
        md += ["", f"## Words most tied to {title} (log-odds, Dirichlet prior)",
               "| word | z | per 10k (class 1) | per 10k (class 0) |", "|---|---|---|---|"]
        for r in report["log_odds_top_words"][side]:
            md.append(f"| {r['word']} | {r['z']} | {r['per10k_class1']} | {r['per10k_class0']} |")
    (out / "TEXT_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nWritten: {out / 'text_audit.json'}, {out / 'TEXT_AUDIT.md'}, {out / 'examples_PRIVATE.md'}")


if __name__ == "__main__":
    main()
