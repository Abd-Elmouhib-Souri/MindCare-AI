"""
CORE 2.7 - Error analysis sample (Bi-LSTM on the TEST split).

Draws a fixed random sample (seed 42) of false negatives (distress posts missed)
and false positives (non-distress posts flagged) from the private prediction file,
and writes a PRIVATE markdown sheet for MANUAL reading. Real Reddit posts: the
sheet is git-ignored (eval/audit/*PRIVATE*). Only the category COUNTS go public.

Categories to assign while reading (one per post):
  NEG    negation changes the meaning ("not suicidal", "never felt better")
  SARC   sarcasm, irony, jokes, song lyrics
  TOPIC  teen-life topic pulls the score (school, crush, gaming) or the reverse
  SHORT  too short / too vague to decide
  LABEL  the subreddit label looks wrong for this post (label noise)
  OTHER  anything else (write a few words)

Usage:  python scripts/sample_errors.py            (writes eval/audit/error_analysis_PRIVATE.md)
"""

from pathlib import Path

import pandas as pd

N_PER_TYPE = 15
SEED = 42
SRC = Path("eval/audit/test_predictions_PRIVATE.csv")
OUT = Path("eval/audit/error_analysis_PRIVATE.md")

df = pd.read_csv(SRC)
fn = df[(df.y == 1) & (df.pred_bilstm == 0)]   # missed distress posts
fp = df[(df.y == 0) & (df.pred_bilstm == 1)]   # non-distress posts flagged as distress
print(f"Bi-LSTM test errors: {len(fn):,} false negatives, {len(fp):,} false positives")

lines = ["# PRIVATE - error analysis sheet (do not commit)", "",
         "Content warning: real posts about suicide and self-harm.", "",
         "For each post, replace `category: ?` by NEG, SARC, TOPIC, SHORT, LABEL or OTHER.", ""]
for name, part in (("FALSE NEGATIVES (label 1 = r/SuicideWatch, model said 0)", fn),
                   ("FALSE POSITIVES (label 0 = r/teenagers, model said 1)", fp)):
    lines += [f"## {name}", ""]
    for i, (_, r) in enumerate(part.sample(min(N_PER_TYPE, len(part)), random_state=SEED).iterrows(), 1):
        text = str(r.text).replace("\n", " ")
        lines += [f"### {i}. score Bi-LSTM {r.score_bilstm:.3f} | TF-IDF {r.score_tfidf:.3f}",
                  text[:700] + ("..." if len(text) > 700 else ""), "", "category: ?", ""]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Written: {OUT}  (open it, fill the categories, then count them)")
