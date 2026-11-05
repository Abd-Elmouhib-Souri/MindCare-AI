"""
Load Suicide_Detection.csv and apply the cleaning rules fixed in CORE 1.3
(same rules as scripts/shortcut_probe.py):
  1. drop posts whose normalized text is empty (emoji-only, non-Latin scripts...)
  2. drop texts that appear with both labels
  3. drop normalized duplicates (keep the first)
  4. assign the frozen hash split (mindcare/splits.py)
"""

import pandas as pd

from mindcare.splits import assign_split, normalize

POSITIVE_LABEL = "suicide"


def load_clean_dataset(csv_path: str) -> pd.DataFrame:
    """Return columns: text, y (1 = suicide subreddit), split ('train' / 'val' / 'test')."""
    df = pd.read_csv(csv_path).dropna(subset=["text", "class"])
    df["text"] = df["text"].astype(str)
    df["y"] = (df["class"] == POSITIVE_LABEL).astype(int)
    df["norm"] = df["text"].map(normalize)
    df = df[df["norm"] != ""]                                        # rule 1
    n_labels = df.groupby("norm")["y"].nunique()
    df = df[~df["norm"].isin(n_labels[n_labels > 1].index)]          # rule 2
    df = df.drop_duplicates("norm", keep="first").reset_index(drop=True)   # rule 3
    df["split"] = df["text"].map(assign_split)                        # rule 4
    return df[["text", "y", "split"]]
