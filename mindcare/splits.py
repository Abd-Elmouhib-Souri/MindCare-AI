"""
Deterministic, duplicate-safe train / validation / test assignment.

Why a hash instead of train_test_split(random_state=42)?
  1. Duplicate-safe: two posts with the same normalized text get the same hash,
     so they ALWAYS land in the same split. A copy can never sit in train and test.
  2. Stable: adding or removing other rows never moves a post to another split
     (random_state splits reshuffle everything when the row count changes).
  3. Auditable: anyone can recompute the split from the text alone.

Limit: the dataset has no author id, so two different posts by the same person
can still fall in different splits. We state this in the README.
"""

import hashlib
import re

SALT = "mindcare-split-v1"   # changing it would create a different split: never change it after use
VAL_PCT = 15                 # 15 % validation (threshold choice, early stopping, model choice)
TEST_PCT = 15                # 15 % test, frozen, used ONCE at the very end


def normalize(text: str) -> str:
    """Lowercase, keep letters and digits, collapse spaces (same rule as the audit)."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def assign_split(text: str, val_pct: int = VAL_PCT, test_pct: int = TEST_PCT) -> str:
    """Return 'train', 'val' or 'test' for one post, from the hash of its normalized text."""
    key = (SALT + "|" + normalize(text)).encode("utf-8")
    bucket = int(hashlib.sha1(key).hexdigest(), 16) % 100   # a stable number in 0..99
    if bucket < 100 - val_pct - test_pct:
        return "train"
    if bucket < 100 - test_pct:
        return "val"
    return "test"
