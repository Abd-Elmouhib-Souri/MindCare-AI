"""
Text cleaning for the neural models (Bi-LSTM + GloVe).

Two rules that matter, both covered by tests/test_preprocess.py:

1. NO stopword removal. "not", "no", "never", "n't" are in NLTK's stopword list;
   removing them turns "I am not happy" into "happy". The audit showed negations
   appear in 83 % of distress posts vs 40 % of the others.

2. Tokenize like GloVe. GloVe 6B was built with the Stanford tokenizer, which
   splits "don't" -> "do n't" and "i'm" -> "i 'm". If we keep "don't" as one
   token, it is NOT in GloVe, gets a zero vector, and (with frozen embeddings)
   the negation is lost again. So we split contractions the same way.
"""

import html
import re

# "don't" -> "do n't", "can't" -> "ca n't", "won't" -> "wo n't" (Stanford/GloVe style)
_NT = re.compile(r"(\w+)n't\b")
# "i'm" -> "i 'm", "you're" -> "you 're", "we've" -> "we 've", "it's" -> "it 's" ...
_CLITIC = re.compile(r"(\w)'(m|s|re|ve|ll|d)\b")


def clean_for_model(text: str) -> str:
    """Lowercase, decode HTML entities, remove URLs/tags, split contractions like GloVe."""
    if not isinstance(text, str):                   # None / NaN -> empty, never the word "none"/"nan"
        return ""
    text = html.unescape(text)               # "&gt;" -> ">", "&amp;" -> "&"  (audit found 'gt', 'amp')
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs carry no distress signal
    text = re.sub(r"<[^>]*>", " ", text)                 # leftover HTML tags
    text = text.replace("’", "'")                        # curly apostrophe -> straight
    text = _NT.sub(r"\1 n't", text)                      # negation stays a separate token
    text = _CLITIC.sub(r"\1 '\2", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)            # keep letters, digits, apostrophes
    return re.sub(r"\s+", " ", text).strip()
