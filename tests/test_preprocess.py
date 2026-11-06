"""Tests for mindcare/preprocess.py: the negations must survive cleaning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.preprocess import UNKNOWN_TOKEN, clean_for_model, model_input  # noqa: E402


def test_model_input_is_never_empty():
    # a URL-only post survives the audit cleaning ("https x com") but is empty after model cleaning;
    # an empty input would give a fully masked sequence, rejected by the GPU LSTM (Kaggle crash, 6 Nov 2026)
    assert clean_for_model("https://example.com/abc") == ""
    assert model_input("https://example.com/abc") == UNKNOWN_TOKEN
    assert model_input("&gt; <br>") == UNKNOWN_TOKEN
    assert model_input("I feel alone") == "i feel alone"


def test_negation_words_are_kept():
    # the old pipeline (NLTK stopwords) turned this into "happy"
    assert clean_for_model("I am NOT happy, never") == "i am not happy never"


def test_contractions_split_like_glove():
    assert clean_for_model("I don't care") == "i do n't care"
    assert clean_for_model("I can't") == "i ca n't"
    assert clean_for_model("I'm tired") == "i 'm tired"


def test_curly_apostrophe_is_handled():
    assert clean_for_model("I don’t know") == "i do n't know"


def test_html_entities_and_urls_removed():
    assert clean_for_model("quote &gt; text https://x.com/a") == "quote text"
    assert clean_for_model("you &amp; me") == "you me"


def test_empty_and_non_string_input():
    assert clean_for_model("") == ""
    assert clean_for_model(None) == ""            # must not become the word "none"
    assert clean_for_model(float("nan")) == ""    # pandas missing value
