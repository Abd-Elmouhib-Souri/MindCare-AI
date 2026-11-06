"""
TF-IDF + logistic regression baseline: ONE definition, used by the validation
script and by the final test script, so both score exactly the same model.
Deterministic: same training texts -> same model -> same scores.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def fit_tfidf_logreg(train_texts, train_y):
    """Fit on TRAIN only. Returns predict(texts) -> probability of class 1."""
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=200_000,
                          sublinear_tf=True, dtype=np.float32)   # NO stopword removal: keeps negations
    clf = LogisticRegression(max_iter=2000)
    clf.fit(vec.fit_transform(train_texts), train_y)

    def predict(texts) -> np.ndarray:
        return clf.predict_proba(vec.transform(texts))[:, 1]

    return predict
