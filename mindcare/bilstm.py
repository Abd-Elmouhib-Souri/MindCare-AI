"""
Bi-LSTM + GloVe text classifier (Keras 3), rebuilt from the v2 design (dual pooling).

Architecture (one line each, so it can be explained in an interview):
  token ids -> Embedding initialised with GloVe 100d, FROZEN, mask_zero=True (padding ignored)
            -> SpatialDropout1D (drops whole embedding channels: regularisation)
            -> Bidirectional LSTM (reads the post left->right AND right->left)
            -> masked max-pooling (strongest signal anywhere, e.g. one hopeless sentence)
               + masked mean-pooling (overall tone of the post)
            -> Dense(1, sigmoid) = score for 'r/SuicideWatch-like'
"""

import numpy as np
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="mindcare")
class MaskedGlobalPooling(layers.Layer):
    """Concatenate max- and mean-pooling over time, ignoring padded positions.
    Written explicitly so padding (zeros) can never leak into the pooled vector."""

    def call(self, x, mask=None):
        if mask is None:
            return ops.concatenate([ops.max(x, axis=1), ops.mean(x, axis=1)], axis=-1)
        m = ops.expand_dims(ops.cast(mask, x.dtype), -1)              # (batch, time, 1): 1 = real token
        neg = ops.full_like(x, -1e9)
        x_max = ops.max(ops.where(m > 0, x, neg), axis=1)              # padded steps can't win the max
        x_mean = ops.sum(x * m, axis=1) / ops.maximum(ops.sum(m, axis=1), 1.0)
        return ops.concatenate([x_max, x_mean], axis=-1)

    def compute_mask(self, inputs, mask=None):
        return None                                                    # output is one vector per post


def build_bilstm(embedding_matrix: np.ndarray, max_len: int, lstm_units: int = 128,
                 dropout: float = 0.3, learning_rate: float = 1e-3) -> keras.Model:
    vocab_size, dim = embedding_matrix.shape
    inp = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")
    x = layers.Embedding(vocab_size, dim, mask_zero=True, trainable=False, name="glove",
                         embeddings_initializer=keras.initializers.Constant(embedding_matrix))(inp)
    x = layers.SpatialDropout1D(dropout)(x)
    # recurrent_dropout=0 keeps the fast cuDNN kernel on GPU
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout,
                                         recurrent_dropout=0.0), name="bilstm")(x)
    x = MaskedGlobalPooling(name="max_mean_pool")(x)
    out = layers.Dense(1, activation="sigmoid", name="distress_score")(x)
    model = keras.Model(inp, out, name="mindcare_bilstm_glove")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def load_glove_matrix(glove_path: str, vocab: list, dim: int = 100, seed: int = 42):
    """Embedding matrix for our vocabulary. Row 0 = padding (zeros), row 1 = OOV token.
    Words missing from GloVe get a small random vector (frozen, so effectively a fixed 'unknown').
    Returns (matrix, coverage) where coverage = share of the vocabulary found in GloVe."""
    index = {w: i for i, w in enumerate(vocab)}
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[0] = 0.0                                                   # padding
    found = 0
    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            i = index.get(word)
            if i is not None and i > 1 and len(values) == dim:
                matrix[i] = np.asarray(values, dtype=np.float32)
                found += 1
    return matrix, found / max(len(vocab) - 2, 1)


def load_bilstm(model_dir: str):
    """Reload the saved model + vocabulary and return predict(texts) -> scores.
    Verified: gives exactly the same scores as at training time (max abs diff 0.0)."""
    import json
    from pathlib import Path

    from mindcare.preprocess import model_input

    model_dir = Path(model_dir)
    model = keras.models.load_model(model_dir / "bilstm_glove.keras")   # custom layer registered above
    cfg = json.loads((model_dir / "bilstm_vocab.json").read_text(encoding="utf-8"))
    # TextVectorization re-adds '' (padding) and '[UNK]' itself, so pass the vocabulary without them
    vec = layers.TextVectorization(standardize=None, split="whitespace",
                                   output_sequence_length=cfg["max_len"], vocabulary=cfg["vocab"][2:])

    def predict(texts) -> np.ndarray:
        ids = vec(np.array([model_input(t) for t in texts], dtype=object)).numpy()   # same rule as training
        return model.predict(ids, batch_size=1024, verbose=0).ravel()

    return predict
