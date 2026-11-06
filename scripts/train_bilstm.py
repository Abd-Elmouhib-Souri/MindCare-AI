"""
CORE 2.5 (c) - Train the fixed Bi-LSTM + GloVe on TRAIN, early-stop on VALIDATION,
choose the threshold on VALIDATION (pre-registered rule), report VALIDATION + CHALLENGE.
The TEST split is not scored here (it is scored once, at the end, for all models).

What was fixed compared with the old pipeline:
  - split BEFORE fitting the vocabulary (frozen hash split, no duplicates across splits)
  - no stopword removal; contractions split like GloVe (negations survive)
  - no SMOTE (the data is 50/50; SMOTE on token ids creates meaningless sequences)
  - no class weights {0: 1.0, 1: 0.4} (they were tuned by hand)
  - threshold from a rule on validation, not 0.75 chosen by hand
  - padding is masked; vocabulary saved as JSON (no pickle)

Run on Kaggle (GPU) or locally:
  python scripts/train_bilstm.py --csv <Suicide_Detection.csv> --glove <glove.6B.100d.txt>
  quick local smoke test: add  --sample 3000 --epochs 1
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

from mindcare.bilstm import build_bilstm, load_glove_matrix  # noqa: E402
from mindcare.data import load_clean_dataset  # noqa: E402
from mindcare.preprocess import clean_for_model  # noqa: E402
from mindcare.report import CHALLENGE, build_report, write_report  # noqa: E402


def sha256(path: str) -> str:
    """Fingerprint of the CSV: proves Kaggle and the local PC used the same data."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--glove", required=True)
    p.add_argument("--out", default="eval/results")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--max-len", type=int, default=256)     # ~p75 of distress posts; keeps the BEGINNING
    p.add_argument("--vocab", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=0, help="debug only: use N train rows")
    args = p.parse_args()
    keras.utils.set_random_seed(args.seed)                 # numpy + python + TF seeds
    t0 = time.time()

    df = load_clean_dataset(args.csv)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if args.sample:
        tr = tr.sample(args.sample, random_state=args.seed)
        va = va.sample(min(args.sample // 4, len(va)), random_state=args.seed)
    ch = pd.read_csv(CHALLENGE)
    print(f"train {len(tr):,} | val {len(va):,} | challenge {len(ch)}")

    # Vocabulary fitted on TRAIN ONLY (no leakage); whitespace split because text is pre-cleaned
    vectorizer = keras.layers.TextVectorization(max_tokens=args.vocab, standardize=None,
                                                split="whitespace", output_sequence_length=args.max_len)
    clean = lambda s: [clean_for_model(t) for t in s]     # noqa: E731
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(clean(tr.text)).batch(1024))
    vocab = vectorizer.get_vocabulary()                    # index 0 = '' (pad), 1 = '[UNK]'
    to_ids = lambda s: vectorizer(np.array(clean(s), dtype=object)).numpy()  # noqa: E731
    Xtr, Xva, Xch = to_ids(tr.text), to_ids(va.text), to_ids(ch.text)

    matrix, coverage = load_glove_matrix(args.glove, vocab)
    print(f"GloVe coverage of the vocabulary: {coverage:.1%}")

    model = build_bilstm(matrix, args.max_len)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    hist = model.fit(Xtr, tr.y.to_numpy(), validation_data=(Xva, va.y.to_numpy()),
                     epochs=args.epochs, batch_size=args.batch, callbacks=[stop], verbose=2)

    val_score = model.predict(Xva, batch_size=1024, verbose=0).ravel()
    ch_score = model.predict(Xch, batch_size=1024, verbose=0).ravel()

    extra = {
        "data_sha256": sha256(args.csv),
        "config": {k: v for k, v in vars(args).items() if k not in ("csv", "glove", "out", "model_dir")},
        "glove_vocab_coverage": round(coverage, 4),
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
        "history": {k: [round(float(x), 4) for x in v] for k, v in hist.history.items()},
        "train_minutes": round((time.time() - t0) / 60, 1),
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": keras.__version__, "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]},
    }
    res = build_report("bilstm_glove100_frozen", va.y, val_score, ch, ch_score, extra)
    md = write_report(res, Path(args.out), "bilstm_glove",
                      "Bi-LSTM + GloVe (frozen) - train -> validation, challenge set")
    print(md)

    # Save the model (.keras) and the vocabulary (JSON, NOT pickle: pickle files can run code)
    mdir = Path(args.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    model.save(mdir / "bilstm_glove.keras")
    (mdir / "bilstm_vocab.json").write_text(json.dumps({"max_len": args.max_len, "vocab": vocab}), encoding="utf-8")
    np.save(mdir / "bilstm_val_scores.npy", val_score)     # for paired comparisons with other models
    print(f"Saved model + vocab in {mdir}. Total time {extra['train_minutes']} min.")


if __name__ == "__main__":
    main()
