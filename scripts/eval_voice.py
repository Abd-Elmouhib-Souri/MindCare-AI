"""
Voice emotion model - honest evaluation on TESS.

TESS = 2 speakers (OAF, YAF) x 7 emotions x 200 words. The original code used a RANDOM
split: the same 2 voices and the same words are in train and test, so the score mostly
measures "recognise this voice saying this word", not emotion.

Protocol (fixed before running, no hyperparameter tuning):
  A. Original protocol, reconstructed: shuffle(random_state=42) + 85/15 stratified split,
     score the EXISTING trained model (speaker-dependent; for comparison only).
  B. Leave-one-speaker-out: re-train the SAME architecture with the SAME two-stage recipe
     on one speaker, test on the other; both directions. This is the honest number.
Report accuracy and macro-F1 with 95 % bootstrap CIs, per-class results, confusion matrices.
Limitation written in the report: 2 speakers only (both female actors), so even B says
little about real-world speakers, microphones or spontaneous speech.

Usage (repo root, .venv active):
  python scripts/eval_voice.py --tess-dir "..\\mindcare_data\\dataset\\TESS Toronto emotional speech set data" --model <best_audio_model_stage2.keras>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.multiclass import (EMOTIONS, markdown_table, multiclass_bootstrap,  # noqa: E402
                                 multiclass_metrics, per_class_table)
from mindcare.voice import extract_3d_mfcc, list_tess, train_two_stage  # noqa: E402


def scored(y, pred) -> dict:
    r = multiclass_metrics(y, pred)
    r["ci95"] = multiclass_bootstrap(y, pred)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tess-dir", required=True)
    p.add_argument("--model", required=True, help="existing best_audio_model_stage2.keras")
    p.add_argument("--out", default="eval/results")
    p.add_argument("--cache", default="../mindcare_runs/tess_features.npz", help="feature cache (outside git)")
    p.add_argument("--weights", default="imagenet", help="'none' only for a quick smoke test")
    args = p.parse_args()
    import keras
    keras.utils.set_random_seed(42)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = list_tess(args.tess_dir)
    speakers = np.array([s for _, _, s, _ in items])
    words = np.array([w for _, _, _, w in items])
    y = np.array([e for _, e, _, _ in items])
    print(f"{len(items):,} files | speakers " + str({str(k): int(v) for k, v in zip(*np.unique(speakers, return_counts=True))}))

    cache = Path(args.cache)
    if cache.exists() and np.load(cache)["X"].shape[0] == len(items):
        X = np.load(cache)["X"]
        print(f"Features loaded from cache {cache}")
    else:
        print("Extracting MFCC features (2-4 min)...")
        X = np.stack([extract_3d_mfcc(path) for path, _, _, _ in items])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X)

    # ---- A. Original random split, existing model -------------------------------------------
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle
    idx = shuffle(np.arange(len(y)), random_state=42)
    tr_idx, te_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y[idx])
    overlap = {
        "test_files_whose_speaker_is_in_train": float(np.isin(speakers[te_idx], speakers[tr_idx]).mean()),
        "test_files_whose_word_is_in_train": float(np.isin(words[te_idx], words[tr_idx]).mean()),
    }
    existing = keras.models.load_model(args.model)
    pred_a = existing.predict(X[te_idx], batch_size=64, verbose=0).argmax(axis=1)
    res = {"n_files": len(items), "random_split_overlap": overlap,
           "A_random_split_existing_model": scored(y[te_idx], pred_a)}
    print(f"A (random split, existing model): accuracy {res['A_random_split_existing_model']['accuracy']:.3f}")

    # ---- B. Leave-one-speaker-out, re-trained with the original recipe ------------------------
    folds = {}
    for held_out in sorted(set(speakers)):
        t0 = time.time()
        train_mask = speakers != held_out
        order = shuffle(np.where(train_mask)[0], random_state=42)     # the original shuffled before fit
        weights = None if args.weights.lower() == "none" else args.weights   # Keras wants None, not 'none'
        model, info = train_two_stage(X[order], y[order], weights=weights, verbose=2)
        pred = model.predict(X[~train_mask], batch_size=64, verbose=0).argmax(axis=1)
        folds[f"train_{'+'.join(sorted(set(speakers[train_mask])))}_test_{held_out}"] = {
            **scored(y[~train_mask], pred), **info, "minutes": round((time.time() - t0) / 60, 1)}
        print(f"B fold, test speaker {held_out}: accuracy {folds[list(folds)[-1]]['accuracy']:.3f}")
    res["B_leave_one_speaker_out"] = folds
    res["B_mean_accuracy"] = float(np.mean([f["accuracy"] for f in folds.values()]))
    res["B_mean_macro_f1"] = float(np.mean([f["macro_f1"] for f in folds.values()]))
    (out / "voice_tess.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    md = ["# Voice emotion model (MobileNetV2 on MFCC) - TESS", "",
          "Generated by `scripts/eval_voice.py`. 7 emotions; 95 % bootstrap CIs (1000 resamples).", "",
          f"Random split overlap: {overlap['test_files_whose_speaker_is_in_train']:.0%} of test files share their "
          f"speaker with train, {overlap['test_files_whose_word_is_in_train']:.0%} share their word.", ""]
    md += markdown_table({"A. random split (original protocol, existing model)": res["A_random_split_existing_model"],
                          **{f"B. leave-one-speaker-out: {k}": v for k, v in folds.items()}})
    md += ["", f"B mean over the 2 folds: accuracy {res['B_mean_accuracy']:.3f}, macro-F1 {res['B_mean_macro_f1']:.3f}",
           "", "Limitation: TESS has 2 speakers (both female actors, acted speech, studio audio). "
           "Leave-one-speaker-out tests generalisation to ONE new voice per fold; it says little about "
           "real users, microphones or spontaneous speech.", ""]
    for k, v in folds.items():
        md += [f"## Per class - {k}", ""] + per_class_table(v) + [""]
    text = "\n".join(md) + "\n"
    (out / "VOICE_TESS.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
