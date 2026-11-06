"""
Try the text models on your own sentences (interactive, local, nothing is stored).

Research prototype: NOT a diagnostic or crisis tool. The score means "how much this
text looks like a r/SuicideWatch post rather than a r/teenagers post", nothing more.
If you or someone you know is struggling: https://findahelpline.com

Usage (repo root, .venv active):
  python scripts/try_text_model.py --bilstm-dir <folder with bilstm_glove.keras> [--csv <Suicide_Detection.csv>]
  --csv also trains the TF-IDF baseline (about 1-2 min) to compare both models.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DISCLAIMER = ("Research prototype - not a diagnostic or crisis tool. Scores reflect similarity to "
              "r/SuicideWatch vs r/teenagers posts.\nIf you are struggling, please reach out: https://findahelpline.com")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bilstm-dir", required=True)
    p.add_argument("--csv", default=None, help="optional: also train the TF-IDF baseline to compare")
    p.add_argument("--results-dir", default="eval/results")
    args = p.parse_args()

    from mindcare.bilstm import load_bilstm
    bilstm = load_bilstm(args.bilstm_dir)
    thr_b = json.loads((Path(args.results_dir) / "bilstm_glove.json").read_text(encoding="utf-8"))["threshold"]

    tfidf, thr_t = None, None
    if args.csv:
        from mindcare.data import load_clean_dataset
        from mindcare.tfidf import fit_tfidf_logreg
        print("Training the TF-IDF baseline on the train split (1-2 min)...")
        tr = load_clean_dataset(args.csv).query("split == 'train'")
        tfidf = fit_tfidf_logreg(tr.text, tr.y)
        thr_t = json.loads((Path(args.results_dir) / "tfidf_logreg.json").read_text(encoding="utf-8"))["threshold"]

    print("\n" + DISCLAIMER + "\nType a sentence (empty line to quit).\n")
    while True:
        text = input("> ").strip()
        if not text:
            break
        s = float(bilstm([text])[0])
        line = f"  Bi-LSTM score {s:.3f} (threshold {thr_b:.3f}) -> {'flagged' if s >= thr_b else 'not flagged'}"
        if tfidf is not None:
            t = float(tfidf([text])[0])
            line += f" | TF-IDF {t:.3f} (threshold {thr_t:.3f}) -> {'flagged' if t >= thr_t else 'not flagged'}"
        print(line)


if __name__ == "__main__":
    main()
