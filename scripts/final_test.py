"""
CORE 2.6 - FINAL evaluation on the frozen TEST split. Run ONCE.

Pre-registered (committed before this script was ever run on the real test set):
  1. Thresholds come from VALIDATION only (pre-registered rule, recall_1 >= 0.90):
       TF-IDF   -> re-derived here, must equal the recorded value (eval/results/tfidf_logreg.json)
       Bi-LSTM  -> recorded value (eval/results/bilstm_glove.json)
  2. Reproducibility gates BEFORE any test score is computed:
       - re-trained TF-IDF must give the recorded validation threshold
       - reloaded Bi-LSTM must give the saved validation scores
     If a gate fails, the script stops and the test set stays unseen.
  3. Primary metric: macro-F1 at the validation-chosen threshold. Also: recall/precision
     of the distress class, ROC-AUC, PR-AUC, ECE, 95 % bootstrap CIs.
  4. Model comparison: paired bootstrap of the difference (Bi-LSTM - TF-IDF) on the SAME
     test posts. "Better" is claimed only if the 95 % interval excludes 0.
  5. Run-once guard: refuses to run if final_test.json already exists in --out.

Outputs:
  <out>/final_test.json, <out>/FINAL_TEST.md                (metrics only, commit them)
  <private-out>/test_predictions_PRIVATE.csv                 (real posts + scores for the
                                                              error analysis; NEVER commit)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.data import load_clean_dataset  # noqa: E402
from mindcare.metrics import choose_threshold, evaluate, paired_bootstrap_diff  # noqa: E402
from mindcare.tfidf import fit_tfidf_logreg  # noqa: E402



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--bilstm-dir", required=True, help="folder with bilstm_glove.keras, vocab, val scores")
    p.add_argument("--out", default="eval/results")
    p.add_argument("--private-out", default="eval/audit")
    p.add_argument("--results-dir", default="eval/results", help="where the recorded validation results are")
    args = p.parse_args()
    out, priv = Path(args.out), Path(args.private_out)
    if (out / "final_test.json").exists():
        sys.exit("final_test.json already exists: the test set has been scored. Not running again.")
    out.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)

    RESULTS = Path(args.results_dir)
    recorded_tfidf = json.loads((RESULTS / "tfidf_logreg.json").read_text(encoding="utf-8"))
    recorded_bilstm = json.loads((RESULTS / "bilstm_glove.json").read_text(encoding="utf-8"))

    df = load_clean_dataset(args.csv)
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    # ---- Gate 1: TF-IDF reproduces its validation threshold ----------------------------
    print("Re-training TF-IDF + LR on TRAIN (deterministic)...")
    tfidf = fit_tfidf_logreg(tr.text, tr.y)
    thr_tfidf = choose_threshold(va.y.to_numpy(), tfidf(va.text))
    if abs(thr_tfidf - recorded_tfidf["threshold"]) > 1e-6:
        sys.exit(f"GATE 1 FAILED: TF-IDF threshold {thr_tfidf} != recorded {recorded_tfidf['threshold']}. Test not scored.")
    print(f"Gate 1 OK: TF-IDF validation threshold reproduced ({thr_tfidf:.4f}).")

    # ---- Gate 2: Bi-LSTM reloads and reproduces its validation scores --------------------
    from mindcare.bilstm import load_bilstm   # imported here: TensorFlow is only needed for this part
    bilstm = load_bilstm(args.bilstm_dir)
    saved_val = np.load(Path(args.bilstm_dir) / "bilstm_val_scores.npy")
    diff = float(np.abs(bilstm(va.text) - saved_val).max())
    if diff > 1e-4:
        sys.exit(f"GATE 2 FAILED: reloaded Bi-LSTM differs from saved validation scores (max diff {diff}). Test not scored.")
    thr_bilstm = recorded_bilstm["threshold"]
    print(f"Gate 2 OK: Bi-LSTM validation scores reproduced (max diff {diff:.2e}); threshold {thr_bilstm:.4f}.")

    # ---- The ONE look at the test set ------------------------------------------------------
    print("Scoring the TEST split (once) and bootstrapping...")
    y = te.y.to_numpy()
    s_tfidf, s_bilstm = tfidf(te.text), bilstm(te.text)
    res = {
        "n_test": int(len(te)),
        "class1_share": float(y.mean()),
        "thresholds_from_validation": {"tfidf": thr_tfidf, "bilstm": thr_bilstm},
        "tfidf": evaluate(y, s_tfidf, thr_tfidf),
        "bilstm": evaluate(y, s_bilstm, thr_bilstm),
        "paired_diff_bilstm_minus_tfidf": paired_bootstrap_diff(y, s_tfidf, thr_tfidf, s_bilstm, thr_bilstm),
    }
    (out / "final_test.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    # Private predictions for the error analysis (real posts: never committed)
    priv_df = te[["text", "y"]].copy()
    priv_df["score_tfidf"], priv_df["score_bilstm"] = s_tfidf.round(4), s_bilstm.round(4)
    priv_df["pred_tfidf"] = (s_tfidf >= thr_tfidf).astype(int)
    priv_df["pred_bilstm"] = (s_bilstm >= thr_bilstm).astype(int)
    priv_df.to_csv(priv / "test_predictions_PRIVATE.csv", index=False)

    def cell(m, k):
        lo, hi = m["ci95"][k]
        return f"{m[k]:.3f} [{lo:.3f}, {hi:.3f}]"

    md = ["# Final test evaluation (frozen test split, scored once)", "",
          f"n = {res['n_test']:,} posts, class-1 share {res['class1_share']:.1%}. "
          "Thresholds chosen on validation. 95 % bootstrap CIs (1000 resamples).", "",
          "| metric | TF-IDF + LR | Bi-LSTM + GloVe | difference (Bi-LSTM - TF-IDF) [paired 95 % CI] |",
          "|---|---|---|---|",
          f"| threshold (from validation) | {thr_tfidf:.4f} | {thr_bilstm:.4f} | |"]
    for k in ("macro_f1", "accuracy", "precision_1", "recall_1", "roc_auc"):
        d = res["paired_diff_bilstm_minus_tfidf"][k]
        sig = "significant" if d["significant"] else "not significant"
        md.append(f"| {k} | {cell(res['tfidf'], k)} | {cell(res['bilstm'], k)} | "
                  f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] ({sig}) |")
    md += [f"| pr_auc | {res['tfidf']['pr_auc']:.3f} | {res['bilstm']['pr_auc']:.3f} | |",
           f"| ECE (10 bins) | {res['tfidf']['calibration']['ece']:.4f} | {res['bilstm']['calibration']['ece']:.4f} | |",
           f"| confusion (tp, fp, fn, tn) | {tuple(res['tfidf']['confusion'].values())} | "
           f"{tuple(res['bilstm']['confusion'].values())} | |"]
    text = "\n".join(md) + "\n"
    (out / "FINAL_TEST.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
