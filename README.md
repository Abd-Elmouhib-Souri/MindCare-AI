# MindCare AI: auditing distress-language and emotion models

> ⚠️ **Research prototype. Not a diagnostic tool, not a crisis tool, not for use on individuals.**
> If you or someone you know is struggling, please contact local emergency services or find a helpline at
> [findahelpline.com](https://findahelpline.com).

MindCare AI started as a "multimodal mental-health detector" (text + voice + face). This version is the
**honest rebuild**: every claim is either measured on a held-out set with confidence intervals, or removed.
The main result is not a high score. It is **what the scores actually measure**.

## Results at a glance

| Module | Task (what the labels really are) | Headline result (held-out, 95 % CI) | What the audit found |
|---|---|---|---|
| **Text** | r/SuicideWatch vs r/teenagers posts (subreddit = weak label) | Bi-LSTM + GloVe **macro-F1 0.938 [0.936, 0.941]**; TF-IDF + LR 0.936 [0.933, 0.938]; difference **not significant** | Post length alone reaches 0.75 accuracy; distress written in a teen context is missed (**0-1/20** on a challenge set) |
| **Face** | 7 acted emotions, FER-2013 | accuracy **0.591** [0.580, 0.602], macro-F1 0.566 | **568 test images (7.9 %) are pixel-identical to training images**; without them: 0.575 / 0.532 |
| **Voice** | 7 acted emotions, TESS (2 speakers) | **0.453** mean accuracy on an unseen speaker (chance 0.143) | The original random split gave **0.981**: it measured the speaker and the word, not the emotion |

Emotion recognition (face, voice) is **not** mental-health detection, and the three modules are **not fused**:
a fusion can't be evaluated without data where the same person is filmed, recorded and writing at the same time.

---

## 1. Text: distress-language classification

### Data and task definition
- **Dataset:** [Suicide and Depression Detection](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
  (N. Komati, CC BY-SA 4.0; originally published in *Suicide Ideation Detection in Social Media Forums*,
  [IEEE](https://ieeexplore.ieee.org/document/9591887)). 232,074 Reddit posts, 50 / 50.
  File SHA-256: `1ec16b065a2fae36488de16e8abfc326586a576a25901e09467223e545ddd295`.
- **What the labels are:** class 1 = posts from r/SuicideWatch; class 0 = posts from **r/teenagers**
  (stated in the data card, and found independently by the audit below). The data card does not say
  whether r/depression posts were merged into class 1. These are **subreddit-level weak labels**, not
  per-post annotations, so the honest task is *"does this post look like r/SuicideWatch rather than r/teenagers?"*

### Audit (before any training): [`eval/audit/TEXT_AUDIT.md`](eval/audit/TEXT_AUDIT.md), [`SHORTCUT_PROBE.md`](eval/audit/SHORTCUT_PROBE.md)
- 0 exact and 212 near duplicates; 50 posts become empty after normalisation (emoji, non-Latin scripts).
  After cleaning: 231,861 posts.
- **Length shortcut:** median 127 words (class 1) vs 31 (class 0). A logistic regression on the
  **length alone** reaches 0.748 accuracy (ROC-AUC 0.807); style features without any topic word reach 0.774.
- **Topic shortcut:** the words most tied to class 0 are `teenagers`, `school`, `crush`, `minecraft`, `discord`.
- **Negations** appear in 83 % of class-1 posts vs 40 % of class-0 posts, which is why stopword removal
  (used in the first version) destroyed a key signal.

### Protocol
- **Frozen split** by hash of the normalised text (70 / 15 / 15): duplicates can never cross splits;
  0 texts found in two splits. No author id exists, so the split cannot be grouped by author.
- **Pre-registered threshold rule:** best precision among thresholds with distress recall ≥ 0.90 **on validation**.
- **Challenge set** (80 synthetic sentences, 4 groups), frozen at commit `07da3bb` **before** any model was
  evaluated on it: implicit distress without keywords, keywords without distress, distress in teen topics,
  negations. 12 seed items + 68 drafted with AI assistance; every label reviewed by the author; no real posts.
- **Test split scored once**, with rules committed beforehand (`31d5385`) and two reproducibility gates
  that stop the run before the test if they fail (see *Reproducibility notes*).

### Models
- **TF-IDF (1-2 grams) + logistic regression**, no stopword removal.
- **Bi-LSTM + GloVe 100d** (frozen, 93.1 % vocabulary coverage): GloVe-style contraction splitting
  (`don't` → `do n't`) so negations exist in the embedding vocabulary; masked padding; max-and-mean pooling;
  vocabulary fitted on train only; early stopping on validation; no SMOTE, no hand-tuned class weights.
  Trained on a Kaggle T4 GPU (TensorFlow 2.21, Keras 3.13.2), 10 epochs (best: 9).

### Results on the test split (n = 34,635, scored once)

| | TF-IDF + LR | Bi-LSTM + GloVe | Paired difference (Bi-LSTM − TF-IDF) |
|---|---|---|---|
| threshold (from validation) | 0.615 | 0.869 | |
| **macro-F1** | 0.936 [0.933, 0.938] | **0.938** [0.936, 0.941] | +0.0024 [−0.0000, +0.0049]: **not significant** |
| distress recall | 0.904 | 0.906 | not significant |
| distress precision | 0.966 | 0.969 | not significant |
| ROC-AUC | 0.985 | 0.987 | +0.0022 [+0.0015, +0.0030]: significant but negligible |
| ECE (calibration) | 0.034 | 0.032 | |

Full tables: [`eval/results/FINAL_TEST.md`](eval/results/FINAL_TEST.md). The recall target set on
validation (0.90) transferred to the test split (0.904 / 0.906).

### Challenge set (accuracy per group, threshold from validation)

| Group (label) | TF-IDF + LR | Bi-LSTM | What it shows |
|---|---|---|---|
| a. implicit distress, no keywords (1) | 7/20 | 3/20 | most distress without explicit words is missed |
| b. keyword without distress (0) | 18/20 | 17/20 | "suicide" in a neutral sentence is *not* blindly flagged |
| **c. distress in a teen topic (1)** | **0/20** | **1/20** | **topic shortcut: school/crush/gaming context hides distress** |
| d. negation that cancels (0) | 9/10 | 10/10 | |
| d. negation that expresses distress (1) | 6/10 | 2/10 | |
| overall | 0.500 | 0.412 | |

Groups a and c have the same label and the same length; only the topic differs. Mean scores drop from
0.628 to 0.201 (TF-IDF) and from 0.597 to 0.148 (Bi-LSTM): **the teen context alone cuts the score by ~3×.**
The models are well calibrated in-domain but their threshold does not transfer to short, out-of-domain text.

**Conclusion for text:** a Bi-LSTM brings no meaningful gain over a well-built TF-IDF baseline, and both
learn *"looks like r/SuicideWatch"*, failing on exactly the population r/teenagers represents.
The limit comes from the labels, not the architecture.

---

## 2. Face: facial expression recognition (FER-2013)
- **Model:** EfficientNetB0 (ImageNet), 96×96 grayscale, two-stage fine-tuning, balanced class weights
  (trained in the earlier prototype; validation taken from train, so the official test split was never used).
- **Safe loading:** the saved model contains a Python `Lambda` layer. It was inspected **without executing it**
  (`marshal` + `dis`: `lambda img: tf.image.grayscale_to_rgb(img)`), replaced by an equivalent layer, and only
  the weights were loaded; all 320 weight arrays were verified identical ([`mindcare/face.py`](mindcare/face.py)).

| Evaluation | n | accuracy [95 % CI] | macro-F1 [95 % CI] |
|---|---|---|---|
| official test | 7,178 | 0.591 [0.580, 0.602] | 0.566 [0.549, 0.580] |
| test without images duplicated in train | 6,610 | **0.575** [0.563, 0.586] | **0.532** [0.515, 0.548] |

- **Leakage:** 568 test images have pixel-identical copies in train (552 with the same label), and 86 are
  duplicated inside test. Duplicated images are classified correctly ~78 % of the time vs 57.5 % for the rest.
- Best classes: happy (F1 0.80), surprise (0.72). Worst: **fear** (recall 0.29, confused with sad/neutral/surprise).
- Top-label ECE 0.033. Details: [`eval/results/FACE_FER2013.md`](eval/results/FACE_FER2013.md).

## 3. Voice: speech emotion recognition (TESS)
- **Model:** MobileNetV2 (ImageNet) on a 3-channel MFCC tensor (MFCC, Δ, ΔΔ; 40×150), two-stage fine-tuning.
- **Data:** TESS, 2,800 recordings = 2 speakers (OAF, YAF) × 7 emotions × 200 words. One file is misnamed
  (`OA_bite_neutral.wav` in an OAF folder); speakers are read from folder names.

| Protocol | n | accuracy [95 % CI] | macro-F1 |
|---|---|---|---|
| random split (original protocol; 100 % of test files share speaker **and** word with train) | 420 | 0.981 [0.967, 0.993] | 0.981 |
| leave-one-speaker-out: train YAF → test OAF | 1,400 | 0.456 [0.431, 0.482] | 0.400 |
| leave-one-speaker-out: train OAF → test YAF | 1,400 | 0.449 [0.422, 0.474] | 0.389 |

Leave-one-speaker-out re-trains the **same** architecture with the **same** recipe (no tuning).
`angry` and `happy` collapse across speakers (F1 ≤ 0.09). One training run per fold: the intervals cover
test sampling, not training randomness. Details: [`eval/results/VOICE_TESS.md`](eval/results/VOICE_TESS.md).

---

## Limitations & ethics
- **Weak labels.** Text labels are subreddits, not clinical or per-post annotations. Nothing here measures
  suicide risk in a person.
- **Shortcuts.** Length, style and topic separate the text classes; the models fail on distress written in a
  teen context, which is a harmful failure mode for the very group the "negative" class comes from.
- **English only**, Reddit only (2008-2021); no author-grouped split possible.
- **Acted emotions ≠ mental state.** FER-2013 and TESS are posed/acted; TESS has 2 female speakers,
  studio audio. Results say little about real users, webcams, microphones or spontaneous speech.
- **No fusion claim.** The modules were never trained or evaluated on paired multimodal data.
- **Intended use:** research and teaching about evaluation, shortcut learning and leakage. Not for screening,
  monitoring or any decision about an individual. No user text, image or audio is stored by this code.
- **AI assistance.** Parts of the code and 68 of the 80 challenge sentences were drafted with AI assistance;
  every result was produced by running the code in this repository, and every label was reviewed by the author.

## Reproducibility notes
- Pinned environment: [`requirements.txt`](requirements.txt) (Python 3.11).
- **Gate 1** (TF-IDF must reproduce its validation threshold) **failed on Kaggle** (0.61568 vs 0.61498:
  different Python / scikit-learn / NumPy versions) and stopped the run before the test was scored.
  The rule was not relaxed: the final test was run in the pinned environment, where the threshold is
  reproduced exactly. **Gate 2**: the Bi-LSTM trained on GPU (Linux, Python 3.13) reproduces its validation
  scores on CPU (Windows, Python 3.11) with a max difference of 5.4e-7.
- Bugs found and fixed along the way are documented in the commit history (a cuDNN crash on URL-only posts,
  the TESS speaker parsing, a `%cd` before `rm -rf` in the notebooks).

## How to reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                                      # 37 tests

# Data (not in git): Suicide_Detection.csv (Kaggle link above), GloVe 6B 100d, FER-2013, TESS
python scripts/audit_data.py      --csv path/to/Suicide_Detection.csv
python scripts/shortcut_probe.py  --csv path/to/Suicide_Detection.csv
python scripts/eval_tfidf_baseline.py --csv path/to/Suicide_Detection.csv
# Bi-LSTM: notebooks/kaggle_train_bilstm.ipynb (GPU) or scripts/train_bilstm.py
python scripts/final_test.py --csv path/to/Suicide_Detection.csv --bilstm-dir path/to/models   # runs once
python scripts/eval_face.py  --fer-dir path/to/images --model path/to/best_vision_stage2.keras
python scripts/eval_voice.py --tess-dir "path/to/TESS Toronto emotional speech set data" --model path/to/best_audio_model_stage2.keras
```
Trained models are published as release assets, not in git.

## Web demo
A small React + TypeScript page served by a FastAPI API. It shows the text score on a 0-1 scale with the
validation threshold, the challenge-set failures and the limitations (all numbers are read from
`eval/results/`). The text is never stored or logged; empty or too long inputs are rejected (HTTP 422).

**Live emotion lab (experimental, optional).** Webcam face and microphone voice, shown separately and never
fused with the text score. The camera and the microphone start only after an explicit consent checkbox, each has
a Stop button, and frames / 3-second WAV windows are analysed in memory and discarded. No face (OpenCV Haar
cascade) or silence (RMS below 0.01) means no prediction. Each panel shows its measured accuracy: 57.5 % for the
face model on FER-2013 test images without train duplicates, 45.3 % for the voice model on a speaker it never
heard (chance 14.3 %). These models recognise posed expressions and acted speech; real-time use on webcams and
everyday voices was **not** evaluated, so expect lower accuracy.

```bash
# terminal 1: the API (needs the Bi-LSTM folder from the v2.0 release)
export MINDCARE_BILSTM_DIR=path/to/bilstm_models      # PowerShell: $env:MINDCARE_BILSTM_DIR = "..."
# optional, live lab: the face and voice .keras files from the v2.0 release
export MINDCARE_FACE_MODEL=path/to/best_vision_stage2.keras
export MINDCARE_VOICE_MODEL=path/to/best_audio_model_stage2.keras
uvicorn mindcare.api:app --port 8000                  # API docs: http://127.0.0.1:8000/docs
# terminal 2: the page
cd web && npm ci && npm run dev                       # open http://localhost:5173
```

## Repository layout
```
mindcare/        shared, tested code: splits, preprocessing, metrics, models, safe loaders
scripts/         one script per step (audit, probes, training, evaluation)
web/             React + TypeScript demo page (talks to mindcare/api.py)
notebooks/       Kaggle notebooks (training, final test)
eval/audit/      dataset audit and shortcut probes (raw posts are git-ignored)
eval/challenge/  frozen synthetic challenge set
eval/results/    every reported number, as JSON + Markdown
tests/           48 unit tests (mutation-checked metrics, API and live-lab guards)
```

## Roadmap
- Error analysis of 30 test errors by category (negation, sarcasm, topic, label noise).
- Out-of-domain evaluation on human-annotated posts (e.g. Dreaddit), licence permitting.
- A transformer baseline under the same split and protocol.
- Several training seeds per voice fold; a paired audio-visual dataset (e.g. CREMA-D) before any fusion claim.

## Data credits
Komati (Suicide and Depression Detection, CC BY-SA 4.0) · Pennington, Socher & Manning (2014), GloVe (PDDL) ·
Goodfellow et al. (2013), FER-2013 · Pichora-Fuller & Dupuis, Toronto Emotional Speech Set (TESS).

## Author
Abd Elmouhib Souri, ESPRIM (Tunisia).
