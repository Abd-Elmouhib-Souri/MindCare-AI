"""
Tests for mindcare/metrics.py and mindcare/splits.py.
Each expected value is computed by hand in the comment, so a wrong formula fails.
Run from the repo root:  python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.metrics import (  # noqa: E402
    bootstrap_ci, choose_threshold, expected_calibration_error, threshold_metrics, wilson_interval,
)
from mindcare.splits import assign_split  # noqa: E402


def test_known_confusion_matrix():
    # y=1,1,0,0 ; scores .9 .4 .6 .1 ; threshold .5 -> preds 1,0,1,0 -> tp=1 fn=1 fp=1 tn=1
    m = threshold_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["recall_1"] == pytest.approx(0.5) and m["precision_1"] == pytest.approx(0.5)


def test_macro_f1_differs_from_accuracy_on_imbalanced_data():
    # 9 negatives, 1 positive, model always says 0 -> accuracy 0.9 but class-1 F1 = 0 -> macro-F1 = (0 + 0.947)/2
    y = [0] * 9 + [1]
    m = threshold_metrics(y, [0.1] * 10, 0.5)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["f1_1"] == 0.0
    assert m["macro_f1"] == pytest.approx((0 + 2 * 0.9 * 1.0 / 1.9) / 2)


def test_threshold_rule_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    score = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 2000), 0, 1)   # noisy but informative scores
    thr = choose_threshold(y, score, target_recall=0.9)
    assert threshold_metrics(y, score, thr)["recall_1"] >= 0.9


def test_ece_is_zero_when_perfectly_calibrated():
    # score .25 on 4 items with 1 positive (25 %), score .75 on 4 items with 3 positives (75 %)
    y = [1, 0, 0, 0, 1, 1, 1, 0]
    s = [0.25] * 4 + [0.75] * 4
    assert expected_calibration_error(y, s)["ece"] == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    # always says 0.95 but only half are positive -> gap 0.45
    assert expected_calibration_error([1, 0] * 50, [0.95] * 100)["ece"] == pytest.approx(0.45)


def test_wilson_interval_known_value():
    # 8/10, z = 1.96 -> about [0.490, 0.943] (hand computation)
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)


def test_wilson_interval_stays_in_0_1_at_the_edges():
    # 0/20 and 20/20 are common on small challenge groups; bounds must stay inside [0, 1]
    lo, _ = wilson_interval(0, 20)
    _, hi = wilson_interval(20, 20)
    assert lo == 0.0 and hi == 1.0


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 500), 0, 1)
    point = threshold_metrics(y, s, 0.5)["macro_f1"]
    lo, hi = bootstrap_ci(y, s, 0.5, n_boot=200)["macro_f1"]
    assert lo <= point <= hi


def test_split_is_duplicate_safe_and_deterministic():
    # same text up to case/punctuation -> same split, every time
    assert assign_split("I feel so ALONE!!") == assign_split("i feel so alone")
    assert assign_split("hello world") == assign_split("hello world")


def test_split_proportions():
    splits = [assign_split(f"post number {i}") for i in range(20000)]
    share = {s: splits.count(s) / len(splits) for s in ("train", "val", "test")}
    assert share["train"] == pytest.approx(0.70, abs=0.02)
    assert share["val"] == pytest.approx(0.15, abs=0.02)
    assert share["test"] == pytest.approx(0.15, abs=0.02)


def test_paired_bootstrap_identical_models_gives_zero_difference():
    from mindcare.metrics import paired_bootstrap_diff
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    s = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 300), 0, 1)
    res = paired_bootstrap_diff(y, s, 0.5, s, 0.5, n_boot=100)
    assert res["macro_f1"]["diff"] == 0.0 and res["macro_f1"]["ci95"] == [0.0, 0.0]
    assert res["macro_f1"]["significant"] is False


def test_paired_bootstrap_detects_a_clearly_better_model():
    from mindcare.metrics import paired_bootstrap_diff
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    weak = rng.uniform(0, 1, 500)                       # random scores
    strong = np.clip(y * 0.6 + rng.normal(0.2, 0.1, 500), 0, 1)
    res = paired_bootstrap_diff(y, weak, 0.5, strong, 0.5, n_boot=200)
    assert res["macro_f1"]["diff"] > 0.3 and res["macro_f1"]["significant"] is True
    assert res["roc_auc"]["ci95"][0] > 0
