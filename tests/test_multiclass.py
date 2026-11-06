"""Tests for mindcare/multiclass.py and the TESS file-name parsing in mindcare/voice.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.multiclass import multiclass_bootstrap, multiclass_metrics, top_label_ece  # noqa: E402
from mindcare.voice import parse_emotion, parse_speaker_word  # noqa: E402


def test_perfect_predictions():
    y = [0, 1, 2, 3, 4, 5, 6]
    m = multiclass_metrics(y, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0


def test_macro_f1_punishes_a_missed_rare_class():
    # 9 'happy' (3) + 1 'disgust' (1); model always says happy -> accuracy 0.9,
    # but macro-F1 averages over the 7 emotions, and 'disgust' gets F1 = 0
    y = [3] * 9 + [1]
    m = multiclass_metrics(y, [3] * 10)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["per_class"]["disgust"]["f1"] == 0.0
    assert m["macro_f1"] < 0.2      # (F1 happy ~0.947 + 0 for the 6 other emotions) / 7


def test_confusion_rows_are_true_labels():
    m = multiclass_metrics([0, 0, 1], [0, 1, 1])
    assert m["confusion"][0][:2] == [1, 1] and m["confusion"][1][:2] == [0, 1]


def test_bootstrap_contains_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 400)
    pred = np.where(rng.random(400) < 0.6, y, rng.integers(0, 7, 400))
    point = multiclass_metrics(y, pred)["accuracy"]
    lo, hi = multiclass_bootstrap(y, pred, n_boot=200)["accuracy"]
    assert lo <= point <= hi


def test_top_label_ece_overconfident():
    # always 90 % confident in class 0, but right only half the time -> ECE 0.4
    probs = np.tile([0.9, 0.1, 0, 0, 0, 0, 0], (100, 1))
    y = [0, 1] * 50
    assert top_label_ece(y, probs) == pytest.approx(0.4)


def test_tess_parsing():
    assert parse_emotion("OAF_Pleasant_surprise") == 6
    assert parse_emotion("YAF_pleasant_surprised") == 6
    assert parse_emotion("OAF_Fear") == 2
    assert parse_emotion("readme") is None
    assert parse_speaker_word("OAF_back_angry.wav") == ("OAF", "back")
    assert parse_speaker_word("YAF_youth_ps.wav") == ("YAF", "youth")


def test_tess_speaker_comes_from_folder_not_file_name(tmp_path):
    # Real TESS has a file named 'OA_...' inside an OAF folder; it must stay speaker OAF
    from mindcare.voice import list_tess, naming_anomalies
    for folder, name in (("OAF_neutral", "OA_bite_neutral.wav"), ("OAF_neutral", "OAF_back_neutral.wav"),
                         ("YAF_angry", "YAF_back_angry.wav")):
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(b"")
    items = list_tess(str(tmp_path))
    assert sorted({spk for _, _, spk, _ in items}) == ["OAF", "YAF"]
    assert naming_anomalies(items) == ["OA_bite_neutral.wav"]
