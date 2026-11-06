"""Safe loading of the face model: the saved Lambda layer is replaced, never executed."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

keras = pytest.importorskip("keras")
tf = pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.face import load_face_model  # noqa: E402


def _save(tmp_path, fn, name):
    inp = keras.Input((96, 96, 1), name="face_input")
    x = keras.layers.Lambda(fn, name=name)(inp)
    x = keras.layers.Conv2D(2, 3, name="conv")(x)
    out = keras.layers.Dense(7, activation="softmax", name="probs")(keras.layers.GlobalAveragePooling2D()(x))
    path = str(tmp_path / f"{name}.keras")
    keras.Model(inp, out).save(path)
    return path


def test_grayscale_lambda_is_replaced_and_weights_are_identical(tmp_path):
    model = load_face_model(_save(tmp_path, lambda img: tf.image.grayscale_to_rgb(img), "repeat_channels"))
    assert model.weights_check["identical"] is True
    assert model.predict(np.zeros((1, 96, 96, 1), "float32"), verbose=0).shape == (1, 7)


def test_unexpected_lambda_is_refused_without_running_it(tmp_path):
    path = _save(tmp_path, lambda img: os.getcwd() and img, "suspicious")
    with pytest.raises(ValueError, match="Refusing to load"):
        load_face_model(path)
