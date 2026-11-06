"""
Face emotion model (EfficientNetB0 on FER-2013) - loading and preprocessing,
reproduced EXACTLY from the original training code (MLL/src/data/vision_pipeline.py):
  grayscale -> resize 96x96 (LANCZOS) -> float32 in [0, 255] (NO division by 255:
  Keras EfficientNet rescales internally; the original docstring saying [0, 1] was wrong).
"""

import hashlib
import os
from glob import glob

import numpy as np
from PIL import Image

from mindcare.multiclass import EMOTIONS

IMAGE_SIZE = (96, 96)


def list_split(split_dir: str) -> list:
    """[(path, label_index), ...] for one FER-2013 split folder (train/ or test/)."""
    items = []
    for label, emotion in enumerate(EMOTIONS):
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            items += [(p, label) for p in sorted(glob(os.path.join(split_dir, emotion, ext)))]
    return items


def pixel_hash(path: str) -> str:
    """Fingerprint of the decoded 48x48 pixels: two files with identical pixels get the same hash,
    even if their file names differ. Used to find the same face in train and test (leakage)."""
    return hashlib.md5(np.asarray(Image.open(path).convert("L")).tobytes()).hexdigest()


def load_image(path: str) -> np.ndarray:
    """Same preprocessing as training: (96, 96, 1) float32 in [0, 255]."""
    img = Image.open(path).convert("L").resize(IMAGE_SIZE, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)[..., None]


EXPECTED_LAMBDA_NAMES = ("tf", "image", "grayscale_to_rgb")


def _register_grayscale_layer():
    """The safe replacement for the original Lambda: same computation, no code from the file."""
    import keras
    import tensorflow as tf

    @keras.saving.register_keras_serializable(package="MLL")
    class GrayscaleToRGB(keras.layers.Layer):
        """Repeats the single gray channel into 3 identical channels (EfficientNet expects RGB)."""
        def call(self, inputs, mask=None):   # the saved Lambda call passed mask=None; accept and ignore it
            return tf.image.grayscale_to_rgb(inputs)

    return GrayscaleToRGB


def _lambda_names(layer_cfg: dict) -> tuple:
    """Names used by a serialized Python lambda, read WITHOUT executing it
    (marshal.loads builds a code object; nothing runs)."""
    import base64
    import marshal
    raw = base64.decodebytes(layer_cfg["config"]["function"]["config"]["code"].encode("ascii"))
    return tuple(marshal.loads(raw).co_names)


def load_face_model(model_path: str):
    """Load the trained face model safely.

    The saved model contains a Keras Lambda layer (a Python lambda compiled into the file).
    Loading it normally would EXECUTE that code. Instead we:
      1. read config.json from the .keras zip,
      2. check the lambda only uses tf.image.grayscale_to_rgb (inspected on 6 Nov 2026:
         `lambda img: tf.image.grayscale_to_rgb(img)`, compiled with Python 3.13),
      3. replace it by the equivalent GrayscaleToRGB layer,
      4. rebuild the graph and load ONLY the weights (model.weights.h5).
    """
    import json
    import tempfile
    import zipfile

    import keras

    _register_grayscale_layer()
    with zipfile.ZipFile(model_path) as z:
        cfg = json.loads(z.read("config.json"))
        replaced = []

        def patch(o):
            if isinstance(o, dict):
                if o.get("class_name") == "Lambda":
                    names = _lambda_names(o)
                    if names != EXPECTED_LAMBDA_NAMES:
                        raise ValueError(f"Unexpected Lambda layer {o['config'].get('name')}: uses {names}. Refusing to load.")
                    keep = {k: o["config"][k] for k in ("name", "trainable", "dtype") if k in o["config"]}
                    o.update({"module": None, "class_name": "GrayscaleToRGB",
                              "registered_name": "MLL>GrayscaleToRGB", "config": keep})
                    replaced.append(keep.get("name"))
                for v in o.values():
                    patch(v)
            elif isinstance(o, list):
                for v in o:
                    patch(v)

        patch(cfg)
        if not replaced:                        # no Lambda: the normal (safe) loader is fine
            return keras.models.load_model(model_path)
        model = keras.saving.deserialize_keras_object(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            z.extract("model.weights.h5", tmp)
            model.load_weights(f"{tmp}/model.weights.h5")
            check = verify_weights_loaded(model, f"{tmp}/model.weights.h5")
    if not check["identical"]:
        raise ValueError(f"Rebuilt model does not hold exactly the saved weights: {check}")
    print(f"Loaded with Lambda layer(s) {replaced} replaced by GrayscaleToRGB (no code executed from the file). "
          f"Weights check: {check['file_arrays']} arrays in the file, all found identical in the model.")
    model.weights_check = check
    return model


def verify_weights_loaded(model, weights_path: str) -> dict:
    """Proof that the rebuilt model holds EXACTLY the saved weights: every layer array stored in
    model.weights.h5 (optimizer state excluded) must appear in the model with the same shape
    and the same values (compared by shape + sum + sum of squares)."""
    import h5py

    file_arrays = []

    def collect(name, obj):
        if isinstance(obj, h5py.Dataset) and not name.startswith("optimizer"):
            file_arrays.append(np.asarray(obj))

    with h5py.File(weights_path, "r") as f:
        f.visititems(collect)
    sig = lambda a: (tuple(a.shape), round(float(a.astype(np.float64).sum()), 4),   # noqa: E731
                     round(float((a.astype(np.float64) ** 2).sum()), 4))
    file_sig = sorted(sig(a) for a in file_arrays)
    model_sig = sorted(sig(np.asarray(w)) for w in model.weights)
    return {"file_arrays": len(file_arrays), "model_variables": len(model.weights),
            "identical": file_sig == model_sig}
