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


def load_face_model(model_path: str):
    """Load the trained .keras model. The custom layer must be registered under the SAME
    package/name as at training time ('MLL>GrayscaleToRGB'), otherwise loading fails."""
    import keras
    import tensorflow as tf

    @keras.saving.register_keras_serializable(package="MLL")
    class GrayscaleToRGB(keras.layers.Layer):
        """Repeats the single gray channel into 3 identical channels (EfficientNet expects RGB)."""
        def call(self, inputs):
            return tf.image.grayscale_to_rgb(inputs)

    return keras.models.load_model(model_path)
