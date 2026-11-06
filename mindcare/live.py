"""
Live emotion lab (experimental): turn one webcam frame or one microphone window into model inputs.

What these models measure (see eval/results):
  - face: 7 POSED expressions, FER-2013 (accuracy 0.575 on test without train duplicates);
  - voice: 7 ACTED emotions, TESS, 2 speakers (about 0.45 accuracy on an unseen speaker).
They do not measure mood, distress, or mental health. Nothing here is stored or logged.
"""

import io

import numpy as np
from PIL import Image

from mindcare.face import IMAGE_SIZE
from mindcare.voice import SAMPLE_RATE, mfcc_from_signal

FER_SIZE = (48, 48)            # FER-2013 faces are 48x48 grayscale; training upscaled them to 96x96
MIN_FACE_PX = 60               # smaller detections are too blurry to be meaningful
SILENCE_RMS = 0.01             # below this RMS (signal in [-1, 1]) the window counts as silence
MIN_SECONDS = 1.0              # shorter windows are refused


def decode_image(data: bytes) -> np.ndarray:
    """Bytes (JPEG/PNG) -> grayscale uint8 array (H, W). Raises ValueError if not an image."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:                 # PIL raises several types; the API turns this into HTTP 422
        raise ValueError("Not a readable image.") from exc
    return np.asarray(img.convert("L"), dtype=np.uint8)


_CASCADE = None


def detect_largest_face(gray: np.ndarray):
    """Largest frontal face as (x, y, w, h) with OpenCV's Haar cascade, or None."""
    global _CASCADE
    import cv2
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_PX, MIN_FACE_PX))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def face_tensor(gray: np.ndarray, box) -> np.ndarray:
    """Crop the box, go down to 48x48 (FER-2013 resolution) then up to 96x96 like training: (96, 96, 1) in [0, 255]."""
    x, y, w, h = box
    crop = Image.fromarray(gray[y:y + h, x:x + w])
    small = crop.resize(FER_SIZE, Image.LANCZOS)
    return np.asarray(small.resize(IMAGE_SIZE, Image.LANCZOS), dtype=np.float32)[..., None]


def voice_signal(data: bytes) -> np.ndarray:
    """WAV bytes -> mono float32 signal at 22,050 Hz in [-1, 1]. Raises ValueError if unreadable or too short."""
    import librosa
    import soundfile as sf
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError("Not a readable WAV file.") from exc
    y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) < MIN_SECONDS * SAMPLE_RATE:
        raise ValueError(f"Audio shorter than {MIN_SECONDS:.0f} s.")
    return y.astype(np.float32)


def rms(y: np.ndarray) -> float:
    """Root-mean-square loudness of a signal in [-1, 1]."""
    return float(np.sqrt(np.mean(np.square(y))))


def is_silent(y: np.ndarray) -> bool:
    return rms(y) < SILENCE_RMS


def voice_tensor(y: np.ndarray) -> np.ndarray:
    return mfcc_from_signal(y)
