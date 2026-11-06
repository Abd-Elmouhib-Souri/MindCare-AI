"""Live emotion lab endpoints, tested with fake models (no TensorFlow needed) and real image/WAV decoding."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cv2")
pytest.importorskip("soundfile")
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.api import MAX_AUDIO_BYTES, MAX_IMAGE_BYTES, create_app  # noqa: E402
from mindcare.live import detect_largest_face, face_tensor  # noqa: E402
from mindcare.multiclass import EMOTIONS  # noqa: E402
from mindcare.voice import SAMPLE_RATE, extract_3d_mfcc  # noqa: E402

SAD = np.array([[0.05, 0.05, 0.05, 0.05, 0.1, 0.6, 0.1]])   # "sad" is index 5
seen = {}


def fake_face(x):
    seen["face_shape"] = x.shape
    return SAD


def fake_voice(x):
    seen["voice_shape"] = x.shape
    return SAD


def jpeg(w=320, h=240, value=128) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), value, np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


def wav(seconds=3.0, sr=48000, amp=0.3) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def client(detector=lambda g: (10, 20, 100, 100), **kw):
    args = dict(predictor=lambda t: [0.1] * len(t), threshold=0.869, face_predictor=fake_face,
                voice_predictor=fake_voice, face_detector=detector, load_live_models=False)
    args.update(kw)
    return TestClient(create_app(**args))


def test_status_reports_models_and_measured_numbers():
    with client() as c:
        r = c.get("/api/live/status").json()
    assert r["face_available"] and r["voice_available"]
    assert r["emotions"] == list(EMOTIONS)
    assert 0.5 < r["face"]["test_accuracy"] < 0.6          # the measured 0.575, read from the result file
    assert 0.4 < r["voice"]["unseen_speaker_accuracy"] < 0.5
    assert "do not measure" in r["note"]


def test_face_found_returns_probabilities_and_96px_input():
    with client() as c:
        r = c.post("/api/live/face", content=jpeg(), headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 200
    body = r.json()
    assert body["face_found"] and body["top"] == "sad"
    assert body["box"] == {"x": 10, "y": 20, "w": 100, "h": 100}
    assert body["frame"] == {"w": 320, "h": 240}
    assert seen["face_shape"] == (1, 96, 96, 1)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3


def test_no_face_means_no_prediction():
    seen.pop("face_shape", None)
    with client(detector=lambda g: None) as c:
        body = c.post("/api/live/face", content=jpeg()).json()
    assert body == {"face_found": False, "note": body["note"]}
    assert "face_shape" not in seen                         # the model was never called


def test_face_rejects_garbage_empty_and_huge():
    with client() as c:
        assert c.post("/api/live/face", content=b"not an image").status_code == 422
        assert c.post("/api/live/face", content=b"").status_code == 422
        assert c.post("/api/live/face", content=b"0" * (MAX_IMAGE_BYTES + 1)).status_code == 413


def test_voice_speech_found_and_mfcc_shape():
    with client() as c:
        body = c.post("/api/live/voice", content=wav(), headers={"Content-Type": "audio/wav"}).json()
    assert body["speech_found"] and body["top"] == "sad"
    assert abs(body["level"] - 0.3 / np.sqrt(2)) < 0.01        # RMS of a 0.3 sine wave
    assert seen["voice_shape"] == (1, 40, 150, 3)


def test_silence_means_no_prediction():
    seen.pop("voice_shape", None)
    with client() as c:
        body = c.post("/api/live/voice", content=wav(amp=0.0)).json()
    assert body["speech_found"] is False and "voice_shape" not in seen
    assert body["level"] == 0.0 and body["silence_threshold"] == 0.01


def test_voice_rejects_short_garbage_and_huge():
    with client() as c:
        assert c.post("/api/live/voice", content=wav(seconds=0.5)).status_code == 422
        assert c.post("/api/live/voice", content=b"RIFFxxxx").status_code == 422
        assert c.post("/api/live/voice", content=b"0" * (MAX_AUDIO_BYTES + 1)).status_code == 413


def test_503_when_live_models_not_loaded():
    with client(face_predictor=None, voice_predictor=None) as c:
        st = c.get("/api/live/status").json()
        assert not st["face_available"] and not st["voice_available"]
        assert c.post("/api/live/face", content=jpeg()).status_code == 503
        assert c.post("/api/live/voice", content=wav()).status_code == 503
        assert c.post("/api/predict", json={"text": "still works"}).status_code == 200   # text part unaffected


def test_real_detector_finds_no_face_in_a_flat_image():
    assert detect_largest_face(np.full((240, 320), 128, np.uint8)) is None


def test_face_tensor_matches_training_format():
    gray = np.random.default_rng(0).integers(0, 256, (240, 320), dtype=np.uint8)
    t = face_tensor(gray, (50, 40, 120, 120))
    assert t.shape == (96, 96, 1) and t.dtype == np.float32 and 0 <= t.min() and t.max() <= 255   # no /255


def test_live_audio_features_equal_file_features(tmp_path):
    """The microphone path (WAV bytes) and the evaluation path (file) must give the same tensor."""
    from mindcare.live import voice_signal, voice_tensor
    data = wav(sr=SAMPLE_RATE)
    f = tmp_path / "a.wav"
    f.write_bytes(data)
    np.testing.assert_allclose(voice_tensor(voice_signal(data)), extract_3d_mfcc(str(f)), atol=1e-4)
