"""API tests with a FAKE predictor (no TensorFlow model needed): contract, validation, privacy."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mindcare.api import MAX_CHARS, create_app  # noqa: E402


def fake_predict(texts):
    # deterministic fake: "alone" -> high score, anything else -> low score
    return [0.95 if "alone" in t.lower() else 0.10 for t in texts]


@pytest.fixture()
def client():
    with TestClient(create_app(predictor=fake_predict, threshold=0.869)) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_flags_above_threshold(client):
    r = client.post("/api/predict", json={"text": "I feel so alone"})
    assert r.status_code == 200
    body = r.json()
    assert body["flagged"] is True and body["score"] == 0.95 and body["threshold"] == 0.869


def test_predict_not_flagged_below_threshold(client):
    assert client.post("/api/predict", json={"text": "bored at school"}).json()["flagged"] is False


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_input_is_rejected(client, text):
    assert client.post("/api/predict", json={"text": text}).status_code == 422


def test_too_long_input_is_rejected(client):
    assert client.post("/api/predict", json={"text": "a" * (MAX_CHARS + 1)}).status_code == 422


def test_missing_field_is_rejected(client):
    assert client.post("/api/predict", json={}).status_code == 422


def test_response_has_disclaimer_and_never_echoes_the_text(client):
    secret = "my private sentence alone"
    r = client.post("/api/predict", json={"text": secret})
    assert "not a diagnostic" in r.json()["disclaimer"].lower()
    assert r.json()["help_url"].startswith("https://")
    assert secret not in r.text                      # privacy: input never returned


def test_model_card_reads_committed_results(client):
    card = client.get("/api/model-card").json()
    f1, lo, hi = card["test_macro_f1"]
    assert lo <= f1 <= hi and 0 < f1 < 1
    assert len(card["limitations"]) >= 3
