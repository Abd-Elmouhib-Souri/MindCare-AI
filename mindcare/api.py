"""
MindCare AI - small HTTP API for the TEXT model (the only module evaluated end to end).

Design rules (responsible-AI):
  - every response carries the disclaimer and a help link;
  - empty / whitespace-only / too long inputs are rejected (HTTP 422);
  - the input text is never stored, logged, or echoed back in the response;
  - the score is described for what it is: similarity to r/SuicideWatch vs r/teenagers posts.

Live emotion lab (experimental, optional): face and voice endpoints under /api/live/*.
  Images and audio are processed in memory and discarded; they are never stored or logged.
  If MINDCARE_FACE_MODEL / MINDCARE_VOICE_MODEL are not set, those endpoints answer 503.

Run (repo root, .venv active):
  $env:MINDCARE_BILSTM_DIR = "C:\\projects\\mindcare_runs\\bilstm_2026-11-06\\models"
  $env:MINDCARE_FACE_MODEL = "...\\best_vision_stage2.keras"            # optional
  $env:MINDCARE_VOICE_MODEL = "...\\best_audio_model_stage2.keras"      # optional
  uvicorn mindcare.api:app --port 8000
Docs: http://127.0.0.1:8000/docs
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field

from mindcare.multiclass import EMOTIONS

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"
MAX_CHARS = 5000
HELP_URL = "https://findahelpline.com"
DISCLAIMER = ("Research prototype. Not a diagnostic or crisis tool, not for use on individuals. "
              "The score only says how much a text resembles r/SuicideWatch posts rather than r/teenagers posts.")


class PredictIn(BaseModel):
    text: str = Field(..., description="English text to analyse", max_length=MAX_CHARS)


class PredictOut(BaseModel):
    score: float            # model output in [0, 1]
    threshold: float        # chosen on validation (pre-registered rule, recall >= 0.90)
    flagged: bool           # score >= threshold
    model: str
    disclaimer: str
    help_url: str


def _load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def model_card() -> dict:
    """Measured numbers shown by the web page, read from the committed result files (never typed by hand)."""
    final = _load_json("final_test.json")
    bilstm = _load_json("bilstm_glove.json")
    b = final["bilstm"]
    return {
        "model": "Bi-LSTM + GloVe 100d (frozen)",
        "task": "r/SuicideWatch vs r/teenagers posts (subreddit-level weak labels)",
        "test_n": final["n_test"],
        "test_macro_f1": [b["macro_f1"], *b["ci95"]["macro_f1"]],
        "test_recall_distress": b["recall_1"],
        "test_precision_distress": b["precision_1"],
        "threshold": final["thresholds_from_validation"]["bilstm"],
        "challenge_groups": {k: {"correct": v["correct"], "n": v["n"]} for k, v in bilstm["challenge_groups"].items()},
        "limitations": [
            "Labels are subreddits, not clinical or per-post annotations.",
            "Distress written in a teen context (school, crush, gaming) is mostly missed.",
            "English Reddit posts (2008-2021) only; short texts get lower scores.",
            "Not for screening, monitoring, or any decision about a person.",
        ],
        "disclaimer": DISCLAIMER,
        "help_url": HELP_URL,
    }


MAX_IMAGE_BYTES = 2_000_000      # one JPEG webcam frame is ~50-150 kB
MAX_AUDIO_BYTES = 3_000_000      # 3 s of 16-bit mono WAV at 48 kHz is ~290 kB
LIVE_NOTE = ("Experimental. These models recognise POSED expressions (FER-2013) and ACTED speech (TESS, 2 actresses). "
             "They do not measure mood, distress, or mental health. Nothing is stored.")


def live_card() -> dict:
    """Measured numbers for the live lab, read from the committed result files."""
    face = _load_json("face_fer2013.json")["test_without_train_duplicates"]
    voice = _load_json("voice_tess.json")
    return {
        "emotions": list(EMOTIONS),
        "face": {"dataset": "FER-2013 (posed, 48x48 grayscale)", "test_accuracy": face["accuracy"],
                 "test_macro_f1": face["macro_f1"], "note": "official test set without images duplicated in train"},
        "voice": {"dataset": "TESS (2 actresses, acted)", "unseen_speaker_accuracy": voice["B_mean_accuracy"],
                  "unseen_speaker_macro_f1": voice["B_mean_macro_f1"],
                  "seen_speaker_accuracy": voice["A_random_split_existing_model"]["accuracy"],
                  "note": "the 0.98 on a random split is inflated: both speakers are in train and test"},
        "chance_accuracy": 1 / len(EMOTIONS),
        "note": LIVE_NOTE,
    }


def _probs(row) -> dict:
    return {e: round(float(p), 4) for e, p in zip(EMOTIONS, row)}


async def _read_body(request: Request, limit: int) -> bytes:
    """Read a raw body with a size cap (413) - the bytes stay in memory and are dropped after the request."""
    data = await request.body()
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Body larger than {limit} bytes.")
    if not data:
        raise HTTPException(status_code=422, detail="Empty body.")
    return data


def create_app(predictor: Optional[Callable] = None, threshold: Optional[float] = None,
               face_predictor: Optional[Callable] = None, voice_predictor: Optional[Callable] = None,
               face_detector: Optional[Callable] = None, load_live_models: bool = True) -> FastAPI:
    """Factory: tests inject fakes; production loads the trained models at startup.
    face_predictor / voice_predictor: batch array -> (n, 7) probabilities. face_detector: gray image -> box or None."""
    state = {"predict": predictor, "threshold": threshold, "face": face_predictor, "voice": voice_predictor,
             "detect": face_detector, "load_live": load_live_models}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["predict"] is None:                       # production: load once, at startup
            from mindcare.bilstm import load_bilstm
            model_dir = os.environ.get("MINDCARE_BILSTM_DIR")
            if not model_dir:
                raise RuntimeError("Set MINDCARE_BILSTM_DIR to the folder containing bilstm_glove.keras")
            state["predict"] = load_bilstm(model_dir)
            state["threshold"] = _load_json("final_test.json")["thresholds_from_validation"]["bilstm"]
        if state["load_live"]:
            if state["face"] is None and os.environ.get("MINDCARE_FACE_MODEL"):
                from mindcare.face import load_face_model
                face_model = load_face_model(os.environ["MINDCARE_FACE_MODEL"])   # safe loader + weights check
                state["face"] = lambda x: face_model.predict(x, verbose=0)
            if state["voice"] is None and os.environ.get("MINDCARE_VOICE_MODEL"):
                import keras
                voice_model = keras.models.load_model(os.environ["MINDCARE_VOICE_MODEL"])  # safe_mode=True default
                state["voice"] = lambda x: voice_model.predict(x, verbose=0)
        if state["detect"] is None:
            from mindcare.live import detect_largest_face
            state["detect"] = detect_largest_face
        yield

    app = FastAPI(title="MindCare AI - text model API (+ experimental live lab)", version="2.0", lifespan=lifespan)
    # Only the local web page may call the API from a browser (Vite dev server by default)
    origins = os.environ.get("MINDCARE_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": state["predict"] is not None}

    @app.get("/api/model-card")
    def card() -> dict:
        return model_card()

    @app.post("/api/predict", response_model=PredictOut)
    def predict(body: PredictIn) -> PredictOut:
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Text is empty.")
        score = float(state["predict"]([text])[0])         # the text is used here and nowhere else
        return PredictOut(score=round(score, 4), threshold=round(state["threshold"], 4),
                          flagged=score >= state["threshold"], model="bilstm_glove100_frozen",
                          disclaimer=DISCLAIMER, help_url=HELP_URL)

    # ---------- Live emotion lab (experimental) ----------
    @app.get("/api/live/status")
    def live_status() -> dict:
        return {"face_available": state["face"] is not None, "voice_available": state["voice"] is not None,
                **live_card()}

    @app.post("/api/live/face")
    async def live_face(request: Request) -> dict:
        if state["face"] is None:
            raise HTTPException(status_code=503, detail="Face model not loaded (set MINDCARE_FACE_MODEL).")
        from mindcare.live import decode_image, face_tensor
        data = await _read_body(request, MAX_IMAGE_BYTES)
        try:
            gray = decode_image(data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        box = state["detect"](gray)
        if box is None:                                    # never predict an emotion on a wall
            return {"face_found": False, "note": LIVE_NOTE}
        probs = np.asarray(state["face"](face_tensor(gray, box)[None, ...]))[0]
        return {"face_found": True, "box": dict(zip(("x", "y", "w", "h"), box)),
                "frame": {"w": int(gray.shape[1]), "h": int(gray.shape[0])},
                "probabilities": _probs(probs), "top": EMOTIONS[int(np.argmax(probs))], "note": LIVE_NOTE}

    @app.post("/api/live/voice")
    async def live_voice(request: Request) -> dict:
        if state["voice"] is None:
            raise HTTPException(status_code=503, detail="Voice model not loaded (set MINDCARE_VOICE_MODEL).")
        from mindcare.live import SILENCE_RMS, is_silent, rms, voice_signal, voice_tensor
        data = await _read_body(request, MAX_AUDIO_BYTES)
        try:
            y = voice_signal(data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        level = round(rms(y), 4)                           # loudness of the window, shown to the user
        if is_silent(y):                                   # never predict an emotion on silence
            return {"speech_found": False, "level": level, "silence_threshold": SILENCE_RMS, "note": LIVE_NOTE}
        probs = np.asarray(state["voice"](voice_tensor(y)[None, ...]))[0]
        return {"speech_found": True, "level": level, "silence_threshold": SILENCE_RMS, "probabilities": _probs(probs),
                "top": EMOTIONS[int(np.argmax(probs))], "note": LIVE_NOTE}

    return app


# Module-level app for `uvicorn mindcare.api:app` (loads the real model at startup)
app = create_app()
