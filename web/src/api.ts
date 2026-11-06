// Typed client for the FastAPI backend (mindcare/api.py).
// The types mirror the Python response models, so a renamed field is a compile error here.

export interface Prediction {
  score: number;       // model output in [0, 1]
  threshold: number;   // chosen on validation (pre-registered rule: distress recall >= 0.90)
  flagged: boolean;    // score >= threshold
  model: string;
  disclaimer: string;
  help_url: string;
}

export interface ChallengeGroup {
  correct: number;
  n: number;
}

export interface ModelCard {
  model: string;
  task: string;
  test_n: number;
  test_macro_f1: [number, number, number]; // [point, CI low, CI high]
  test_recall_distress: number;
  test_precision_distress: number;
  threshold: number;
  challenge_groups: Record<string, ChallengeGroup>;
  limitations: string[];
  disclaimer: string;
  help_url: string;
}

// Errors the page knows how to explain to the user
export class ApiError extends Error {
  constructor(public kind: "offline" | "invalid" | "server", message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    // fetch only throws when the server cannot be reached at all
    throw new ApiError("offline", "The model server is not running.");
  }
  if (res.status === 422) throw new ApiError("invalid", "Type a sentence first.");
  if (res.status === 502 || res.status === 503 || res.status === 504) {
    throw new ApiError("offline", "The model server is not running.");
  }
  if (!res.ok) throw new ApiError("server", `The server answered with HTTP ${res.status}.`);
  return (await res.json()) as T;
}

export const predict = (text: string) =>
  request<Prediction>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

export const getModelCard = () => request<ModelCard>("/api/model-card");

// ---------- Live emotion lab (experimental) ----------
export type Emotion = "angry" | "disgust" | "fear" | "happy" | "neutral" | "sad" | "surprise";

export interface LiveStatus {
  face_available: boolean;
  voice_available: boolean;
  emotions: Emotion[];
  face: { dataset: string; test_accuracy: number; test_macro_f1: number; note: string };
  voice: { dataset: string; unseen_speaker_accuracy: number; unseen_speaker_macro_f1: number;
           seen_speaker_accuracy: number; note: string };
  chance_accuracy: number;
  note: string;
}

export interface FaceResult {
  face_found: boolean;
  box?: { x: number; y: number; w: number; h: number };
  frame?: { w: number; h: number };
  probabilities?: Record<Emotion, number>;
  top?: Emotion;
}

export interface VoiceResult {
  speech_found: boolean;
  level: number;              // RMS loudness of the 3 s window
  silence_threshold: number;
  probabilities?: Record<Emotion, number>;
  top?: Emotion;
}

export const getLiveStatus = () => request<LiveStatus>("/api/live/status");

// Raw bytes in the body (an image or a WAV), not JSON: no base64 overhead
async function postRaw<T>(path: string, body: Blob): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", headers: { "Content-Type": body.type }, body });
  } catch {
    throw new ApiError("offline", "The model server is not running.");
  }
  if (res.status === 503) throw new ApiError("offline", "This model is not loaded on the server.");
  if (!res.ok) throw new ApiError(res.status === 422 ? "invalid" : "server", `The server answered with HTTP ${res.status}.`);
  return (await res.json()) as T;
}

export const postFace = (jpeg: Blob) => postRaw<FaceResult>("/api/live/face", jpeg);
export const postVoice = (wav: Blob) => postRaw<VoiceResult>("/api/live/voice", wav);
