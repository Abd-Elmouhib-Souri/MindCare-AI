// Live emotion lab (experimental): webcam face and microphone voice, shown SEPARATELY.
// Rules: explicit consent before any device starts, a Stop button for each, nothing stored,
// no prediction when there is no face or only silence, and the measured accuracy next to every result.
import { useEffect, useRef, useState } from "react";
import {
  ApiError, getLiveStatus, postFace, postVoice,
  type Emotion, type FaceResult, type LiveStatus, type VoiceResult,
} from "./api";
import { addTapWorklet, encodeWav } from "./wav";

const FRAME_MAX_WIDTH = 480;   // smaller frames = faster upload; the face is cut out at 48x48 anyway
const VOICE_SECONDS = 3;       // TESS sentences last about 2 seconds; 3 s windows cover one phrase
const pct = (v: number) => `${Math.round(v * 100)} %`;
const acc = (v: number) => `${(v * 100).toFixed(1)} %`;   // measured accuracies: one decimal, as in the README

function EmotionBars({ probs, top, label }: { probs: Record<Emotion, number>; top: Emotion; label: string }) {
  return (
    <ul className="bars" aria-label={label}>
      {(Object.entries(probs) as [Emotion, number][]).map(([e, p]) => (
        <li key={e} className={e === top ? "bar is-top" : "bar"}>
          <span className="bar-name">{e}</span>
          <span className="bar-track"><span className="bar-fill" style={{ width: `${p * 100}%` }} /></span>
          <span className="bar-value">{pct(p)}</span>
        </li>
      ))}
    </ul>
  );
}

function deviceError(e: unknown): string {
  if (e instanceof DOMException && e.name === "NotAllowedError") return "Permission was refused in the browser.";
  if (e instanceof DOMException && e.name === "NotFoundError") return "No device of this kind was found.";
  if (e instanceof ApiError) return e.message;
  return "The device could not be started.";
}

export function LiveLab() {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [statusError, setStatusError] = useState(false);
  const [consent, setConsent] = useState(false);

  // ----- camera -----
  const [camOn, setCamOn] = useState(false);
  const [face, setFace] = useState<FaceResult | null>(null);
  const [camMsg, setCamMsg] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const camStream = useRef<MediaStream | null>(null);
  const camRunning = useRef(false);

  // ----- microphone -----
  const [micOn, setMicOn] = useState(false);
  const [voice, setVoice] = useState<VoiceResult | null>(null);
  const [micMsg, setMicMsg] = useState("");
  const [fill, setFill] = useState(0);              // how much of the current 3 s window is recorded
  const micStream = useRef<MediaStream | null>(null);
  const audioCtx = useRef<AudioContext | null>(null);

  useEffect(() => {
    getLiveStatus().then(setStatus).catch(() => setStatusError(true));
    return () => { stopCamera(); stopMic(); };     // leaving the page always releases the devices
  }, []);

  // ---------- camera ----------
  async function startCamera() {
    setCamMsg("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      camStream.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      setCamOn(true);
      camRunning.current = true;
      cameraLoop();
    } catch (e) {
      setCamMsg(deviceError(e));
      stopCamera();
    }
  }

  // One request at a time: the next frame is taken only after the previous answer arrived (no pile-up)
  async function cameraLoop() {
    const canvas = document.createElement("canvas");   // never attached to the page, never saved
    while (camRunning.current) {
      const video = videoRef.current;
      if (video && video.videoWidth > 0) {
        const scale = Math.min(1, FRAME_MAX_WIDTH / video.videoWidth);
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
        const jpeg = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/jpeg", 0.85));
        if (jpeg && camRunning.current) {
          try {
            const r = await postFace(jpeg);
            if (camRunning.current) { setFace(r); setCamMsg(""); }   // an answer arriving after Stop is ignored
          } catch (e) {
            if (camRunning.current) setCamMsg(deviceError(e));
          }
        }
      }
      await new Promise((r) => setTimeout(r, 400));
    }
  }

  function stopCamera() {
    camRunning.current = false;
    camStream.current?.getTracks().forEach((t) => t.stop());   // turns the camera light off
    camStream.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamOn(false);
    setFace(null);
  }

  // ---------- microphone ----------
  async function startMic() {
    setMicMsg("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 }, video: false });
      micStream.current = stream;
      const ctx = new AudioContext();
      audioCtx.current = ctx;
      await addTapWorklet(ctx);
      const source = ctx.createMediaStreamSource(stream);
      const tap = new AudioWorkletNode(ctx, "mindcare-tap");
      const mute = ctx.createGain();
      mute.gain.value = 0;                                   // keeps the graph running without playing your voice back
      source.connect(tap).connect(mute).connect(ctx.destination);

      const target = VOICE_SECONDS * ctx.sampleRate;
      let chunks: Float32Array[] = [];
      let length = 0;
      let busy = false;
      tap.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        chunks.push(ev.data);
        length += ev.data.length;
        setFill(Math.min(1, length / target));
        if (length < target) return;
        const win = new Float32Array(length);              // the 3 s window, in memory only
        let offset = 0;
        for (const c of chunks) { win.set(c, offset); offset += c.length; }
        chunks = [];
        length = 0;
        if (busy) return;                                  // previous window still being analysed: skip this one
        busy = true;
        postVoice(encodeWav(win, ctx.sampleRate))
          .then((r) => { if (micStream.current === stream) { setVoice(r); setMicMsg(""); } })  // ignored after Stop
          .catch((e) => { if (micStream.current === stream) setMicMsg(deviceError(e)); })
          .finally(() => { busy = false; });
      };
      setMicOn(true);
    } catch (e) {
      setMicMsg(deviceError(e));
      stopMic();
    }
  }

  function stopMic() {
    micStream.current?.getTracks().forEach((t) => t.stop());
    micStream.current = null;
    audioCtx.current?.close();
    audioCtx.current = null;
    setMicOn(false);
    setVoice(null);
    setFill(0);
  }

  function toggleConsent(checked: boolean) {
    setConsent(checked);
    if (!checked) { stopCamera(); stopMic(); }             // withdrawing consent stops everything
  }

  const box = face?.face_found && face.box && face.frame ? {
    left: `${(face.box.x / face.frame.w) * 100}%`, top: `${(face.box.y / face.frame.h) * 100}%`,
    width: `${(face.box.w / face.frame.w) * 100}%`, height: `${(face.box.h / face.frame.h) * 100}%`,
  } : null;

  return (
    <section className="live" aria-labelledby="live-title">
      <h2 id="live-title">Live emotion lab <span className="tag">experimental</span></h2>
      <p>
        The earlier version of this project also read faces and voices in real time. The two models are kept here,
        separately, with their measured accuracy. They recognise <strong>posed</strong> expressions and{" "}
        <strong>acted</strong> speech; they do not measure mood, distress or mental health, and their outputs are never
        combined with the text score.
      </p>

      {statusError && (
        <div className="error" role="alert"><p>The model server is not running, so the live lab is unavailable.</p></div>
      )}

      {status && (
        <>
          <label className="consent">
            <input type="checkbox" checked={consent} onChange={(e) => toggleConsent(e.target.checked)} />
            <span>
              I agree to turn on my camera or microphone. Frames and 3-second sound windows are sent to my local API,
              analysed in memory and discarded. Nothing is recorded or stored.
            </span>
          </label>

          <div className="live-grid">
            {/* ---------- face ---------- */}
            <div className="live-panel">
              <h3>Face</h3>
              <p className="measured">
                {acc(status.face.test_accuracy)} accuracy on FER-2013 test images (chance {pct(status.chance_accuracy)}).
                Webcam faces look different from those 48×48 photos, so expect less.
              </p>
              {!status.face_available ? (
                <p className="unavailable">Face model not loaded. Set <code>MINDCARE_FACE_MODEL</code> and restart the API.</p>
              ) : (
                <>
                  <div className="cam">
                    {/* mirrored like a mirror; the box is inside the mirrored layer so it stays on the face */}
                    <div className="cam-mirror">
                      <video ref={videoRef} muted playsInline aria-label="Your camera preview" />
                      {box && <span className="cam-box" style={box} />}
                    </div>
                    {camOn && <span className="rec" role="status">● Camera on</span>}
                    {!camOn && <span className="cam-off">Camera off</span>}
                  </div>
                  <div className="actions">
                    {!camOn
                      ? <button type="button" onClick={startCamera} disabled={!consent}>Start camera</button>
                      : <button type="button" className="stop" onClick={stopCamera}>Stop camera</button>}
                  </div>
                  <div className="live-out" aria-live="polite">
                    {camMsg && <p className="unavailable">{camMsg}</p>}
                    {camOn && face && !face.face_found && <p className="waiting">No face found: nothing is predicted.</p>}
                    {face?.face_found && face.probabilities && face.top && (
                      <EmotionBars probs={face.probabilities} top={face.top} label="Face expression probabilities" />
                    )}
                  </div>
                </>
              )}
            </div>

            {/* ---------- voice ---------- */}
            <div className="live-panel">
              <h3>Voice</h3>
              <p className="measured">
                {acc(status.voice.unseen_speaker_accuracy)} accuracy on a voice the model never heard (chance{" "}
                {pct(status.chance_accuracy)}). It was trained on two actresses only; yours is a new voice.
              </p>
              {!status.voice_available ? (
                <p className="unavailable">Voice model not loaded. Set <code>MINDCARE_VOICE_MODEL</code> and restart the API.</p>
              ) : (
                <>
                  <div className="mic">
                    <span className="mic-label">{micOn ? `Listening: ${VOICE_SECONDS} s window` : "Microphone off"}</span>
                    <span className="mic-track"><span className="mic-fill" style={{ width: `${fill * 100}%` }} /></span>
                    {micOn && <span className="rec" role="status">● Microphone on</span>}
                  </div>
                  <div className="actions">
                    {!micOn
                      ? <button type="button" onClick={startMic} disabled={!consent}>Start microphone</button>
                      : <button type="button" className="stop" onClick={stopMic}>Stop microphone</button>}
                  </div>
                  <div className="live-out" aria-live="polite">
                    {micMsg && <p className="unavailable">{micMsg}</p>}
                    {micOn && voice && !voice.speech_found && <p className="waiting">Only silence: nothing is predicted.</p>}
                    {micOn && voice && (
                      <p className="level">Loudness of the last window: {voice.level.toFixed(3)} (silence below {voice.silence_threshold})</p>
                    )}
                    {voice?.speech_found && voice.probabilities && voice.top && (
                      <EmotionBars probs={voice.probabilities} top={voice.top} label="Voice emotion probabilities" />
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
          <p className="scale-note">{status.note}</p>
        </>
      )}
    </section>
  );
}
