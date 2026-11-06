// Encode raw microphone samples as a 16-bit mono WAV file, so the server can read it without ffmpeg.
// (Browsers record WebM/Opus by default; decoding that server-side would need an extra native tool.)

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); };

  text(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);  // file size - 8
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);                       // PCM header size
  view.setUint16(20, 1, true);                        // format 1 = PCM
  view.setUint16(22, 1, true);                        // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);           // bytes per second
  view.setUint16(32, 2, true);                        // bytes per sample
  view.setUint16(34, 16, true);                       // bits per sample
  text(36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let i = 0; i < samples.length; i++) {          // float [-1, 1] -> int16
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// AudioWorklet that forwards each block of 128 samples to the page (runs off the main thread).
// Inlined as a Blob URL so no extra file has to be served.
const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("mindcare-tap", Tap);
`;

export async function addTapWorklet(ctx: AudioContext): Promise<void> {
  const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}
