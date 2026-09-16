// Whisper inference worker — model loading and transcription run here so the
// page stays responsive while hour-long files grind through.
import { pipeline, env } from '../vendor/transformers.min.js';

// Everything ships from vendor/ except the model weights themselves, which
// are fetched from the Hugging Face Hub once and cached by the browser.
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;

let transcriber = null;
let loadedKey = '';

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      const key = `${msg.model}|${msg.device}`;
      if (transcriber && loadedKey === key) {
        self.postMessage({ type: 'ready', device: msg.device });
        return;
      }
      if (transcriber?.dispose) { try { await transcriber.dispose(); } catch {} }
      transcriber = null;
      loadedKey = '';
      transcriber = await pipeline('automatic-speech-recognition', msg.model, {
        device: msg.device,
        progress_callback: (p) => {
          if (p.status === 'initiate' || p.status === 'progress' || p.status === 'done') {
            self.postMessage({
              type: 'model-progress',
              file: p.file || '',
              status: p.status,
              loaded: p.loaded || 0,
              total: p.total || 0,
            });
          }
        },
      });
      loadedKey = key;
      self.postMessage({ type: 'ready', device: msg.device });
    } else if (msg.type === 'chunk') {
      const opts = { return_timestamps: true };
      if (msg.language) {
        opts.language = msg.language;
        opts.task = 'transcribe';
      }
      const out = await transcriber(msg.audio, opts);
      self.postMessage({
        type: 'result',
        id: msg.id,
        text: out.text || '',
        chunks: (out.chunks || []).map((c) => ({ timestamp: c.timestamp, text: c.text })),
      });
    }
  } catch (err) {
    self.postMessage({
      type: 'error',
      during: msg.type,
      id: msg.id,
      message: err?.message || String(err),
    });
  }
};
