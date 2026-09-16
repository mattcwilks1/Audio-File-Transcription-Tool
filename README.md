# Audio Transcriber

Transcribe large audio files — meetings, interviews, hearings, podcasts —
**entirely in your browser**. Speech-to-text runs locally with
[Whisper](https://github.com/openai/whisper) via
[transformers.js](https://huggingface.co/docs/transformers.js): your audio
never leaves your machine. The only thing fetched at runtime is the model
weights, downloaded once from the Hugging Face Hub and cached by the browser.

## Running it

You need any static file server (module workers can't load from a plain
`file://` page). Two easy options from the project folder:

```bash
# option 1 (Python, preinstalled on Mac/Linux)
python3 -m http.server 8000

# option 2 (Node)
npx serve -l 8000
```

Then open <http://localhost:8000> in Chrome/Edge/Firefox.

## Workflow

1. **Open Audio…** (or drag & drop anywhere) an audio file — MP3, WAV, M4A,
   FLAC, OGG, Opus, or the audio track of an MP4/WebM.
2. Pick a **model** — *Base · English* is a good default. Bigger models are
   more accurate but slower; multilingual variants handle other languages
   (with auto-detect or a fixed language).
3. Press **Transcribe**. The transcript streams in live with progress and an
   ETA. **Pause/Resume** or **Cancel** any time — everything transcribed so
   far is kept and exportable.
4. Click any **timestamp** to jump the built-in player there; the current
   segment highlights during playback.
5. Export as **TXT**, **SRT**, or **VTT**, or **Copy** the plain text.

## How large files are handled

- The file is decoded to 16 kHz mono (Whisper's native format), then split
  into ~30-second chunks — with each cut placed at the quietest point near
  the boundary, so words aren't sliced mid-syllable. Hours-long recordings
  are fine.
- Chunks are transcribed sequentially in a **Web Worker**, so the page stays
  responsive throughout.
- **WebGPU** is used automatically when the browser has it (much faster),
  falling back to CPU/WASM otherwise.
- Near-silent chunks are skipped, which also avoids Whisper hallucinating
  text on silence.

## Tech

Plain HTML/CSS/JS, no build step, no server-side anything.
[transformers.js](https://huggingface.co/docs/transformers.js) and the ONNX
Runtime Web wasm build are vendored in `vendor/`, so apart from the one-time
model download the tool works without a CDN.
