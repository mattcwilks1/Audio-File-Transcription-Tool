// TakeOffs Transcriber — transcribe large audio files locally with Whisper.
//
// The file is decoded to 16 kHz mono, split into ~30 s chunks at the quietest
// point near each boundary (so words aren't cut mid-syllable), and the chunks
// are transcribed sequentially in a worker. Segments stream into the page as
// they finish, so a 3-hour recording shows useful text within seconds and can
// be paused/cancelled at any chunk boundary with everything so far exportable.

const TARGET_SR = 16000;      // Whisper's expected sample rate
const CHUNK_S = 30;           // Whisper's native window
const SEARCH_S = 2.5;         // hunt this far around a boundary for silence
const SILENT_PEAK = 0.0015;   // chunks quieter than this are skipped entirely

const MODELS = [
  { id: 'Xenova/whisper-tiny.en',  label: 'Tiny · English — fastest (~40 MB)',        en: true },
  { id: 'Xenova/whisper-base.en',  label: 'Base · English — balanced (~80 MB)',       en: true },
  { id: 'Xenova/whisper-small.en', label: 'Small · English — most accurate (~250 MB)', en: true },
  { id: 'Xenova/whisper-tiny',     label: 'Tiny · multilingual (~40 MB)',             en: false },
  { id: 'Xenova/whisper-base',     label: 'Base · multilingual (~80 MB)',             en: false },
  { id: 'Xenova/whisper-small',    label: 'Small · multilingual (~250 MB)',           en: false },
  { id: 'onnx-community/whisper-large-v3-turbo', label: 'Large v3 Turbo · multilingual (~800 MB, WebGPU recommended)', en: false },
];

const LANGUAGES = [
  ['', 'Auto-detect'], ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'],
  ['de', 'German'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'],
  ['ru', 'Russian'], ['zh', 'Chinese'], ['ja', 'Japanese'], ['ko', 'Korean'],
  ['ar', 'Arabic'], ['hi', 'Hindi'], ['tr', 'Turkish'], ['pl', 'Polish'],
  ['uk', 'Ukrainian'], ['vi', 'Vietnamese'], ['th', 'Thai'], ['sv', 'Swedish'],
];

const $ = (id) => document.getElementById(id);

const state = {
  file: null,
  audio: null,          // Float32Array, 16 kHz mono
  duration: 0,
  segments: [],         // { start, end, text }
  running: false,
  paused: false,
  cancelled: false,
  resumeGate: null,
  worker: null,
  workerReadyKey: '',
  pending: new Map(),   // request id -> { resolve, reject }
  nextId: 1,
};

// ---------- UI setup ----------

const modelSel = $('model');
for (const m of MODELS) {
  const opt = document.createElement('option');
  opt.value = m.id;
  opt.textContent = m.label;
  modelSel.appendChild(opt);
}
modelSel.value = 'Xenova/whisper-base.en';

const langSel = $('language');
for (const [code, label] of LANGUAGES) {
  const opt = document.createElement('option');
  opt.value = code;
  opt.textContent = label;
  langSel.appendChild(opt);
}

function syncLanguageEnabled() {
  const m = MODELS.find((x) => x.id === modelSel.value);
  langSel.disabled = !!m?.en;
  langSel.title = m?.en ? 'English-only model — language is fixed' : 'Spoken language (auto-detect works well)';
}
modelSel.addEventListener('change', syncLanguageEnabled);
syncLanguageEnabled();

const hasWebGPU = !!navigator.gpu;
$('device-pill').textContent = hasWebGPU ? 'WebGPU available' : 'CPU (WASM) — slower';
$('device-pill').classList.add(hasWebGPU ? 'ok' : 'warn');
if (!hasWebGPU) $('device').value = 'wasm';

// ---------- file loading ----------

$('btn-open').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', (e) => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

const dropZone = document.body;
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); $('drop-hint').classList.add('over'); });
dropZone.addEventListener('dragleave', () => $('drop-hint').classList.remove('over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  $('drop-hint').classList.remove('over');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

async function loadFile(file) {
  if (state.running) { setStatus('Cancel the current run before loading a new file.', 'warn'); return; }
  setStatus(`Decoding ${file.name}…`);
  $('file-name').textContent = file.name;
  $('file-meta').textContent = `${fmtBytes(file.size)} — decoding…`;
  try {
    const audio = await decodeToMono16k(file);
    state.file = file;
    state.audio = audio;
    state.duration = audio.length / TARGET_SR;
    state.segments = [];
    renderSegments();
    updateExportButtons();
    $('file-meta').textContent = `${fmtBytes(file.size)} · ${fmtClock(state.duration)}`;
    $('player').src && URL.revokeObjectURL($('player').src);
    $('player').src = URL.createObjectURL(file);
    $('player-wrap').hidden = false;
    $('btn-start').disabled = false;
    setProgress(0, '');
    setStatus(`Ready — ${fmtClock(state.duration)} of audio. Press Transcribe.`);
  } catch (err) {
    state.file = null;
    state.audio = null;
    $('btn-start').disabled = true;
    $('file-meta').textContent = fmtBytes(file.size);
    setStatus(`Couldn't decode "${file.name}" — is it an audio file this browser can play? (${err.message || err})`, 'warn');
  }
}

async function decodeToMono16k(file) {
  const buf = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC({ sampleRate: TARGET_SR });
  let ab;
  try {
    ab = await ctx.decodeAudioData(buf);
  } finally {
    ctx.close();
  }
  const n = ab.length;
  const mono = new Float32Array(n);
  for (let c = 0; c < ab.numberOfChannels; c++) {
    const d = ab.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += d[i];
  }
  if (ab.numberOfChannels > 1) {
    const inv = 1 / ab.numberOfChannels;
    for (let i = 0; i < n; i++) mono[i] *= inv;
  }
  // Browsers normally decode straight to the context rate; resample if not.
  if (ab.sampleRate !== TARGET_SR) return resampleLinear(mono, ab.sampleRate, TARGET_SR);
  return mono;
}

function resampleLinear(input, fromSr, toSr) {
  const outLen = Math.round(input.length * toSr / fromSr);
  const out = new Float32Array(outLen);
  const ratio = fromSr / toSr;
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    out[i] = input[i0] + (input[i1] - input[i0]) * (pos - i0);
  }
  return out;
}

// ---------- chunk planning ----------

function planChunks(audio, sr) {
  const total = audio.length / sr;
  const chunks = [];
  let start = 0;
  while (start < total) {
    // Absorb a short tail into the last chunk rather than making a 2 s runt.
    if (total - start <= CHUNK_S + SEARCH_S) {
      chunks.push({ start, end: total });
      break;
    }
    const cutSample = quietestCut(audio, sr, start + CHUNK_S, SEARCH_S);
    const end = cutSample / sr;
    chunks.push({ start, end });
    start = end;
  }
  return chunks;
}

// Find the middle of the quietest 200 ms window within ±searchSec of tSec.
function quietestCut(audio, sr, tSec, searchSec) {
  const half = Math.floor(searchSec * sr);
  const center = Math.floor(tSec * sr);
  const lo = Math.max(0, center - half);
  const hi = Math.min(audio.length, center + half);
  const win = Math.floor(0.2 * sr);
  const hop = Math.floor(0.05 * sr);
  let best = center;
  let bestE = Infinity;
  for (let s = lo; s + win <= hi; s += hop) {
    let e = 0;
    for (let i = s; i < s + win; i++) e += audio[i] * audio[i];
    if (e < bestE) { bestE = e; best = s + (win >> 1); }
  }
  return best;
}

function chunkPeak(audio, sr, chunk) {
  const s = Math.floor(chunk.start * sr);
  const e = Math.min(audio.length, Math.floor(chunk.end * sr));
  let peak = 0;
  for (let i = s; i < e; i++) {
    const a = Math.abs(audio[i]);
    if (a > peak) peak = a;
  }
  return peak;
}

// ---------- worker ----------

function transcribeRequest(msg, transfer) {
  return new Promise((resolve, reject) => {
    const id = state.nextId++;
    state.pending.set(id, { resolve, reject });
    state.worker.postMessage({ ...msg, id }, transfer || []);
  });
}

function ensureWorker() {
  if (state.worker) return;
  const w = new Worker('./src/transcribe-worker.js', { type: 'module' });
  w.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'model-progress') {
      renderModelProgress(msg);
      return;
    }
    // 'ready' answers the init request (no id of its own); results and errors
    // carry the id of the request they answer.
    const key = msg.type === 'ready' ? state.initReqId : (msg.id ?? state.initReqId);
    const p = state.pending.get(key);
    if (!p) {
      if (msg.type === 'error') setStatus(`Worker error: ${msg.message}`, 'warn');
      return;
    }
    state.pending.delete(key);
    if (msg.type === 'error') p.reject(new Error(msg.message));
    else p.resolve(msg);
  };
  w.onerror = (e) => {
    for (const [, p] of state.pending) p.reject(new Error(e.message || 'Worker crashed'));
    state.pending.clear();
    state.worker = null;
    state.workerReadyKey = '';
  };
  state.worker = w;
}

async function initModel(model, device) {
  const key = `${model}|${device}`;
  if (state.workerReadyKey === key) return;
  ensureWorker();
  $('model-progress').hidden = false;
  const id = state.nextId++;
  state.initReqId = id;
  await new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
    state.worker.postMessage({ type: 'init', model, device });
  });
  state.workerReadyKey = key;
  clearModelProgress();
}

function clearModelProgress() {
  $('model-progress').hidden = true;
  $('model-progress').innerHTML = '';
  progressRows.clear();
}

const progressRows = new Map();
function renderModelProgress(msg) {
  if (!msg.file) return;
  let row = progressRows.get(msg.file);
  if (!row) {
    row = document.createElement('div');
    row.className = 'dl-row';
    row.innerHTML = `<span class="dl-name"></span><span class="dl-pct"></span><div class="dl-bar"><div></div></div>`;
    row.querySelector('.dl-name').textContent = msg.file.split('/').pop();
    $('model-progress').appendChild(row);
    progressRows.set(msg.file, row);
  }
  if (msg.status === 'done') {
    row.remove();
    progressRows.delete(msg.file);
    return;
  }
  const pct = msg.total ? Math.round(100 * msg.loaded / msg.total) : 0;
  row.querySelector('.dl-pct').textContent = msg.total ? `${pct}% of ${fmtBytes(msg.total)}` : fmtBytes(msg.loaded);
  row.querySelector('.dl-bar > div').style.width = pct + '%';
}

// ---------- transcription run ----------

$('btn-start').addEventListener('click', run);
$('btn-pause').addEventListener('click', togglePause);
$('btn-cancel').addEventListener('click', () => {
  state.cancelled = true;
  if (state.paused) togglePause();
  setStatus('Cancelling after the current chunk…');
});

function togglePause() {
  state.paused = !state.paused;
  $('btn-pause').textContent = state.paused ? 'Resume' : 'Pause';
  if (!state.paused && state.resumeGate) {
    state.resumeGate();
    state.resumeGate = null;
  }
  if (state.paused) setStatus('Paused — progress is kept; Resume to continue.');
}

async function run() {
  if (!state.audio || state.running) return;
  state.running = true;
  state.cancelled = false;
  state.paused = false;
  state.segments = [];
  renderSegments();
  updateExportButtons();
  setControls('running');

  const model = modelSel.value;
  const wantDevice = $('device').value === 'auto' ? (hasWebGPU ? 'webgpu' : 'wasm') : $('device').value;
  const isEn = !!MODELS.find((m) => m.id === model)?.en;
  const language = isEn ? '' : langSel.value;

  try {
    setStatus('Loading model — first use downloads it, then it\'s cached by the browser…');
    try {
      await initModel(model, wantDevice);
    } catch (err) {
      if (wantDevice === 'webgpu') {
        setStatus(`WebGPU failed (${err.message}); retrying on CPU…`, 'warn');
        state.workerReadyKey = '';
        await initModel(model, 'wasm');
      } else {
        throw err;
      }
    }

    const chunks = planChunks(state.audio, TARGET_SR);
    let done = 0;
    let emaRate = null; // seconds of compute per second of audio

    for (const chunk of chunks) {
      if (state.cancelled) break;
      if (state.paused) await new Promise((res) => { state.resumeGate = res; });
      if (state.cancelled) break;

      const len = chunk.end - chunk.start;
      if (chunkPeak(state.audio, TARGET_SR, chunk) >= SILENT_PEAK) {
        const s = Math.floor(chunk.start * TARGET_SR);
        const e = Math.min(state.audio.length, Math.floor(chunk.end * TARGET_SR));
        const slice = state.audio.slice(s, e); // copy — the buffer is transferred away
        const t0 = performance.now();
        const result = await transcribeRequest(
          { type: 'chunk', audio: slice, language },
          [slice.buffer],
        );
        const rate = (performance.now() - t0) / 1000 / len;
        emaRate = emaRate === null ? rate : emaRate * 0.7 + rate * 0.3;
        appendSegments(result, chunk);
      }

      done += len;
      const remaining = state.duration - done;
      const eta = emaRate ? ` · ~${fmtClock(remaining * emaRate)} left` : '';
      setProgress(done / state.duration, `${fmtClock(done)} / ${fmtClock(state.duration)}${eta}`);
      setStatus(`Transcribing… ${Math.round(100 * done / state.duration)}%${eta}`);
    }

    if (state.cancelled) {
      setStatus(`Cancelled — ${fmtClock(done)} transcribed. Exports include everything so far.`);
    } else {
      setProgress(1, `${fmtClock(state.duration)} / ${fmtClock(state.duration)}`);
      setStatus(`Done — ${state.segments.length} segments from ${fmtClock(state.duration)} of audio.`, 'ok');
    }
  } catch (err) {
    console.error(err);
    let hint = '';
    if (/fetch|network|Failed to load|404/i.test(err.message || '')) {
      hint = ' Model files download from huggingface.co on first use — check your connection.';
    }
    setStatus(`Transcription failed: ${err.message || err}.${hint}`, 'warn');
  } finally {
    state.running = false;
    clearModelProgress();
    setControls('idle');
    updateExportButtons();
  }
}

function appendSegments(result, chunk) {
  const parts = result.chunks?.length
    ? result.chunks
    : (result.text.trim() ? [{ timestamp: [0, chunk.end - chunk.start], text: result.text }] : []);
  for (const p of parts) {
    const text = (p.text || '').trim();
    if (!text) continue;
    const start = chunk.start + (p.timestamp?.[0] ?? 0);
    const end = chunk.start + (p.timestamp?.[1] ?? (chunk.end - chunk.start));
    state.segments.push({ start, end: Math.min(end, chunk.end), text });
  }
  renderSegments(true);
}

// ---------- transcript rendering ----------

function renderSegments(append) {
  const box = $('transcript');
  if (!append) box.innerHTML = '';
  const from = append ? box.children.length : 0;
  for (let i = from; i < state.segments.length; i++) {
    const seg = state.segments[i];
    const row = document.createElement('div');
    row.className = 'seg';
    row.dataset.start = seg.start;
    const ts = document.createElement('button');
    ts.className = 'seg-ts';
    ts.textContent = fmtClock(seg.start);
    ts.title = 'Jump the player here';
    ts.addEventListener('click', () => {
      const player = $('player');
      player.currentTime = seg.start;
      player.play();
    });
    const tx = document.createElement('span');
    tx.className = 'seg-text';
    tx.textContent = seg.text;
    row.append(ts, tx);
    box.appendChild(row);
  }
  if (state.segments.length) {
    $('transcript-empty').hidden = true;
    if (nearBottom(box.parentElement)) box.lastElementChild?.scrollIntoView({ block: 'end' });
  } else {
    $('transcript-empty').hidden = false;
  }
}

function nearBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 120;
}

$('player').addEventListener('timeupdate', () => {
  const t = $('player').currentTime;
  const rows = $('transcript').children;
  let active = null;
  for (const row of rows) {
    const isNow = +row.dataset.start <= t;
    if (isNow) active = row;
  }
  for (const row of rows) row.classList.toggle('active', row === active);
});

// ---------- exports ----------

function baseName() {
  return (state.file?.name || 'transcript').replace(/\.[^.]+$/, '');
}

$('btn-txt').addEventListener('click', () => {
  download(`${baseName()}.txt`, state.segments.map((s) => s.text).join('\n'), 'text/plain');
});
$('btn-srt').addEventListener('click', () => {
  const srt = state.segments
    .map((s, i) => `${i + 1}\n${fmtStamp(s.start, ',')} --> ${fmtStamp(s.end, ',')}\n${s.text}\n`)
    .join('\n');
  download(`${baseName()}.srt`, srt, 'text/plain');
});
$('btn-vtt').addEventListener('click', () => {
  const vtt = 'WEBVTT\n\n' + state.segments
    .map((s) => `${fmtStamp(s.start, '.')} --> ${fmtStamp(s.end, '.')}\n${s.text}\n`)
    .join('\n');
  download(`${baseName()}.vtt`, vtt, 'text/vtt');
});
$('btn-copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText(state.segments.map((s) => s.text).join('\n'));
  setStatus('Transcript copied to clipboard.', 'ok');
});

function updateExportButtons() {
  const off = state.segments.length === 0;
  for (const id of ['btn-txt', 'btn-srt', 'btn-vtt', 'btn-copy']) $(id).disabled = off;
}

function download(name, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ---------- small helpers ----------

function setControls(mode) {
  const running = mode === 'running';
  $('btn-start').disabled = running || !state.audio;
  $('btn-pause').disabled = !running;
  $('btn-cancel').disabled = !running;
  $('btn-pause').textContent = 'Pause';
  modelSel.disabled = running;
  $('device').disabled = running;
  langSel.disabled = running || !!MODELS.find((m) => m.id === modelSel.value)?.en;
}

function setStatus(text, kind) {
  const el = $('status');
  el.textContent = text;
  el.className = kind || '';
}

function setProgress(frac, label) {
  $('progress-bar').style.width = (100 * frac).toFixed(1) + '%';
  $('progress-label').textContent = label;
}

function fmtClock(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}

function fmtStamp(sec, msSep) {
  sec = Math.max(0, sec);
  const h = String(Math.floor(sec / 3600)).padStart(2, '0');
  const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
  const s = String(Math.floor(sec % 60)).padStart(2, '0');
  const ms = String(Math.round((sec % 1) * 1000)).padStart(3, '0');
  return `${h}:${m}:${s}${msSep}${ms}`;
}

function fmtBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(0) + ' KB';
  if (b < 1024 * 1024 * 1024) return (b / 1024 / 1024).toFixed(1) + ' MB';
  return (b / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

setStatus('Drop an audio file (or click Open Audio…) to begin.');
