# VoxStudio — AI Voice Cloning & Video Dubbing Studio

A production-shaped, **100% local** studio for voice cloning and video dubbing.
No paid APIs: it wires open-source models together behind a FastAPI backend and
ships a dark, self-contained web UI.

| Capability | Engine | HTTP |
| --- | --- | --- |
| Voice cloning / TTS | **Coqui XTTS-v2** (`torch`, CUDA or CPU) | `requirements-ai.txt` |
| Speech-to-text | **faster-whisper** | `requirements-ai.txt` |
| Translation | **NLLB-200** (`transformers`) | `requirements-ai.txt` |
| Background separation | **Demucs** | `requirements-ai.txt` |
| Speaker diarization | **pyannote.audio** (optional, needs `HF_TOKEN`) | `requirements-ai.txt` |
| Media muxing / DSP | **ffmpeg** + **ffprobe** | system binary |

Every AI engine is **lazily imported and optional**. The API boots instantly on a
CPU-only machine and reports exactly what is missing through
`GET /api/system/capabilities` (the UI renders this as header badges).

---

## Quickstart

```bash
# 1. Base web stack (fast, ~15 s)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Run the studio
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The compiled `app/static/css/tailwind.css` and the vendored browser libraries
are **committed**, so no Node.js step is needed just to run the app. Only
rebuild after changing `assets/tailwind.css`, the templates, or `studio.js`:

```bash
npm install          # once (Tailwind CLI)
npm run build:css    # recompile Tailwind
npm run vendor       # re-fetch Alpine / Lucide / Wavesurfer
```

Open <http://localhost:8000> for the studio UI and
<http://localhost:8000/docs> for the interactive OpenAPI docs.

### Enabling the AI pipeline

```bash
# ffmpeg is required for any audio/video work
sudo apt-get install -y ffmpeg        # or: brew install ffmpeg

# Local models (PyTorch — download accordingly)
.venv/bin/pip install -r requirements-ai.txt
```

On a CUDA 12.1 host, install the GPU wheels first so pip does not pull the CPU
build:

```bash
.venv/bin/pip install torch==2.1.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements-ai.txt
```

`VOX_TTS_DEVICE=auto` (default) uses CUDA when available and transparently falls
back to CPU — no code change needed.

---

## Architecture

```
app/
├── api/                    # HTTP + WebSocket layer
│   ├── deps.py             # shared dependencies (session, engines, pagination)
│   ├── voices.py           # POST/GET/PATCH/DELETE /api/voices
│   ├── synthesis.py        # POST /api/tts/quick  (JSON or multipart)
│   ├── dubbing.py          # project CRUD, subtitles, /render
│   ├── tasks.py            # task polling, cancellation, /ws/tasks/{id}
│   └── system.py           # /api/health, /api/system/capabilities
├── core/
│   ├── config.py           # typed settings (VOX_* env vars, cached)
│   ├── database.py         # SQLAlchemy engine/session, SQLite FK pragma
│   ├── exceptions.py       # AppError hierarchy → consistent JSON envelope
│   └── logging.py          # one logging configuration for the process
├── models/                 # SQLAlchemy 2.0 typed models
│   ├── voice_profile.py    # VoiceProfile
│   ├── dubbing_project.py  # DubbingProject + DubbingStatus
│   └── subtitle_segment.py # SubtitleSegment
├── schemas/                # Pydantic v2 request/response contracts
├── services/               # the engines (all lazily initialised singletons)
│   ├── tts_engine.py       # XTTS-v2 wrapper: chunking, CUDA→CPU fallback
│   ├── audio_processor.py  # ffmpeg/ffprobe: extract, fit, mix, mux, Demucs
│   ├── transcription_service.py
│   ├── translation_service.py
│   ├── diarization_service.py # optional pyannote speaker labels
│   ├── subtitle_parser.py  # SRT/VTT parsing (+ export)
│   ├── storage.py          # safe streaming uploads, path↔URL mapping
│   ├── task_manager.py     # in-memory async task registry with pub/sub
│   └── dubbing_service.py  # the 4-step pipeline orchestrator
├── static/
│   ├── css/tailwind.css    # COMPILED Tailwind build (committed)
│   ├── js/studio.js        # Alpine component + MediaRecorder/Wavesurfer glue
│   └── vendor/             # Alpine, Lucide, Wavesurfer (committed, no CDN)
├── templates/index.html    # Jinja2 shell
└── main.py                 # app factory, lifespan, error handlers

assets/tailwind.css         # Tailwind source (→ npm run build:css)
scripts/vendor-assets.mjs   # Re-downloads the vendored libs (→ npm run vendor)
tailwind.config.js · package.json
```

### Database (SQLite + SQLAlchemy)

* **`VoiceProfile`** — `id, name, description, sample_audio_path, language,
  duration_seconds, is_active, created_at, updated_at`
* **`DubbingProject`** — `id, title, video_path, original_audio_path,
  vocals_audio_path, background_audio_path, dubbed_audio_path, final_video_path,
  status, progress, error_message, duration_seconds, source_language,
  target_language, voice_profile_id, speaker_reference_path, segment_count,
  last_task_id, created_at, updated_at`
* **`SubtitleSegment`** — `id, project_id (FK, cascade), start_time, end_time,
  order_index, original_text, translated_text, speaker_id, voice_profile_id,
  dubbed_audio_path, created_at, updated_at`

Paths are persisted **relative to `data/`** so the workspace stays portable.

### Background tasks

Long jobs never block the request loop. `POST /api/tts/quick` and
`POST /api/dubbing/{id}/render` return `202 Accepted` with a `task_id`:

```jsonc
{
  "task_id": "6991…", "status": "running", "progress": 0.0,
  "message": "Dubbing queued.",
  "status_url": "/api/tasks/6991…/status",
  "websocket_url": "/ws/tasks/6991…",
  "resource_id": 1
}
```

Track it by **polling** `GET /api/tasks/{id}/status` (progress `0–100`) or by
**streaming** `/ws/tasks/{id}` (the UI uses WebSocket with automatic polling
fallback). Cancellation via `DELETE /api/tasks/{id}` is cooperative.

---

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/voices` | Create a reusable voice profile (multipart upload) |
| `GET` | `/api/voices` | List profiles (`skip`, `limit`, `search`, `active_only`) |
| `GET` | `/api/voices/{id}` | Fetch one profile |
| `PATCH` | `/api/voices/{id}` | Rename / re-describe / archive |
| `DELETE` | `/api/voices/{id}` | Delete profile + reference clip |
| `POST` | `/api/tts/quick` | Clone & synthesise — **JSON** or **multipart** |
| `GET` | `/api/tts/languages` | Languages supported by XTTS-v2 |
| `POST` | `/api/dubbing/upload` | Upload a video → extract · separate · transcribe (`202`) |
| `GET` | `/api/dubbing` | List projects (`status` filter) |
| `GET` | `/api/dubbing/{id}` | Project detail incl. segments |
| `DELETE` | `/api/dubbing/{id}` | Delete project + all artifacts |
| `PUT` | `/api/dubbing/segments/{sid}` | Edit a cue's text / assign its voice |
| `POST` | `/api/dubbing/segments/{sid}/preview` | Clone one line for auditioning (`202`) |
| `POST` | `/api/dubbing/{id}/translate` | Auto-translate the track, no synthesis (`202`) |
| `GET`/`POST` | `/api/dubbing/{id}/segments` | List / bulk-add subtitle segments |
| `DELETE` | `/api/dubbing/{id}/segments/{sid}` | Remove a segment |
| `POST` | `/api/dubbing/{id}/subtitles` | Import an `.srt` / `.vtt` file |
| `POST` | `/api/dubbing/render/{id}` | Synthesise voices + render the video (`202`) |
| `GET` | `/api/tasks` | List tasks (`status`, `kind` filters) |
| `GET` | `/api/tasks/stats` | Registry counters |
| `GET` | `/api/tasks/{id}/status` | Real-time status |
| `DELETE` | `/api/tasks/{id}` | Request cancellation |
| `WS` | `/ws/tasks/{id}` | Live status events |
| `GET` | `/api/health` | Liveness + database probe |
| `GET` | `/api/system/capabilities` | Which local engines are installed |

`POST /api/tts/quick` accepts either shape on the same route:

```bash
# JSON — clone from a saved profile
curl -X POST localhost:8000/api/tts/quick -H 'Content-Type: application/json' \
  -d '{"text":"Salom dunyo","language":"en","voice_profile_id":1,"speed":1.0}'

# multipart — clone from a freshly uploaded clip
curl -X POST localhost:8000/api/tts/quick \
  -F 'text=Salom dunyo' -F 'language=en' -F 'reference_audio=@ref.wav'
```

### Dubbing pipeline

The engine is two endpoints over one resumable, four-step pipeline. Progress is
published through `/ws/tasks/{id}` under these exact labels:

| Step | Label | What happens |
| --- | --- | --- |
| 1/4 | **Separating Music** | FFmpeg extracts the audio, then Demucs writes `vocals.wav` + `background.wav` (falls back to the original track when Demucs is absent) |
| 2/4 | **Transcribing** | `faster-whisper` returns timed segments; pyannote adds speakers when a token is configured |
| 3/4 | **Generating Voices** | NLLB translates (when the languages differ), XTTS-v2 clones the assigned voice per segment, and each clip is time-stretched to fit its cue |
| 4/4 | **Rendering Video** | the timeline is rebuilt, `background.wav` is mixed back in, and FFmpeg muxes the final video |

* `POST /api/dubbing/upload` runs steps 1–2 in the background and returns the
  project id immediately. Pass `analyze=false` to create the project without
  running them.
* `PUT /api/dubbing/segments/{segment_id}` edits a cue's `translated_text` or
  assigns it a different saved `voice_profile_id` (send `null` for the project
  default). Changing either drops the segment's stale audio so a re-render
  regenerates it.
* `POST /api/dubbing/render/{project_id}` runs steps 3–4. It re-enters steps 1–2
  automatically if their artifacts are missing, which makes the whole pipeline
  resumable — a render is safe to call on a freshly uploaded project.

Failures mark the project `failed` and store an actionable `error_message`
instead of leaking a traceback.

## Studio frontend

A single-page studio at `/` — **HTML5 + Alpine.js + Tailwind CSS + Lucide
icons**, with **Wavesurfer.js** for waveforms. Dark theme built on slate
surfaces with violet/indigo accents.

**1 · Voice Studio**
* Microphone recorder with a live `AnalyserNode` canvas visualizer, a duration
timer, and instant playback / discard.
* Drag-and-drop (or click-to-browse) audio uploader.
* Reference source switch: **Library** persona dropdown · **Microphone** · **Upload**.
* Script textarea with a live word + character counter (5 000 char backend cap).
* Language selector driven by `/api/tts/languages` — the 17 XTTS-v2 languages are
  selectable, and languages the model cannot speak (Uzbek, Kazakh, …) are listed
  but disabled with a reason instead of silently producing gibberish.
* **Speed (0.5×–2.0×) and Pitch (−12…+12 semitones)** sliders, loudness
  normalisation, and a WAV/MP3 output selector.
* Generated audio rendered with Wavesurfer.js (scrub, play/pause, timecode) plus
  a download button and copy-path action.

**2 · Dubbing Studio** — a two-column editorial workspace.

*Upload*: video dropzone, source/target languages, optional project voice and a
**Run analysis now** toggle; the pipeline streams `Step 1/4: Separating Music` →
`Step 2/4: Transcribing` character-for-character from the task.

*Left — player & controls*
* HTML5 `<video>` for the source clip, **synchronised with the active cue**: the
  cue under the playhead is highlighted automatically.
* A four-tile **stage track** (separate → transcribe → generate → render) that
  lights up from the task's reported step, plus the overall progress bar.
* Custom transport (previous · play/pause · next · timecode), the **Render
  dubbed video** primary button and a **Download final video** link.
* A clickable **cue map** strip: every detected line, tinted once it has audio.

*Right — segment & translation editor*
* Scrollable table with five columns: **time range** (`0:03.0 – 0:07.0`),
  **original** transcript, **editable translation**, **voice selector** and a
  per-line **Preview** button that clones that one sentence
  (`POST /api/dubbing/segments/{id}/preview`) and plays it back immediately.
* Clicking a row seeks the player to that cue; **Auto-translate** fills the
  translation column (`POST /api/dubbing/{id}/translate`) without rendering.
* SRT/VTT import and inline playback of the finished dub.

*Keyboard*: **Space** play/pause · **Tab** next cue · **Shift+Tab** previous cue.
Shortcuts are suppressed while a text field has focus, so typing stays normal.

**3 · Voice Library**
* Persona cards with an inline preview player, language/duration chips, a
  **Use in studio** shortcut and delete.
* Modal to **record or upload** a new persona and save it permanently.

Every long-running action streams real progress: the UI subscribes to
`/ws/tasks/{id}` and falls back to polling `/api/tasks/{id}/status`, driving the
progress bars, while failures surface as toast notifications carrying the
backend's actionable error message.

### Error contract

Every error — validation, not-found, engine-unavailable, unexpected — returns:

```json
{ "error": { "code": "validation_error", "message": "…", "details": { } } }
```

---

## Deployment

### GitHub Pages (static UI only)

GitHub Pages serves **static files** — it cannot run this backend (uvicorn,
XTTS-v2, Demucs, faster-whisper, ffmpeg). `npm run build:pages` therefore
publishes the studio *interface* only:

```bash
npm run build        # compile Tailwind + vendor the browser libraries
npm run build:pages  # → ./pages/
```

`.github/workflows/pages.yml` runs exactly that on every push to `main` and
deploys `pages/` with the official Pages actions.

#### Repository settings you must set once

1. **Settings → Pages → Build and deployment → Source → `GitHub Actions`.**
   Enabling a Pages site requires repository-admin rights, so the workflow's own
   token cannot do it for you. Until this is set, the *Configure Pages* step
   fails on purpose with that instruction rather than deploying an empty site.
   (Do **not** pick "Deploy from a branch": the published artifact is built by
   the workflow, not committed to a `gh-pages` branch.)
2. **(Optional, only for a split deployment)** *Settings → Secrets and variables
   → Actions → Variables → New repository variable* named `VOX_API_BASE`, set to
   the origin of a running backend, e.g. `https://voxstudio.example.com` (no
   trailing slash). The export injects it as `window.VOX_API_BASE`, and the UI
   resolves every API call, every `/media/...` URL and the `/docs` link against
   it. Set the same `VOX_API_BASE` for any other split deployment; leave it unset
   for the normal same-origin setup.
3. **Settings → Actions → General → Workflow permissions**: leave the default
   *Read repository contents* — the workflow requests the extra `pages: write`
   and `id-token: write` scopes itself.

The site lands at `https://<owner>.github.io/<repo>/`. Assets are exported with
relative paths, so that sub-path works without any extra configuration, and a
`.nojekyll` file keeps Jekyll away from the artifact.

A static build with no `VOX_API_BASE` has no backend to talk to, so it renders
the interface with an honest "static preview" badge and every request fails.
That is expected; it is a UI preview, not the dubbing engine.

For the **full** dubbing engine, host the FastAPI app somewhere that runs Python
and expose it with `VOX_API_BASE` (the Pages UI will drive it), or just run it
locally.

### Hugging Face Spaces (Gradio, phone-friendly)

GitHub Pages cannot run Python, and Freebuff's managed hosting is a Node-only
image — but a Hugging Face Space runs a real Python container, so the dubbing
engine can live there for free and be driven entirely from a phone browser.
`hf-space/` is a **two-file** Gradio build of the studio:

| File | Purpose |
| --- | --- |
| `hf-space/app.py` | Gradio UI + the whole pipeline (XTTS-v2, faster-whisper, NLLB-200, ffmpeg) |
| `hf-space/requirements.txt` | AI stack only — the Space image already ships Gradio and ffmpeg |
| `hf-space/README.md` | Space card (YAML frontmatter) + usage notes |

Deploying from a phone — no local machine, no git, no CLI:

1. **huggingface.co → New → Space** → pick a name, SDK **Gradio**, hardware
   **CPU basic (free)** → *Create Space*.
2. Open the Space's **Files** tab → **Add file → Upload files**, and upload
   `hf-space/app.py` **and** `hf-space/requirements.txt`. Both must land in the
   **root** of the Space repo, not in a subfolder. Upload `hf-space/README.md`
   too if you want the Space card (it replaces the generated one).
3. Wait for the build (a few minutes — PyTorch is the big download), then open
   the Space. Microphone recording works from a phone browser.

The first request downloads XTTS-v2 (~1.9 GB) and NLLB-200 (~2.4 GB) into the
container's **ephemeral** disk, so they are fetched again after a restart;
`app.py` warms XTTS-v2 up in a background thread at startup to hide that cost.
The free tier gives 2 vCPU, and a single video is capped at
`VOX_MAX_VIDEO_SECONDS` (10 minutes by default) because CPU inference is roughly
2–5 minutes per minute of footage. Two pieces of the full stack are deliberately
left out of this build: **Demucs** background separation and **pyannote**
diarization are far too slow for a free CPU container, so the dubbed voice
replaces the original audio track, and there are no speaker labels. XTTS-v2 is
licensed under **CPML** (non-commercial); `app.py` sets `COQUI_TOS_AGREED=1`,
which records acceptance of that licence.

---

## Configuration

All settings use the `VOX_` prefix and can be supplied as environment variables
or in a `.env` file at the project root. Defaults work out of the box.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOX_ENVIRONMENT` | `development` | `development` / `production` / `test` |
| `VOX_DEBUG` | `true` | Reload + verbose errors |
| `VOX_LOG_LEVEL` | `INFO` | `DEBUG`…`CRITICAL` |
| `VOX_HOST` / `VOX_PORT` | `0.0.0.0` / `8000` | Used by `python -m app.main` |
| `VOX_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `VOX_DATA_DIR` | `./data` | Root for DB, uploads, voices, outputs |
| `VOX_DATABASE_URL` | `sqlite:///./data/studio.db` | Any SQLAlchemy URL |
| `VOX_MAX_UPLOAD_SIZE_MB` | `512` | Streamed-upload ceiling |
| `VOX_TTS_MODEL_NAME` | `…/xtts_v2` | XTTS-v2 checkpoint |
| `VOX_TTS_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `VOX_TTS_DEFAULT_LANGUAGE` | `en` | Fallback target language |
| `VOX_TTS_MAX_TEXT_LENGTH` | `5000` | Per-request character limit |
| `VOX_AUDIO_SAMPLE_RATE` | `24000` | Working sample rate |
| `VOX_FFMPEG_BINARY` / `VOX_FFPROBE_BINARY` | `ffmpeg` / `ffprobe` | Binary names/paths |
| `VOX_ENABLE_BACKGROUND_SEPARATION` | `true` | Toggle the Demucs step |
| `VOX_BACKGROUND_SEPARATION_MODEL` | `htdemucs` | Demucs checkpoint |
| `VOX_TRANSLATION_MODEL_NAME` | `facebook/nllb-200-distilled-600M` | Translation model |
| `VOX_WHISPER_MODEL_SIZE` | `small` | `tiny`…`large-v3` (the dubbing floor) |
| `VOX_ENABLE_DIARIZATION` | `true` | Toggle speaker labels |
| `VOX_DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization pipeline |
| `VOX_HF_TOKEN` | _unset_ | Hugging Face token for the gated pyannote model |
| `VOX_TASK_RETENTION_SECONDS` | `3600` | How long finished tasks are kept |
| `VOX_TASK_MAX_HISTORY` | `500` | Task registry ceiling |

---

## Testing

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

The suite runs **without a GPU, ffmpeg or any model download**: it asserts the
real HTTP contract (voice CRUD, task lifecycle, SRT import, upload → analysis,
per-segment edits, project render, pitch plumbing, language availability) and
that optional engines fail with clear, actionable errors. The four pipeline
steps are exercised end to end with stubbed engines (the `Step n/4:` labels must
stream in order), and the Demucs/Pyannote contracts are unit-tested — including
that stems are normalised to `vocals.wav` / `background.wav` and that
unavailable diarization leaves speakers unlabelled instead of guessing. Tests
use an isolated temporary data directory, so your workspace is never touched.

`tests/test_frontend.py` adds Node-backed tests that run against the real
frontend sources, including the GitHub Pages export:

| Harness | What it proves |
| --- | --- |
| `node --check` | `studio.js` parses |
| `tests/frontend_check.mjs` | Computed getters, guard rails, subtitle-timeline geometry, toast/Tailwind class mappings, **script load order**, `init()` idempotency, and that no Tailwind class is built by string interpolation |
| `tests/boot_check.mjs` | The page **actually mounts**: loads the real HTML + assets in jsdom, then asserts Alpine initialized `x-data="studio()"`, Lucide replaced every icon, and the live API supplied capabilities and languages |
| `tests/test_frontend.py` | The Pages export is servable (`npm run build:pages` → relative asset paths, injected `VOX_API_BASE`, `.nojekyll`, zero unsubstituted Jinja tokens) and the Actions workflow is wired to it |

They skip automatically when Node.js or `node_modules` is missing, and
`boot_check.mjs` also skips (exit 3) when no server is reachable.

> The boot check guards a real footgun: the Alpine CDN build calls
> `Alpine.start()` from a `queueMicrotask` that drains as soon as its own
> `<script>` finishes — *before* the next deferred script runs. Loading
> `studio.js` after Alpine leaves `x-data="studio()"` pointing at an undefined
> factory and kills the entire page with cascading `ReferenceError`s. That is
> why `studio.js` is the first script tag, and why the check asserts on a
> rendered DOM rather than just parsing the file.

---

## Notes

* Uploads and generated media are served from `/media/{uploads,voices,outputs}`;
  the SQLite database and temp scratch directories are deliberately not exposed.
* `data/` is git-ignored — the whole workspace is disposable.
* Freebuff's managed hosting builds with a Node-only image, so deploying this
  Python/ASGI service needs a Python-capable host (Render, Fly.io, a VPS, …).
  The managed **preview** runs it locally via
  `.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
