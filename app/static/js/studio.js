/* ==========================================================================
   VoxStudio — studio frontend
   Alpine.js component + vanilla media plumbing (MediaRecorder, Wavesurfer).
   Talks to the FastAPI backend over REST; live task progress arrives on a
   WebSocket with an automatic polling fallback.
   ========================================================================== */

const TERMINAL_STATES = new Set(["completed", "failed", "cancelled"]);
const TEXT_LIMIT = 5000;
const MAX_RECORDING_SECONDS = 120;

/**
 * Base URL for the FastAPI backend.
 *
 * Empty by default, which means "same origin" — the normal way the app runs.
 * A static deployment (e.g. GitHub Pages) has no backend, so it injects
 * `window.VOX_API_BASE` pointing at a hosted API instead.
 */
const API_BASE = String(window.VOX_API_BASE || "").replace(/\/+$/, "");

const apiUrl = (path) => `${API_BASE}${path}`;

/** The four pipeline stages the backend reports, mirrored for the UI. */
const DUBBING_STAGES = [
  { step: 1, label: "Separate music", icon: "music-4" },
  { step: 2, label: "Transcribe", icon: "captions" },
  { step: 3, label: "Generate voices", icon: "mic" },
  { step: 4, label: "Render video", icon: "clapperboard" },
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function refreshIcons() {
  if (window.lucide?.createIcons) window.lucide.createIcons();
}

class ApiError extends Error {
  constructor(message, status, code, details) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function apiRequest(path, { method = "GET", body, headers = {} } = {}) {
  const options = { method, headers: { ...headers } };
  if (body instanceof FormData) {
    options.body = body;
  } else if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  const response = await fetch(apiUrl(path), options);
  if (response.status === 204) return null;

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    const error = payload && typeof payload === "object" ? payload.error : null;
    throw new ApiError(
      error?.message || `Request failed with status ${response.status}`,
      response.status,
      error?.code,
      error?.details,
    );
  }
  return payload;
}

/**
 * Follow a background task to completion.
 * Prefers the WebSocket; falls back to polling if the socket cannot be used.
 * Resolves with the final terminal snapshot.
 */
function watchTask(taskId, onUpdate) {
  return new Promise((resolve) => {
    let settled = false;
    let polling = false;
    let socket = null;

    const finish = (snapshot) => {
      if (settled) return;
      settled = true;
      try {
        socket?.close();
      } catch {
        /* already closed */
      }
      resolve(snapshot);
    };

    const poll = async () => {
      if (polling || settled) return;
      polling = true;
      while (!settled) {
        try {
          const snapshot = await apiRequest(`/api/tasks/${taskId}/status`);
          onUpdate?.(snapshot);
          if (TERMINAL_STATES.has(snapshot.status)) {
            finish(snapshot);
            return;
          }
        } catch (error) {
          if (error.status === 404) {
            finish({ status: "failed", message: "Task expired before completing." });
            return;
          }
        }
        await sleep(1100);
      }
    };

    try {
      // Follow the API host: a static build may point at a remote backend.
      const target = API_BASE ? new URL(API_BASE) : window.location;
      const scheme = target.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${target.host}/ws/tasks/${taskId}`);
      socket.onmessage = (event) => {
        if (polling || settled) return;
        let snapshot;
        try {
          snapshot = JSON.parse(event.data);
        } catch {
          return;
        }
        if (snapshot?.error) {
          poll();
          return;
        }
        onUpdate?.(snapshot);
        if (TERMINAL_STATES.has(snapshot.status)) finish(snapshot);
      };
      socket.onerror = () => poll();
      socket.onclose = () => {
        if (!settled) poll();
      };
    } catch {
      poll();
    }
  });
}

function studio() {
  return {
    /* ------------------------------------------------------------------ */
    /* state                                                              */
    /* ------------------------------------------------------------------ */
    tab: "voice",

    capabilities: null,
    capabilitiesFailed: false,
    capabilityBadges: [],
    capabilityList: [],
    languages: [],
    unsupportedLanguages: [],

    voices: [],

    // Voice Studio
    voiceSource: "library",
    selectedVoiceId: "",
    text: "",
    language: "en",
    speed: 1,
    pitch: 0,
    normalize: false,
    format: "wav",

    reference: { kind: null, blob: null, name: "", url: null },
    referencePlaying: false,
    dragging: false,
    draggingVideo: false,
    // dragenter/dragleave also fire when the pointer crosses a *child* of the
    // dropzone, so a plain boolean flickers. Count enter/leave pairs instead.
    dragDepth: {},

    recording: false,
    recSeconds: 0,

    synthesizing: false,
    taskProgress: 0,
    taskStatus: "",
    statusMessage: "",
    result: null,

    waveReady: false,
    playing: false,
    playhead: 0,
    waveDuration: 0,

    // Dubbing Studio
    projects: [],
    selectedProject: null,
    segments: [],
    newProject: {
      title: "",
      file: null,
      sourceLanguage: "auto",
      targetLanguage: "en",
      voiceProfileId: "",
      analyze: true,
    },
    creatingProject: false,
    analyzing: false,
    analysisProgress: 0,
    analysisMessage: "",
    rendering: false,
    savingSegment: null,
    translating: false,
    createOpen: true,

    // Dubbing player / editor
    playerTime: 0,
    playerDuration: 0,
    playerPlaying: false,
    activeSegmentIndex: -1,
    activeSegmentId: null,
    previewingSegment: null,
    previewAudio: null,
    stageStep: 0,

    // Voice Library modal
    modalOpen: false,
    newVoice: { name: "", description: "", language: "en", mode: "record", file: null },
    savingVoice: false,
    modalError: "",

    toasts: [],

    // private
    _toastId: 0,
    _recorder: null,
    _chunks: [],
    _stream: null,
    _audioCtx: null,
    _analyser: null,
    _vizFrame: null,
    _recTimer: null,
    _wavesurfer: null,
    _objectUrl: null,
    _referenceAudio: null,
    _initialized: false,

    /* ------------------------------------------------------------------ */
    /* computed                                                           */
    /* ------------------------------------------------------------------ */
    get wordCount() {
      const trimmed = this.text.trim();
      return trimmed ? trimmed.split(/\s+/).length : 0;
    },

    get blockedReason() {
      if (!this.capabilitiesFailed) {
        if (!this.capabilities) return "Checking local engines…";
        if (!this.capabilities.tts?.available) {
          return "XTTS-v2 is not installed — run: pip install -r requirements-ai.txt";
        }
      }
      if (!this.text.trim()) return "Type a script to synthesise.";
      if (this.voiceSource === "library" && !this.selectedVoiceId) {
        return "Choose a saved voice from the library.";
      }
      if (this.voiceSource !== "library" && !this.reference.blob) {
        return "Record or upload a reference clip first.";
      }
      return "";
    },

    get canSynthesize() {
      return !this.blockedReason;
    },

    get downloadName() {
      if (!this.result?.audio_url) return "";
      return this.result.audio_url.split("/").pop() || "speech";
    },

    get resultVoiceLabel() {
      if (!this.result) return "—";
      if (!this.result.voice_profile_id) return "Uploaded sample";
      const match = this.voices.find((voice) => voice.id === this.result.voice_profile_id);
      return match ? match.name : `Profile #${this.result.voice_profile_id}`;
    },

    get projectDuration() {
      const ends = this.segments.map((segment) => Number(segment.end_time) || 0);
      const lastCue = ends.length ? Math.max(...ends) : 0;
      return this.selectedProject?.duration_seconds || lastCue || 0;
    },

    get timelineCues() {
      const total = this.projectDuration;
      if (!total) return [];
      return this.segments.map((segment, index) => {
        const start = Math.max(0, Number(segment.start_time) || 0);
        const end = Math.max(start, Number(segment.end_time) || 0);
        return {
          id: segment.id,
          index: index + 1,
          left: (start / total) * 100,
          width: Math.max(1, ((end - start) / total) * 100),
          title: segment.translated_text || segment.original_text || "",
          hasAudio: Boolean(segment.dubbed_audio_url),
        };
      });
    },

    get dubbingStatusText() {
      const project = this.selectedProject;
      if (!project) return "";
      if (project.error_message) return `Failed · ${project.error_message}`;
      if (project.status === "completed") return "Dubbing complete.";
      if (project.status === "pending") return "Ready — press Render dubbed video.";
      return `${Math.round(project.progress)}% · ${project.status}`;
    },

    /** The four labelled pipeline stages, for the editor's stage track. */
    get stageTrack() {
      const project = this.selectedProject;
      const current = this.renderStep; // 1..4, or 0 when idle
      return DUBBING_STAGES.map((stage) => ({
        ...stage,
        done: current > stage.step || project?.status === "completed",
        active: current === stage.step && project?.status !== "completed",
      }));
    },

    /** Which of the four stages the last task reported (0 = not running). */
    get renderStep() {
      const project = this.selectedProject;
      if (!project) return 0;
      if (project.status === "completed") return DUBBING_STAGES.length;
      if (!this.rendering && !this.analyzing) return 0;
      return Number(this.stageStep) || 0;
    },

    get activeSegment() {
      return this.activeSegmentIndex >= 0 ? this.segments[this.activeSegmentIndex] : null;
    },

    get finalVideoUrl() {
      return this.selectedProject?.final_video_url || "";
    },

    /* ------------------------------------------------------------------ */
    /* helpers                                                            */
    /* ------------------------------------------------------------------ */
    formatClock(seconds) {
      const total = Number(seconds);
      if (!Number.isFinite(total) || total <= 0) return "0:00.0";
      const minutes = Math.floor(total / 60);
      const rest = total - minutes * 60;
      return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
    },

    initials(name) {
      const cleaned = String(name || "").trim();
      if (!cleaned) return "V";
      return cleaned
        .split(/\s+/)
        .slice(0, 2)
        .map((part) => part.charAt(0).toUpperCase())
        .join("");
    },

    progressClass(status) {
      if (status === "failed" || status === "cancelled") return "progress-bar-failed";
      if (status === "completed") return "progress-bar-done";
      return "progress-bar";
    },

    /**
     * Media URLs the backend hands out are root-relative (`/media/...`). On a
     * static host (GitHub Pages) they must be resolved against the API host.
     */
    mediaUrl(url) {
      if (!url) return "";
      if (!API_BASE || /^(https?:|data:|blob:)/.test(url)) return url;
      return `${API_BASE}${url.startsWith("/") ? url : `/${url}`}`;
    },

    /** True when this page is a static build with no backend to talk to. */
    get staticPreview() {
      return Boolean(this.capabilitiesFailed) && !API_BASE;
    },

    // Literal class names (never interpolated) so the Tailwind scanner emits them.
    stageClass(step) {
      const current = this.renderStep;
      if (this.selectedProject?.status === "completed" || current > step) return "stage-done";
      if (current === step) return "stage-active";
      return "stage-idle";
    },

    statusChipClass(status) {
      if (status === "completed") return "chip-ok";
      if (status === "failed") return "chip-danger";
      if (status === "pending") return "chip-neutral";
      return "chip-warn";
    },

    // Returned as complete literal class names (never interpolated) so the
    // Tailwind content scanner can see and emit them.
    toastClass(kind) {
      if (kind === "error") return "toast-error";
      if (kind === "success") return "toast-success";
      return "toast-info";
    },

    toastIconClass(kind) {
      if (kind === "error") return "mt-0.5 shrink-0 text-rose-300";
      if (kind === "success") return "mt-0.5 shrink-0 text-emerald-300";
      return "mt-0.5 shrink-0 text-violet-300";
    },

    toast(message, kind = "info", detail = "") {
      const id = ++this._toastId;
      const icons = { success: "check-circle-2", error: "alert-triangle", info: "info" };
      this.toasts.push({ id, kind, message, detail, icon: icons[kind] || "info" });
      this.$nextTick(refreshIcons);
      const ttl = kind === "error" ? 9000 : 4200;
      setTimeout(() => this.dismissToast(id), ttl);
    },

    dismissToast(id) {
      this.toasts = this.toasts.filter((item) => item.id !== id);
    },

    /* ------------------------------------------------------------------ */
    /* lifecycle                                                          */
    /* ------------------------------------------------------------------ */
    async init() {
      // Alpine calls init() automatically *and* the template wires it through
      // x-init, so guard against running twice (which would double every fetch).
      if (this._initialized) return;
      this._initialized = true;

      refreshIcons();
      await Promise.all([
        this.loadCapabilities(),
        this.loadLanguages(),
        this.loadVoices(),
        this.loadProjects(),
      ]);
      this.$nextTick(refreshIcons);
    },

    destroy() {
      this.stopVisualizer();
      this.releaseStream();
      if (this._recTimer) clearInterval(this._recTimer);
      if (this._objectUrl) URL.revokeObjectURL(this._objectUrl);
      this._wavesurfer?.destroy?.();
    },

    /* ------------------------------------------------------------------ */
    /* bootstrap data                                                     */
    /* ------------------------------------------------------------------ */
    async loadCapabilities() {
      try {
        const capabilities = await apiRequest("/api/system/capabilities");
        this.capabilities = capabilities;
        this.capabilitiesFailed = false;
        this.capabilityBadges = [
          { key: "device", label: capabilities.device === "cuda" ? "GPU" : "CPU", ok: capabilities.device === "cuda", detail: capabilities.gpu?.detail },
          { key: "ffmpeg", label: "ffmpeg", ok: capabilities.ffmpeg?.available, detail: capabilities.ffmpeg?.detail },
          { key: "tts", label: "XTTS-v2", ok: capabilities.tts?.available, detail: capabilities.tts?.detail },
          { key: "asr", label: "Whisper", ok: capabilities.transcription?.available, detail: capabilities.transcription?.detail },
          { key: "mt", label: "NLLB", ok: capabilities.translation?.available, detail: capabilities.translation?.detail },
          { key: "demucs", label: "Demucs", ok: capabilities.background_separation?.available, detail: capabilities.background_separation?.detail },
        ];
        this.capabilityList = this.capabilityBadges.map((badge) => ({
          key: badge.key,
          label: badge.label,
          ok: badge.ok,
          detail: badge.detail,
        }));
      } catch (error) {
        this.capabilitiesFailed = true;
        this.capabilityBadges = [{ key: "api", label: "backend offline", ok: false, detail: error.message }];
        this.capabilityList = [{ key: "api", label: "Backend", ok: false, detail: error.message }];
      }
      this.$nextTick(refreshIcons);
    },

    async loadLanguages() {
      try {
        const payload = await apiRequest("/api/tts/languages");
        this.languages = payload?.languages || [];
        this.unsupportedLanguages = payload?.unsupported || [];
        if (this.languages.length) {
          const codes = this.languages.map((item) => item.code);
          if (!codes.includes(this.language)) this.language = payload.default || codes[0];
          if (!codes.includes(this.newVoice.language)) this.newVoice.language = payload.default || codes[0];
        }
      } catch (error) {
        this.languages = [
          { code: "en", name: "English" },
          { code: "es", name: "Spanish" },
          { code: "ru", name: "Russian" },
        ];
        this.toast("Could not load languages", "error", error.message);
      }
    },

    /* ------------------------------------------------------------------ */
    /* voice library                                                      */
    /* ------------------------------------------------------------------ */
    async loadVoices() {
      try {
        const payload = await apiRequest("/api/voices?limit=100");
        this.voices = payload?.items || [];
      } catch (error) {
        this.toast("Could not load voices", "error", error.message);
      }
      this.$nextTick(refreshIcons);
    },

    useVoice(voiceId) {
      this.selectedVoiceId = String(voiceId);
      this.voiceSource = "library";
      this.tab = "voice";
      this.toast("Voice selected", "success");
      this.$nextTick(refreshIcons);
    },

    async deleteVoice(voice) {
      if (!window.confirm(`Delete the voice persona “${voice.name}”?`)) return;
      try {
        await apiRequest(`/api/voices/${voice.id}`, { method: "DELETE" });
        if (this.selectedVoiceId === String(voice.id)) this.selectedVoiceId = "";
        this.toast("Voice profile deleted", "success");
        await this.loadVoices();
      } catch (error) {
        this.toast("Could not delete voice", "error", error.message);
      }
    },

    openVoiceModal() {
      this.newVoice = { name: "", description: "", language: this.language, mode: "record", file: null };
      this.modalError = "";
      this.recSeconds = 0;
      this.clearReference();
      this.modalOpen = true;
      this.$nextTick(refreshIcons);
    },

    closeVoiceModal() {
      if (this.recording) this.stopRecording();
      this.modalOpen = false;
    },

    async saveVoice() {
      const name = this.newVoice.name.trim();
      if (!name) {
        this.modalError = "A name is required.";
        return;
      }

      const source = this.newVoice.mode === "record" ? this.reference.blob : this.newVoice.file;
      if (!source) {
        this.modalError =
          this.newVoice.mode === "record"
            ? "Record a sample before saving."
            : "Attach a reference clip before saving.";
        return;
      }

      const form = new FormData();
      form.append("name", name);
      form.append("language", this.newVoice.language);
      if (this.newVoice.description.trim()) form.append("description", this.newVoice.description.trim());
      form.append("sample_audio", source, source.name || "reference.webm");

      this.savingVoice = true;
      this.modalError = "";
      try {
        const created = await apiRequest("/api/voices", { method: "POST", body: form });
        this.toast("Voice persona saved", "success", created?.name);
        this.modalOpen = false;
        this.clearReference();
        await this.loadVoices();
        this.selectedVoiceId = String(created.id);
        this.voiceSource = "library";
      } catch (error) {
        this.modalError = error.message;
        this.toast("Could not save voice", "error", error.message);
      } finally {
        this.savingVoice = false;
        this.$nextTick(refreshIcons);
      }
    },

    /* ------------------------------------------------------------------ */
    /* reference audio (recording + upload)                               */
    /* ------------------------------------------------------------------ */
    setReference({ kind, blob, name }) {
      this.releaseReferenceUrl();
      const url = URL.createObjectURL(blob);
      this._objectUrl = url;
      this.reference = { kind, blob, name: name || "reference.webm", url };
      this.referencePlaying = false;
    },

    clearReference() {
      this.releaseReferenceUrl();
      this.reference = { kind: null, blob: null, name: "", url: null };
      this.referencePlaying = false;
      this._referenceAudio?.pause();
      this._referenceAudio = null;
      if (this.voiceSource !== "library") this.voiceSource = "library";
    },

    releaseReferenceUrl() {
      if (this._objectUrl) URL.revokeObjectURL(this._objectUrl);
      this._objectUrl = null;
    },

    toggleReferencePlayback() {
      if (!this.reference.url) return;
      if (!this._referenceAudio) {
        this._referenceAudio = new Audio(this.reference.url);
        this._referenceAudio.onended = () => {
          this.referencePlaying = false;
        };
      }
      if (this.referencePlaying) {
        this._referenceAudio.pause();
        this.referencePlaying = false;
      } else {
        this._referenceAudio.play();
        this.referencePlaying = true;
      }
    },

    startDrag(flag) {
      this.dragDepth[flag] = (this.dragDepth[flag] || 0) + 1;
      this[flag] = true;
    },

    endDrag(flag) {
      this.dragDepth[flag] = Math.max(0, (this.dragDepth[flag] || 0) - 1);
      if (this.dragDepth[flag] === 0) this[flag] = false;
    },

    _resetDrag(flag) {
      this.dragDepth[flag] = 0;
      this[flag] = false;
    },

    handleFileInput(event) {
      const file = event.target.files?.[0];
      if (file) this.acceptReferenceFile(file);
      event.target.value = "";
    },

    handleDrop(event) {
      this._resetDrag("dragging");
      const file = event.dataTransfer?.files?.[0];
      if (file) this.acceptReferenceFile(file);
    },

    acceptReferenceFile(file) {
      this.setReference({ kind: "upload", blob: file, name: file.name });
      this.voiceSource = "upload";
      this.toast("Reference clip attached", "success", file.name);
    },

    handleModalFileInput(event) {
      const file = event.target.files?.[0];
      if (file) this.newVoice.file = file;
      event.target.value = "";
      this.$nextTick(refreshIcons);
    },

    handleModalDrop(event) {
      this._resetDrag("dragging");
      const file = event.dataTransfer?.files?.[0];
      if (file) {
        this.newVoice.file = file;
        this.newVoice.mode = "upload";
        this.$nextTick(refreshIcons);
      }
    },

    /* ------------------------------------------------------------------ */
    /* microphone recorder                                                */
    /* ------------------------------------------------------------------ */
    async startRecording() {
      if (this.recording) return;
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        this.toast("Recording unsupported", "error", "This browser cannot capture audio.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
        this._stream = stream;
        this._chunks = [];

        const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
        const mimeType = candidates.find((type) => MediaRecorder.isTypeSupported?.(type)) || "";
        const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
        recorder.ondataavailable = (event) => {
          if (event.data?.size) this._chunks.push(event.data);
        };
        recorder.onstop = () => this.finishRecording();
        recorder.onerror = () => this.toast("Recorder error", "error");

        this._recorder = recorder;
        recorder.start(250);
        this.recording = true;
        this.recSeconds = 0;
        this._recTimer = setInterval(() => {
          this.recSeconds += 1;
          if (this.recSeconds >= MAX_RECORDING_SECONDS) this.stopRecording();
        }, 1000);

        this.startVisualizer(stream);
      } catch (error) {
        this.recording = false;
        this.toast("Microphone unavailable", "error", error?.message || "Permission was denied.");
      }
      this.$nextTick(refreshIcons);
    },

    stopRecording() {
      if (this._recTimer) {
        clearInterval(this._recTimer);
        this._recTimer = null;
      }
      this.recording = false;
      try {
        this._recorder?.state !== "inactive" && this._recorder?.stop();
      } catch {
        /* already stopped */
      }
      this.$nextTick(refreshIcons);
    },

    finishRecording() {
      const type = this._recorder?.mimeType || this._chunks[0]?.type || "audio/webm";
      const blob = new Blob(this._chunks, { type });
      this._chunks = [];
      this._recorder = null;
      this.stopVisualizer();
      this.releaseStream();

      if (!blob.size) {
        this.toast("Nothing recorded", "error", "The microphone produced no audio.");
        return;
      }

      const container = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      this.setReference({ kind: "recording", blob, name: `recording.${container}` });
      this.voiceSource = "recording";
      this.toast("Recording ready", "success", `${this.formatClock(this.recSeconds)} captured`);
      this.$nextTick(refreshIcons);
    },

    releaseStream() {
      this._stream?.getTracks().forEach((track) => track.stop());
      this._stream = null;
    },

    /* ------------------------------------------------------------------ */
    /* live visualizer                                                    */
    /* ------------------------------------------------------------------ */
    startVisualizer(stream) {
      const canvases = Array.from(document.querySelectorAll("[data-viz]"));
      if (!canvases.length) return;

      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        this._audioCtx = new AudioContextClass();
        const source = this._audioCtx.createMediaStreamSource(stream);
        const analyser = this._audioCtx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.75;
        source.connect(analyser);
        this._analyser = analyser;
      } catch {
        return; // visualization is a nicety, never a blocker
      }

      const data = new Uint8Array(this._analyser.frequencyBinCount);

      const draw = () => {
        this._vizFrame = requestAnimationFrame(draw);
        this._analyser.getByteFrequencyData(data);
        canvases.forEach((canvas) => this.drawBars(canvas, data));
      };
      draw();
    },

    drawBars(canvas, data) {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      if (!width || !height) return;

      const ratio = window.devicePixelRatio || 1;
      if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
        canvas.width = Math.floor(width * ratio);
        canvas.height = Math.floor(height * ratio);
      }

      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      const bars = 44;
      const gap = 2;
      const barWidth = Math.max(1.5, (width - gap * (bars - 1)) / bars);
      const step = Math.max(1, Math.floor(data.length / bars));
      const gradient = context.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, "#c4b5fd");
      gradient.addColorStop(1, "#6366f1");
      context.fillStyle = gradient;

      for (let index = 0; index < bars; index += 1) {
        const magnitude = data[index * step] / 255;
        const barHeight = Math.max(2, magnitude * height * 0.92);
        const x = index * (barWidth + gap);
        const y = (height - barHeight) / 2;
        if (typeof context.roundRect === "function") {
          context.beginPath();
          context.roundRect(x, y, barWidth, barHeight, barWidth / 2);
          context.fill();
        } else {
          context.fillRect(x, y, barWidth, barHeight);
        }
      }
    },

    stopVisualizer() {
      if (this._vizFrame) cancelAnimationFrame(this._vizFrame);
      this._vizFrame = null;
      this._analyser = null;
      this._audioCtx?.close?.().catch?.(() => {});
      this._audioCtx = null;
    },

    /* ------------------------------------------------------------------ */
    /* synthesis                                                          */
    /* ------------------------------------------------------------------ */
    async generate() {
      const blocked = this.blockedReason;
      if (blocked) {
        this.toast("Cannot generate yet", "error", blocked);
        return;
      }

      const form = new FormData();
      form.append("text", this.text.trim());
      form.append("language", this.language);
      form.append("speed", String(this.speed));
      form.append("pitch", String(this.pitch));
      form.append("output_format", this.format);
      if (this.normalize) form.append("normalize", "true");

      if (this.voiceSource === "library") {
        form.append("voice_profile_id", this.selectedVoiceId);
      } else {
        form.append("reference_audio", this.reference.blob, this.reference.name);
      }

      this.synthesizing = true;
      this.result = null;
      this.waveReady = false;
      this.taskProgress = 1;
      this.taskStatus = "running";
      this.statusMessage = "Queued…";

      try {
        const accepted = await apiRequest("/api/tts/quick", { method: "POST", body: form });
        const snapshot = await watchTask(accepted.task_id, (task) => {
          this.taskProgress = task.progress;
          this.taskStatus = task.status;
          this.statusMessage = task.message || task.status;
        });

        if (snapshot.status !== "completed") {
          this.taskStatus = snapshot.status;
          this.taskProgress = 100;
          this.statusMessage = snapshot.error || snapshot.message || "Synthesis failed.";
          this.toast("Synthesis failed", "error", snapshot.error || snapshot.message);
          return;
        }

        this.result = snapshot.result || {};
        this.taskStatus = "completed";
        this.taskProgress = 100;
        this.statusMessage = "Synthesis complete.";
        this.loadWaveform(this.result.audio_url);
        this.toast("Speech generated", "success", this.downloadName);
      } catch (error) {
        this.taskStatus = "failed";
        this.taskProgress = 100;
        this.statusMessage = error.message;
        this.toast("Synthesis failed", "error", error.message);
      } finally {
        this.synthesizing = false;
        this.$nextTick(refreshIcons);
      }
    },

    loadWaveform(url) {
      if (!url || !window.WaveSurfer) {
        this.waveReady = false;
        return;
      }
      const wavesurfer = this.ensureWavesurfer();
      if (!wavesurfer) return;
      wavesurfer.load(url);
    },

    ensureWavesurfer() {
      if (this._wavesurfer) return this._wavesurfer;
      if (!window.WaveSurfer || !this.$refs.waveform) return null;

      this._wavesurfer = window.WaveSurfer.create({
        container: this.$refs.waveform,
        height: 96,
        waveColor: "#312e81",
        progressColor: "#8b5cf6",
        cursorColor: "#c4b5fd",
        cursorWidth: 2,
        barWidth: 3,
        barGap: 2,
        barRadius: 2,
        normalize: true,
      });

      this._wavesurfer.on("ready", () => {
        this.waveReady = true;
        this.waveDuration = this._wavesurfer.getDuration();
      });
      this._wavesurfer.on("play", () => {
        this.playing = true;
      });
      this._wavesurfer.on("pause", () => {
        this.playing = false;
      });
      this._wavesurfer.on("finish", () => {
        this.playing = false;
      });
      this._wavesurfer.on("timeupdate", (time) => {
        this.playhead = time;
      });
      this._wavesurfer.on("error", (error) => {
        this.toast("Waveform unavailable", "error", String(error?.message || error));
      });

      return this._wavesurfer;
    },

    togglePlayback() {
      this._wavesurfer?.playPause();
    },

    async copyResultPath() {
      if (!this.result?.audio_path) return;
      try {
        await navigator.clipboard.writeText(this.result.audio_path);
        this.toast("Path copied", "success", this.result.audio_path);
      } catch {
        this.toast("Could not copy", "error", this.result.audio_path);
      }
    },

    /* ------------------------------------------------------------------ */
    /* dubbing                                                            */
    /* ------------------------------------------------------------------ */
    async loadProjects() {
      try {
        const payload = await apiRequest("/api/dubbing?limit=100");
        this.projects = payload?.items || [];
      } catch (error) {
        this.toast("Could not load projects", "error", error.message);
      }
      this.$nextTick(refreshIcons);
    },

    handleVideoInput(event) {
      const file = event.target.files?.[0];
      if (file) this.newProject.file = file;
      event.target.value = "";
      this.$nextTick(refreshIcons);
    },

    handleVideoDrop(event) {
      this._resetDrag("draggingVideo");
      const file = event.dataTransfer?.files?.[0];
      if (file) {
        this.newProject.file = file;
        this.$nextTick(refreshIcons);
      }
    },

    async createProject() {
      if (!this.newProject.title.trim()) {
        this.toast("Title required", "error", "Give the project a name.");
        return;
      }
      if (!this.newProject.file) {
        this.toast("Video required", "error", "Attach a source video.");
        return;
      }

      const form = new FormData();
      form.append("title", this.newProject.title.trim());
      form.append("video", this.newProject.file, this.newProject.file.name);
      form.append("source_language", this.newProject.sourceLanguage);
      form.append("target_language", this.newProject.targetLanguage);
      form.append("analyze", this.newProject.analyze ? "true" : "false");
      if (this.newProject.voiceProfileId) {
        form.append("voice_profile_id", this.newProject.voiceProfileId);
      }

      this.creatingProject = true;
      this.analysisMessage = "Uploading…";
      this.analysisProgress = 0;
      try {
        // The upload endpoint runs steps 1-2 (separate + transcribe) in the
        // background and returns the project id immediately.
        const accepted = await apiRequest("/api/dubbing/upload", { method: "POST", body: form });
        const projectId = accepted.project_id;
        this.toast("Upload accepted", "success", accepted.message || "Processing…");
        this.newProject.title = "";
        this.newProject.file = null;
        await this.loadProjects();
        await this.selectProject(projectId);

        if (accepted.task_id) {
          this.analyzing = true;
          const snapshot = await watchTask(accepted.task_id, (task) => {
            this.analysisProgress = task.progress || 0;
            // Backend publishes "Step 1/4: Separating Music" … verbatim.
            this.analysisMessage = task.message || "";
          });
          if (snapshot.status === "completed") {
            this.toast("Analysis complete", "success", snapshot.message || "Subtitles are ready.");
          } else {
            this.toast("Analysis failed", "error", snapshot.error || snapshot.message);
          }
          await this.selectProject(projectId);
        }
      } catch (error) {
        this.toast("Could not create project", "error", error.message);
      } finally {
        this.creatingProject = false;
        this.analyzing = false;
        this.analysisMessage = "";
        this.createOpen = false;
        await this.loadProjects();
        this.$nextTick(refreshIcons);
      }
    },

    async selectProject(projectId) {
      try {
        const project = await apiRequest(`/api/dubbing/${projectId}`);
        this.selectedProject = project;
        this.segments = project.segments || [];
        this.resetPlayerState();
      } catch (error) {
        this.toast("Could not load project", "error", error.message);
      }
      this.$nextTick(refreshIcons);
    },

    resetPlayerState() {
      try {
        this.previewAudio?.pause();
      } catch {
        /* nothing playing */
      }
      this.previewAudio = null;
      this.playerTime = 0;
      this.playerDuration = 0;
      this.playerPlaying = false;
      this.activeSegmentIndex = -1;
      this.activeSegmentId = null;
    },

    /* ---------------- player <-> timeline sync ---------------- */
    /** Zero-padded `MM:SS` for cue boundaries, e.g. `00:03`. */
    formatTimestamp(seconds) {
      const total = Math.max(0, Math.floor(Number(seconds) || 0));
      const minutes = Math.floor(total / 60);
      const rest = total % 60;
      return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
    },

    /** The editor's `Time` column, e.g. `00:03 - 00:07`. */
    formatRange(segment) {
      return `${this.formatTimestamp(segment.start_time)} - ${this.formatTimestamp(segment.end_time)}`;
    },

    onVideoLoaded(event) {
      this.playerDuration = Number(event.target.duration) || 0;
      this.syncActiveSegment();
    },

    onVideoTimeUpdate(event) {
      this.playerTime = Number(event.target.currentTime) || 0;
      this.syncActiveSegment();
    },

    onVideoPlay() {
      this.playerPlaying = true;
    },

    onVideoPause() {
      this.playerPlaying = false;
    },

    /** Highlight whichever cue's span contains the playhead. */
    syncActiveSegment() {
      const time = this.playerTime;
      let index = -1;
      for (let i = 0; i < this.segments.length; i += 1) {
        const start = Number(this.segments[i].start_time) || 0;
        const end = Number(this.segments[i].end_time) || start;
        if (time >= start && time <= end) {
          index = i;
          break;
        }
      }
      if (index !== this.activeSegmentIndex) {
        this.activeSegmentIndex = index;
        this.activeSegmentId = index >= 0 ? this.segments[index].id : null;
      }
    },

    /** Clicking a row jumps the player to that cue's exact start. */
    seekToSegment(segment, index) {
      const player = this.$refs.player;
      const start = Number(segment.start_time) || 0;
      if (player) {
        player.currentTime = start;
        this.playerTime = start;
      }
      this.activeSegmentIndex = index;
      this.activeSegmentId = segment.id;
      this.scrollToActiveSegment();
    },

    scrollToActiveSegment() {
      this.$nextTick(() => {
        const list = this.$refs.segmentList;
        if (!list || this.activeSegmentId == null) return;
        const row = list.querySelector(`[data-segment-id="${this.activeSegmentId}"]`);
        row?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      });
    },

    togglePlay() {
      const player = this.$refs.player;
      if (!player) return;
      if (player.paused) player.play().catch(() => {});
      else player.pause();
    },

    /**
     * Editor shortcuts. Space toggles playback and Tab walks the cues — both
     * are suppressed while a field has focus so typing stays normal.
     */
    onEditorKeydown(event) {
      if (this.tab !== "dubbing" || !this.selectedProject) return;
      const tag = String(event.target?.tagName || "").toLowerCase();
      const inField = tag === "input" || tag === "textarea" || tag === "select";
      // Space: play/pause. Never steal it from a text field or the video itself
      // (the native control already toggles playback there).
      if (event.key === " " && !inField && tag !== "video") {
        event.preventDefault();
        this.togglePlay();
        return;
      }
      // Tab: walk the cues. Left alone inside a field so editing stays normal.
      if (event.key === "Tab" && !inField) {
        event.preventDefault();
        this.stepSegment(event.shiftKey ? -1 : 1);
      }
    },

    stepSegment(delta) {
      if (!this.segments.length) return;
      const next =
        this.activeSegmentIndex < 0
          ? (delta > 0 ? 0 : this.segments.length - 1)
          : Math.min(this.segments.length - 1, Math.max(0, this.activeSegmentIndex + delta));
      const segment = this.segments[next];
      if (segment) this.seekToSegment(segment, next);
    },

    /** Clone one line in isolation and immediately audition it. */
    async previewLine(segment) {
      if (this.previewingSegment) return;
      this.previewingSegment = segment.id;
      try {
        const accepted = await apiRequest(`/api/dubbing/segments/${segment.id}/preview`, {
          method: "POST",
          body: { translate: false },
        });
        const snapshot = await watchTask(accepted.task_id);
        if (snapshot.status !== "completed") {
          this.toast("Preview failed", "error", snapshot.error || snapshot.message);
          return;
        }
        await this.selectProject(this.selectedProject.id);
        const fresh = this.segments.find((row) => row.id === segment.id);
        if (fresh?.dubbed_audio_url) {
          this.playPreview(fresh.dubbed_audio_url);
          this.toast("Line ready", "success", `Cue ${fresh.order_index + 1} cloned`);
        }
      } catch (error) {
        this.toast("Could not preview the line", "error", error.message);
      } finally {
        this.previewingSegment = null;
        this.$nextTick(refreshIcons);
      }
    },

    playPreview(url) {
      const source = this.mediaUrl(url);
      if (!source) return;
      try {
        this.previewAudio?.pause();
        this.previewAudio = new Audio(source);
        this.previewAudio.play().catch(() => {});
      } catch {
        /* autoplay can be blocked until the user has interacted */
      }
    },

    /** Fill the translation column with NLLB before rendering. */
    async autoTranslate() {
      const project = this.selectedProject;
      if (!project || this.translating) return;
      this.translating = true;
      this.stageStep = 3;
      try {
        const accepted = await apiRequest(`/api/dubbing/${project.id}/translate`, {
          method: "POST",
          body: { translate: true, target_language: project.target_language },
        });
        const snapshot = await watchTask(accepted.task_id, (task) => {
          this.stageStep = task.metadata?.step || this.stageStep;
        });
        if (snapshot.status === "completed") {
          const count = snapshot.result?.translated;
          this.toast("Translation ready", "success", count ? `${count} line(s) translated` : "Done");
        } else {
          this.toast("Translation failed", "error", snapshot.error || snapshot.message);
        }
      } catch (error) {
        this.toast("Could not translate", "error", error.message);
      } finally {
        this.translating = false;
        await this.selectProject(project.id);
        this.$nextTick(refreshIcons);
      }
    },

    async renderProject() {
      const project = this.selectedProject;
      if (!project) return;

      const payload = { translate: true, mix_background: true };
      if (this.newProject.voiceProfileId) payload.voice_profile_id = Number(this.newProject.voiceProfileId);

      this.rendering = true;
      project.status = "extracting";
      project.progress = 1;

      try {
        const accepted = await apiRequest(`/api/dubbing/render/${project.id}`, {
          method: "POST",
          body: payload,
        });
        const snapshot = await watchTask(accepted.task_id, (task) => {
          this.selectedProject = { ...this.selectedProject, progress: task.progress, status: task.status };
          this.stageStep = task.metadata?.step || this.stageStep;
        });

        if (snapshot.status === "completed") {
          this.toast("Dubbing complete", "success", "The dubbed video is ready.");
        } else {
          this.toast("Dubbing failed", "error", snapshot.error || snapshot.message);
        }
      } catch (error) {
        this.toast("Could not start dubbing", "error", error.message);
      } finally {
        this.rendering = false;
        await this.loadProjects();
        await this.selectProject(project.id);
        this.$nextTick(refreshIcons);
      }
    },

    async saveSegmentText(segment) {
      const value = (segment.translated_text ?? "").trim();
      this.savingSegment = segment.id;
      try {
        const updated = await apiRequest(`/api/dubbing/segments/${segment.id}`, {
          method: "PUT",
          body: { translated_text: value },
        });
        Object.assign(segment, updated);
        this.toast("Translation saved", "success", "Re-render to hear the new line.");
        await this.selectProject(this.selectedProject.id);
      } catch (error) {
        this.toast("Could not save segment", "error", error.message);
      } finally {
        this.savingSegment = null;
        this.$nextTick(refreshIcons);
      }
    },

    async assignSegmentVoice(segment, voiceId) {
      try {
        const updated = await apiRequest(`/api/dubbing/segments/${segment.id}`, {
          method: "PUT",
          body: { voice_profile_id: voiceId ? Number(voiceId) : null },
        });
        Object.assign(segment, updated);
        this.toast(
          "Voice assigned",
          "success",
          updated.voice_profile_name || "Falling back to the project voice.",
        );
        await this.selectProject(this.selectedProject.id);
      } catch (error) {
        this.toast("Could not assign voice", "error", error.message);
      } finally {
        this.$nextTick(refreshIcons);
      }
    },

    async uploadSubtitles(event) {
      const input = event.target;
      const file = input.files?.[0];
      if (!file || !this.selectedProject) return;

      const form = new FormData();
      form.append("file", file, file.name);
      try {
        await apiRequest(`/api/dubbing/${this.selectedProject.id}/subtitles`, {
          method: "POST",
          body: form,
        });
        this.toast("Subtitles imported", "success", file.name);
        await this.selectProject(this.selectedProject.id);
        await this.loadProjects();
      } catch (error) {
        this.toast("Could not import subtitles", "error", error.message);
      } finally {
        input.value = "";
      }
    },
  };
}

window.studio = studio;

// studio.js is deliberately the first deferred script, which means Lucide (and
// the rest of the bundles) have not executed yet when Alpine initialises the
// tree. Render the first icon pass on DOMContentLoaded, by which point every
// deferred script has run.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", refreshIcons);
} else {
  refreshIcons();
}
