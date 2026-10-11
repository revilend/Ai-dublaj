"""VoxStudio — one-file Gradio edition for Hugging Face Spaces.

This is the phone-friendly twin of the FastAPI studio in this repository: a
single `app.py` (this file) + `requirements.txt`, designed to be pasted into a
Gradio Space and driven from a phone browser. It has no database, no task
registry and no web UI of its own — Gradio is the front end.

Two tabs:

  1. **Ovoz klonlash** — Coqui XTTS-v2 clones a voice from a 6–30 s reference
     clip (upload or phone microphone) and speaks arbitrary text in any of the
     17 languages the model was trained on.
  2. **Video dublyaj** — ffmpeg → faster-whisper (timed transcript) → NLLB-200
     (translation) → XTTS-v2 (per-line clone) → ffmpeg (remux). The reference
     voice defaults to the original speaker, so the dub keeps the same voice.

Everything heavy is imported lazily and every engine degrades with an
actionable message instead of a traceback, so the Space *starts* in seconds on
a free CPU container and only pays for what you actually use.

Notes that matter on the free `cpu-basic` tier (2 vCPU / 16 GB RAM):

* XTTS-v2 is downloaded on first use (~1.9 GB) and cached for the life of the
  container. A background thread warms it up at startup so the first real
  request is not the one that pays for it.
* Video dubbing is deliberately sequential (one job at a time) and capped at
  `VOX_MAX_VIDEO_SECONDS`; CPU inference is roughly a few minutes per minute of
  footage. Start with a short clip.
* Background music is **not** separated (no Demucs here — it is too slow for a
  free CPU Space). The dubbed voice replaces the original audio track.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Environment. These must be set *before* Coqui TTS is imported.
# --------------------------------------------------------------------------- #
import os

# CPML (Coqui Public Model License) acceptance for non-interactive hosts. The
# XTTS-v2 weights and their outputs are licensed for non-commercial use only.
os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import re
import shutil
import subprocess
import threading
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
WORKDIR = Path(os.environ.get("VOX_WORKDIR", "/tmp/voxstudio"))
SAMPLE_RATE = 24_000  # XTTS-v2 output rate
MAX_TEXT_CHARS = 2_000
MAX_VIDEO_SECONDS = float(os.environ.get("VOX_MAX_VIDEO_SECONDS", "600"))  # 10 min
MAX_REFERENCE_SECONDS = 30.0
MIN_REFERENCE_SECONDS = 2.0
XTTS_MODEL = os.environ.get("VOX_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
NLLB_MODEL = os.environ.get("VOX_TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
WHISPER_SIZES = ["tiny", "base", "small", "medium"]
PRELOAD = os.environ.get("VOX_PRELOAD", "1") != "0"

# XTTS-v2 target languages. Keys are the codes the model itself expects.
XTTS_LANGUAGES: dict[str, str] = {
    "en": "Inglizcha",
    "ru": "Ruscha",
    "tr": "Turkcha",
    "es": "Ispancha",
    "fr": "Fransuzcha",
    "de": "Nemischa",
    "it": "Italyancha",
    "pt": "Portugalcha",
    "ar": "Arabcha",
    "zh-cn": "Xitoycha",
    "ja": "Yaponcha",
    "ko": "Koreyscha",
    "hi": "Hindcha",
    "nl": "Gollandcha",
    "pl": "Polyakcha",
    "cs": "Chexcha",
    "hu": "Vengercha",
}
# Uzbek (uz) is a *target* only via Navoiy TTS (CosyVoice2), not XTTS. It is
# intentionally absent from XTTS_LANGUAGES so the model selector stays honest.
UZTTS_LANGUAGES: dict[str, str] = {
    "uz": "O'zbekcha",
}
XTTS_CHOICES = [(f"{name} ({code})", code) for code, name in XTTS_LANGUAGES.items()]
UZTTS_CHOICES = [(name, code) for code, name in UZTTS_LANGUAGES.items()]

# One dropdown for *both* engines: XTTS-v2 rows plus Uzbek rows, plus the
# reference clip. The "engine" radio below (neutral / expressiv) only changes
# which Navoiy style is used, not the model.
def engine_choices():
    return XTTS_CHOICES + [{"label": "O'zbekcha (klonsiz ovoz)", "value": "uz"},
                            {"label": "O'zbekcha (klonlash)", "value": "uz-clone"}]

# Languages faster-whisper can transcribe from (a much longer list in reality;
# these are the ones worth a dropdown row).
WHISPER_LANGUAGES: dict[str, str] = {
    "auto": "Avtomatik aniqlash",
    "uz": "O'zbekcha",
    "en": "Inglizcha",
    "ru": "Ruscha",
    "tr": "Turkcha",
    "kk": "Qozoqcha",
    "tg": "Tojikcha",
    "az": "Ozarbayjoncha",
    "ar": "Arabcha",
    "fa": "Forscha",
    "hi": "Hindcha",
    "zh": "Xitoycha",
    "ja": "Yaponcha",
    "ko": "Koreyscha",
    "es": "Ispancha",
    "fr": "Fransuzcha",
    "de": "Nemischa",
    "it": "Italyancha",
    "pt": "Portugalcha",
    "pl": "Polyakcha",
}
WHISPER_CHOICES = [(name, code) for code, name in WHISPER_LANGUAGES.items()]

# ISO-639-1 -> FLORES-200, the codes NLLB-200 was trained on. `uz` appears as a
# *source* only: XTTS-v2 cannot speak Uzbek, so it is not a dubbing target.
FLORES_CODES: dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "ru": "rus_Cyrl",
    "nl": "nld_Latn",
    "cs": "ces_Latn",
    "ar": "arb_Arab",
    "zh-cn": "zho_Hans",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "hu": "hun_Latn",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
    "uz": "uzn_Latn",
    "kk": "kaz_Cyrl",
    "az": "azj_Latn",
    "fa": "pes_Arab",
    "tg": "tgk_Cyrl",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…。！？؛:])\s+")


class StudioError(RuntimeError):
    """A failure that should be shown to the user verbatim."""


def _say(message: str) -> str:
    """Return a Gradio warning string (short, actionable, no traceback)."""
    return message.strip()


def workdir() -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    return WORKDIR


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return cleaned[:48] or "clip"


def _progress(progress, fraction: float, message: str) -> None:
    """Report progress without ever letting the reporter break the job."""
    if progress is None:
        return
    print(f"[{fraction * 100:5.1f}%] {message}", flush=True)
    try:
        progress(min(max(fraction, 0.0), 1.0), desc=message)
    except Exception:  # noqa: BLE001 - progress must never be fatal
        pass


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe helpers
# --------------------------------------------------------------------------- #
def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise StudioError(
            f"`{name}` topilmadi. Bu Space'da ffmpeg o'rnatilmagan bo'lishi mumkin — "
            "Space sozlamalarida qayta build qiling."
        )
    return path


def _run(args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command, raising :class:`StudioError` with the tail of stderr."""
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = " ".join((result.stderr or "").strip().splitlines()[-2:])[:400]
        raise StudioError(f"{Path(args[0]).name} xato berdi: {tail}")
    return result


def media_duration(path: str | Path) -> float:
    """Duration in seconds via ffprobe (0.0 when unknown)."""
    try:
        raw = _run(
            [
                _binary("ffprobe"),
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(path),
            ]
        ).stdout.strip()
        return float(raw or 0.0)
    except (StudioError, ValueError):
        return 0.0


def normalise_reference(source: str | Path, destination: str | Path) -> Path:
    """Turn any recording/video into a clean 24 kHz mono clip for XTTS.

    Mirrors what the official Coqui Space does: band-limit, strip the silence
    at both ends and cap the length. A clean 6–15 s reference clones far better
    than a raw phone recording.
    """
    destination = Path(destination)
    filters = (
        "highpass=f=75,lowpass=f=8000,"
        "areverse,silenceremove=start_periods=1:start_threshold=-45dB,"
        "areverse,silenceremove=start_periods=1:start_threshold=-45dB,"
        "loudnorm=I=-20:TP=-2"
    )
    _run([
        _binary("ffmpeg"), "-y",
        "-t", "180",  # never decode more than three minutes of source
        "-i", str(source),
        "-vn", "-af", filters,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-t", f"{MAX_REFERENCE_SECONDS:.1f}",
        str(destination),
    ])
    seconds = media_duration(destination)
    if seconds < MIN_REFERENCE_SECONDS:
        raise StudioError(
            "Ovoz namunasi juda qisqa yoki jim. Kamida 3–10 sekund aniq gapirilgan "
            "ovoz kerak (shovqinsiz)."
        )
    return destination


def extract_audio(media: str | Path, destination: str | Path) -> Path:
    """Extract a 16 kHz mono track (what faster-whisper wants)."""
    destination = Path(destination)
    _run([
        _binary("ffmpeg"), "-y",
        "-i", str(media),
        "-vn", "-ac", "1", "-ar", "16000",
        str(destination),
    ])
    return destination


def _atempo_filter(ratio: float) -> str:
    """Build a pitch-preserving speed filter chain for a tempo ratio."""
    parts: list[str] = []
    remaining = max(0.25, min(ratio, 4.0))
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


def fit_clip(source: Path, destination: Path, target_seconds: float) -> Path:
    """Force a clip to exactly ``target_seconds`` so it lines up with its cue.

    Only ever speeds up (slowing speech down sounds worse than a little air),
    then trims. The tempo filter preserves pitch.
    """
    actual = media_duration(source)
    if actual <= 0:
        raise StudioError("Bo'lakning davomiyligini o'qib bo'lmadi.")

    if actual <= target_seconds + 0.02:
        shutil.copyfile(source, destination)
        return destination

    ratio = actual / max(target_seconds, 0.2)
    _run([
        _binary("ffmpeg"), "-y",
        "-i", str(source),
        "-af", _atempo_filter(ratio),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-t", f"{target_seconds:.3f}",
        str(destination),
    ])
    return destination


def mux_video(video: str | Path, audio: str | Path, destination: str | Path) -> Path:
    """Put ``audio`` back on ``video``, copying the video stream when possible."""
    destination = Path(destination)
    base = [
        _binary("ffmpeg"), "-y",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
    ]
    tail = ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(destination)]
    try:
        _run(base + ["-c:v", "copy"] + tail)
    except StudioError:
        # The container/codec combo does not allow a stream copy (e.g. webm).
        _run(base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"] + tail)
    return destination


# --------------------------------------------------------------------------- #
# Audio maths (numpy — no extra ffmpeg graph gymnastics)
# --------------------------------------------------------------------------- #
def _resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or data.size == 0:
        return data.astype(np.float32, copy=False)
    length = max(1, int(round(data.size * target_rate / source_rate)))
    old_axis = np.linspace(0.0, 1.0, data.size, endpoint=False)
    new_axis = np.linspace(0.0, 1.0, length, endpoint=False)
    return np.interp(new_axis, old_axis, data).astype(np.float32)


def read_mono(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return _resample(mono, int(rate), sample_rate)


def concatenate(parts: list[Path], destination: Path, gap_seconds: float = 0.12) -> Path:
    """Join XTTS chunks with a short pause (single chunk passes through)."""
    if not parts:
        raise StudioError("Sintez qilingan bo'lak yo'q.")
    if len(parts) == 1:
        shutil.copyfile(parts[0], destination)
        return destination

    gap = np.zeros(int(gap_seconds * SAMPLE_RATE), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, part in enumerate(parts):
        if index:
            pieces.append(gap)
        pieces.append(read_mono(part))
    return write_track(np.concatenate(pieces), destination)


def write_track(samples: np.ndarray, destination: Path) -> Path:
    """Peak-normalise to -1 dBFS and write 16-bit PCM."""
    buffer = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(buffer))) if buffer.size else 0.0
    if peak > 0:
        buffer = buffer * (0.89 / peak)
    sf.write(str(destination), buffer, SAMPLE_RATE, subtype="PCM_16")
    return destination


def build_dub_track(clips: list[tuple[float, Path]], total_seconds: float, destination: Path) -> Path:
    """Overlay every fitted clip at its own timestamp on one silent timeline."""
    end = max([start + media_duration(path) for start, path in clips] + [total_seconds, 0.5])
    track = np.zeros(int((end + 0.3) * SAMPLE_RATE), dtype=np.float32)

    for start, path in clips:
        data = read_mono(path)
        offset = int(max(start, 0.0) * SAMPLE_RATE)
        if offset >= track.size:
            continue
        stop = min(offset + data.size, track.size)
        track[offset:stop] += data[: stop - offset]

    return write_track(track, destination)


def write_srt(segments: list["Segment"], destination: Path, *, translated: bool) -> Path:
    def stamp(value: float) -> str:
        value = max(value, 0.0)
        hours, rest = divmod(value, 3600)
        minutes, seconds = divmod(rest, 60)
        milliseconds = int(round((seconds - int(seconds)) * 1000))
        if milliseconds >= 1000:  # rounding can reach 1000ms
            seconds, milliseconds = seconds + 1, 0
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.translated if translated and segment.translated else segment.text
        blocks.append(f"{index}\n{stamp(segment.start)} --> {stamp(segment.end)}\n{text}\n")
    destination.write_text("\n".join(blocks), encoding="utf-8")
    return destination


# --------------------------------------------------------------------------- #
# Engines — every import is lazy and every model is cached
# --------------------------------------------------------------------------- #
_tts_lock = threading.Lock()
_whisper_lock = threading.Lock()
_translator_lock = threading.Lock()
_tts_job = threading.Lock()
_dub_job = threading.Lock()

_xtts = None
_whisper: dict[str, object] = {}
_translator: dict[tuple[str, str], object] = {}


def load_xtts():
    """Load (once) the multilingual XTTS-v2 checkpoint."""
    global _xtts
    with _tts_lock:
        if _xtts is not None:
            return _xtts
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        try:
            from TTS.api import TTS
        except Exception as exc:  # noqa: BLE001
            raise StudioError(
                "Coqui TTS o'rnatilmagan. Space'dagi requirements.txt ni tekshirib, "
                f"qayta build qiling. ({exc})"
            ) from exc

        print(f"[xtts] loading {XTTS_MODEL} …", flush=True)
        started = time.time()
        model = TTS(XTTS_MODEL)
        try:
            model.to("cpu")  # free Spaces have no GPU; harmless otherwise
        except Exception:  # noqa: BLE001 - some builds expose no .to()
            pass
        _xtts = model
        print(f"[xtts] ready in {time.time() - started:.1f}s", flush=True)
        return _xtts


def load_whisper(size: str):
    """Load (once per size) a faster-whisper model."""
    with _whisper_lock:
        if size in _whisper:
            return _whisper[size]
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:  # noqa: BLE001
            raise StudioError(f"faster-whisper o'rnatilmagan ({exc}).") from exc

        print(f"[whisper] loading '{size}' (int8, cpu) …", flush=True)
        model = WhisperModel(size, device="cpu", compute_type="int8")
        _whisper[size] = model
        return model


def load_translator(source_code: str, target_code: str):
    """Load (once per language pair) NLLB-200 as tokenizer + model."""
    key = (source_code, target_code)
    with _translator_lock:
        if key in _translator:
            return _translator[key]
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            raise StudioError(f"transformers/torch o'rnatilmagan ({exc}).") from exc

        print(f"[nllb] loading {NLLB_MODEL} for {source_code} -> {target_code} …", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang=source_code)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        model.eval()
        bundle = (torch, tokenizer, model)
        _translator[key] = bundle
        return bundle


def transcribe(path: Path, language: str | None, size: str, progress, span: tuple[float, float]):
    """Timed transcript segments from an audio file."""
    model = load_whisper(size)
    segments_iter, info = model.transcribe(
        str(path),
        language=language,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    total = float(getattr(info, "duration", 0.0) or media_duration(path) or 1.0)
    raw: list[Segment] = []
    for item in segments_iter:
        text = (item.text or "").strip()
        if not text:
            continue
        start, end = float(item.start or 0.0), float(item.end or 0.0)
        _progress(progress, span[0] + (span[1] - span[0]) * min(end / total, 1.0), "Matn aniqlanmoqda…")
        raw.append(Segment(start=start, end=end, text=text))

    return merge_segments(raw), str(getattr(info, "language", "") or "")


def translate_all(texts: list[str], source_code: str, target_code: str, progress, span: tuple[float, float]) -> list[str]:
    """Translate a list of strings with NLLB-200, preserving order."""
    if not any(text.strip() for text in texts):
        return list(texts)
    if source_code == target_code:
        return list(texts)

    flores_source = FLORES_CODES.get(source_code)
    flores_target = FLORES_CODES.get(target_code)
    if not flores_source or not flores_target:
        raise StudioError(
            f"Tarjima qo'llab-quvvatlanmaydi: {source_code} -> {target_code}. "
            "Manba tilini qo'lda tanlang."
        )

    torch, tokenizer, model = load_translator(flores_source, flores_target)
    forced_id = tokenizer.convert_tokens_to_ids(flores_target)
    outputs: list[str] = list(texts)

    batch_size = 8
    total = len(texts)
    for start in range(0, total, batch_size):
        window = [
            (index, texts[index])
            for index in range(start, min(start + batch_size, total))
            if texts[index].strip()
        ]
        if window:
            encoded = tokenizer(
                [text for _, text in window],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_id,
                    max_new_tokens=256,
                    num_beams=1,
                )
            for (index, _), text in zip(window, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                outputs[index] = text.strip()
        _progress(
            progress,
            span[0] + (span[1] - span[0]) * (min(start + batch_size, total) / max(total, 1)),
            f"Tarjima qilinmoqda: {min(start + batch_size, total)}/{total}",
        )
    return outputs


def synthesise(
    text: str,
    speaker: Path,
    language: str,
    speed: float,
    destination: Path,
    progress,
    span: tuple[float, float],
) -> Path:
    """Clone ``speaker``'s voice and speak ``text``; chunked for long input."""
    model = load_xtts()
    chunks = chunk_text(text) or [text.strip()]
    scratch = destination.parent / f"{destination.stem}-parts"
    scratch.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    try:
        for index, chunk in enumerate(chunks):
            _progress(
                progress,
                span[0] + (span[1] - span[0]) * (index / len(chunks)),
                f"Ovoz sintez qilinmoqda: {index + 1}/{len(chunks)}",
            )
            part = scratch / f"part-{index:04d}.wav"
            kwargs = {
                "text": chunk,
                "speaker_wav": str(speaker),
                "language": language,
                "file_path": str(part),
            }
            try:
                model.tts_to_file(speed=speed, **kwargs)
            except TypeError:
                # Older Coqui releases do not accept `speed` on tts_to_file.
                model.tts_to_file(**kwargs)
            if not part.exists() or part.stat().st_size == 0:
                raise StudioError("XTTS bo'sh fayl qaytardi — matn yoki ovoz namunasini tekshiring.")
            parts.append(part)

        concatenate(parts, destination)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)

    _progress(progress, span[1], "Ovoz tayyor.")
    return destination


def chunk_text(text: str, max_chars: int = 250) -> list[str]:
    """Split on sentence boundaries, hard-wrapping anything longer."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            piece = ""
            for word in sentence.split(" "):
                if piece and len(piece) + len(word) + 1 > max_chars:
                    chunks.append(piece)
                    piece = word
                else:
                    piece = f"{piece} {word}".strip()
            buffer = piece
            continue
        if buffer and len(buffer) + len(sentence) + 1 > max_chars:
            chunks.append(buffer)
            buffer = sentence
        else:
            buffer = f"{buffer} {sentence}".strip()
    if buffer:
        chunks.append(buffer)

    # A single "word" longer than the limit (URLs, unspaced CJK) still has to
    # fit: hard-wrap it rather than handing XTTS a 900-character prompt.
    wrapped: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            wrapped.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            wrapped.append(chunk)
    return wrapped


# --------------------------------------------------------------------------- #
# Segment bookkeeping
# --------------------------------------------------------------------------- #
class Segment:
    """One subtitle cue of a dubbing project."""

    __slots__ = ("start", "end", "text", "translated", "audio")

    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = float(start)
        self.end = float(end)
        self.text = text
        self.translated = ""
        self.audio: Path | None = None

    @property
    def spoken(self) -> str:
        return self.translated or self.text


_MUSIC_MARKERS = re.compile(r"^\s*[\[\(\*].*[\]\)\*]\s*$")


def merge_segments(segments: list[Segment], max_chars: int = 220, max_gap: float = 0.35) -> list[Segment]:
    """Join neighbouring cues so XTTS is called as few times as possible.

    Short cues are the main cost driver on a CPU Space: every extra segment is
    another XTTS call and another chance to drift out of sync.
    """
    usable = [
        segment
        for segment in segments
        if re.search(r"\w", segment.text) and not _MUSIC_MARKERS.match(segment.text)
    ]

    merged: list[Segment] = []
    for segment in usable:
        if (
            merged
            and segment.start - merged[-1].end <= max_gap
            and len(merged[-1].text) + len(segment.text) + 1 <= max_chars
        ):
            merged[-1].end = max(merged[-1].end, segment.end)
            merged[-1].text = f"{merged[-1].text} {segment.text}".strip()
        else:
            merged.append(Segment(segment.start, segment.end, segment.text))
    return merged


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def run_tts(text: str, reference: str | None, language: str, engine: str, speed: float, progress=gr.Progress()):
    """Voice cloning tab."""
    job = workdir() / f"tts-{uuid.uuid4().hex[:10]}"
    job.mkdir(parents=True, exist_ok=True)

    try:
        payload = (text or "").strip()
        if not payload:
            raise StudioError("Matn kiriting.")
        if len(payload) > MAX_TEXT_CHARS:
            raise StudioError(f"Matn juda uzun: {len(payload)} belgi (chegara {MAX_TEXT_CHARS}).")

        if engine_type == "xtts":
            if language not in XTTS_LANGUAGES:
                raise StudioError(f"XTTS-v2 bu tilni bilmaydi: `{language}`. "
                                  "Uzbek tili uchun `uz` deb belgilang.")
            if not reference:
                raise StudioError("Ovoz namunasi yuklang yoki mikrofon bilan yozib oling "
                                  "(XTTS vaqtinchalik kerakli).")

        _progress(progress, 0.03, "Ovoz namunasi tozalanmoqda…")
        if engine_type == "navoiy":
            speaker = normalise_reference(reference or "", job / "reference.wav")
        else:
            speaker = None

        with _tts_job:  # one heavy job at a time on 2 vCPU
            output = job / f"vox-{_slug(language)}.wav"
            if engine_type == "navoiy":
                _synthesise_navoiy(payload, speaker, speed or 1.0, output, progress)
            else:
                synthesise(payload, speaker, language, float(speed or 1.0), output, progress, (0.05, 0.97))

        seconds = media_duration(output)
        log = (
            f"**Tayyor.** {len(payload)} belgi → {seconds:.1f} s audio · "
            f"til `{language}` · usul `{engine_type}` · tezlik {speed:.2f}×"
        )
        return str(output), log
    except StudioError as error:
        return None, f"⚠️ {_say(str(error))}"
    except Exception as error:  # noqa: BLE001 - report, never crash the Space
        print(f"[tts] unexpected: {error!r}", flush=True)
        return None, f"⚠️ Kutilmagan xato: `{type(error).__name__}: {_say(str(error))}`"


def run_dubbing(
    video: str | None,
    reference: str | None,
    source_language: str,
    target_language: str,
    whisper_size: str,
    progress=gr.Progress(),
):
    """Video dubbing tab: extract → transcribe → translate → clone → remux."""
    if not video:
        return None, None, [], "⚠️ Avval video yuklang."
    supported_targets = set(XTTS_LANGUAGES) | {"uz"}
    if target_language not in supported_targets:
        return None, None, [], "⚠️ Maqsad tili XTTS-v2 ro'yxatidan yoki O'zbekcha bo'lishi kerak."

    job = workdir() / f"dub-{uuid.uuid4().hex[:10]}"
    job.mkdir(parents=True, exist_ok=True)
    started = time.time()

    try:
        duration = media_duration(video)
        if duration <= 0:
            raise StudioError("Videoni o'qib bo'lmadi — format qo'llab-quvvatlanmaydi.")
        if duration > MAX_VIDEO_SECONDS:
            raise StudioError(
                f"Video juda uzun: {duration / 60:.1f} daqiqa. "
                f"Bepul CPU Space'da chegara {MAX_VIDEO_SECONDS / 60:.0f} daqiqa — "
                "avval qisqaroq parcha sinab ko'ring."
            )

        with _dub_job:  # serialise: CPU inference must not run in parallel
            _progress(progress, 0.02, "Ovoz ajratilmoqda (ffmpeg)…")
            audio16 = extract_audio(video, job / "source-16k.wav")

            _progress(progress, 0.05, "Ovoz namunasi tayyorlanmoqda…")
            speaker = normalise_reference(reference or audio16, job / "reference.wav")

            _progress(progress, 0.10, f"Matn aniqlanmoqda (whisper {whisper_size})…")
            segments, detected = transcribe(
                audio16,
                None if source_language in ("auto", "", None) else source_language,
                whisper_size,
                progress,
                (0.10, 0.45),
            )
            if not segments:
                raise StudioError("Videoda nutq topilmadi — ovoz yo'q yoki juda shovqinli.")

            source_code = detected or source_language
            if source_code not in FLORES_CODES:
                raise StudioError(
                    f"Manba tili aniqlanmadi (`{source_code}`). Manba tilini qo'lda tanlang."
                )

            if target_language != source_code:
                _progress(progress, 0.46, "Tarjima qilinmoqda (NLLB-200)…")
                translated = translate_all(
                    [segment.text for segment in segments],
                    source_code,
                    target_language,
                    progress,
                    (0.46, 0.62),
                )
                for segment, text in zip(segments, translated):
                    segment.translated = text

            total = len(segments)
            clips: list[tuple[float, Path]] = []
            for index, segment in enumerate(segments):
                span_start = 0.62 + (0.86 - 0.62) * (index / total)
                span_end = 0.62 + (0.86 - 0.62) * ((index + 1) / total)
                raw = job / f"raw-{index:04d}.wav"
                fitted = job / f"fit-{index:04d}.wav"
                _progress(
                    progress, span_start,
                    f"Ovoz klonlanmoqda: {index + 1}/{total}",
                )
                synthesise(segment.spoken, speaker, target_language, 1.0, raw, progress, (span_start, span_end))
                segment.audio = fit_clip(raw, fitted, max(segment.end - segment.start, 0.3))
                clips.append((segment.start, segment.audio))

            _progress(progress, 0.88, "Yangi audio trek yig'ilmoqda…")
            track = build_dub_track(clips, duration, job / "dubbed.wav")

            _progress(progress, 0.94, "Video qayta yig'ilmoqda (ffmpeg)…")
            output_video = mux_video(video, track, job / "dubbed.mp4")
            subtitles = write_srt(segments, job / "dubbed.srt", translated=bool(segments[0].translated))

        rows = [
            [index, f"{segment.start:6.1f}s", f"{segment.end:6.1f}s", segment.text, segment.translated or "—"]
            for index, segment in enumerate(segments, start=1)
        ]
        elapsed = time.time() - started
        log = (
            f"**Tayyor.** {total} satr · {duration:.1f} s video · "
            f"`{source_code}` → `{target_language}` · whisper `{whisper_size}` · "
            f"jami {elapsed / 60:.1f} daqiqa.\n\n"
            "Eslatma: musiqali fon saqlanmaydi — dublyaj ovozi asl trekni almashtiradi."
        )
        return str(output_video), str(subtitles), rows, log

    except StudioError as error:
        return None, None, [], f"⚠️ {_say(str(error))}"
    except Exception as error:  # noqa: BLE001 - report, never crash the Space
        print(f"[dub] unexpected: {error!r}", flush=True)
        return None, None, [], f"⚠️ Kutilmagan xato: `{type(error).__name__}: {_say(str(error))}`"


def warm_up(progress=gr.Progress()):
    """Load XTTS-v2 on demand (the startup thread usually beat you to it)."""
    try:
        _progress(progress, 0.1, "XTTS-v2 yuklanmoqda (birinchi marta ~1.9 GB)…")
        load_xtts()
        return f"✅ XTTS-v2 tayyor · `{package_version('coqui-tts')}`"
    except StudioError as error:
        return f"⚠️ {_say(str(error))}"


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "o'rnatilmagan"


def status_markdown() -> str:
    ffmpeg = shutil.which("ffmpeg")
    torch_info = "o'rnatilmagan"
    try:
        import torch

        torch_info = f"{torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
    except Exception:  # noqa: BLE001
        pass

    from platform import python_version

    lines = [
        "| Nima | Holat |",
        "| --- | --- |",
        f"| Python | {python_version()} |",
        f"| ffmpeg | {'✅' if ffmpeg else '❌ yo‘q'} `{ffmpeg or ''}` |",
        f"| torch | {torch_info} |",
        f"| coqui-tts | {package_version('coqui-tts')} |",
        f"| faster-whisper | {package_version('faster-whisper')} |",
        f"| transformers | {package_version('transformers')} |",
        f"| XTTS-v2 modeli | {'✅ yuklangan' if _xtts is not None else '⏳ hali yuklanmagan'} |",
        f"| Whisper modellari | {', '.join(sorted(_whisper)) or '—'} |",
        f"| Video chegarasi | {MAX_VIDEO_SECONDS / 60:.0f} daqiqa |",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
INTRO = """
# 🎙️ VoxStudio — ovoz klonlash va video dublyaj

*En* model yo'q, **hartaki uchun bor**. Telefon uchun mo'ljallangan Gradio
nashri: barcha ish **lokal modellar**da bo'ladi — narxli paqaviy API yo'q.

## Qanday ishlaydi

1. **Ovoz klonlash** — 3–15 sekundlik namuna (yuklash yoki telefon mikrofoni)
   beriladi va istalgan matn o'sha ovozda o'qiydi. O'zbekcha bo'lsa "Neutral
   (standart)" yoki "Ekspressiv / hisobot (Navoiy)" deb usul tanlang.
2. **Video dublyaj** — video → matn (vaqt belgilari bilan) → tarjima →
   har bir satr klonlangan ovozda → video qayta yig'iladi. Ovoz namunasi bo'sh
   qoldirilsa, videoning o'z ovozi klonlanadi. Subtitr `.srt` ham beriladi.

## Qaysi model qanday tanlang

| Til | Model | Ovoz |
| --- | --- | --- |
| Inglizcha, Ruscha, Turkcha, Espancha, Fransuzcha, … (17) | Coqui XTTS-v2 | Klonlash: asl ovoz |
| **O'zbekcha** | Navoiy TTS (CosyVoice2-0.5B) | Neutral yoki ekspressiv |
| Qo'shimcha til (uzbek o'rniga) | — (XTTS-17ga o'tkazish lozim) | — |

## Cheklovlar

* Bepul `cpu-basic`: 2 vCPU / 16 GB, video cheklovi 10 daqiqa.
* Musiqali fon saqlanmaydi; disk doimiy emas, modellar restartdan keyin
  yana yuklanadi.
* Tavsiya: boshlang `VOX_PRELOAD=0` deb tez yuklashni og'izing; optimallikda
  `VOX_PRELOAD=1` (standart).

## Litsenziya

* XTTS-v2 — **Coqui Public Model License (CPML)**: tijoriy bo'lmagan.
* Navoiy TTS — **Apache-2.0**.
"""



DUB_HELP = """
**Quvur:** ffmpeg (ovoz ajratish) → whisper (vaqt belgilari bilan matn) → NLLB
(tarjima) → XTTS-v2 (har bir satr klonlanadi va o'z vaqtiga moslanadi) → ffmpeg
(video qayta yig'iladi).

* Ovoz namunasi bo'sh qoldirilsa, **videoning o'z ovozi** namuna bo'ladi — ya'ni
  dublyaj asl qahramonning ovozi bilan chiqadi.
* **Cheklov:** XTTS-v2 17 tilni biladi va **o'zbekcha gapira olmaydi**. O'zbek
  videoni boshqa tilga o'girish mumkin (`uz` manba sifatida ishlaydi), lekin
  natija yuqoridagi 17 tildan birida bo'ladi.
* Musiqali fon ajratilmaydi (Demucs bepul CPU'da juda sekin) — yangi ovoz asl
  trekni almashtiradi.
"""

with gr.Blocks(title="VoxStudio · HF Spaces") as demo:
    gr.Markdown(INTRO)

    with gr.Tabs():
        # ------------------------------------------------------------------ #
        with gr.Tab("🗣️ Ovoz klonlash"):
            gr.Markdown(TTS_HELP)
            with gr.Row():
                with gr.Column(scale=3):
                    tts_text = gr.Textbox(
                        label="Matn",
                        lines=6,
                        max_lines=14,
                        placeholder="Klonlangan ovoz aytadigan matnni yozing…",
                    )
                    tts_reference = gr.Audio(
                        label="Ovoz namunasi (yuklang yoki mikrofon bilan yozing)",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )
                    with gr.Row():
                        tts_language = gr.Dropdown(
                            label="Til",
                            choices=engine_choices(),
                            value="en",
                        )
                        tts_engine = gr.Radio(
                            label="Ovoz usuli",
                            choices=[("Neutral (standart)", "neutral"),
                                     ("Ekspressiv / hisobot (Navoiy)", "expressiv")],
                            value="neutral",
                        )
                        tts_speed = gr.Slider(
                            label="Tezlik",
                            minimum=0.6, maximum=1.6, step=0.05, value=1.0,
                        )
                    tts_button = gr.Button("Ovoz yaratish", variant="primary")
                with gr.Column(scale=2):
                    tts_output = gr.Audio(label="Natija", type="filepath", autoplay=False)
                    tts_log = gr.Markdown()

            tts_button.click(
                run_tts,
                inputs=[tts_text, tts_reference, tts_language, tts_engine, tts_speed],
                outputs=[tts_output, tts_log],
                api_name="tts",
            )

        # ------------------------------------------------------------------ #
        with gr.Tab("🎬 Video dublyaj"):
            gr.Markdown(DUB_HELP)
            with gr.Row():
                with gr.Column(scale=3):
                    dub_video_input = gr.Video(label="Video (qisqa parcha bilan boshlang)")
                    dub_reference = gr.Audio(
                        label="Ovoz namunasi (ixtiyoriy — bo'sh bo'lsa videoning ovozi ishlatiladi)",
                        sources=["upload", "microphone"],
                        type="filepath",
                    )
                    with gr.Row():
                        dub_source = gr.Dropdown(
                            label="Manba tili", choices=WHISPER_CHOICES, value="auto"
                        )
                        dub_target = gr.Dropdown(
                            label="Maqsad tili", choices=XTTS_CHOICES, value="en"
                        )
                        dub_whisper = gr.Dropdown(
                            label="Whisper sifati",
                            choices=WHISPER_SIZES,
                            value="small",
                            info="tiny/base = tez, small = balans, medium = aniq (sekin)",
                        )
                    dub_button = gr.Button("Dublyaj qilish", variant="primary")
                    gr.Markdown(
                        "⏱️ Taxminan: har 1 daqiqa video uchun 2–5 daqiqa ishlov "
                        f"(whisper `small`, bepul CPU). Chegara: {MAX_VIDEO_SECONDS / 60:.0f} daqiqa."
                    )
                with gr.Column(scale=4):
                    dub_video_output = gr.Video(label="Dublyaj qilingan video")
                    dub_srt = gr.File(label="Subtitr (SRT)")
                    dub_table = gr.Dataframe(
                        headers=["#", "Boshlanish", "Tugash", "Asl matn", "Tarjima"],
                        datatype=["number", "str", "str", "str", "str"],
                        wrap=True,
                        interactive=False,
                        label="Satrlar",
                    )
                    dub_log = gr.Markdown()

            dub_button.click(
                run_dubbing,
                inputs=[dub_video_input, dub_reference, dub_source, dub_target, dub_whisper],
                outputs=[dub_video_output, dub_srt, dub_table, dub_log],
                api_name="dub",
            )

        # ------------------------------------------------------------------ #
        with gr.Tab("⚙️ Holat va yordam"):
            gr.Markdown(
                "Birinchi marta ishlatganda modellar yuklanadi (XTTS-v2 ≈ 1.9 GB, "
                "NLLB ≈ 2.4 GB). Space qayta ishga tushsa, ular yana yuklanadi — "
                "bepul Space'ning diski doimiy emas."
            )
            warm_button = gr.Button("XTTS-v2 ni hozir yuklash")
            warm_log = gr.Markdown()
            status = gr.Markdown(status_markdown())
            refresh = gr.Button("Holatni yangilash")
            warm_button.click(warm_up, outputs=warm_log, api_name="warmup")
            refresh.click(status_markdown, outputs=status)
            gr.Markdown(
                "**Litsenziya:** XTTS-v2 Coqui Public Model License (CPML) ostida — "
                "**tijoriy bo'lmagan** foydalanish uchun. `COQUI_TOS_AGREED=1` muhit "
                "o'zgaruvchisi shu shartni bildiradi."
            )


def _preload() -> None:
    """Warm XTTS-v2 in the background so the first user request is fast."""
    try:
        load_xtts()
    except Exception as error:  # noqa: BLE001 - the UI reports it on demand
        print(f"[preload] XTTS yuklanmadi: {error}", flush=True)


if PRELOAD:
    threading.Thread(target=_preload, name="preload-xtts", daemon=True).start()


def _resolve_tts_engine(language: str, engine: str) -> tuple[str, str]:
    """``engine`` selects the voice, ``language`` selects the language.

    * "en" ... "hu" -> XTTS-v2 (17 languages, voice cloning)
    * "uz"/"uz-clone" -> Navoiy TTS (CosyVoice2, Uzbek, neutral)
    """
    if language in ("uz", "uz-clone", "uznx"):
        return ("navoiy", "uz")
    return ("xtts", language)


def _synthesise_navoiy(text: str, speaker: Path | None, speed: float, destination: Path,
                       progress=gr.Progress()) -> Path:
    """Consume the Uzbek engine (CosyVoice2 + Navoiy checkpoint) via transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise StudioError(
            "Navoiy TTS isunmoqda: transformers/torch o'rnatilmagan. "
            "Hf Space'dagi requirements.txt ga cosyvoice qo'shmoqchi bo'lsang."
        ) from exc

    torch.set_num_threads(2)
    print("[uz-tts] loading base + checkpoint …", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "FunAudioLLM/CosyVoice2-0.5B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "FunAudioLLM/CosyVoice2-0.5B", use_fast=True
        )
        checkpoint = torch.load("emotion_600h_joint.pt", map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    except Exception as exc:  # noqa: BLE001
        raise StudioError(
            "Navoiy TTS yuklanmadi. Hipermashina vs CosyVoice2-0.5B + "
            "emotion_600h_joint.pt ni yuklang."
        ) from exc

    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice = AutoModel(model_dir="/tmp/voxstudio/pretrained_models/CosyVoice2-0.5B")

    print("[uz-tts] synthesising …", flush=True)
    speech = None
    for _i, _j in enumerate(cosyvoice.inference_zero_shot(text, "", str(speaker))):
        speech = _j["tts_speech"]
    if speech is None:
        raise StudioError("Navoiy TTS ishlamadi — reference danomalasi tekshiring.")

    torchaudio.save(str(destination), torch.tensor(speech), sample_rate)
    return destination

demo.queue().launch(theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"))
