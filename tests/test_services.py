"""Unit tests for the standalone services (no external binaries required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import ServiceUnavailableError, ValidationError
from app.schemas.task import TaskState
from app.services.audio_processor import AudioProcessor
from app.services.diarization_service import DiarizationService, SpeakerTurn
from app.services.subtitle_parser import parse_subtitles, to_srt
from app.services.task_manager import TaskCancelled, TaskContext, TaskManager
from app.services.tts_engine import TTSEngine, TTSValidationError, normalize_language

SRT = """1
00:00:01,000 --> 00:00:03,500
Hello <i>there</i>

2
00:01:02,250 --> 00:01:06,000
Second line
wrapped over two rows
"""

VTT = """WEBVTT

00:00.500 --> 00:02.000
VTT cue one

00:02.000 --> 00:04.000
VTT cue two
"""


def test_parse_srt_strips_markup_and_newlines() -> None:
    cues = parse_subtitles(SRT)
    assert len(cues) == 2
    assert cues[0].text == "Hello there"
    assert cues[0].start == 1.0
    assert cues[0].end == 3.5
    assert cues[1].text == "Second line wrapped over two rows"
    assert cues[1].start == pytest.approx(62.25)


def test_parse_vtt() -> None:
    cues = parse_subtitles(VTT)
    assert [cue.text for cue in cues] == ["VTT cue one", "VTT cue two"]
    assert cues[0].start == pytest.approx(0.5)


def test_parse_rejects_empty_and_garbage() -> None:
    with pytest.raises(ValidationError):
        parse_subtitles("   ")
    with pytest.raises(ValidationError):
        parse_subtitles("not a subtitle file at all")


def test_to_srt_round_trip() -> None:
    cues = parse_subtitles(SRT)
    reparsed = parse_subtitles(to_srt(cues))
    assert [cue.text for cue in reparsed] == [cue.text for cue in cues]


def test_chunk_text_respects_limit() -> None:
    text = " ".join(["This is a sentence."] * 40)
    chunks = TTSEngine.chunk_text(text, max_chars=120)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_normalize_language_aliases() -> None:
    assert normalize_language("EN") == "en"
    assert normalize_language("zh") == "zh-cn"
    assert normalize_language("pt-BR") == "pt"
    with pytest.raises(TTSValidationError):
        normalize_language("klingon")


def test_unsupported_language_is_rejected_with_a_reason() -> None:
    """Uzbek is offered by the studio UI but XTTS-v2 cannot speak it."""
    with pytest.raises(TTSValidationError) as excinfo:
        normalize_language("uz")
    assert excinfo.value.details["reason"] == "not_trained_by_xtts"
    assert "Uzbek" in excinfo.value.message


def test_pitch_ratio_math() -> None:
    assert AudioProcessor.pitch_ratio(0) == pytest.approx(1.0)
    assert AudioProcessor.pitch_ratio(12) == pytest.approx(2.0)
    assert AudioProcessor.pitch_ratio(-12) == pytest.approx(0.5)
    # values outside the supported range are clamped, not extrapolated
    assert AudioProcessor.pitch_ratio(48) == pytest.approx(2.0)
    assert AudioProcessor.pitch_ratio(-48) == pytest.approx(0.5)


def test_atempo_chain_stays_within_ffmpeg_limits() -> None:
    for ratio in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0):
        chain = AudioProcessor._atempo_chain(ratio)
        values = [float(part.split("=")[1]) for part in chain.split(",")]
        assert all(0.5 <= value <= 2.0 for value in values), chain
        product = 1.0
        for value in values:
            product *= value
        assert product == pytest.approx(ratio, rel=1e-6)


def test_shift_pitch_requires_ffmpeg(tmp_path) -> None:
    processor = AudioProcessor()
    source = tmp_path / "in.wav"
    source.write_bytes(b"RIFF")  # contents are irrelevant when ffmpeg is absent
    if processor.ffmpeg_available:  # pragma: no cover - only on machines with ffmpeg
        result = processor.shift_pitch(source, tmp_path / "out.wav", 0.0)
        assert result.exists()
    else:
        with pytest.raises(ServiceUnavailableError):
            processor.shift_pitch(source, tmp_path / "out.wav", 5.0)


def test_separate_stems_normalises_demucs_output_names(tmp_path) -> None:
    """Demucs emits ``vocals.wav`` / ``no_vocals.wav``; the pipeline must
    receive them as ``vocals.wav`` and ``background.wav`` in the destination."""
    from app.core.config import Settings

    class FakeDemucs(AudioProcessor):
        def __init__(self) -> None:
            super().__init__(Settings(enable_background_separation=True))
            self.commands: list[list[str]] = []

        @property
        def background_separation_available(self) -> bool:  # pretend demucs is installed
            return True

        def _run(self, args, *, timeout=0, action="process media"):  # type: ignore[override]
            self.commands.append(list(args))
            out = Path(args[args.index("--out") + 1])
            model = args[args.index("-n") + 1]
            produced = out / model / Path(args[-1]).stem
            produced.mkdir(parents=True, exist_ok=True)
            (produced / "vocals.wav").write_bytes(b"VOCALS")
            (produced / "no_vocals.wav").write_bytes(b"BACKING")
            return None

    processor = FakeDemucs()
    source = tmp_path / "original.wav"
    source.write_bytes(b"RIFF")

    stems = processor.separate_stems(source, tmp_path / "out")

    assert stems.ok
    assert stems.vocals is not None and stems.vocals.name == "vocals.wav"
    assert stems.background is not None and stems.background.name == "background.wav"
    assert stems.vocals.read_bytes() == b"VOCALS"
    assert stems.background.read_bytes() == b"BACKING"
    # Two-stem mode must be requested so the model only splits dialogue vs. bed.
    assert "--two-stems=vocals" in processor.commands[0]
    # The intermediate Demucs tree is cleaned up, leaving only the named stems.
    assert not (tmp_path / "out" / "_demucs").exists()


def test_separate_stems_is_empty_when_disabled(tmp_path) -> None:
    from app.core.config import Settings

    processor = AudioProcessor(Settings(enable_background_separation=False))
    stems = processor.separate_stems(tmp_path / "original.wav", tmp_path / "out")
    assert stems.vocals is None and stems.background is None
    assert stems.ok is False


def test_diarization_is_honestly_unavailable_without_pyannote(tmp_path) -> None:
    """No pyannote => no speakers guessed; every span is left unlabelled."""
    from app.services.diarization_service import DiarizationService

    service = DiarizationService()
    available, reason = service.dependency_status()
    if available:  # pragma: no cover - only where pyannote + a token exist
        pytest.skip("pyannote is configured in this environment")
    assert reason
    source = tmp_path / "vocals.wav"
    source.write_bytes(b"RIFF")
    assert service.assign_speakers(source, [(0.0, 1.0), (1.0, 2.0)]) == [None, None]


def test_diarization_assigns_the_overlapping_speaker() -> None:
    from app.services.diarization_service import DiarizationService, SpeakerTurn

    class FakeDiarization(DiarizationService):
        def timeline(self, audio_path):  # type: ignore[override]
            return [
                SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00"),
                SpeakerTurn(start=2.0, end=4.0, speaker="SPEAKER_01"),
            ]

    service = FakeDiarization()
    labels = service.assign_speakers(
        "unused.wav",
        [
            (0.1, 1.9),  # wholly inside SPEAKER_00
            (2.2, 3.8),  # wholly inside SPEAKER_01
            (0.5, 2.5),  # straddles both; 1.5s vs 0.5s overlap
            (7.0, 8.0),  # no overlap at all
        ],
    )
    assert labels == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", None]


def test_task_manager_lifecycle() -> None:
    manager = TaskManager(retention_seconds=0)
    record = manager.create("unit task", kind="test")
    assert manager.get(record.id).status.value == "pending"

    manager.update(record.id, progress=45.0, message="halfway")
    snapshot = manager.get(record.id)
    assert snapshot.progress == 45.0
    assert snapshot.message == "halfway"

    # progress is clamped to the 0-100 contract
    manager.update(record.id, progress=900)
    assert manager.get(record.id).progress == 100.0

    manager.update(record.id, progress=-10)
    assert manager.get(record.id).progress == 0.0

    stats = manager.stats()
    assert stats["total"] == 1


def test_task_context_reports_and_completes() -> None:
    manager = TaskManager(retention_seconds=0)
    record = manager.create("ctx task")
    ctx = TaskContext(manager, record.id)

    ctx.report(10, "starting")
    ctx.complete({"ok": True}, message="done")

    stored = manager.get(record.id)
    assert stored.status.value == "completed"
    assert stored.progress == 100.0
    assert stored.result == {"ok": True}


def test_task_context_raise_if_cancelled() -> None:
    manager = TaskManager(retention_seconds=0)
    record = manager.create("cancel task")
    manager.update(record.id, status=TaskState.RUNNING)
    manager.cancel(record.id)

    ctx = TaskContext(manager, record.id)
    with pytest.raises(TaskCancelled):
        ctx.raise_if_cancelled()


def test_clone_voice_signature_is_stable() -> None:
    """The four documented positional parameters must not move."""
    import inspect

    params = list(inspect.signature(TTSEngine.clone_voice).parameters)
    assert params[:5] == ["self", "text", "reference_audio", "language", "speed"]
    assert "pitch" in params


def test_tts_engine_degrades_without_runtime() -> None:
    engine = TTSEngine()
    available, reason = engine.dependency_status()
    if available:  # pragma: no cover - only when the AI runtime is installed
        pytest.skip("Coqui TTS is installed in this environment")
    assert reason
    assert engine.resolve_device() in {"cpu", "cuda"}
