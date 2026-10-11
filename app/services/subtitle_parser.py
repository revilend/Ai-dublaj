"""Parse and serialise SubRip (``.srt``) and WebVTT (``.vtt``) subtitles.

Only the subset needed by the dubbing pipeline is implemented, but it is
tolerant of the usual real-world quirks: BOMs, CRLF, missing indices,
positioning suffixes and ``-->`` alignment tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.exceptions import ValidationError

# SRT uses ``HH:MM:SS,mmm``; WebVTT also allows the short ``MM:SS.mmm`` form.
_TIMESTAMP = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{2}):(?P<s>\d{2})(?:[,.](?P<ms>\d{1,3}))?"
)
_ARROW = re.compile(r"-->")
_HTML_TAG = re.compile(r"<[^>]+>")


@dataclass(slots=True)
class SubtitleCue:
    index: int
    start: float
    end: float
    text: str


def _to_seconds(match: re.Match[str]) -> float:
    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = match.group("ms") or "0"
    millis = int(millis.ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _clean_text(lines: list[str]) -> str:
    joined = " ".join(line.strip() for line in lines if line.strip())
    joined = _HTML_TAG.sub("", joined)
    return " ".join(joined.split())


def parse_subtitles(content: str) -> list[SubtitleCue]:
    """Parse SRT or VTT content into ordered cues.

    Raises:
        ValidationError: when no usable cue could be parsed.
    """
    if not content or not content.strip():
        raise ValidationError("The subtitle file is empty.")

    normalised = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalised.split("\n")

    cues: list[SubtitleCue] = []
    pending: list[str] = []
    current: dict[str, object] = {}

    def flush() -> None:
        nonlocal pending, current
        if "start" in current and pending:
            cues.append(
                SubtitleCue(
                    index=len(cues),
                    start=float(current["start"]),  # type: ignore[arg-type]
                    end=float(current["end"]),  # type: ignore[arg-type]
                    text=_clean_text(pending),
                )
            )
        pending = []
        current = {}

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush()
            continue

        if _ARROW.search(line):
            flush()
            start_part, _, end_part = line.partition("-->")
            end_part = end_part.strip().split(" ")[0]
            start_match = _TIMESTAMP.search(start_part)
            end_match = _TIMESTAMP.search(end_part)
            if not start_match or not end_match:
                continue
            current = {
                "start": _to_seconds(start_match),
                "end": _to_seconds(end_match),
            }
            continue

        if line.isdigit() and not pending:
            # SubRip cue index (ignored — order is positional).
            continue
        if line.upper().startswith("WEBVTT") or line.upper().startswith("NOTE"):
            continue

        if "start" in current:
            pending.append(line)

    flush()

    cues = [cue for cue in cues if cue.text and cue.end > cue.start]
    if not cues:
        raise ValidationError("No valid subtitle cues were found in the uploaded file.")
    cues.sort(key=lambda cue: cue.start)
    for position, cue in enumerate(cues):
        cue.index = position
    return cues


def _format_timestamp(seconds: float) -> str:
    millis = int(round(max(0.0, seconds) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(cues: list[SubtitleCue]) -> str:
    blocks = [
        f"{index + 1}\n{_format_timestamp(cue.start)} --> {_format_timestamp(cue.end)}\n{cue.text}"
        for index, cue in enumerate(cues)
    ]
    return "\n\n".join(blocks) + "\n"


def estimate_cue_from_duration(index: int, start: float, duration: float, text: str) -> SubtitleCue:
    return SubtitleCue(index=index, start=start, end=start + max(0.05, duration), text=text)


__all__ = ["SubtitleCue", "estimate_cue_from_duration", "parse_subtitles", "to_srt"]
