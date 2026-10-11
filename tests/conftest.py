"""Test configuration.

Environment variables are set *before* the application is imported so the whole
process (settings cache, SQLite engine, storage roots) points at a throwaway
data directory instead of the real ``./data`` workspace.
"""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="voxstudio-tests-"))

os.environ["VOX_ENVIRONMENT"] = "test"
os.environ["VOX_DEBUG"] = "false"
os.environ["VOX_LOG_LEVEL"] = "WARNING"
os.environ["VOX_DATA_DIR"] = str(TEST_DATA_DIR)
os.environ["VOX_DATABASE_URL"] = f"sqlite:///{(TEST_DATA_DIR / 'test.db').as_posix()}"
os.environ["VOX_TASK_RETENTION_SECONDS"] = "0"
os.environ["VOX_ENABLE_BACKGROUND_SEPARATION"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


def make_wav_bytes(seconds: float = 0.5, rate: int = 8000) -> bytes:
    """Build a tiny but structurally valid mono 16-bit PCM WAV file."""
    frames = int(seconds * rate)
    payload = b"\x00\x00" * frames
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(payload))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(payload))
    )
    return header + payload


SAMPLE_SRT = """1
00:00:00,000 --> 00:00:02,500
Salom, dunyo!

2
00:00:02,500 --> 00:00:05,000
Bu ikkinchi qator.
"""


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def audio_file() -> tuple[str, bytes, str]:
    return ("reference.wav", make_wav_bytes(), "audio/wav")


@pytest.fixture()
def video_file() -> tuple[str, bytes, str]:
    # ffmpeg is not required to *upload* a video; the pipeline is what needs it.
    return ("clip.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 512, "video/mp4")
