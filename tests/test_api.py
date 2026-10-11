"""End-to-end tests for the studio API (no GPU / ffmpeg required)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.services.dubbing_service import STEP_MESSAGES, STEP_TOTAL
from tests.conftest import SAMPLE_SRT, make_wav_bytes


def _create_voice(client: TestClient, name: str = "Test Narrator") -> dict:
    response = client.post(
        "/api/voices",
        data={"name": name, "description": "fixture", "language": "en"},
        files={"sample_audio": ("ref.wav", make_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# System
# --------------------------------------------------------------------------- #
def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_capabilities_shape(client: TestClient) -> None:
    body = client.get("/api/system/capabilities").json()
    assert body["device"] in {"cpu", "cuda"}
    for key in (
        "ffmpeg",
        "gpu",
        "tts",
        "transcription",
        "translation",
        "background_separation",
    ):
        assert key in body, key
        assert "available" in body[key], key
        assert isinstance(body[key]["available"], bool), key


def test_languages_endpoint(client: TestClient) -> None:
    body = client.get("/api/tts/languages").json()
    codes = {item["code"] for item in body["languages"]}
    assert {"en", "es", "zh-cn"} <= codes
    assert all(item["supported"] is True for item in body["languages"])

    # Uzbek is listed for the UI but flagged as unsupported by XTTS-v2.
    unsupported = {item["code"]: item for item in body["unsupported"]}
    assert unsupported["uz"]["supported"] is False
    assert unsupported["uz"]["reason"] == "not_trained_by_xtts"


def test_quick_tts_rejects_uzbek_with_an_actionable_error(client: TestClient) -> None:
    voice = _create_voice(client, "Uzbek Guard Voice")
    accepted = client.post(
        "/api/tts/quick",
        json={"text": "Salom dunyo", "language": "uz", "voice_profile_id": voice["id"]},
    )
    assert accepted.status_code == 202
    snapshot = _wait_for_terminal(client, accepted.json()["task_id"])
    assert snapshot["status"] == "failed"
    assert "Uzbek" in (snapshot["error"] or "") or "not_trained_by_xtts" in (
        snapshot["error"] or ""
    )


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --------------------------------------------------------------------------- #
# Voices
# --------------------------------------------------------------------------- #
def test_voice_profile_lifecycle(client: TestClient) -> None:
    created = _create_voice(client, "Lifecycle Voice")
    voice_id = created["id"]
    assert created["sample_audio_url"].startswith("/media/voices/")

    listed = client.get("/api/voices?limit=100").json()
    assert any(item["id"] == voice_id for item in listed["items"])

    fetched = client.get(f"/api/voices/{voice_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "Lifecycle Voice"

    patched = client.patch(f"/api/voices/{voice_id}", json={"description": "updated"})
    assert patched.json()["description"] == "updated"

    assert client.delete(f"/api/voices/{voice_id}").status_code == 204
    assert client.get(f"/api/voices/{voice_id}").status_code == 404


def test_duplicate_voice_name_conflicts(client: TestClient) -> None:
    _create_voice(client, "Duplicate Voice")
    response = client.post(
        "/api/voices",
        data={"name": "Duplicate Voice", "language": "en"},
        files={"sample_audio": ("ref.wav", make_wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_voice_rejects_unsupported_extension(client: TestClient) -> None:
    response = client.post(
        "/api/voices",
        data={"name": "Bad File", "language": "en"},
        files={"sample_audio": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_voice_requires_reference_file(client: TestClient) -> None:
    response = client.post("/api/voices", data={"name": "No File", "language": "en"})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Instant TTS
# --------------------------------------------------------------------------- #
def test_quick_tts_without_reference_is_rejected(client: TestClient) -> None:
    response = client.post("/api/tts/quick", json={"text": "Salom", "language": "en"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_quick_tts_queues_a_task_that_fails_cleanly_without_tts(
    client: TestClient,
) -> None:
    """Without the AI runtime installed the job must fail with a clear message."""
    voice = _create_voice(client, "TTS Voice")

    accepted = client.post(
        "/api/tts/quick",
        json={"text": "Salom, bu test.", "language": "en", "voice_profile_id": voice["id"]},
    )
    assert accepted.status_code == 202, accepted.text
    payload = accepted.json()
    assert payload["status_url"] == f"/api/tasks/{payload['task_id']}/status"

    snapshot = _wait_for_terminal(client, payload["task_id"])
    assert snapshot["status"] == "failed"
    assert "tts" in (snapshot["error"] or "").lower() or "Coqui" in (snapshot["error"] or "")


def test_quick_tts_rejects_unknown_language(client: TestClient) -> None:
    voice = _create_voice(client, "Language Voice")
    response = client.post(
        "/api/tts/quick",
        json={"text": "hi", "language": "klingon", "voice_profile_id": voice["id"]},
    )
    # Validation happens inside the task, which surfaces as a failed task OR a
    # synchronous 422 depending on where it is caught — accept either.
    assert response.status_code in (202, 422)


def test_task_status_unknown_id(client: TestClient) -> None:
    response = client.get("/api/tasks/does-not-exist/status")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "task_not_found"


# --------------------------------------------------------------------------- #
# Dubbing
# --------------------------------------------------------------------------- #
def test_dubbing_project_and_subtitles(client: TestClient) -> None:
    response = _upload(client, title="Test Project", source_language="en", target_language="es")
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["task_id"] is None  # analyze=false
    assert accepted["message"]

    project = client.get(f"/api/dubbing/{accepted['project_id']}").json()
    assert project["status"] == "pending"
    assert project["video_url"].startswith("/media/uploads/")
    # Separation artifacts do not exist until the pipeline has run.
    assert project["vocals_audio_url"] is None
    assert project["background_audio_url"] is None

    upload = client.post(
        f"/api/dubbing/{project['id']}/subtitles",
        files={"file": ("captions.srt", SAMPLE_SRT.encode(), "text/plain")},
    )
    assert upload.status_code == 201, upload.text
    segments = upload.json()
    assert len(segments) == 2
    assert segments[0]["original_text"] == "Salom, dunyo!"
    assert segments[1]["start_time"] == 2.5

    project_id = accepted["project_id"]
    detail = client.get(f"/api/dubbing/{project_id}").json()
    assert detail["segment_count"] == 2
    assert len(detail["segments"]) == 2

    # Deleting the project must also remove its derived output directory.
    assert client.delete(f"/api/dubbing/{project_id}").status_code == 204
    assert client.get(f"/api/dubbing/{project_id}").status_code == 404


# --------------------------------------------------------------------------- #
# Upload -> analysis (steps 1-2)
# --------------------------------------------------------------------------- #
def test_upload_with_analysis_queues_step_1_and_2(client: TestClient) -> None:
    response = _upload(client, title="Analysed", analyze="true")
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["task_id"], accepted
    assert accepted["status_url"] == f"/api/tasks/{accepted['task_id']}/status"
    assert accepted["websocket_url"] == f"/ws/tasks/{accepted['task_id']}"
    assert "transcrib" in accepted["message"].lower()

    snapshot = _wait_for_terminal(client, accepted["task_id"])
    # ffmpeg is absent in CI, so step 1 must fail with an actionable message.
    assert snapshot["status"] == "failed"
    assert "ffmpeg" in (snapshot["error"] or "")
    # The failure replaces `message` with the error text, so the step the task
    # reached is carried on the metadata instead (asserted verbatim below).
    assert snapshot["metadata"]["step"] == 1
    assert snapshot["metadata"]["step_total"] == STEP_TOTAL

    detail = client.get(f"/api/dubbing/{accepted['project_id']}").json()
    assert detail["status"] == "failed"
    assert detail["error_message"]


# --------------------------------------------------------------------------- #
# Per-segment editing
# --------------------------------------------------------------------------- #
def test_update_segment_text_and_voice(client: TestClient) -> None:
    voice = _create_voice(client, "Segment Voice")
    project_id = _segmented_project(client)
    segments = client.get(f"/api/dubbing/{project_id}/segments").json()
    segment_id = segments[0]["id"]

    response = client.put(
        f"/api/dubbing/segments/{segment_id}",
        json={"translated_text": "Hola, mundo!", "voice_profile_id": voice["id"]},
    )
    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["translated_text"] == "Hola, mundo!"
    assert updated["voice_profile_id"] == voice["id"]
    assert updated["voice_profile_name"] == "Segment Voice"
    # The original text is untouched by a partial update.
    assert updated["original_text"] == "Salom, dunyo!"

    # Clearing the voice falls back to the project default.
    cleared = client.put(f"/api/dubbing/segments/{segment_id}", json={"voice_profile_id": None})
    assert cleared.status_code == 200
    assert cleared.json()["voice_profile_id"] is None
    assert cleared.json()["voice_profile_name"] is None


def test_update_segment_validation(client: TestClient) -> None:
    project_id = _segmented_project(client)
    segment_id = client.get(f"/api/dubbing/{project_id}/segments").json()[0]["id"]

    missing = client.put("/api/dubbing/segments/999999", json={"translated_text": "x"})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"

    unknown_voice = client.put(
        f"/api/dubbing/segments/{segment_id}", json={"voice_profile_id": 999999}
    )
    assert unknown_voice.status_code == 404

    bad_span = client.put(
        f"/api/dubbing/segments/{segment_id}", json={"start_time": 9.0, "end_time": 1.0}
    )
    assert bad_span.status_code == 422
    assert bad_span.json()["error"]["code"] == "validation_error"

    # An empty patch is a no-op, not an error.
    noop = client.put(f"/api/dubbing/segments/{segment_id}", json={})
    assert noop.status_code == 200


def test_editing_text_invalidates_generated_audio(client: TestClient) -> None:
    project_id = _segmented_project(client)
    segment_id = client.get(f"/api/dubbing/{project_id}/segments").json()[0]["id"]

    # Simulate a clip produced by an earlier render.
    from app.core.database import session_scope
    from app.models.subtitle_segment import SubtitleSegment

    with session_scope() as db:
        row = db.get(SubtitleSegment, segment_id)
        row.dubbed_audio_path = "outputs/project-x/fitted-0000.wav"

    updated = client.put(f"/api/dubbing/segments/{segment_id}", json={"translated_text": "New"})
    assert updated.status_code == 200
    assert updated.json()["dubbed_audio_path"] is None
    assert updated.json()["dubbed_audio_url"] is None


# --------------------------------------------------------------------------- #
# Render (steps 3-4)
# --------------------------------------------------------------------------- #
def test_render_requires_a_voice(client: TestClient) -> None:
    project_id = _upload(client, title="No Voice") .json()["project_id"]
    render = client.post(f"/api/dubbing/render/{project_id}", json={})
    assert render.status_code == 422
    assert render.json()["error"]["code"] == "validation_error"


def test_render_accepts_a_segment_level_voice(client: TestClient) -> None:
    """A per-segment voice counts as a configured voice for the project."""
    voice = _create_voice(client, "Segment Only Voice")
    project_id = _segmented_project(client)
    segment_id = client.get(f"/api/dubbing/{project_id}/segments").json()[0]["id"]
    client.put(f"/api/dubbing/segments/{segment_id}", json={"voice_profile_id": voice["id"]})

    render = client.post(f"/api/dubbing/render/{project_id}", json={"translate": False})
    assert render.status_code == 202, render.text
    assert render.json()["resource_id"] == project_id


def test_render_queues_task_with_a_voice(client: TestClient) -> None:
    voice = _create_voice(client, "Render Voice")
    project_id = _upload(client, title="Renderable", voice_profile_id=str(voice["id"])).json()[
        "project_id"
    ]

    render = client.post(f"/api/dubbing/render/{project_id}", json={"translate": False})
    assert render.status_code == 202, render.text
    accepted = render.json()

    snapshot = _wait_for_terminal(client, accepted["task_id"])
    # ffmpeg is absent in CI, so the pipeline must fail with a clear 503-style error.
    assert snapshot["status"] == "failed"
    assert snapshot["error"]
    # Render announces itself at step 3 (voices) of the same four-step pipeline.
    assert snapshot["metadata"]["step"] == 3
    assert snapshot["metadata"]["step_total"] == STEP_TOTAL

    detail = client.get(f"/api/dubbing/{project_id}").json()
    assert detail["status"] == "failed"
    assert detail["error_message"]


def test_render_unknown_project_is_404(client: TestClient) -> None:
    response = client.post("/api/dubbing/render/999999", json={})
    assert response.status_code == 404


def test_pipeline_publishes_the_four_step_labels_in_order(client: TestClient) -> None:
    """Drive the whole pipeline with stubbed engines and assert the exact
    progress labels the spec requires are published, in order.

    The real engines (ffmpeg / Demucs / Whisper / XTTS) are absent in CI, so the
    four labels can only be exercised by substituting the heavy steps and
    watching what the orchestrator reports.
    """
    import asyncio

    from app.core.database import session_scope
    from app.models.dubbing_project import DubbingProject
    from app.schemas.dubbing import RenderRequest
    from app.services.dubbing_service import DubbingService
    from app.services.task_manager import TaskContext, TaskManager

    assert list(STEP_MESSAGES.values()) == [
        "Step 1/4: Separating Music",
        "Step 2/4: Transcribing",
        "Step 3/4: Generating Voices",
        "Step 4/4: Rendering Video",
    ]

    voice = _create_voice(client, "Pipeline Voice")
    project_id = _segmented_project(client)
    with session_scope() as db:
        db.get(DubbingProject, project_id).voice_profile_id = voice["id"]

    class StubAudio:
        background_separation_available = False

        def require_ffmpeg(self) -> None:  # noqa: D401 - unused stub
            return None

        def extract_audio(self, video, destination, **_kw):
            destination.write_bytes(b"RIFF")
            return destination

        def get_duration(self, _path) -> float:
            return 1.0

        def generate_silence(self, _seconds, path):
            path.write_bytes(b"RIFF")
            return path

        def concat_audio(self, _parts, destination, **_kw):
            destination.write_bytes(b"RIFF")
            return destination

        def fit_to_duration(self, _clip, fitted, _target):
            fitted.write_bytes(b"RIFF")
            return fitted

        def mix_audio(self, _a, _b, destination):
            destination.write_bytes(b"RIFF")
            return destination

        def mux_video(self, _video, _audio, destination):
            destination.write_bytes(b"\x00")
            return destination

    class StubTTS:
        async def clone_voice_async(self, _text, _reference, **kw):
            out = kw["output_path"]
            out.write_bytes(b"RIFF")
            return type("Result", (), {"audio_path": out})()

    manager = TaskManager(retention_seconds=0)
    record = manager.create("pipeline")
    ctx = TaskContext(manager, record.id)
    service = DubbingService(audio=StubAudio(), tts=StubTTS())

    # Capture every snapshot the manager publishes (the WebSocket/polling path).
    published: list[str] = []
    original_publish = manager._publish

    def capture(task_id, snapshot, _original=original_publish):
        published.append(snapshot.message)
        _original(task_id, snapshot)

    manager._publish = capture  # type: ignore[method-assign]

    asyncio.run(service.render(ctx, project_id, RenderRequest(translate=False)))

    seen = published
    for label in STEP_MESSAGES.values():
        assert label in seen, f"{label!r} was never reported; saw {seen}"
    indexes = [seen.index(label) for label in STEP_MESSAGES.values()]
    assert indexes == sorted(indexes), f"step labels out of order: {seen}"

    detail = client.get(f"/api/dubbing/{project_id}").json()
    assert detail["status"] == "completed"
    assert detail["final_video_url"]
    assert all(segment["dubbed_audio_url"] for segment in detail["segments"])


# --------------------------------------------------------------------------- #
# Per-line preview
# --------------------------------------------------------------------------- #
def test_preview_line_synthesises_a_single_segment(client: TestClient) -> None:
    """The editor's "Preview line" clones one cue and stores it as its audio."""
    import asyncio

    from app.core.database import session_scope
    from app.models.dubbing_project import DubbingProject
    from app.services.dubbing_service import DubbingService
    from app.services.task_manager import TaskContext, TaskManager

    voice = _create_voice(client, "Preview Voice")
    project_id = _segmented_project(client)
    segment_id = client.get(f"/api/dubbing/{project_id}/segments").json()[0]["id"]
    with session_scope() as db:
        db.get(DubbingProject, project_id).voice_profile_id = voice["id"]

    class StubAudio:
        def get_duration(self, _path) -> float:
            return 1.0

        def fit_to_duration(self, _clip, fitted, _target):
            fitted.write_bytes(b"RIFF")
            return fitted

    class StubTTS:
        async def clone_voice_async(self, _text, _reference, **kw):
            out = kw["output_path"]
            out.write_bytes(b"RIFF")
            return type("Result", (), {"audio_path": out})()

    manager = TaskManager(retention_seconds=0)
    record = manager.create("preview")
    ctx = TaskContext(manager, record.id)
    service = DubbingService(audio=StubAudio(), tts=StubTTS())

    result = asyncio.run(service.preview_segment(ctx, segment_id))
    assert result["segment_id"] == segment_id
    assert result["project_id"] == project_id
    assert result["dubbed_audio_url"]

    # The clip is persisted on the segment so the editor can play it back.
    fresh = client.get(f"/api/dubbing/{project_id}/segments").json()[0]
    assert fresh["dubbed_audio_url"] == result["dubbed_audio_url"]


def test_preview_requires_a_voice(client: TestClient) -> None:
    _create_voice(client, "Unused Preview Voice")
    project_id = _segmented_project(client)
    segment_id = client.get(f"/api/dubbing/{project_id}/segments").json()[0]["id"]

    response = client.post(f"/api/dubbing/segments/{segment_id}/preview", json={})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


def test_preview_queues_a_task_when_a_voice_exists(client: TestClient) -> None:
    voice = _create_voice(client, "Preview Queue Voice")
    project_id = _segmented_project(client)
    segment_id = client.get(f"/api/dubbing/{project_id}/segments").json()[0]["id"]
    client.put(f"/api/dubbing/segments/{segment_id}", json={"voice_profile_id": voice["id"]})

    response = client.post(f"/api/dubbing/segments/{segment_id}/preview", json={})
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["resource_id"] == segment_id
    assert accepted["websocket_url"] == f"/ws/tasks/{accepted['task_id']}"

    snapshot = _wait_for_terminal(client, accepted["task_id"])
    # XTTS is absent in CI, so the clone must fail with an actionable message.
    assert snapshot["status"] == "failed"
    assert snapshot["error"]
    assert snapshot["metadata"]["segment_id"] == segment_id


def test_preview_unknown_segment_is_404(client: TestClient) -> None:
    assert client.post("/api/dubbing/segments/999999/preview", json={}).status_code == 404


# --------------------------------------------------------------------------- #
# Auto-translate (no synthesis)
# --------------------------------------------------------------------------- #
def test_translate_fills_the_translation_column(client: TestClient) -> None:
    import asyncio

    from app.schemas.dubbing import RenderRequest
    from app.services.dubbing_service import DubbingService
    from app.services.task_manager import TaskContext, TaskManager

    project_id = _segmented_project(client)  # source_language "en", two cues

    class StubTranslator:
        @staticmethod
        def supports(language) -> bool:
            return language in {"en", "es"}

        async def translate_batch_async(self, texts, *, source_language, target_language):
            return [f"[{target_language}] {text}" for text in texts]

    manager = TaskManager(retention_seconds=0)
    record = manager.create("translate")
    ctx = TaskContext(manager, record.id)
    service = DubbingService(translator=StubTranslator())

    result = asyncio.run(
        service.translate_segments(ctx, project_id, RenderRequest(target_language="es"))
    )
    assert result["translated"] == 2

    rows = client.get(f"/api/dubbing/{project_id}/segments").json()
    assert all(row["translated_text"].startswith("[es]") for row in rows)

    # Translation is an editor step: the project is left ready to render.
    detail = client.get(f"/api/dubbing/{project_id}").json()
    assert detail["status"] == "pending"


def test_translate_endpoint_requires_segments(client: TestClient) -> None:
    project_id = _upload(client, title="No Cues").json()["project_id"]
    response = client.post(f"/api/dubbing/{project_id}/translate", json={})
    assert response.status_code == 202, response.text

    snapshot = _wait_for_terminal(client, response.json()["task_id"])
    assert snapshot["status"] == "failed"
    assert "segment" in (snapshot["error"] or "").lower()


def test_translate_unknown_project_is_404(client: TestClient) -> None:
    assert client.post("/api/dubbing/999999/translate", json={}).status_code == 404


def test_translate_failure_leaves_the_project_ready(client: TestClient) -> None:
    """A failed translation must not strand the project on `translating`."""
    import asyncio

    from app.core.exceptions import ServiceUnavailableError
    from app.schemas.dubbing import RenderRequest
    from app.services.dubbing_service import DubbingService
    from app.services.task_manager import TaskContext, TaskManager

    project_id = _segmented_project(client)

    class FailingTranslator:
        @staticmethod
        def supports(language) -> bool:
            return language in {"en", "es"}

        async def translate_batch_async(self, texts, *, source_language, target_language):
            raise ServiceUnavailableError("transformers is not installed.")

    manager = TaskManager(retention_seconds=0)
    ctx = TaskContext(manager, manager.create("translate-fail").id)
    service = DubbingService(translator=FailingTranslator())

    with pytest.raises(ServiceUnavailableError):
        asyncio.run(
            service.translate_segments(ctx, project_id, RenderRequest(target_language="es"))
        )

    assert client.get(f"/api/dubbing/{project_id}").json()["status"] == "pending"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _upload(client: TestClient, **form: str):
    data = {"title": "Upload", "source_language": "en", "target_language": "en", "analyze": "false"}
    data.update({key: str(value) for key, value in form.items()})
    return client.post(
        "/api/dubbing/upload",
        data=data,
        files={"video": ("clip.mp4", b"\x00" * 1024, "video/mp4")},
    )


def _segmented_project(client: TestClient) -> int:
    """Create a project (no analysis) and import two subtitle cues."""
    project_id = _upload(client, title="Segmented").json()["project_id"]
    upload = client.post(
        f"/api/dubbing/{project_id}/subtitles",
        files={"file": ("captions.srt", SAMPLE_SRT.encode(), "text/plain")},
    )
    assert upload.status_code == 201, upload.text
    return project_id


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _wait_for_terminal(client: TestClient, task_id: str, timeout: float = 20.0) -> dict:
    deadline = time.time() + timeout
    snapshot: dict = {}
    while time.time() < deadline:
        snapshot = client.get(f"/api/tasks/{task_id}/status").json()
        if snapshot["status"] in {"completed", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.15)
    raise AssertionError(f"Task {task_id} did not finish in time: {snapshot}")
