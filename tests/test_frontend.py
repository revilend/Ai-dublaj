"""Run the headless frontend harness (tests/frontend_check.mjs) under Node.

The studio UI is plain JavaScript with no bundler, so its logic is verified by
executing the real source in a stubbed browser context rather than by mocking
it. Skipped when Node.js is not installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS = PROJECT_ROOT / "tests" / "frontend_check.mjs"
BOOT_HARNESS = PROJECT_ROOT / "tests" / "boot_check.mjs"
STUDIO_JS = PROJECT_ROOT / "app" / "static" / "js" / "studio.js"

# boot_check.mjs exits with this code when it cannot find a running server
BOOT_SKIP_EXIT_CODE = 3


def test_frontend_harness_passes() -> None:
    node = shutil.which("node")
    if node is None:  # pragma: no cover - depends on the host
        pytest.skip("Node.js is not installed")

    result = subprocess.run(
        [node, str(HARNESS)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"harness failed:\n{result.stdout}\n{result.stderr}"
    assert "All frontend checks passed." in result.stdout


def test_studio_js_parses() -> None:
    node = shutil.which("node")
    if node is None:  # pragma: no cover - depends on the host
        pytest.skip("Node.js is not installed")

    result = subprocess.run(
        [node, "--check", str(STUDIO_JS)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_compiled_tailwind_is_present_and_used() -> None:
    """The page must reference the compiled Tailwind build, not the old CSS."""
    css = PROJECT_ROOT / "app" / "static" / "css" / "tailwind.css"
    assert css.is_file(), "run `npm run build:css`"
    assert css.stat().st_size > 8_000, "compiled stylesheet looks empty"

    template = (PROJECT_ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "/static/css/tailwind.css" in template
    assert "styles.css" not in template
    assert "app.js" not in template

    # Components the UI relies on must have survived tree-shaking.
    compiled = css.read_text(encoding="utf-8")
    for selector in (".panel{", ".btn-primary", ".dropzone", ".tab-active", ".timeline-cue"):
        assert selector in compiled, f"{selector} was tree-shaken out of tailwind.css"


def test_page_boots_in_a_dom_engine() -> None:
    """Boot the real page in jsdom against a running server.

    This is the regression guard for script load order: Alpine starts itself
    from a microtask that drains before the next deferred script, so the
    component factory must be defined before Alpine runs. When it is not, the
    whole page dies with cascading ReferenceErrors.
    """
    node = shutil.which("node")
    if node is None:  # pragma: no cover - depends on the host
        pytest.skip("Node.js is not installed")
    if not (PROJECT_ROOT / "node_modules" / "jsdom").is_dir():
        pytest.skip("jsdom is not installed — run `npm install`")

    result = subprocess.run(
        [node, str(BOOT_HARNESS)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env={**os.environ, "VOXSTUDIO_BASE_URL": os.environ.get("VOXSTUDIO_BASE_URL", "")},
    )
    if result.returncode == BOOT_SKIP_EXIT_CODE:
        pytest.skip("no VoxStudio server reachable — start the preview first")
    assert result.returncode == 0, f"boot check failed:\n{result.stdout}\n{result.stderr}"
    assert "Boot check passed" in result.stdout


def test_pages_export_is_static_and_self_contained() -> None:
    """`npm run build:pages` must emit a site GitHub Pages can actually serve.

    Pages hosts static files only and serves a project site from `/<repo>/`, so
    the export has to use relative asset paths, carry the runtime API base, and
    contain no unsubstituted Jinja tokens.
    """
    node = shutil.which("node")
    if node is None:  # pragma: no cover - depends on the host
        pytest.skip("Node.js is not installed")

    script = PROJECT_ROOT / "scripts" / "build-pages.mjs"
    assert script.is_file(), "scripts/build-pages.mjs is missing"

    result = subprocess.run(
        [node, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "VOX_API_BASE": "https://api.example.test"},
    )
    assert result.returncode == 0, f"pages build failed:\n{result.stdout}\n{result.stderr}"

    out = PROJECT_ROOT / "pages"
    index = (out / "index.html").read_text(encoding="utf-8")

    assert (out / ".nojekyll").is_file()
    assert (out / "static" / "css" / "tailwind.css").is_file()
    assert (out / "static" / "js" / "studio.js").is_file()
    assert (out / "static" / "vendor" / "alpine.min.js").is_file()

    # Relative assets — an absolute /static/... 404s on a project Pages site.
    assert 'href="./static/css/tailwind.css"' in index
    assert '"/static/' not in index

    # Jinja is rendered away, and the API host is injected for the runtime.
    assert "{{" not in index and "{%" not in index
    assert 'window.VOX_API_BASE = "https://api.example.test"' in index
    assert 'href="https://api.example.test/docs"' in index
    assert 'src="./static/js/studio.js"' in index


def test_pages_export_without_a_backend_is_still_valid() -> None:
    """With no VOX_API_BASE the export must still be a complete, honest shell."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - depends on the host
        pytest.skip("Node.js is not installed")

    env = {key: value for key, value in os.environ.items() if key != "VOX_API_BASE"}
    result = subprocess.run(
        [node, str(PROJECT_ROOT / "scripts" / "build-pages.mjs")],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert result.returncode == 0, f"pages build failed:\n{result.stdout}\n{result.stderr}"

    index = (PROJECT_ROOT / "pages" / "index.html").read_text(encoding="utf-8")
    assert 'window.VOX_API_BASE = ""' in index
    # Without a backend the API-docs link must not point at a dead route.
    assert 'href="/docs"' not in index


def test_pages_workflow_and_script_are_wired() -> None:
    """The Actions workflow must build and deploy the static export."""
    workflow = PROJECT_ROOT / ".github" / "workflows" / "pages.yml"
    assert workflow.is_file(), ".github/workflows/pages.yml is missing"
    text = workflow.read_text(encoding="utf-8")
    for needle in (
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
        "npm run build:pages",
        "path: pages",
        "pages: write",
        "VOX_API_BASE",
        # upload-pages-artifact v5+ excludes top-level dotfiles unless asked
        # not to, which would silently drop the .nojekyll marker.
        'include-hidden-files: "true"',
    ):
        assert needle in text, f"{needle!r} missing from the Pages workflow"



def test_vendored_libraries_are_committed() -> None:
    vendor = PROJECT_ROOT / "app" / "static" / "vendor"
    for name, marker in (
        ("alpine.min.js", "Alpine"),
        ("lucide.min.js", "createIcons"),
        ("wavesurfer.min.js", "WaveSurfer"),
    ):
        path = vendor / name
        assert path.is_file(), f"{name} is missing — run `npm run vendor`"
        assert marker in path.read_text(encoding="utf-8", errors="replace")
