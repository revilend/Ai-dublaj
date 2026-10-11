"""Optional machine-translation service (NLLB-200 via transformers).

The rest of the codebase speaks ISO-639-1 codes (``en``, ``pt``, ``zh-cn``)
because that is what XTTS-v2 expects. NLLB wants FLORES-200 codes
(``eng_Latn``), so this module owns the mapping in one place.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ServiceUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ISO-639-1 -> FLORES-200
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
    "ja": "jpn_Jpan",
    "hu": "hun_Latn",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
}


class TranslationUnavailable(ServiceUnavailableError):
    code = "translation_unavailable"
    default_message = "Machine translation is not available on this host."


class TranslationService:
    """Batch translation between the languages supported by the studio."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pipeline: Any | None = None
        self._pipeline_key: tuple[str, str] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def dependency_status(self) -> tuple[bool, str | None]:
        try:
            import transformers  # noqa: F401
        except Exception:  # noqa: BLE001
            return False, (
                "transformers is not installed. Install the local AI runtime "
                "with `pip install -r requirements-ai.txt`."
            )
        return True, None

    def detail(self) -> str:
        available, reason = self.dependency_status()
        if available:
            return f"{self.settings.translation_model_name} ready"
        return reason or "unavailable"

    @staticmethod
    def supports(language: str | None) -> bool:
        return (language or "").strip().lower() in FLORES_CODES

    # ------------------------------------------------------------------ #
    def _load(self, source: str, target: str) -> Any:
        key = (source, target)
        with self._lock:
            if self._pipeline is not None and self._pipeline_key == key:
                return self._pipeline

            available, reason = self.dependency_status()
            if not available:
                raise TranslationUnavailable(reason or "Translation is unavailable.")

            from transformers import pipeline

            logger.info(
                "Loading translation pipeline %s (%s -> %s)…",
                self.settings.translation_model_name, source, target,
            )
            self._pipeline = pipeline(
                "translation",
                model=self.settings.translation_model_name,
                src_lang=source,
                tgt_lang=target,
            )
            self._pipeline_key = key
            return self._pipeline

    def translate_batch(
        self,
        texts: list[str],
        *,
        source_language: str,
        target_language: str,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[str]:
        """Translate every non-empty string, preserving list order."""
        if not texts:
            return []
        source = source_language.strip().lower()
        target = target_language.strip().lower()
        if source not in FLORES_CODES or target not in FLORES_CODES:
            raise TranslationUnavailable(
                f"Unsupported language pair '{source_language}' -> '{target_language}'.",
                details={"supported": sorted(FLORES_CODES)},
            )
        if source == target:
            return list(texts)

        translator = self._load(FLORES_CODES[source], FLORES_CODES[target])

        results: list[str] = []
        total = len(texts)
        for index, text in enumerate(texts):
            if not text.strip():
                results.append("")
                continue
            output = translator(
                text,
                src_lang=FLORES_CODES[source],
                tgt_lang=FLORES_CODES[target],
                max_length=512,
            )
            results.append(str(output[0]["translation_text"]).strip())
            if progress_callback and (index + 1) % 5 == 0:
                progress_callback(
                    ((index + 1) / total) * 100.0,
                    f"Translating segment {index + 1}/{total}…",
                )

        logger.info("Translated %d segment(s) %s -> %s", total, source, target)
        return results

    async def translate_batch_async(self, *args: Any, **kwargs: Any) -> list[str]:
        return await asyncio.to_thread(self.translate_batch, *args, **kwargs)


__all__ = ["FLORES_CODES", "TranslationService", "TranslationUnavailable"]
