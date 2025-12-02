"""
Speech‑to‑text handler for the medical assistant.

This module defines a class ``STTHandler`` which provides a ``transcribe``
method for converting audio files into text.  It either calls a user
provided callback or falls back to OpenAI Whisper.  The Whisper model is
loaded lazily on first use to conserve resources.
"""

from __future__ import annotations

from typing import Callable, Optional, Any


class STTHandler:
    """Handle speech‑to‑text operations using Whisper.

    If a custom ``stt_fn`` is provided via the constructor, it will be used
    instead of Whisper.  Otherwise, Whisper will be loaded lazily on the
    first call to ``transcribe``.  Errors during transcription return
    ``None``.
    """

    def __init__(self, stt_fn: Optional[Callable[[str], Optional[str]]] = None, model_name: str = "base"):
        self.stt_fn = stt_fn
        self.model_name = model_name
        self._whisper_model: Optional[Any] = None

    def _load_whisper(self):
        if self._whisper_model is not None or self.stt_fn is not None:
            return
        try:
            import whisper  # type: ignore
            self._whisper_model = whisper.load_model(self.model_name)
        except Exception:
            self._whisper_model = None

    def transcribe(self, audio_path: str) -> Optional[str]:
        if self.stt_fn is not None:
            try:
                return self.stt_fn(audio_path)
            except Exception:
                return None
        self._load_whisper()
        if self._whisper_model is None:
            return None
        try:
            result = self._whisper_model.transcribe(audio_path, fp16=False)
            text = (result.get("text") or "").strip()
            return text if text else None
        except Exception:
            return None


__all__ = ['STTHandler']