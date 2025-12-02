"""
Text‑to‑speech (TTS) handler.

This module defines a wrapper around the Facebook MMS TTS model.  It
provides a ``synthesize`` method that converts text into a WAV file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import pipeline


class TTSHandler:
    """Simple wrapper around the Facebook MMS TTS model."""

    def __init__(self, model_name: str = "facebook/mms-tts-eng"):
        self.model_name = model_name
        self.tts_pipe = pipeline("text-to-speech", model=self.model_name)

    def synthesize(self, text: str, output_path: str) -> str:
        speech = self.tts_pipe(text)
        audio = np.asarray(speech["audio"])
        audio = np.squeeze(audio)
        if audio.ndim == 2 and audio.shape[1] > 32 and audio.shape[0] <= 32:
            audio = audio.T
        audio = audio.astype(np.float32, copy=False)
        rate = int(speech["sampling_rate"])
        from scipy.io import wavfile  # type: ignore
        wavfile.write(output_path, rate=rate, data=audio)
        return output_path


__all__ = ['TTSHandler']