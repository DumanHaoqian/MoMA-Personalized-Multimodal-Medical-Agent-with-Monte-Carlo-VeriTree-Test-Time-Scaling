"""
Input aggregation for the medical assistant.

The ``InputHandler`` composes OCR and STT handlers to process multimodal
inputs (text, images and audio).  It returns a dictionary containing the
concatenated text and any loaded PIL images.  Consumers of this class do
not need to worry about how images or audio are converted into text.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

from .ocr_handler import OCRHandler
from .stt_handler import STTHandler


class InputHandler:
    def __init__(
        self,
        ocr_handler: Optional[OCRHandler] = None,
        stt_handler: Optional[STTHandler] = None,
    ):
        self.ocr_handler = ocr_handler or OCRHandler()
        self.stt_handler = stt_handler or STTHandler()

    def prepare(self, text: Optional[str], image_paths: List[str], audio_paths: List[str]) -> Dict[str, Any]:
        texts: List[str] = []
        if text:
            texts.append(text.strip())
        images_loaded: List[Image.Image] = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                images_loaded.append(img)
            except Exception:
                pass
            ocr_text = self.ocr_handler.run(img_path)
            if ocr_text:
                texts.append(ocr_text.strip())
        for audio_path in audio_paths:
            stt_text = self.stt_handler.transcribe(audio_path)
            if stt_text:
                texts.append(stt_text.strip())
        combined_text = "\n\n".join([t for t in texts if t])
        multi_modal_data: Dict[str, Any] = {}
        if images_loaded:
            multi_modal_data["image"] = images_loaded if len(images_loaded) > 1 else images_loaded[0]
        return {"prompt": combined_text, "multi_modal_data": multi_modal_data}


__all__ = ['InputHandler']