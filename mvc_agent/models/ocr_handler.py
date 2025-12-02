"""
OCR handler for the medical assistant.

This module defines a default OCR implementation using DeepSeek and wraps
it into a class for easy replacement.  The ``OCRHandler`` provides a
``run`` method that accepts an image path and returns extracted text.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch


def ti2t_ocr_default(input_path: str, output_path: str) -> Optional[str]:
    """Fallback OCR implementation using DeepSeek.

    Args:
        input_path: path to the input image file.
        output_path: directory where intermediate files may be written.
    Returns:
        Recognised text or ``None`` if no text is found.
    """
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    model_name = 'deepseek-ai/DeepSeek-OCR'
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=input_path,
        output_path=output_path,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True,
    )
    text: Optional[str] = None
    if isinstance(res, str):
        text = res.strip()
    elif isinstance(res, dict):
        candidate = res.get("text") or res.get("output") or res.get("result")
        if isinstance(candidate, str):
            text = candidate.strip()
        elif isinstance(candidate, (list, tuple)):
            text = "\n".join([str(x) for x in candidate]).strip()
    elif isinstance(res, (list, tuple)):
        text = "\n".join([str(x) for x in res]).strip()
    if not text:
        result_file = os.path.join(output_path, 'result.mmd')
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    text = content if content else None
            except Exception:
                text = None
    return text


class OCRHandler:
    """Encapsulate OCR functionality.

    This handler accepts a callable for performing OCR and falls back to
    ``ti2t_ocr_default`` if none is provided.  It exposes a ``run`` method
    that takes an image path and returns extracted text.  By isolating the
    OCR logic in its own class, the rest of the system can remain agnostic
    to how images are processed.
    """

    def __init__(self, ocr_fn: Optional[Callable[[str, str], Optional[str]]] = None):
        self.ocr_fn = ocr_fn or ti2t_ocr_default

    def run(self, image_path: str, output_dir: Optional[str] = None) -> Optional[str]:
        output_dir = output_dir or os.path.dirname(image_path) or "."
        try:
            return self.ocr_fn(image_path, output_dir)
        except Exception:
            return None


__all__ = ['OCRHandler', 'ti2t_ocr_default']