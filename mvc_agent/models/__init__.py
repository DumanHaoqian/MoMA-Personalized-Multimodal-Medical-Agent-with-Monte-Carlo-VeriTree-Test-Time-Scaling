"""
mvc_agent.models
================

The ``models`` package defines data structures and domain logic for the
medical assistant.  It includes modules for OCR, STT, input aggregation,
retrieval augmented generation (RAG) over text and images, web search,
image generation, user profiling, chat logging and text‑to‑speech.
"""

from .ocr_handler import OCRHandler, ti2t_ocr_default
from .stt_handler import STTHandler
from .input_handler import InputHandler
from .text_rag import TextRAG
from .image_rag import ImageRAG
from .web_engine import WebSearchEngine
from .image_generator import ImageGenerator
from .user_profile import UserProfile
from .chat_logger import ChatLogger
from .tts_handler import TTSHandler

__all__ = [
    'OCRHandler',
    'ti2t_ocr_default',
    'STTHandler',
    'InputHandler',
    'TextRAG',
    'ImageRAG',
    'WebSearchEngine',
    'ImageGenerator',
    'UserProfile',
    'ChatLogger',
    'TTSHandler',
]