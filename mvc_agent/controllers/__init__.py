"""
mvc_agent.controllers
=====================

Controllers coordinate between models and views.  The primary controller in
this package is ``MedicalAssistant``, which orchestrates input handling,
retrieval, web search, GPT‑4o interaction, image generation, TTS, user
profiling and logging.
"""

from .medical_assistant import MedicalAssistant

__all__ = ['MedicalAssistant']