"""
mvc_agent package
=================

This package implements a multi‑modal medical assistant using the
Model–View–Controller (MVC) architectural pattern.  The ``models``
submodule contains the data structures and domain logic (e.g. RAG index
builders, OCR and STT handlers), the ``controllers`` submodule orchestrates
the interactions between models and view, and the ``views`` submodule
exposes the user interface using Gradio.  Utility functions such as
communication with GPT‑4o are placed under ``utils``.
"""

__all__ = [
    'models',
    'controllers',
    'views',
    'utils',
]