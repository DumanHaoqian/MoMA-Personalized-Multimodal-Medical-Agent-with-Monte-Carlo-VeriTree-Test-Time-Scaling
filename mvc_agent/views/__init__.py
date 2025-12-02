"""
mvc_agent.views
==============

View layer for the medical assistant.  The primary view in this package
is the Gradio interface constructed by ``InterfaceManager``.  It binds
UI components to controller callbacks and exposes the application to the
user.
"""

from .interface_manager import InterfaceManager

__all__ = ['InterfaceManager']