"""
Chat log model.

This module defines ``ChatLogger``, a simple class that appends chat
messages with timestamps to a log file and reads back the log.  It is
useful for providing conversational memory or generating reports.
"""

from __future__ import annotations

import datetime
import os
from typing import Any


class ChatLogger:
    """Append chat messages with timestamps to a log file."""

    def __init__(self, log_path: str = "chat_logs.txt"):
        self.log_path = log_path

    def append(self, role: str, content: str):
        timestamp = datetime.datetime.now().isoformat()
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {role.upper()}: {content}\n")

    def read(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, 'r', encoding='utf-8') as f:
            return f.read()


__all__ = ['ChatLogger']