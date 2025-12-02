"""
User profile model.

This module defines a simple class ``UserProfile`` that persists user
demographic data (sex, age, height, weight) to a JSON file.  It exposes
methods to update and check completeness of the profile.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


class UserProfile:
    """Persist simple demographic data for the user."""

    def __init__(self, profile_path: str = "user_profile.json"):
        self.profile_path = profile_path
        self.data: Dict[str, Any] = {}
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def update(self, sex: str, age: str, height: str, weight: str):
        self.data.update({"sex": sex, "age": age, "height": height, "weight": weight})
        self.save()

    def is_complete(self) -> bool:
        return all(key in self.data and self.data[key] for key in ["sex", "age", "height", "weight"])


__all__ = ['UserProfile']