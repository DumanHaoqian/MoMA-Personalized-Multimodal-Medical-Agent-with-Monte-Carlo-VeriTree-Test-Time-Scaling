"""
Image generator wrapper for the medical assistant.

This module defines ``ImageGenerator``, a wrapper around the Jimeng 4.0
API for text‑to‑image generation.  It submits tasks and polls for results
until completion.  A static method ``save_images`` downloads the
generated images.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import datetime


class ImageGenerator:
    """Wrapper around the Jimeng 4.0 API for text‑to‑image generation."""

    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key

    def generate(self, prompt_text: str, image_urls: Optional[List[str]] = None) -> Optional[List[str]]:
        try:
            from volcengine.visual.VisualService import VisualService  # type: ignore
        except Exception:
            return None
        visual_service = VisualService()
        visual_service.set_ak(self.access_key)
        visual_service.set_sk(self.secret_key)
        submit_task_body = {
            "req_key": "jimeng_t2i_v40",
            "prompt": prompt_text,
            "image_urls": image_urls if image_urls else [],
        }
        try:
            resp_submit = visual_service.cv_sync2async_submit_task(submit_task_body)
            if resp_submit.get("code") == 10000 and resp_submit.get("data", {}).get("task_id"):
                task_id = resp_submit["data"]["task_id"]
            else:
                return None
        except Exception:
            return None
        query_extra_params = {"return_url": True, "logo_info": {"add_logo": False}}
        query_task_body = {
            "req_key": "jimeng_t2i_v40",
            "task_id": task_id,
            "req_json": json.dumps(query_extra_params),
        }
        max_attempts = 30
        poll_interval = 5
        for _ in range(max_attempts):
            try:
                resp_query = visual_service.cv_sync2async_get_result(query_task_body)
            except Exception:
                return None
            if resp_query.get("code") == 10000 and resp_query.get("data"):
                status = resp_query["data"].get("status")
                if status == "done":
                    return resp_query["data"].get("image_urls")
                elif status in ["in_queue", "generating"]:
                    time.sleep(poll_interval)
                else:
                    return None
            else:
                return None
        return None

    @staticmethod
    def save_images(image_urls: List[str], output_dir: str = "generated_images") -> List[str]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        saved_files: List[str] = []
        for i, url in enumerate(image_urls):
            try:
                import requests  # local import
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"jimeng_v4_{timestamp}_{i}.jpg"
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                saved_files.append(os.path.abspath(file_path))
            except Exception:
                continue
        return saved_files


__all__ = ['ImageGenerator']