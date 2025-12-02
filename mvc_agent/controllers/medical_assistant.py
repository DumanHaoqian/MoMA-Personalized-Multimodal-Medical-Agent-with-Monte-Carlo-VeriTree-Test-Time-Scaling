"""
Medical assistant controller.

The ``MedicalAssistant`` orchestrates the entire workflow of processing
user inputs, retrieving knowledge, consulting the web, invoking GPT‑4o,
generating illustrations, synthesising speech, logging conversations and
managing user profiles.  It sits between the models and the view layer.
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional, Tuple

from models import (
    InputHandler,
    TextRAG,
    ImageRAG,
    WebSearchEngine,
    ImageGenerator,
    UserProfile,
    ChatLogger,
    TTSHandler,
)
from utils import call_azure_chat, refine_answer_with_gpt4o


class MedicalAssistant:
    def __init__(
        self,
        text_dataset_path: str,
        image_dataset_path: str,
        azure_api_url: str,
        azure_api_key: str,
        jimeng_access_key: Optional[str] = None,
        jimeng_secret_key: Optional[str] = None,
        enable_web: bool = False,
        max_retrieved: int = 3,
    ):
        self.user_profile = UserProfile()
        self.chat_logger = ChatLogger()
        self.azure_api_url = azure_api_url
        self.azure_api_key = azure_api_key
        self.enable_web = enable_web
        self.input_handler = InputHandler()
        self.text_rag = TextRAG(dataset_path=text_dataset_path, top_k=max_retrieved)
        try:
            self.text_rag.build_index()
        except Exception as e:
            print(f"Warning: failed to build text RAG index: {e}")
        self.image_rag = ImageRAG(dataset_jsonl_path=image_dataset_path, top_k=max_retrieved)
        try:
            self.image_rag.build_index()
        except Exception as e:
            print(f"Warning: failed to build image RAG index: {e}")
        self.web_engine = WebSearchEngine() if enable_web else None
        self.image_generator: Optional[ImageGenerator] = None
        if jimeng_access_key and jimeng_secret_key:
            self.image_generator = ImageGenerator(jimeng_access_key, jimeng_secret_key)
        self.tts_handler = TTSHandler()

    def _call_gpt4o(self, prompt: str) -> Tuple[str, str]:
        system_instruction = (
            "You are Moma, a professor at the Faculty of Medicine of The Hong Kong Polytechnic University. "
            "You are an extremely friendly, warm, and compassionate medical expert who loves to help people. "
            "You should respond to users' questions with enthusiasm and kindness, providing detailed and thorough answers. "
            "Feel free to use emojis and expressive language to make your responses more engaging and friendly. "
            "Be proactive in providing comprehensive explanations - don't be brief, share your knowledge generously! "
            "Use any provided conversation history, retrieved contexts and image captions to provide a thorough and reliable answer to the user's medical question. "
            "Always remind the user that your advice cannot replace professional medical consultation. "
            "After answering, include the token <IMAGE_PROMPT> followed by a concise one‑sentence description of a helpful medical illustration to aid understanding."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        response = call_azure_chat(messages, self.azure_api_url, self.azure_api_key, max_tokens=1024)
        if "<IMAGE_PROMPT>" in response:
            answer_text, illustration_prompt = response.split("<IMAGE_PROMPT>", 1)
        else:
            answer_text = response
            illustration_prompt = ""
        return answer_text.strip(), illustration_prompt.strip()

    def respond(
        self,
        user_text: str,
        user_images: List[str],
        user_audios: List[str],
        use_rag: bool,
        use_web: bool,
        generate_illustration: bool,
    ) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        self.chat_logger.append("user", user_text)
        prepared = self.input_handler.prepare(user_text, user_images, user_audios)
        prompt = prepared.get("prompt", "")
        retrieved_contexts: List[Tuple[Dict[str, Any], float]] = []
        if use_rag and prompt:
            try:
                retrieved_contexts = self.text_rag.retrieve(prompt, top_k=self.text_rag.top_k)
            except Exception:
                pass
        if use_rag and user_images:
            try:
                image_results = self.image_rag.retrieve_by_image(user_images[0], top_k=self.image_rag.top_k)
                if image_results:
                    caption_prompt = self.image_rag.format_prompt(user_text, image_results)
                    prompt += "\n\n" + caption_prompt
            except Exception:
                pass
        if use_web and self.web_engine is not None:
            try:
                web_answer = self.web_engine.search_and_answer(prompt)
                prompt = web_answer + "\n\n" + prompt
            except Exception:
                pass
        if retrieved_contexts:
            prompt = self.text_rag.format_prompt(prompt, retrieved_contexts)
        try:
            logs = self.chat_logger.read().strip().splitlines()
            if logs:
                logs_without_latest = logs[:-1]
            else:
                logs_without_latest = []
            recent = logs_without_latest[-6:] if len(logs_without_latest) >= 6 else logs_without_latest
            if recent:
                memory_context = "Here is the recent conversation history:\n" + "\n".join(recent) + "\n\n"
                prompt = memory_context + prompt
        except Exception:
            pass
        answer_text, illustration_prompt = self._call_gpt4o(prompt)
        refined_text = refine_answer_with_gpt4o(answer_text, self.azure_api_url, self.azure_api_key)
        illustration_path: Optional[str] = None
        if generate_illustration and illustration_prompt and self.image_generator is not None:
            urls = self.image_generator.generate(illustration_prompt)
            if urls:
                saved = self.image_generator.save_images(urls)
                if saved:
                    illustration_path = saved[0]
        tts_path: Optional[str] = None
        if refined_text:
            tts_file = f"tts_output_{int(time.time())}.wav"
            try:
                tts_path = self.tts_handler.synthesize(refined_text, tts_file)
            except Exception:
                tts_path = None
        self.chat_logger.append("assistant", refined_text)
        return refined_text, illustration_path, tts_path, None


__all__ = ['MedicalAssistant']