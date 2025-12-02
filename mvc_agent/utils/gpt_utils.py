"""
GPT‑4o utilities.

This module defines helper functions to communicate with Azure GPT‑4o
endpoints and refine generated answers.  It also includes a function to
generate personalised reports summarising a conversation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import requests


def call_azure_chat(messages: List[Dict[str, str]], api_url: str, api_key: str, max_tokens: int = 1024) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Azure API error: {e}]"


def refine_answer_with_gpt4o(raw_answer: str, api_url: str, api_key: str) -> str:
    sentences = [s.strip() for s in raw_answer.replace('\n', ' ').split('.') if s.strip()]
    if not sentences:
        return raw_answer
    seg_len = max(1, len(sentences) // 4)
    segments = [". ".join(sentences[i:i+seg_len]) + ("." if i + seg_len < len(sentences) else '') for i in range(0, len(sentences), seg_len)]
    refined_parts: List[str] = []
    for segment in segments:
        candidates: List[str] = []
        for _ in range(3):
            messages = [
                {"role": "system", "content": "You are a helpful assistant who rewrites text to improve clarity and correctness."},
                {"role": "user", "content": f"Please rewrite the following medical explanation to improve its clarity, conciseness and accuracy. Ensure that the meaning is preserved.\n\nText:\n{segment}"},
            ]
            candidate = call_azure_chat(messages, api_url, api_key, max_tokens=512)
            candidates.append(candidate.strip())
        comparison_prompt = "\n\n".join([f"Option {i+1}: {cand}" for i, cand in enumerate(candidates)])
        messages = [
            {"role": "system", "content": "You are an expert editor selecting the best rewritten paragraph."},
            {"role": "user", "content": (
                f"Below are three rewritten versions of a medical explanation. Choose the best option that is most clear, accurate and concise. "
                f"Respond only with the option number (1, 2 or 3).\n\n{comparison_prompt}"
            )},
        ]
        choice = call_azure_chat(messages, api_url, api_key, max_tokens=10).strip()
        try:
            choice_idx = int(choice[0]) - 1
        except Exception:
            choice_idx = 0
        selected = candidates[choice_idx] if 0 <= choice_idx < len(candidates) else candidates[0]
        refined_parts.append(selected)
    return "\n\n".join(refined_parts)


def generate_personalised_report(user_profile: 'mvc_agent.models.UserProfile', chat_logger: 'mvc_agent.models.ChatLogger', api_url: str, api_key: str) -> str:
    profile_lines = [
        f"Sex: {user_profile.data.get('sex', 'N/A')}",
        f"Age: {user_profile.data.get('age', 'N/A')}",
        f"Height: {user_profile.data.get('height', 'N/A')}",
        f"Weight: {user_profile.data.get('weight', 'N/A')}",
    ]
    logs = chat_logger.read()
    messages = [
        {"role": "system", "content": "You are a medical assistant generating a personalised health report based on a conversation."},
        {"role": "user", "content": (
            "Here are the user's demographic details:\n" + "\n".join(profile_lines) + "\n\n" +
            "Here is a log of the conversation between the user and the assistant:\n" + logs + "\n\n" +
            "Please provide a concise summary of the topics discussed, emphasise important medical recommendations, "
            "and remind the user that this report does not replace professional medical advice."
        )},
    ]
    summary = call_azure_chat(messages, api_url, api_key, max_tokens=512)
    return summary


__all__ = ['call_azure_chat', 'refine_answer_with_gpt4o', 'generate_personalised_report']