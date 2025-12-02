"""
Web search engine wrapper for the medical assistant.

This module defines a ``WebSearchEngine`` class that wraps DuckDuckGo
search and DeepSeek LLM.  It performs a search for a given query and
constrains DeepSeek to answer using the search results.
"""

from __future__ import annotations

from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class WebSearchEngine:
    """Perform web search using DuckDuckGo and synthesise results with DeepSeek LLM."""

    def __init__(self, model_name: str = "deepseek-ai/deepseek-llm-7b-chat"):
        self.model_name = model_name
        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self._model.generation_config = GenerationConfig.from_pretrained(self.model_name)
            self._model.generation_config.pad_token_id = self._model.generation_config.eos_token_id

    def _duckduckgo_search(self, query: str, num_results: int = 3) -> str:
        try:
            from ddgs import DDGS  # type: ignore
        except Exception:
            raise ImportError(
                "ddgs module is required for web search. Please install ddgs to enable web search."
            )
        results_text: List[str] = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            for r in results:
                snippet = f"Title: {r['title']}\nBody: {r['body']}\nURL: {r['href']}\n"
                results_text.append(snippet)
        return "\n\n".join(results_text)

    def search_and_answer(self, prompt: str, num_results: int = 3) -> str:
        web_context = self._duckduckgo_search(prompt, num_results=num_results)
        system_prompt = (
            "You are a professional, friendly, and cautious medical health assistant. "
            "You have access to the following web search summaries. "
            "Use the information responsibly to answer the user’s question accurately. "
            "Always remind the user that your advice cannot replace medical consultation.\n\n"
            f"--- Web Search Results ---\n{web_context}\n\n"
            "Now, based on the above information, please provide a clear and fact-based answer:"
        )
        self._load_model()
        assert self._tokenizer is not None and self._model is not None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_tensor = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = self._model.generate(
            input_tensor.to(self._model.device), max_new_tokens=500, temperature=0.7, top_p=0.9
        )
        result = self._tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result


__all__ = ['WebSearchEngine']