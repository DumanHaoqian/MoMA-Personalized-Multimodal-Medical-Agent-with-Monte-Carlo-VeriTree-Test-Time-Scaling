"""
mvc_agent.utils
==============

Utility functions used across the MVC architecture.  This submodule
contains helpers for interacting with external services (e.g. Azure GPT‑4o)
and generating personalised reports.  Keeping these functions in their
own module emphasises their stateless nature and simplifies testing.
"""

from .gpt_utils import call_azure_chat, refine_answer_with_gpt4o, generate_personalised_report

__all__ = ['call_azure_chat', 'refine_answer_with_gpt4o', 'generate_personalised_report']