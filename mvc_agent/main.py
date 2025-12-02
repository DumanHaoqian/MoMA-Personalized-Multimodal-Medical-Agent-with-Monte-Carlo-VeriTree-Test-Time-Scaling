"""
Entry point for the MVC medical assistant.

This script creates a ``MedicalAssistant`` instance with appropriate
configuration, constructs an ``InterfaceManager`` to build the Gradio UI,
and launches the application.

Update the dataset paths and API keys according to your environment before
running this script.
"""

from __future__ import annotations

from controllers import MedicalAssistant
from views import InterfaceManager


def main():
    # Paths to local datasets used for RAG.  Adjust these paths to your environment.
    text_dataset_path = "/Dataset/PT/ori_pqal.json"
    image_dataset_path = "/Dataset/TaI/ti_datasubset.jsonl"
    # Azure configuration for GPT‑4o inference
    azure_api_url = ""
    azure_api_key = ""
    assistant = MedicalAssistant(
        text_dataset_path=text_dataset_path,
        image_dataset_path=image_dataset_path,
        azure_api_url=azure_api_url,
        azure_api_key=azure_api_key,
        enable_web=True,
    )
    interface = InterfaceManager(assistant)
    interface.launch()


if __name__ == "__main__":
    main()