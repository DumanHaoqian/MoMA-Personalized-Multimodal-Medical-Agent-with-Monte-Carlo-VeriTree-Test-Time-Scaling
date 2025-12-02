"""
Gradio interface manager.

This module defines ``InterfaceManager``, which constructs and launches
a Gradio interface for interacting with the medical assistant.  It
handles user inputs, displays outputs and manages chat state.  All UI
logic is contained within this class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from controllers import MedicalAssistant
from utils import generate_personalised_report


class InterfaceManager:
    def __init__(self, assistant: MedicalAssistant):
        self.assistant = assistant

    def launch(self):
        agent = self.assistant
        custom_css = """
        #title {
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            color: #2A3E5C;
            margin-bottom: 10px;
        }
        #subtitle {
            text-align: center;
            font-size: 16px;
            color: #444;
            margin-bottom: 20px;
        }
        #chatbox_section {
            background-color: #F0F8FF;
            border: 1px solid #CCE0F5;
            border-radius: 8px;
            padding: 10px;
        }
        #input_section {
            background-color: #F7FCF7;
            border: 1px solid #D5E8D4;
            border-radius: 8px;
            padding: 10px;
        }
        #media_section {
            background-color: #FFF7F5;
            border: 1px solid #F5D5D5;
            border-radius: 8px;
            padding: 10px;
        }
        #info_section {
            background-color: #FCFCFC;
            border: 1px solid #EEEEEE;
            border-radius: 6px;
            padding: 6px;
            margin-top: 8px;
            font-size: 12px;
            color: #666666;
        }
        """
        with gr.Blocks(title="MoMA: Your Personalized Multimodal Medical Agent", css=custom_css) as demo:
            gr.Markdown("# MoMA: Your Personalized Multimodal Medical Agent 🩺🤖", elem_id="title")
            gr.Markdown(
                "Welcome to **MoMA**! Ask your medical question, upload relevant images (e.g. lab reports) or audio recordings, "
                "and select additional options below. Your privacy is respected and this tool provides information only—always consult a doctor for medical advice. ✨",
                elem_id="subtitle",
            )
            with gr.Accordion("👤 User Information", open=not agent.user_profile.is_complete()):
                sex_in = gr.Textbox(label="Sex (e.g. male/female)", value=agent.user_profile.data.get("sex", ""))
                age_in = gr.Textbox(label="Age", value=agent.user_profile.data.get("age", ""))
                height_in = gr.Textbox(label="Height (cm)", value=agent.user_profile.data.get("height", ""))
                weight_in = gr.Textbox(label="Weight (kg)", value=agent.user_profile.data.get("weight", ""))
                save_btn = gr.Button("Save Information")
                save_msg = gr.Markdown("", visible=False)
            with gr.Row():
                with gr.Column(scale=2, elem_id="chatbox_section"):
                    history_state = gr.State([])
                    chatbot = gr.Chatbot(label="Conversation", height=650)
                with gr.Column(scale=1, elem_id="input_section"):
                    user_text = gr.Textbox(label="🗣️ Your question", lines=4)
                    user_images = gr.File(label="🖼️ Upload image(s)", file_count="multiple", type="filepath")
                    user_images_paste = gr.Image(
                        label="📋 Or paste image(s) here (Ctrl+V)",
                        type="filepath",
                        sources=["clipboard", "upload"],
                        height=200,
                    )
                    user_audios = gr.File(label="🎤 Upload audio(s)", file_count="multiple", type="filepath")
                    use_rag = gr.Checkbox(label="📚 Use domain knowledge (RAG)", value=True)
                    use_web = gr.Checkbox(label="🔎 Use web search", value=False)
                    generate_illustration = gr.Checkbox(label="🎨 Generate illustration", value=False)
                    submit_btn = gr.Button("Send ➡️")
            with gr.Column(elem_id="media_section"):
                gr.Markdown("### 🎨 & 🔊 Outputs")
                illustration_out = gr.Image(label="Generated Illustration", visible=False)
                audio_out = gr.Audio(label="Answer Speech", type="filepath", visible=False)
                progress_bar = gr.HTML(value="", visible=False)
            with gr.Accordion("📝 Generate Personalised Report"):
                report_btn = gr.Button("Generate Report")
                report_out = gr.Markdown("", visible=False)
            def save_info(sex, age, height, weight):
                agent.user_profile.update(sex, age, height, weight)
                return gr.update(value="✅ Information saved.", visible=True)
            save_btn.click(save_info, inputs=[sex_in, age_in, height_in, weight_in], outputs=[save_msg])
            def process_query(q_text, imgs, pasted_img, auds, rag_flag, web_flag, ill_flag, history):
                if history is None:
                    history = []
                history.append((q_text, "🤖 Generating..."))
                yield {
                    progress_bar: gr.update(value="<progress style='width:100%'></progress>", visible=True),
                    chatbot: history,
                    history_state: history,
                }
                image_paths = [f.name for f in imgs] if imgs else []
                if pasted_img:
                    if isinstance(pasted_img, str):
                        image_paths.append(pasted_img)
                    elif isinstance(pasted_img, (list, tuple)) and len(pasted_img) > 0:
                        for img in pasted_img:
                            if isinstance(img, str):
                                image_paths.append(img)
                    else:
                        import tempfile
                        if isinstance(pasted_img, Image.Image):
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            pasted_img.save(temp_file.name)
                            image_paths.append(temp_file.name)
                        elif isinstance(pasted_img, np.ndarray):
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            Image.fromarray(pasted_img).save(temp_file.name)
                            image_paths.append(temp_file.name)
                audio_paths = [f.name for f in auds] if auds else []
                answer, illustration_path, tts_path, _ = agent.respond(
                    user_text=q_text,
                    user_images=image_paths,
                    user_audios=audio_paths,
                    use_rag=rag_flag,
                    use_web=web_flag,
                    generate_illustration=ill_flag,
                )
                history[-1] = (q_text, answer)
                outputs = {
                    chatbot: history,
                    history_state: history,
                }
                if illustration_path:
                    outputs[illustration_out] = gr.update(value=illustration_path, visible=True)
                else:
                    outputs[illustration_out] = gr.update(visible=False)
                if tts_path:
                    outputs[audio_out] = gr.update(value=tts_path, visible=True)
                else:
                    outputs[audio_out] = gr.update(visible=False)
                outputs[progress_bar] = gr.update(value="", visible=False)
                yield outputs
            submit_btn.click(
                process_query,
                inputs=[user_text, user_images, user_images_paste, user_audios, use_rag, use_web, generate_illustration, history_state],
                outputs=[chatbot, illustration_out, audio_out, history_state, progress_bar],
            )
            def produce_report():
                if not agent.user_profile.is_complete():
                    return gr.update(value="Please save your user information before generating a report.", visible=True)
                summary = generate_personalised_report(agent.user_profile, agent.chat_logger, agent.azure_api_url, agent.azure_api_key)
                return gr.update(value=summary, visible=True)
            report_btn.click(produce_report, inputs=[], outputs=[report_out])
        demo.launch()


__all__ = ['InterfaceManager']