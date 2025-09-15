import os
from base64 import b64encode
import gradio as gr

from config import LLM_CHECKPOINTS, FLOW_CHECKPOINTS, DEFAULT_FLOW, DEFAULT_LLM, VOICE_TEMPLATES, ASSET_PATH
from model_cache import handle_load_model_button
from inference import (
    load_voice_template,
    clone_voice_zero_shot_with_download,
    batch_inference,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(SCRIPT_DIR, "TL-logo.png")


def _build_logo_html():
    try:
        if not os.path.exists(LOGO_PATH):
            return '<div style="text-align:center;"><h1>TinyLinguist: CosyVoice2</h1></div>'
        with open(LOGO_PATH, "rb") as f:
            b64 = b64encode(f.read()).decode("ascii")
        return f"""
        <div style="text-align:center; padding:22px 10px 8px;">
            <img src="data:image/png;base64,{b64}"
                 alt="TinyLinguist Logo"
                 style="max-width:240px;width:25%;min-width:130px;display:block;margin:0 auto 6px auto;">
            <h1 style="margin:6px 0 4px 0;">TinyLinguist: CosyVoice2</h1>
            <p style="margin:0;">This page provides inference service of our Cerence-internal versions of CosyVoice2 TTS Models.</p>
        </div>
        """
    except Exception as e:
        return f'<div style="text-align:center;"><h1>TinyLinguist: CosyVoice2</h1><p style="color:#f66;">Logo load error: {e}</p></div>'


def create_interface():
    custom_theme = gr.Theme.from_hub("shivi/calm_seafoam").set(
        input_background_fill_dark="#020614",
        checkbox_background_color_selected_dark="#267EAA",
        body_background_fill_dark="#020614",
        background_fill_secondary_dark="#1a1a1a",
        block_background_fill_dark="#122335",
        button_secondary_background_fill_dark="#0D131B",
        checkbox_label_background_fill_selected_dark="#122335",
        border_color_primary_dark="#267EAA",
        input_border_width="#803482",
        input_background_fill_focus="#E8F4FD",
        slider_color="#00B4BE",
        slider_color_dark="#267EAA",
    )

    with gr.Blocks(title="CosyVoice2 Inference Service", theme=custom_theme) as demo:
        gr.HTML(_build_logo_html())

        gr.HTML("""
        <style>
        .bold-dropdown label { font-weight: bold !important; }
        .larger-title label { font-size: larger; }
        .card-button { width: 100% !important; padding: 15px !important; margin: 8px !important; border: 2px solid #ddd !important; border-radius: 8px !important; background: white !important; cursor: pointer !important; transition: all 0.3s ease !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; text-align: left !important; font-size: 14px !important; line-height: 1.4 !important; white-space: pre-line !important; min-height: 100px !important; font-weight: normal !important; }
        .card-button:hover { border-color: #2196F3 !important; background: #f8fffe !important; box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important; transform: translateY(-2px) !important; }
        @media (prefers-color-scheme: dark) { .card-button { background: #122335 !important; border-color: #555 !important; color: #e0e0e0 !important; } .card-button:hover { border-color: #64b5f6 !important; background: #1a2332 !important; } }
        </style>
        <h3>Recommended Model Combinations</h3>
        """)

        with gr.Row():
            combination_btn_1 = gr.Button(
                "Base CosyVoice2\nLLM: Base Model\nFlow: Base Model\nVoice: choose any voice",
                elem_classes=["card-button"],
                variant="secondary",
            )
            combination_btn_2 = gr.Button(
                "Base CosyVoice2\nLLM: Custom Model\nFlow: Custom Model\nVoice: choose any voice",
                elem_classes=["card-button"],
                variant="secondary",
            )

        gr.HTML("""
            <div class="usage-tips">
                <small><strong>Usage Tips:</strong> Click a combination, then "Load Model". Use Single Audio or Batch tabs.</small>
            </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>Model Configuration</h3>")
                with gr.Row():
                    llm_dropdown = gr.Dropdown(
                        choices=list(LLM_CHECKPOINTS.keys()),
                        value=DEFAULT_LLM,
                        label="LLM Checkpoint",
                        elem_classes=["bold-dropdown"],
                    )
                    flow_dropdown = gr.Dropdown(
                        choices=list(FLOW_CHECKPOINTS.keys()),
                        value=DEFAULT_FLOW,
                        label="Flow Checkpoint",
                        elem_classes=["bold-dropdown"],
                    )
                load_model_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(label="Model Status", lines=2, interactive=False, elem_classes=["bold-dropdown"])

        with gr.Tabs():
            with gr.TabItem("Single Audio Generation / Zero-shot Cloning", elem_classes=["larger-title"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>Step 1: Record Your Own Prompt Audio</h3>")
                        audio_input_zs = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Prompt Audio", format="wav")
                        gr.HTML("<h4>Or Choose an Existing Prompt Audio:</h4>")
                        with gr.Row():
                            template_buttons_zs = []
                            for i in range(0, len(VOICE_TEMPLATES), 3):
                                with gr.Row():
                                    for j in range(3):
                                        if i + j < len(VOICE_TEMPLATES):
                                            voice_name = VOICE_TEMPLATES[i + j]
                                            display_name = voice_name.replace("_", " ").replace("cvyo", "").replace("mixed", "M").strip()
                                            if len(display_name) > 15:
                                                display_name = display_name[:15] + "..."
                                            btn = gr.Button(f"{display_name}", size="sm", variant="secondary")
                                            template_buttons_zs.append((btn, voice_name))
                        prompt_text_zs = gr.Textbox(label="Prompt Text", placeholder="Enter exactly what you said in the recording...", lines=2)
                        target_text_zs = gr.Textbox(label="Target Text", placeholder="Enter the text you want to synthesize...", lines=3)
                        with gr.Row():
                            speed_zs = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                            trim_ending_zs = gr.Slider(minimum=0, maximum=200, value=40, step=1, label="Trim Ending (ms)", info="Truncate from end of generated audio")
                        gr.HTML("<h4>Step 2: Post-processing Options</h4>")
                        with gr.Row():
                            sample_rate_zs = gr.Dropdown(choices=[("Original (24000 Hz)", None), ("16000 Hz", 16000), ("22050 Hz", 22050)], value=None, label="Output Sample Rate")
                            bit_depth_zs = gr.Dropdown(choices=[("Original (32-bit)", None), ("16-bit", 16)], value=None, label="Output Bit Depth")
                        clone_btn_zs = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.HTML("<h3>Generated Audio</h3>")
                        output_audio_zs = gr.Audio(label="Generated Audio", type="filepath")
                        download_file_zs = gr.File(label="Download Generated Audio", interactive=False)
                        status_output_zs = gr.Textbox(label="Status", lines=8, interactive=False)

            with gr.TabItem("Batch Inference", elem_classes=["larger-title"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>Step 1: Record Your Own Prompt Audio</h3>")
                        audio_input_batch = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Prompt Audio")
                        with gr.Group() as zero_shot_group:
                            prompt_text_batch = gr.Textbox(label="Prompt Text", placeholder="Enter exactly what you said...", lines=2)
                        gr.HTML("<h4>Or Choose an Existing Prompt Audio:</h4>")
                        with gr.Row():
                            template_buttons_batch = []
                            for i in range(0, len(VOICE_TEMPLATES), 3):
                                with gr.Row():
                                    for j in range(3):
                                        if i + j < len(VOICE_TEMPLATES):
                                            voice_name = VOICE_TEMPLATES[i + j]
                                            display_name = voice_name.replace("_", " ").replace("cvyo", "").replace("mixed", "M").strip()
                                            if len(display_name) > 15:
                                                display_name = display_name[:15] + "..."
                                            btn = gr.Button(f"{display_name}", size="sm", variant="secondary")
                                            template_buttons_batch.append((btn, voice_name))
                        gr.HTML("<h3>Step 2: Upload Text File</h3>")
                        text_file_input = gr.File(label="Upload Text File", file_types=[".txt"], file_count="single")
                        gr.HTML("<small>Upload a .txt file with one sentence per line</small>")
                        gr.HTML("<h3>Step 3: Configure Settings</h3>")
                        with gr.Row():
                            speed_batch = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                            trim_ending_batch = gr.Slider(minimum=0, maximum=200, value=40, step=1, label="Trim Ending (ms)", info="Truncate from end of generated audio")
                        gr.HTML("<h4>Step 4: Post-processing Options</h4>")
                        with gr.Row():
                            sample_rate_batch = gr.Dropdown(choices=[("Original (24000 Hz)", None), ("16000 Hz", 16000), ("22050 Hz", 22050)], value=None, label="Output Sample Rate")
                            bit_depth_batch = gr.Dropdown(choices=[("Original (32-bit)", None), ("16-bit", 16)], value=None, label="Output Bit Depth")
                        batch_btn = gr.Button("Start Batch Inference", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.HTML("<h3>Batch Results</h3>")
                        download_file = gr.File(label="Download ZIP File", interactive=False)
                        status_output_batch = gr.Textbox(label="Status", lines=10, interactive=False)

        load_model_btn.click(fn=handle_load_model_button, inputs=[llm_dropdown, flow_dropdown], outputs=[model_status])

        for btn, voice_name in template_buttons_zs:
            btn.click(fn=load_voice_template, inputs=[gr.State(voice_name)], outputs=[audio_input_zs, status_output_zs, prompt_text_zs])

        for btn, voice_name in template_buttons_batch:
            def load_template_batch(vn):
                audio_path, status, prompt_text = load_voice_template(vn)
                return audio_path, status, prompt_text if prompt_text else ""
            btn.click(fn=lambda vn=voice_name: load_template_batch(vn), inputs=[], outputs=[audio_input_batch, status_output_batch, prompt_text_batch])

        clone_btn_zs.click(
            fn=clone_voice_zero_shot_with_download,
            inputs=[llm_dropdown, flow_dropdown, audio_input_zs, prompt_text_zs, target_text_zs, speed_zs, trim_ending_zs, gr.State(False), sample_rate_zs, bit_depth_zs],
            outputs=[output_audio_zs, status_output_zs, download_file_zs],
        )

        batch_btn.click(
            fn=batch_inference,
            inputs=[llm_dropdown, flow_dropdown, audio_input_batch, prompt_text_batch, text_file_input, speed_batch, trim_ending_batch, gr.State(False), gr.State("用中文说这句话"), sample_rate_batch, bit_depth_batch],
            outputs=[download_file, status_output_batch],
        )




        # Connect combination buttons using Gradio's event system
        def set_combination_1():
            return "Base Model", "Base Model"
        
        def set_combination_2():
            return "Custom Model", "Custom Model"

        combination_btn_1.click(
            fn=set_combination_1,
            inputs=[],
            outputs=[llm_dropdown, flow_dropdown]
        )
        
        combination_btn_2.click(
            fn=set_combination_2,
            inputs=[],
            outputs=[llm_dropdown, flow_dropdown]
        )

        gr.Examples(
            examples=[["Overall, when companies actually put some thought into it, these wellness programs can really make a difference and turn the workplace into a better place for everyone!"],
                      ["Ah, a whole new world! A new fantastic point of view."],
                      ["今天我们这个讨论呐，主要是想说一下，现在我们这个team，这一段时间的发展目标怎么定。"],
                      ["哎呦喂，皇军托我给您带个话儿，只要你能够投降皇军，保证你一辈子荣华富贵，金票大大地！"],
                      ["四郎，那年杏花微雨，你说你是果郡王，也许从一开始，便都是错的。"],
                      ["It's only after we've lost everything that we're free to do anything."],
                      ["You know, slow is smooth, smooth is fast. Just focus on driving."]],
            inputs=[target_text_zs],
            label="Click to load some quick tests.",
        )

    return demo


