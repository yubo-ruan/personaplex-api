#!/usr/bin/env python3
"""
PersonaPlex Gradio Web UI

A simple web interface for testing PersonaPlex speech-to-speech API.

Usage:
    pip install gradio requests
    python app.py --server http://35.239.24.211:8000
"""

import argparse
import tempfile
import requests
import gradio as gr

# Default server URL
SERVER_URL = "http://35.239.24.211:8000"

VOICES = {
    "Natural Female": ["NATF0", "NATF1", "NATF2", "NATF3"],
    "Natural Male": ["NATM0", "NATM1", "NATM2", "NATM3"],
    "Variety Female": ["VARF0", "VARF1", "VARF2", "VARF3", "VARF4"],
    "Variety Male": ["VARM0", "VARM1", "VARM2", "VARM3", "VARM4"],
}

ALL_VOICES = [v for voices in VOICES.values() for v in voices]


def check_server_health():
    """Check if the server is healthy."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.ok:
            data = response.json()
            return f"✅ Server Online | GPU: {data.get('gpu_name', 'N/A')} | VRAM: {data.get('gpu_memory', 'N/A')}"
        return "❌ Server returned error"
    except Exception as e:
        return f"❌ Server offline: {e}"


def speech_to_speech(audio_path, voice, text_prompt, seed):
    """Send audio to PersonaPlex API and get response."""
    if audio_path is None:
        return None, "Please record or upload audio first."

    try:
        with open(audio_path, "rb") as f:
            files = {"audio": ("input.wav", f, "audio/wav")}
            data = {
                "voice": voice,
                "text_prompt": text_prompt,
                "seed": int(seed),
            }

            response = requests.post(
                f"{SERVER_URL}/v1/audio/speech",
                files=files,
                data=data,
                timeout=300,
            )

        if response.ok:
            # Save response audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
                out.write(response.content)
                output_path = out.name

            # Get transcript if available
            transcript = response.headers.get("X-Transcript", "{}")
            return output_path, f"✅ Success! Transcript: {transcript}"
        else:
            return None, f"❌ Error {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        return None, "❌ Request timed out. The model may still be loading."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_ui():
    """Create the Gradio interface."""
    with gr.Blocks(title="PersonaPlex Speech-to-Speech", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎙️ PersonaPlex Speech-to-Speech

        Upload or record audio, and PersonaPlex will respond with speech!
        """)

        # Server status
        server_status = gr.Textbox(
            label="Server Status",
            value=check_server_health(),
            interactive=False,
        )
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(check_server_health, outputs=server_status)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Your Audio",
                    type="filepath",
                    sources=["microphone", "upload"],
                )

                voice = gr.Dropdown(
                    choices=ALL_VOICES,
                    value="NATF2",
                    label="Voice",
                )

                text_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful assistant.",
                    lines=2,
                )

                seed = gr.Number(
                    label="Seed (for reproducibility)",
                    value=42,
                    precision=0,
                )

                submit_btn = gr.Button("🎤 Generate Response", variant="primary")

            with gr.Column():
                gr.Markdown("### Output")
                audio_output = gr.Audio(
                    label="PersonaPlex Response",
                    type="filepath",
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

        # Connect the button
        submit_btn.click(
            speech_to_speech,
            inputs=[audio_input, voice, text_prompt, seed],
            outputs=[audio_output, status_output],
        )

        # Examples
        gr.Markdown("""
        ### Voice Categories
        - **Natural Female (NATF0-3)**: Natural sounding female voices
        - **Natural Male (NATM0-3)**: Natural sounding male voices
        - **Variety Female (VARF0-4)**: More diverse female voices
        - **Variety Male (VARM0-4)**: More diverse male voices

        ### Tips
        - First request may take longer as the model loads
        - Speak clearly into the microphone
        - Try different voices and prompts for varied responses
        """)

    return demo


def main():
    global SERVER_URL

    parser = argparse.ArgumentParser(description="PersonaPlex Gradio UI")
    parser.add_argument("--server", default=SERVER_URL, help="PersonaPlex API server URL")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()
    SERVER_URL = args.server

    print(f"Connecting to PersonaPlex server: {SERVER_URL}")
    print(check_server_health())

    demo = create_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
