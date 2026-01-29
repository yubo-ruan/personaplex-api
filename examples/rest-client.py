#!/usr/bin/env python3
"""
PersonaPlex REST Client Example

Simple speech-to-speech using the REST API.

Usage:
    python rest-client.py <server_ip> <input.wav> [--output output.wav]

Requirements:
    pip install requests
"""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def speech_to_speech(
    server_url: str,
    input_file: str,
    output_file: str,
    voice: str = "NATF2",
    text_prompt: str = "You are a helpful assistant.",
    seed: int = 42,
):
    """Send audio and get speech response."""
    endpoint = f"{server_url}/v1/audio/speech"

    print(f"Sending {input_file} to {endpoint}")
    print(f"Voice: {voice}")
    print(f"Prompt: {text_prompt[:50]}...")

    with open(input_file, "rb") as f:
        files = {"audio": (input_file, f, "audio/wav")}
        data = {
            "voice": voice,
            "text_prompt": text_prompt,
            "seed": seed,
        }

        response = requests.post(endpoint, files=files, data=data, timeout=120)

    if response.ok:
        # Save audio
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"Response saved to {output_file}")

        # Print transcript if available
        transcript = response.headers.get("X-Transcript")
        if transcript:
            try:
                data = json.loads(transcript)
                print(f"\nTranscript: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError:
                print(f"\nTranscript: {transcript}")

        return True
    else:
        print(f"Error {response.status_code}: {response.text}")
        return False


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex REST Client")
    parser.add_argument("server_ip", help="Server IP address")
    parser.add_argument("input", help="Input WAV file")
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--voice", default="NATF2", help="Voice preset")
    parser.add_argument("--prompt", default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    server_url = f"http://{args.server_ip}:{args.port}"

    # Check health first
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        if health.ok:
            data = health.json()
            print(f"Server: {data.get('model')} (GPU: {data.get('gpu_name', 'N/A')})")
            if not data.get("model_loaded"):
                print("Warning: Model not loaded yet. First request may take a while.")
        else:
            print(f"Server health check failed: {health.status_code}")
    except Exception as e:
        print(f"Could not reach server: {e}")
        return

    print()

    success = speech_to_speech(
        server_url,
        args.input,
        args.output,
        args.voice,
        args.prompt,
        args.seed,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
