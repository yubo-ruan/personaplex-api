#!/usr/bin/env python3
"""
PersonaPlex WebSocket Client Example

Real-time speech-to-speech conversation with PersonaPlex.

Usage:
    python websocket-client.py <server_ip> [--voice NATF2] [--prompt "You are helpful."]

Requirements:
    pip install websockets pyaudio numpy
"""

import asyncio
import sys
import json
import argparse
import struct

try:
    import websockets
    import pyaudio
    import numpy as np
except ImportError:
    print("Please install dependencies:")
    print("  pip install websockets pyaudio numpy")
    sys.exit(1)


# Audio configuration (PersonaPlex uses 24kHz)
SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SIZE = 2400  # 100ms chunks
FORMAT = pyaudio.paInt16


class PersonaPlexClient:
    def __init__(self, server_url: str, voice: str = "NATF2", text_prompt: str = "You are a helpful assistant."):
        self.server_url = server_url
        self.voice = voice
        self.text_prompt = text_prompt
        self.audio = pyaudio.PyAudio()
        self.running = False

    async def run(self):
        """Main conversation loop."""
        print(f"Connecting to {self.server_url}...")

        async with websockets.connect(self.server_url) as ws:
            print("Connected!")

            # Send configuration
            config = {
                "voice": self.voice,
                "text_prompt": self.text_prompt,
            }
            await ws.send(json.dumps(config))
            print(f"Config sent: voice={self.voice}")

            # Wait for config acknowledgment
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("Configuration accepted")
            else:
                print(f"Unexpected response: {data}")
                return

            print("\nStart speaking... (Press Ctrl+C to stop)\n")
            self.running = True

            # Start audio input/output tasks
            send_task = asyncio.create_task(self.send_audio(ws))
            receive_task = asyncio.create_task(self.receive_responses(ws))

            try:
                await asyncio.gather(send_task, receive_task)
            except asyncio.CancelledError:
                pass
            finally:
                self.running = False

    async def send_audio(self, ws):
        """Capture and send microphone audio."""
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        try:
            while self.running:
                # Read audio chunk
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Send raw PCM to server
                await ws.send(data)

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.05)
        finally:
            stream.stop_stream()
            stream.close()

    async def receive_responses(self, ws):
        """Receive and handle server responses."""
        output_stream = None

        try:
            async for message in ws:
                if isinstance(message, bytes):
                    # Audio data - play it
                    if output_stream is None:
                        output_stream = self.audio.open(
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=SAMPLE_RATE,
                            output=True,
                            frames_per_buffer=CHUNK_SIZE
                        )

                    # Skip WAV header if present
                    if message[:4] == b'RIFF':
                        # Find data chunk
                        idx = message.find(b'data')
                        if idx != -1:
                            # Skip "data" + size (8 bytes)
                            message = message[idx + 8:]

                    output_stream.write(message)
                else:
                    # JSON message
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "processing":
                            print("🎤 Processing your speech...")
                        elif msg_type == "transcript":
                            transcript = data.get("data", {})
                            if "user" in transcript:
                                print(f"You: {transcript['user']}")
                            if "assistant" in transcript:
                                print(f"AI: {transcript['assistant']}")
                        elif msg_type == "done":
                            print("✓ Response complete\n")
                        elif msg_type == "ping":
                            pass  # Keep-alive
                        elif msg_type == "error":
                            print(f"❌ Error: {data.get('message')}")
                        else:
                            print(f"[{msg_type}]: {data}")
                    except json.JSONDecodeError:
                        print(f"[Unknown]: {message}")
        finally:
            if output_stream:
                output_stream.stop_stream()
                output_stream.close()

    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex WebSocket Client")
    parser.add_argument("server_ip", help="Server IP address")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--voice", default="NATF2", help="Voice preset (default: NATF2)")
    parser.add_argument("--prompt", default="You are a helpful assistant.", help="System prompt")

    args = parser.parse_args()

    server_url = f"ws://{args.server_ip}:{args.port}/ws/conversation"

    print("=" * 50)
    print("PersonaPlex Real-time Conversation Client")
    print("=" * 50)
    print(f"Server: {server_url}")
    print(f"Voice: {args.voice}")
    print(f"Prompt: {args.prompt[:50]}...")
    print("=" * 50)

    client = PersonaPlexClient(server_url, args.voice, args.prompt)

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n\nDisconnecting...")
    finally:
        client.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
