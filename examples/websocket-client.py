#!/usr/bin/env python3
"""
PersonaPlex WebSocket Client Example

Real-time speech-to-speech conversation with PersonaPlex.

Usage:
    python websocket-client.py <server_ip>

Requirements:
    pip install websockets pyaudio numpy
"""

import asyncio
import sys
import json
import struct
import wave
import io

try:
    import websockets
    import pyaudio
    import numpy as np
except ImportError:
    print("Please install dependencies:")
    print("  pip install websockets pyaudio numpy")
    sys.exit(1)


# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16


class PersonaPlexClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.audio = pyaudio.PyAudio()
        self.running = False

    async def run(self):
        """Main conversation loop."""
        print(f"Connecting to {self.server_url}...")

        async with websockets.connect(self.server_url) as ws:
            print("Connected! Start speaking...")
            print("Press Ctrl+C to stop")

            self.running = True

            # Start audio input/output tasks
            send_task = asyncio.create_task(self.send_audio(ws))
            receive_task = asyncio.create_task(self.receive_audio(ws))

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

                # Send to server
                await ws.send(data)

                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        finally:
            stream.stop_stream()
            stream.close()

    async def receive_audio(self, ws):
        """Receive and play audio responses."""
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )

        try:
            async for message in ws:
                if isinstance(message, bytes):
                    # Audio data - play it
                    stream.write(message)
                else:
                    # JSON message (transcript, etc.)
                    try:
                        data = json.loads(message)
                        if data.get("type") == "transcript":
                            print(f"\n[Transcript]: {data.get('text', '')}")
                        elif data.get("type") == "response":
                            print(f"\n[Response]: {data.get('text', '')}")
                    except json.JSONDecodeError:
                        print(f"[Message]: {message}")
        finally:
            stream.stop_stream()
            stream.close()

    def cleanup(self):
        """Clean up audio resources."""
        self.audio.terminate()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python websocket-client.py <server_ip>")
        print("Example: python websocket-client.py 35.238.3.33")
        sys.exit(1)

    server_ip = sys.argv[1]
    server_url = f"ws://{server_ip}:8000/ws/conversation"

    client = PersonaPlexClient(server_url)

    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nDisconnecting...")
    finally:
        client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
