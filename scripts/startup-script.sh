#!/bin/bash
set -e
exec > >(tee /var/log/startup-script.log) 2>&1

echo "=== PersonaPlex Setup Starting ==="
echo "Time: $(date)"

# Get CPU offload setting from metadata
CPU_OFFLOAD=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/cpu-offload" -H "Metadata-Flavor: Google" || echo "false")
echo "CPU Offload: $CPU_OFFLOAD"

# Create app directory
mkdir -p /opt/personaplex
cd /opt/personaplex

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y libopus-dev ffmpeg git

# Clone PersonaPlex repository
echo "Cloning PersonaPlex repository..."
if [ ! -d "/opt/personaplex/personaplex" ]; then
  git clone https://github.com/NVIDIA/personaplex.git /opt/personaplex/personaplex
fi

cd /opt/personaplex/personaplex

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install ./moshi/.
pip install fastapi uvicorn python-multipart aiofiles accelerate

# Create the API server
echo "Creating API server..."
mkdir -p /opt/personaplex/server
cat > /opt/personaplex/server/main.py << 'SERVEREOF'
"""
PersonaPlex API Server
"""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
import struct
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, "/opt/personaplex/personaplex")

model_state = {
    "loaded": False,
    "device": None,
    "cpu_offload": os.environ.get("CPU_OFFLOAD", "false").lower() == "true",
}

VOICE_PROMPTS = [
    "NATF0", "NATF1", "NATF2", "NATF3",
    "NATM0", "NATM1", "NATM2", "NATM3",
    "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",
    "VARM0", "VARM1", "VARM2", "VARM3", "VARM4",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"PersonaPlex API starting...")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CPU Offload: {model_state['cpu_offload']}")
    yield

app = FastAPI(
    title="PersonaPlex API",
    description="Real-time speech-to-speech conversational AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = None
    if gpu_available:
        props = torch.cuda.get_device_properties(0)
        gpu_memory = f"{props.total_memory / 1024**3:.1f} GB"

    return {
        "status": "ok",
        "model": "personaplex-7b-v1",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory": gpu_memory,
        "cpu_offload": model_state["cpu_offload"],
    }

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "personaplex-7b-v1",
                "object": "model",
                "owned_by": "nvidia",
                "permission": [],
            }
        ]
    }

@app.get("/v1/voices")
async def list_voices():
    return {
        "voices": VOICE_PROMPTS,
        "categories": {
            "natural_female": ["NATF0", "NATF1", "NATF2", "NATF3"],
            "natural_male": ["NATM0", "NATM1", "NATM2", "NATM3"],
            "variety_female": ["VARF0", "VARF1", "VARF2", "VARF3", "VARF4"],
            "variety_male": ["VARM0", "VARM1", "VARM2", "VARM3", "VARM4"],
        }
    }

@app.post("/v1/audio/speech")
async def speech_to_speech(
    audio: UploadFile = File(...),
    voice: str = Form(default="NATF2"),
    text_prompt: str = Form(default="You are a helpful assistant."),
    seed: int = Form(default=42),
):
    if voice not in VOICE_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Invalid voice. Choose from: {VOICE_PROMPTS}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
        content = await audio.read()
        input_file.write(content)
        input_path = input_file.name

    output_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    output_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

    try:
        cmd = [
            "python3", "-m", "moshi.offline",
            "--voice-prompt", f"{voice}.pt",
            "--text-prompt", text_prompt,
            "--input-wav", input_path,
            "--seed", str(seed),
            "--output-wav", output_wav,
            "--output-text", output_json,
        ]

        if model_state["cpu_offload"]:
            cmd.append("--cpu-offload")

        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd="/opt/personaplex/personaplex",
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")

        transcript = {}
        if os.path.exists(output_json):
            with open(output_json) as f:
                transcript = json.load(f)

        return FileResponse(
            output_wav,
            media_type="audio/wav",
            headers={"X-Transcript": json.dumps(transcript)},
        )

    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    config = {"voice": "NATF2", "text_prompt": "You are a helpful assistant."}

    try:
        first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        try:
            config = json.loads(first_msg)
            print(f"Config received: {config}")
            await websocket.send_text(json.dumps({"type": "config_ack", "config": config}))
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"type": "error", "message": "First message must be JSON config"}))
            return

        audio_buffer = bytearray()
        silence_counter = 0

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                audio_buffer.extend(data)

                if len(audio_buffer) > 48000:
                    last_chunk = audio_buffer[-1024:]
                    avg_amplitude = sum(abs(b - 128) for b in last_chunk) / len(last_chunk)

                    if avg_amplitude < 10:
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    if silence_counter > 3:
                        await websocket.send_text(json.dumps({"type": "processing"}))

                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            sample_rate = 24000
                            num_channels = 1
                            bits_per_sample = 16
                            data_size = len(audio_buffer)

                            f.write(b'RIFF')
                            f.write(struct.pack('<I', 36 + data_size))
                            f.write(b'WAVE')
                            f.write(b'fmt ')
                            f.write(struct.pack('<I', 16))
                            f.write(struct.pack('<H', 1))
                            f.write(struct.pack('<H', num_channels))
                            f.write(struct.pack('<I', sample_rate))
                            f.write(struct.pack('<I', sample_rate * num_channels * bits_per_sample // 8))
                            f.write(struct.pack('<H', num_channels * bits_per_sample // 8))
                            f.write(struct.pack('<H', bits_per_sample))
                            f.write(b'data')
                            f.write(struct.pack('<I', data_size))
                            f.write(bytes(audio_buffer))
                            input_path = f.name

                        output_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                        output_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

                        try:
                            cmd = [
                                "python3", "-m", "moshi.offline",
                                "--voice-prompt", f"{config.get('voice', 'NATF2')}.pt",
                                "--text-prompt", config.get("text_prompt", "You are helpful."),
                                "--input-wav", input_path,
                                "--output-wav", output_wav,
                                "--output-text", output_json,
                            ]

                            if model_state["cpu_offload"]:
                                cmd.append("--cpu-offload")

                            result = subprocess.run(
                                cmd,
                                cwd="/opt/personaplex/personaplex",
                                capture_output=True,
                                text=True,
                                timeout=120,
                            )

                            if result.returncode == 0 and os.path.exists(output_wav):
                                if os.path.exists(output_json):
                                    with open(output_json) as f:
                                        transcript = json.load(f)
                                    await websocket.send_text(json.dumps({"type": "transcript", "data": transcript}))

                                with open(output_wav, "rb") as f:
                                    audio_data = f.read()
                                await websocket.send_bytes(audio_data)
                                await websocket.send_text(json.dumps({"type": "done"}))
                            else:
                                await websocket.send_text(json.dumps({"type": "error", "message": result.stderr or "Processing failed"}))
                        finally:
                            for f in [input_path, output_wav, output_json]:
                                if os.path.exists(f):
                                    os.unlink(f)

                        audio_buffer = bytearray()
                        silence_counter = 0

            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
SERVEREOF

# Create start script
cat > /opt/personaplex/start.sh << 'STARTSCRIPT'
#!/bin/bash
cd /opt/personaplex

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Please run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

CPU_OFFLOAD=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/cpu-offload" -H "Metadata-Flavor: Google" 2>/dev/null || echo "false")
export CPU_OFFLOAD

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi

echo "Starting PersonaPlex API server..."
echo "CPU Offload: $CPU_OFFLOAD"
cd /opt/personaplex/server
python main.py
STARTSCRIPT
chmod +x /opt/personaplex/start.sh

# Create systemd service
cat > /etc/systemd/system/personaplex.service << 'SVCFILE'
[Unit]
Description=PersonaPlex API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/personaplex/server
Environment="HF_TOKEN="
ExecStart=/opt/personaplex/start.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCFILE

systemctl daemon-reload

echo "=== PersonaPlex Setup Complete ==="
echo "To start: export HF_TOKEN=your_token && /opt/personaplex/start.sh"
