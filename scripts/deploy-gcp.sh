#!/bin/bash
# Deploy PersonaPlex to Google Cloud Compute Engine with GPU
# Prerequisites: gcloud CLI authenticated, project configured
#
# PersonaPlex requires significant GPU memory - A100 80GB recommended
# Uses Spot VM for ~70% cost savings

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-personaplex-server}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"  # A100 40GB
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-a100}"
DISK_SIZE="${DISK_SIZE:-200}"  # PersonaPlex models are large (~14GB)
USE_SPOT="${USE_SPOT:-true}"
HF_TOKEN="${HF_TOKEN:-}"

echo "=== Deploying PersonaPlex to GCP ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "Machine: $MACHINE_TYPE with $GPU_TYPE"
echo "Disk: ${DISK_SIZE}GB"
echo "Spot VM: $USE_SPOT"
echo ""

if [ -z "$HF_TOKEN" ]; then
  echo "WARNING: HF_TOKEN not set. You'll need to set it on the instance."
  echo "  export HF_TOKEN=your_token"
  echo ""
fi

# Build spot VM flags
SPOT_FLAGS=""
if [ "$USE_SPOT" = "true" ]; then
  SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
  echo "Using Spot VM (preemptible) for cost savings"
  echo ""
fi

# Check if instance exists
if gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
  echo "Instance $INSTANCE_NAME already exists."
  read -p "Delete and recreate? (y/N): " confirm
  if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "Deleting existing instance..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet
  else
    echo "Aborting."
    exit 1
  fi
fi

# Create instance with GPU
echo "Creating VM instance..."
gcloud compute instances create "$INSTANCE_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=$GPU_TYPE,count=1" \
  --maintenance-policy=TERMINATE \
  $SPOT_FLAGS \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size="${DISK_SIZE}GB" \
  --boot-disk-type=pd-ssd \
  --scopes=cloud-platform \
  --metadata=startup-script='#!/bin/bash
set -e
exec > >(tee /var/log/startup-script.log) 2>&1

echo "=== PersonaPlex Setup Starting ==="
echo "Time: $(date)"

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
pip install fastapi uvicorn python-multipart aiofiles

# Create the API server
echo "Creating API server..."
mkdir -p /opt/personaplex/server
cat > /opt/personaplex/server/main.py << 'SERVEREOF'
"""
PersonaPlex API Server

Provides REST and WebSocket endpoints for PersonaPlex speech-to-speech model.
"""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add personaplex to path
sys.path.insert(0, "/opt/personaplex/personaplex")

# Global model state
model_state = {
    "loaded": False,
    "lm_gen": None,
    "mimi": None,
    "device": None,
}

VOICE_PROMPTS = [
    "NATF0", "NATF1", "NATF2", "NATF3",
    "NATM0", "NATM1", "NATM2", "NATM3",
    "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",
    "VARM0", "VARM1", "VARM2", "VARM3", "VARM4",
]

def load_model():
    """Load PersonaPlex model."""
    global model_state

    if model_state["loaded"]:
        return

    print("Loading PersonaPlex model...")

    try:
        from moshi.models import loaders

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load model from HuggingFace
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable not set")

        mimi, lm_gen = loaders.get_moshi_mimi(
            "nvidia/personaplex-7b-v1",
            device=device,
            dtype=torch.bfloat16,
        )

        model_state["mimi"] = mimi
        model_state["lm_gen"] = lm_gen
        model_state["device"] = device
        model_state["loaded"] = True

        print("PersonaPlex model loaded successfully!")

    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request if HF_TOKEN is set.")
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
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "ok",
        "model": "personaplex-7b-v1",
        "model_loaded": model_state["loaded"],
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
    }

@app.get("/v1/models")
async def list_models():
    """List available models."""
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
    """List available voice prompts."""
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
    text_prompt: str = Form(default="You are a helpful assistant. Answer questions clearly and concisely."),
    seed: int = Form(default=42),
):
    """
    Process speech input and generate speech response.

    - audio: Input WAV file (24kHz recommended)
    - voice: Voice prompt (e.g., NATF2, NATM1)
    - text_prompt: System prompt for the AI persona
    - seed: Random seed for reproducibility
    """
    if not model_state["loaded"]:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

    if voice not in VOICE_PROMPTS:
        raise HTTPException(status_code=400, detail=f"Invalid voice. Choose from: {VOICE_PROMPTS}")

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
        content = await audio.read()
        input_file.write(content)
        input_path = input_file.name

    # Create output paths
    output_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    output_json = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name

    try:
        # Run offline inference using the moshi CLI
        cmd = [
            "python", "-m", "moshi.offline",
            "--voice-prompt", f"{voice}.pt",
            "--text-prompt", text_prompt,
            "--input-wav", input_path,
            "--seed", str(seed),
            "--output-wav", output_wav,
            "--output-text", output_json,
        ]

        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd="/opt/personaplex/personaplex",
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {result.stderr}"
            )

        # Read transcript
        transcript = {}
        if os.path.exists(output_json):
            with open(output_json) as f:
                transcript = json.load(f)

        # Return audio file
        return FileResponse(
            output_wav,
            media_type="audio/wav",
            headers={
                "X-Transcript": json.dumps(transcript),
            }
        )

    finally:
        # Cleanup input file
        if os.path.exists(input_path):
            os.unlink(input_path)

@app.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    """
    Real-time WebSocket conversation endpoint.

    Protocol:
    1. Client sends config JSON: {"voice": "NATF2", "text_prompt": "..."}
    2. Client sends audio chunks (raw PCM, 24kHz, 16-bit, mono)
    3. Server sends back audio chunks and transcripts
    """
    await websocket.accept()
    print("WebSocket connection established")

    config = {
        "voice": "NATF2",
        "text_prompt": "You are a helpful assistant.",
    }

    try:
        # First message should be config
        first_msg = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
        try:
            config = json.loads(first_msg)
            print(f"Config received: {config}")
            await websocket.send_text(json.dumps({"type": "config_ack", "config": config}))
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({"type": "error", "message": "First message must be JSON config"}))
            return

        # For now, use offline mode for each turn
        # TODO: Implement streaming with moshi.server integration

        audio_buffer = bytearray()
        silence_counter = 0
        SILENCE_THRESHOLD = 500  # bytes of near-silence to trigger processing

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                audio_buffer.extend(data)

                # Simple VAD: check if we have enough audio and detect silence
                if len(audio_buffer) > 48000:  # ~1 second at 24kHz 16-bit
                    # Check last chunk for silence
                    last_chunk = audio_buffer[-1024:]
                    avg_amplitude = sum(abs(b - 128) for b in last_chunk) / len(last_chunk)

                    if avg_amplitude < 10:  # Near silence
                        silence_counter += 1
                    else:
                        silence_counter = 0

                    # If enough silence, process the audio
                    if silence_counter > 3:
                        await websocket.send_text(json.dumps({"type": "processing"}))

                        # Save audio to temp file and process
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            # Write WAV header
                            import struct
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
                                "python", "-m", "moshi.offline",
                                "--voice-prompt", f"{config.get('voice', 'NATF2')}.pt",
                                "--text-prompt", config.get("text_prompt", "You are helpful."),
                                "--input-wav", input_path,
                                "--output-wav", output_wav,
                                "--output-text", output_json,
                            ]

                            result = subprocess.run(
                                cmd,
                                cwd="/opt/personaplex/personaplex",
                                capture_output=True,
                                text=True,
                                timeout=60,
                            )

                            if result.returncode == 0 and os.path.exists(output_wav):
                                # Send transcript
                                if os.path.exists(output_json):
                                    with open(output_json) as f:
                                        transcript = json.load(f)
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "data": transcript
                                    }))

                                # Send audio
                                with open(output_wav, "rb") as f:
                                    audio_data = f.read()
                                await websocket.send_bytes(audio_data)
                                await websocket.send_text(json.dumps({"type": "done"}))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "error",
                                    "message": result.stderr or "Processing failed"
                                }))
                        finally:
                            for f in [input_path, output_wav, output_json]:
                                if os.path.exists(f):
                                    os.unlink(f)

                        # Reset buffer
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

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Please run: export HF_TOKEN=your_huggingface_token"
    echo "You need to accept the license at https://huggingface.co/nvidia/personaplex-7b-v1"
    exit 1
fi

# Activate conda environment if available
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base
fi

# Start the API server
echo "Starting PersonaPlex API server..."
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
echo ""
echo "To start the server:"
echo "  1. Set your HuggingFace token: export HF_TOKEN=your_token"
echo "  2. Run: /opt/personaplex/start.sh"
echo ""
echo "Or configure systemd service:"
echo "  1. Edit /etc/systemd/system/personaplex.service"
echo "  2. Set Environment=\"HF_TOKEN=your_token\""
echo "  3. systemctl start personaplex"
echo ""
'

# Create firewall rule if it does not exist
echo "Configuring firewall..."
gcloud compute firewall-rules create allow-personaplex-api \
  --project="$PROJECT_ID" \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server \
  --description="Allow PersonaPlex API access" 2>/dev/null || true

# Add network tag to instance
gcloud compute instances add-tags "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --tags=http-server

echo ""
echo "=== Deployment initiated ==="
echo ""
echo "The instance will:"
echo "  1. Install dependencies (~5 min)"
echo "  2. Clone PersonaPlex repository"
echo "  3. Set up API server"
echo ""
echo "Total setup time: ~10-15 minutes"
echo ""

# Wait a moment for instance to be accessible
sleep 10

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "pending...")

echo "External IP: $EXTERNAL_IP"
echo ""
echo "Commands:"
echo "  # SSH into instance"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "  # Check startup logs"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command='tail -f /var/log/startup-script.log'"
echo ""
echo "  # Set HF token and start server"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command='export HF_TOKEN=your_token && /opt/personaplex/start.sh'"
echo ""
echo "Once ready, test at:"
echo "  curl http://$EXTERNAL_IP:8000/health"
echo ""

if [ "$USE_SPOT" = "true" ]; then
  echo "Note: This is a Spot VM. If preempted, restart with:"
  echo "  gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
fi
