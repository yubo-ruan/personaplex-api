#!/bin/bash
# Deploy PersonaPlex to Google Cloud Compute Engine with GPU
# Prerequisites: gcloud CLI authenticated, project configured
#
# PersonaPlex requires significant GPU memory - L4 (24GB) recommended
# Uses Spot VM for ~70% cost savings (~$0.21/hr vs ~$0.70/hr for L4)

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-personaplex-server}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"  # g2 for L4 GPU
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
DISK_SIZE="${DISK_SIZE:-200}"  # PersonaPlex models are large
USE_SPOT="${USE_SPOT:-true}"

echo "=== Deploying PersonaPlex to GCP ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "Machine: $MACHINE_TYPE with $GPU_TYPE"
echo "Disk: ${DISK_SIZE}GB"
echo "Spot VM: $USE_SPOT"
echo ""

# Build spot VM flags
SPOT_FLAGS=""
if [ "$USE_SPOT" = "true" ]; then
  SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
  echo "Using Spot VM (preemptible) for ~70% cost savings"
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
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_SIZE}GB" \
  --boot-disk-type=pd-ssd \
  --scopes=cloud-platform \
  --metadata=startup-script='#!/bin/bash
set -e
exec > >(tee /var/log/startup-script.log) 2>&1

echo "=== PersonaPlex Setup Starting ==="
echo "Time: $(date)"

# Install NVIDIA drivers
echo "Installing NVIDIA drivers..."
apt-get update
apt-get install -y linux-headers-$(uname -r)

# Add NVIDIA driver repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA driver
apt-get update
apt-get install -y nvidia-driver-550 nvidia-container-toolkit

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Configure Docker for NVIDIA
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Create app directory
mkdir -p /opt/personaplex
mkdir -p /opt/personaplex/models

# Create docker-compose file
cat > /opt/personaplex/docker-compose.yml << 'COMPOSEEOF'
version: "3.8"
services:
  personaplex:
    image: nvcr.io/nvidia/nemo:24.01.speech
    container_name: personaplex
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - ./app:/app
    command: python /app/server.py
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
COMPOSEEOF

# Create the server application
mkdir -p /opt/personaplex/app
cat > /opt/personaplex/app/server.py << PYEOF
"""
PersonaPlex Server - Speech-to-Speech API

This is a placeholder server. PersonaPlex requires:
1. Access to NVIDIA NGC for model weights
2. NeMo framework setup
3. Specific model checkpoints

For actual deployment, replace with official PersonaPlex implementation.
"""

import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="PersonaPlex API")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "personaplex", "gpu": "available"}

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "personaplex-v1", "object": "model", "owned_by": "nvidia"}
        ]
    }

@app.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    """
    Real-time speech-to-speech conversation endpoint.

    Client sends: Raw audio chunks (16kHz, 16-bit PCM)
    Server sends: Response audio chunks
    """
    await websocket.accept()
    print("WebSocket connection established")

    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            print(f"Received {len(data)} bytes of audio")

            # TODO: Process with PersonaPlex model
            # 1. Run ASR to get text
            # 2. Generate response with LLM
            # 3. Synthesize speech with TTS
            # 4. Send audio back

            # For now, echo back a placeholder response
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": "[Speech recognition placeholder]"
            }))

    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.post("/v1/audio/speech")
async def text_to_speech(request: dict):
    """
    OpenAI-compatible TTS endpoint.
    """
    text = request.get("input", "")
    voice = request.get("voice", "alloy")

    # TODO: Implement TTS with PersonaPlex
    # For now, return error indicating not implemented
    return JSONResponse(
        status_code=501,
        content={"error": "TTS not yet implemented. Use WebSocket for full speech-to-speech."}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
PYEOF

# Create requirements file
cat > /opt/personaplex/app/requirements.txt << REQEOF
fastapi>=0.104.0
uvicorn>=0.24.0
websockets>=12.0
numpy>=1.24.0
REQEOF

# Create a simple start script
cat > /opt/personaplex/start.sh << STARTEOF
#!/bin/bash
cd /opt/personaplex

# Stop any existing container
docker rm -f personaplex-placeholder 2>/dev/null || true

# Check if using full PersonaPlex or placeholder
if [ -f /opt/personaplex/models/personaplex/model.nemo ]; then
  echo "Starting full PersonaPlex server..."
  docker compose up -d
else
  echo "PersonaPlex model not found. Starting placeholder server..."
  # Run simple placeholder server (no GPU needed for placeholder)
  docker run -d --name personaplex-placeholder \
    -p 8000:8000 \
    -v /opt/personaplex/app:/app \
    -w /app \
    python:3.11-slim \
    bash -c "pip install --no-cache-dir fastapi uvicorn websockets && python server.py"
fi
STARTEOF
chmod +x /opt/personaplex/start.sh

# Create systemd service for auto-start after reboot
cat > /etc/systemd/system/personaplex.service << SVCEOF
[Unit]
Description=PersonaPlex Server
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/opt/personaplex/start.sh
ExecStop=/usr/bin/docker stop personaplex-placeholder

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable personaplex.service

echo "=== Waiting for NVIDIA driver to be ready ==="
# Wait for driver installation to complete
sleep 30

# Reboot to load NVIDIA driver
echo "Rebooting to load NVIDIA driver..."
reboot
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
echo "  1. Install NVIDIA drivers (~5 min)"
echo "  2. Reboot to load drivers"
echo "  3. Auto-start placeholder server (via systemd)"
echo ""
echo "Total setup time: ~10 minutes"
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
echo "  # Check GPU status (after reboot)"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command='nvidia-smi'"
echo ""
echo "  # Check server logs (auto-starts after reboot)"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command='docker logs personaplex-placeholder -f'"
echo ""
echo "Once ready, test at:"
echo "  curl http://$EXTERNAL_IP:8000/health"
echo ""

if [ "$USE_SPOT" = "true" ]; then
  echo "Note: This is a Spot VM. If preempted, restart with:"
  echo "  gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
fi
