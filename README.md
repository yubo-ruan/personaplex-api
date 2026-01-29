# PersonaPlex API

Self-hosted PersonaPlex speech-to-speech model on GCP with GPU support.

## Overview

PersonaPlex is NVIDIA's real-time speech-to-speech conversational AI model. This repository provides deployment scripts for running PersonaPlex on Google Cloud Platform with GPU acceleration.

**Note:** PersonaPlex currently only supports English. No Chinese or Japanese fine-tuned versions exist.

## Requirements

- Google Cloud Platform account with GPU quota
- `gcloud` CLI installed and authenticated
- GPU: NVIDIA L4 or better (L4 recommended for cost/performance)

## Quick Start

### 1. Deploy to GCP

```bash
# Deploy with default settings (L4 GPU, Spot VM)
./scripts/deploy-gcp.sh

# Or customize deployment
ZONE=us-west1-a GPU_TYPE=nvidia-l4 ./scripts/deploy-gcp.sh
```

### 2. Test the API

```bash
# Get external IP
gcloud compute instances describe personaplex-server --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Test health endpoint
curl http://<EXTERNAL_IP>:8000/health

# Test speech-to-speech (WebSocket)
# See examples/websocket-client.py
```

## API Endpoints

### WebSocket: `/ws/conversation`

Real-time bidirectional speech conversation.

```javascript
const ws = new WebSocket('ws://<host>:8000/ws/conversation');

// Send audio chunks (16kHz, 16-bit PCM)
ws.send(audioChunk);

// Receive audio responses
ws.onmessage = (event) => {
  playAudio(event.data);
};
```

### REST: `/v1/audio/speech` (TTS only)

OpenAI-compatible text-to-speech endpoint.

```bash
curl -X POST http://<host>:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "alloy"}' \
  --output speech.wav
```

## Configuration

Environment variables for deployment:

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | (from gcloud) | GCP project ID |
| `GCP_ZONE` | `us-central1-a` | Deployment zone |
| `INSTANCE_NAME` | `personaplex-server` | VM instance name |
| `MACHINE_TYPE` | `g2-standard-8` | Machine type (g2 for L4) |
| `GPU_TYPE` | `nvidia-l4` | GPU type |
| `USE_SPOT` | `true` | Use Spot VM for savings |

## Cost Estimates

Using Spot VMs for ~70% cost savings:

| GPU | On-Demand | Spot | Notes |
|-----|-----------|------|-------|
| L4 | ~$0.70/hr | ~$0.21/hr | Recommended |
| T4 | ~$0.35/hr | ~$0.11/hr | Budget option |
| A100 | ~$3.67/hr | ~$1.10/hr | High performance |

## Architecture

```
┌─────────────────────────────────────────┐
│              GCP VM (g2-standard-8)      │
│  ┌─────────────────────────────────────┐ │
│  │         PersonaPlex Server          │ │
│  │  ┌─────────┐      ┌─────────────┐  │ │
│  │  │ ASR     │ ──── │ LLM         │  │ │
│  │  │ (Speech │      │ (Response   │  │ │
│  │  │  Input) │      │  Generation)│  │ │
│  │  └─────────┘      └─────────────┘  │ │
│  │       │                  │         │ │
│  │       └──────┬───────────┘         │ │
│  │              ▼                     │ │
│  │        ┌─────────┐                 │ │
│  │        │ TTS     │                 │ │
│  │        │ (Speech │                 │ │
│  │        │  Output)│                 │ │
│  │        └─────────┘                 │ │
│  └─────────────────────────────────────┘ │
│                 │                        │
│            NVIDIA L4 GPU                 │
└─────────────────────────────────────────┘
```

## Monitoring

```bash
# SSH into instance
gcloud compute ssh personaplex-server --zone=us-central1-a

# Check logs
docker logs personaplex -f

# Check GPU utilization
nvidia-smi -l 1
```

## Stopping/Starting

```bash
# Stop instance (saves cost)
gcloud compute instances stop personaplex-server --zone=us-central1-a

# Start instance
gcloud compute instances start personaplex-server --zone=us-central1-a
```

## License

MIT
