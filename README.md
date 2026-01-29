# PersonaPlex API

Self-hosted [PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) speech-to-speech model API on GCP with GPU support.

## Overview

PersonaPlex is NVIDIA's real-time, full-duplex speech-to-speech conversational AI model. It enables natural conversations with persona control through text-based role prompts and audio-based voice conditioning.

**Key Features:**
- Real-time speech-to-speech conversation
- 18 voice presets (natural and variety voices)
- Custom persona prompts
- WebSocket API for streaming
- REST API for batch processing

**Requirements:**
- NVIDIA A100 GPU (40GB+ VRAM recommended)
- HuggingFace account with accepted [NVIDIA Open Model License](https://huggingface.co/nvidia/personaplex-7b-v1)

## Quick Start

### 1. Accept the License

Visit [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) and accept the license agreement.

### 2. Get HuggingFace Token

Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Deploy to GCP

```bash
# Set your HuggingFace token
export HF_TOKEN=your_huggingface_token

# Option 1: Deploy with L4 GPU (cheaper, ~$0.21/hr Spot)
GPU_TYPE=l4 ./scripts/deploy-gcp.sh

# Option 2: Deploy with A100 GPU (faster, ~$1.10/hr Spot)
GPU_TYPE=a100 ./scripts/deploy-gcp.sh

# Option 3: Deploy both to compare
./scripts/deploy-both.sh
```

**GPU Comparison:**
| GPU | VRAM | Spot Cost | CPU Offload | Performance |
|-----|------|-----------|-------------|-------------|
| L4 | 24GB | ~$0.21/hr | Required | Slower but cheaper |
| A100 | 40GB | ~$1.10/hr | Not needed | Recommended |

### 4. Start the Server

```bash
# SSH into the instance
gcloud compute ssh personaplex-server --zone=us-central1-a

# Set token and start server
export HF_TOKEN=your_token
/opt/personaplex/start.sh
```

### 5. Test the API

```bash
# Health check
curl http://<EXTERNAL_IP>:8000/health

# List voices
curl http://<EXTERNAL_IP>:8000/v1/voices

# Speech-to-speech (REST)
curl -X POST http://<EXTERNAL_IP>:8000/v1/audio/speech \
  -F "audio=@input.wav" \
  -F "voice=NATF2" \
  -F "text_prompt=You are a helpful assistant." \
  --output response.wav
```

## API Reference

### Health Check

```
GET /health
```

Returns server status and GPU info.

### List Voices

```
GET /v1/voices
```

Returns available voice presets:
- **Natural Female**: NATF0, NATF1, NATF2, NATF3
- **Natural Male**: NATM0, NATM1, NATM2, NATM3
- **Variety Female**: VARF0, VARF1, VARF2, VARF3, VARF4
- **Variety Male**: VARM0, VARM1, VARM2, VARM3, VARM4

### Speech-to-Speech (REST)

```
POST /v1/audio/speech
Content-Type: multipart/form-data
```

**Parameters:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| audio | file | required | Input WAV file (24kHz recommended) |
| voice | string | NATF2 | Voice preset |
| text_prompt | string | "You are a helpful assistant..." | System prompt |
| seed | int | 42 | Random seed |

**Response:** WAV audio file with `X-Transcript` header containing JSON transcript.

### WebSocket Conversation

```
WS /ws/conversation
```

**Protocol:**
1. Connect to WebSocket
2. Send config JSON: `{"voice": "NATF2", "text_prompt": "..."}`
3. Send raw PCM audio chunks (24kHz, 16-bit, mono)
4. Receive responses:
   - `{"type": "processing"}` - Processing started
   - `{"type": "transcript", "data": {...}}` - Transcript
   - Binary data - Response audio
   - `{"type": "done"}` - Turn complete

## Example Prompts

**QA Assistant:**
```
You are a wise and friendly teacher. Answer questions or provide advice clearly.
```

**Casual Conversation:**
```
You enjoy having a good conversation.
```

**Customer Service:**
```
You work for CitySan Services (waste management). Your name is Ayelen Lucero.
You need to verify customer Omar Torres before proceeding with their request.
```

## Python Client Example

```python
import requests

# Speech-to-speech
with open("input.wav", "rb") as f:
    response = requests.post(
        "http://your-server:8000/v1/audio/speech",
        files={"audio": f},
        data={
            "voice": "NATF2",
            "text_prompt": "You are a helpful assistant.",
        }
    )

if response.ok:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("Transcript:", response.headers.get("X-Transcript"))
```

See [examples/](examples/) for more client examples.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | required | HuggingFace token |
| `GCP_PROJECT_ID` | (from gcloud) | GCP project ID |
| `GCP_ZONE` | us-central1-a | Deployment zone |
| `INSTANCE_NAME` | personaplex-server | VM instance name |
| `GPU_TYPE` | l4 | GPU type: `l4` or `a100` |
| `USE_SPOT` | true | Use Spot VM for savings |

## Cost Estimates

| GPU | Machine | On-Demand | Spot | Notes |
|-----|---------|-----------|------|-------|
| L4 24GB | g2-standard-8 | ~$0.70/hr | ~$0.21/hr | Budget option, needs CPU offload |
| A100 40GB | a2-highgpu-1g | ~$3.67/hr | ~$1.10/hr | Recommended |

## Monitoring

```bash
# SSH into instance
gcloud compute ssh personaplex-server --zone=us-central1-a

# Check logs
tail -f /var/log/startup-script.log

# Check GPU utilization
nvidia-smi -l 1

# Check server logs
journalctl -u personaplex -f
```

## Stopping/Starting

```bash
# Stop instance (saves cost)
./scripts/stop-instance.sh

# Start instance
./scripts/start-instance.sh
```

## References

- [PersonaPlex on HuggingFace](https://huggingface.co/nvidia/personaplex-7b-v1)
- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [NVIDIA Research Page](https://research.nvidia.com/labs/adlr/personaplex/)
- [PersonaPlex Paper](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)

## License

Code: MIT
Model: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
