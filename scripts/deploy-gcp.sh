#!/bin/bash
# Deploy PersonaPlex to Google Cloud Compute Engine with GPU
# Supports both A100 (guaranteed) and L4 (cheaper, with CPU offload)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-personaplex-server}"
GPU_TYPE="${GPU_TYPE:-l4}"
DISK_SIZE="${DISK_SIZE:-200}"
USE_SPOT="${USE_SPOT:-true}"

# Set machine type based on GPU
case "$GPU_TYPE" in
  l4|L4|nvidia-l4)
    MACHINE_TYPE="g2-standard-8"
    GPU_ACCELERATOR="nvidia-l4"
    CPU_OFFLOAD="true"
    echo "Using L4 GPU (24GB VRAM) - cheaper but needs CPU offload"
    ;;
  a100|A100|nvidia-tesla-a100)
    MACHINE_TYPE="a2-highgpu-1g"
    GPU_ACCELERATOR="nvidia-tesla-a100"
    CPU_OFFLOAD="false"
    echo "Using A100 GPU (40GB VRAM) - recommended"
    ;;
  *)
    echo "Unknown GPU_TYPE: $GPU_TYPE"
    echo "Supported: l4, a100"
    exit 1
    ;;
esac

echo ""
echo "=== Deploying PersonaPlex to GCP ==="
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Instance: $INSTANCE_NAME"
echo "Machine: $MACHINE_TYPE with $GPU_ACCELERATOR"
echo "Disk: ${DISK_SIZE}GB"
echo "Spot VM: $USE_SPOT"
echo "CPU Offload: $CPU_OFFLOAD"
echo ""

# Build spot VM flags
SPOT_FLAGS=""
if [ "$USE_SPOT" = "true" ]; then
  SPOT_FLAGS="--provisioning-model=SPOT --instance-termination-action=STOP"
  echo "Using Spot VM for cost savings"
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
  --accelerator="type=$GPU_ACCELERATOR,count=1" \
  --maintenance-policy=TERMINATE \
  $SPOT_FLAGS \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size="${DISK_SIZE}GB" \
  --boot-disk-type=pd-ssd \
  --scopes=cloud-platform \
  --metadata="cpu-offload=$CPU_OFFLOAD" \
  --metadata-from-file="startup-script=$SCRIPT_DIR/startup-script.sh"

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
echo "Setup time: ~10-15 minutes"
echo ""

# Wait for instance
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
echo "Test at: curl http://$EXTERNAL_IP:8000/health"
echo ""

if [ "$USE_SPOT" = "true" ]; then
  echo "Note: Spot VM. If preempted, restart with:"
  echo "  gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
fi
