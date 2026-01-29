#!/bin/bash
# Deploy both L4 and A100 PersonaPlex instances for comparison
#
# This deploys two VMs:
# - personaplex-l4: L4 GPU (24GB) with CPU offload - cheaper (~$0.21/hr Spot)
# - personaplex-a100: A100 GPU (40GB) - faster (~$1.10/hr Spot)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Deploying both L4 and A100 PersonaPlex VMs"
echo "=========================================="
echo ""

# Deploy L4 instance
echo ">>> Deploying L4 instance (cheaper, with CPU offload)..."
GPU_TYPE=l4 INSTANCE_NAME=personaplex-l4 "$SCRIPT_DIR/deploy-gcp.sh"

echo ""
echo ">>> Waiting 30 seconds before deploying A100..."
sleep 30

# Deploy A100 instance
echo ""
echo ">>> Deploying A100 instance (faster, more VRAM)..."
GPU_TYPE=a100 INSTANCE_NAME=personaplex-a100 "$SCRIPT_DIR/deploy-gcp.sh"

echo ""
echo "=========================================="
echo "Both instances deployed!"
echo "=========================================="
echo ""
echo "Instance comparison:"
echo ""
echo "  personaplex-l4:"
echo "    - GPU: NVIDIA L4 (24GB VRAM)"
echo "    - Machine: g2-standard-8"
echo "    - Cost: ~\$0.21/hr (Spot)"
echo "    - CPU Offload: Enabled"
echo "    - Performance: Slower but cheaper"
echo ""
echo "  personaplex-a100:"
echo "    - GPU: NVIDIA A100 (40GB VRAM)"
echo "    - Machine: a2-highgpu-1g"
echo "    - Cost: ~\$1.10/hr (Spot)"
echo "    - CPU Offload: Disabled"
echo "    - Performance: Faster, recommended"
echo ""
echo "To check instance IPs:"
echo "  gcloud compute instances list --filter='name~personaplex'"
echo ""
echo "To test each instance:"
echo "  curl http://<L4_IP>:8000/health"
echo "  curl http://<A100_IP>:8000/health"
