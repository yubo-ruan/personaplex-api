#!/bin/bash
# Start PersonaPlex instance

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-personaplex-server}"

echo "Starting $INSTANCE_NAME..."
gcloud compute instances start "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID"

# Wait for instance to be running
echo "Waiting for instance to be ready..."
sleep 30

# Get external IP
EXTERNAL_IP=$(gcloud compute instances describe "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "Instance started!"
echo "External IP: $EXTERNAL_IP"
echo ""
echo "Start the PersonaPlex server:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='/opt/personaplex/start.sh'"
echo ""
echo "Test at:"
echo "  curl http://$EXTERNAL_IP:8000/health"
