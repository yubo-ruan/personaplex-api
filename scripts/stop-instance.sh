#!/bin/bash
# Stop PersonaPlex instance to save costs

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-personaplex-server}"

echo "Stopping $INSTANCE_NAME..."
gcloud compute instances stop "$INSTANCE_NAME" \
  --zone="$ZONE" \
  --project="$PROJECT_ID"

echo "Instance stopped. Restart with:"
echo "  gcloud compute instances start $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
