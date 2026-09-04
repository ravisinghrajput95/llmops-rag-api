#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# gcloud-only provisioning, as an alternative to terraform/ for anyone who
# wants to see exactly which API calls create the infrastructure.
#
# Everything created here is either Always Free or free at demo scale.
#
# Usage:
#   export PROJECT_ID=my-project
#   export GITHUB_REPO=owner/repo
#   export OPENAI_API_KEY=sk-...
#   ./scripts/bootstrap.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
GITHUB_REPO="${GITHUB_REPO:?set GITHUB_REPO as owner/repo}"
OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"
# us-central1/us-east1/us-west1 only: the 5 GB Always Free storage allowance
# does not apply anywhere else.
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-llmops-rag-api}"
BUCKET="${PROJECT_ID}-mlflow-artifacts"
RUNTIME_SA="${SERVICE}-run"
DEPLOYER_SA="${SERVICE}-deployer"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

gcloud config set project "$PROJECT_ID" >/dev/null

say "Enabling APIs (free; only usage is billed)"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  logging.googleapis.com \
  --quiet

say "Artifact Registry repository (Always Free up to 0.5 GB)"
gcloud artifacts repositories create "$SERVICE" \
  --repository-format=docker \
  --location="$REGION" \
  --description="Container images for $SERVICE" \
  --quiet 2>/dev/null || echo "  already exists, skipping"

say "GCS bucket for MLflow artifacts (Always Free up to 5 GB in US regions)"
gcloud storage buckets create "gs://${BUCKET}" \
  --location="$REGION" \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access \
  --quiet 2>/dev/null || echo "  already exists, skipping"

# Without a lifecycle rule, artifacts accumulate until they cross the free 5 GB.
say "Applying a 30-day deletion rule to the bucket"
cat > /tmp/lifecycle.json <<'JSON'
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30}}]}
JSON
gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file=/tmp/lifecycle.json --quiet
rm -f /tmp/lifecycle.json

say "Runtime service account (least privilege, not the default compute SA)"
gcloud iam service-accounts create "$RUNTIME_SA" \
  --display-name="Runtime identity for $SERVICE" \
  --quiet 2>/dev/null || echo "  already exists, skipping"

RUNTIME_EMAIL="${RUNTIME_SA}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/storage.objectAdmin" --quiet >/dev/null

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/logging.logWriter" --quiet >/dev/null

say "Storing the OpenAI key in Secret Manager (Always Free up to 6 versions)"
if ! gcloud secrets describe "${SERVICE}-openai-api-key" >/dev/null 2>&1; then
  gcloud secrets create "${SERVICE}-openai-api-key" \
    --replication-policy=user-managed --locations="$REGION" --quiet
fi
printf '%s' "$OPENAI_API_KEY" | \
  gcloud secrets versions add "${SERVICE}-openai-api-key" --data-file=- --quiet >/dev/null

gcloud secrets add-iam-policy-binding "${SERVICE}-openai-api-key" \
  --member="serviceAccount:${RUNTIME_EMAIL}" \
  --role="roles/secretmanager.secretAccessor" --quiet >/dev/null

say "Deploying a placeholder Cloud Run service"
# --min-instances=0 is the single most important flag on this page: it means
# the service costs nothing while idle. --max-instances caps the worst case.
gcloud run deploy "$SERVICE" \
  --image="us-docker.pkg.dev/cloudrun/container/hello" \
  --region="$REGION" \
  --platform=managed \
  --service-account="$RUNTIME_EMAIL" \
  --min-instances=0 \
  --max-instances=2 \
  --cpu=1 \
  --memory=512Mi \
  --timeout=60s \
  --concurrency=80 \
  --allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},SERVICE_NAME=${SERVICE},CHROMA_DIR=/tmp/chroma,MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow.db,MLFLOW_ARTIFACT_LOCATION=gs://${BUCKET}/mlflow,MLFLOW_EXPERIMENT=llmops-rag-demo,LOG_LEVEL=INFO" \
  --set-secrets="OPENAI_API_KEY=${SERVICE}-openai-api-key:latest" \
  --quiet

URL=$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')

say "Done"
cat <<SUMMARY

  Service URL : ${URL}
  Registry    : ${REGION}-docker.pkg.dev/${PROJECT_ID}/${SERVICE}
  Bucket      : gs://${BUCKET}

  Next:
    1. ./scripts/setup_wif.sh          # keyless GitHub Actions deploys
    2. ./scripts/set_budget.sh         # budget alert -- do this today
    3. git push origin main            # CI builds and deploys the real image

  Before your trial credits expire:
    ./scripts/cost_check.sh            # audit what is running
    ./scripts/teardown.sh              # delete everything billable
SUMMARY
