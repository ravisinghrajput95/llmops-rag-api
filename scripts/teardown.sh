#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Delete every billable resource this project created.
#
# Run this before the trial credits expire, or any time you want spending to
# stop. It asks for confirmation and prints exactly what it will remove first.
#
#   PROJECT_ID=my-project ./scripts/teardown.sh
#
# Add --yes to skip the prompt (for automation).
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-llmops-rag-api}"
BUCKET="${PROJECT_ID}-mlflow-artifacts"
# Read the pool id from terraform.tfvars rather than defaulting to the first
# one this project ever used. Pool ids get bumped after every destroy (they are
# reserved for 30 days and cannot be reused), so a hardcoded default silently
# leaves the live pool running while this script reports a clean teardown --
# and the pool is the one resource whose name you must not lose track of.
_TFVARS="$(dirname "$0")/../terraform/terraform.tfvars"
if [ -z "${WIF_POOL:-}" ] && [ -f "$_TFVARS" ]; then
  WIF_POOL="$(sed -n 's/^[[:space:]]*wif_pool_id[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$_TFVARS" | tail -1)"
fi
WIF_POOL="${WIF_POOL:-llmops-github-pool}"
: "${PROJECT_ID:?set PROJECT_ID or run: gcloud config set project <id>}"

AUTO_APPROVE=false
[ "${1:-}" = "--yes" ] && AUTO_APPROVE=true

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
gone() { printf '  \033[0;32mdeleted\033[0m %s\n' "$1"; }
skip() { printf '  \033[0;90mabsent \033[0m %s\n' "$1"; }

cat <<PLAN

  This will PERMANENTLY DELETE from project ${PROJECT_ID}:

    - Cloud Run service        ${SERVICE} (${REGION})
    - Artifact Registry repo   ${SERVICE} and every image in it
    - GCS bucket               gs://${BUCKET} and every object in it
    - Secrets                  ${SERVICE}-openai-api-key, ${SERVICE}-app-api-key
    - Service accounts         ${SERVICE}-run, ${SERVICE}-deployer
    - Workload identity pool   ${WIF_POOL}

  Kept (free, and annoying to recreate):
    - Enabled APIs
    - The project itself

PLAN

if [ "$AUTO_APPROVE" = false ]; then
  read -r -p "  Type 'delete' to confirm: " REPLY
  if [ "$REPLY" != "delete" ]; then
    echo "  Aborted. Nothing was changed."
    exit 0
  fi
fi

gcloud config set project "$PROJECT_ID" >/dev/null

say "Cloud Run"
if gcloud run services describe "$SERVICE" --region="$REGION" >/dev/null 2>&1; then
  gcloud run services delete "$SERVICE" --region="$REGION" --quiet
  gone "cloud run service ${SERVICE}"
else
  skip "cloud run service ${SERVICE}"
fi

say "Artifact Registry"
if gcloud artifacts repositories describe "$SERVICE" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories delete "$SERVICE" --location="$REGION" --quiet
  gone "artifact registry repo ${SERVICE}"
else
  skip "artifact registry repo ${SERVICE}"
fi

say "Cloud Storage"
if gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
  # --recursive removes the objects too; a non-empty bucket cannot be deleted.
  gcloud storage rm --recursive "gs://${BUCKET}" --quiet
  gone "bucket gs://${BUCKET}"
else
  skip "bucket gs://${BUCKET}"
fi

say "Secrets"
for secret in "${SERVICE}-openai-api-key" "${SERVICE}-app-api-key"; do
  if gcloud secrets describe "$secret" >/dev/null 2>&1; then
    gcloud secrets delete "$secret" --quiet
    gone "secret ${secret}"
  else
    skip "secret ${secret}"
  fi
done

say "Workload identity federation"
if gcloud iam workload-identity-pools describe ${WIF_POOL} --location=global >/dev/null 2>&1; then
  # Pools are soft-deleted and the id stays reserved for 30 days.
  gcloud iam workload-identity-pools delete ${WIF_POOL} --location=global --quiet
  gone "workload identity pool ${WIF_POOL}"
else
  skip "workload identity pool ${WIF_POOL}"
fi

say "Service accounts"
for sa in "${SERVICE}-run" "${SERVICE}-deployer"; do
  EMAIL="${sa}@${PROJECT_ID}.iam.gserviceaccount.com"
  if gcloud iam service-accounts describe "$EMAIL" >/dev/null 2>&1; then
    gcloud iam service-accounts delete "$EMAIL" --quiet
    gone "service account ${EMAIL}"
  else
    skip "service account ${EMAIL}"
  fi
done

say "Verifying nothing billable is left"
PROJECT_ID="$PROJECT_ID" REGION="$REGION" "$(dirname "$0")/cost_check.sh" || true

cat <<'DONE'

  Teardown complete.

  Still worth doing manually:
    - Revoke the OpenAI API key at https://platform.openai.com/api-keys
      (it is billed by OpenAI and is NOT covered by GCP credits)
    - Delete the GitHub repository secrets if you are done with the project
    - Consider deleting the whole GCP project for a guaranteed zero bill:
        gcloud projects delete PROJECT_ID

DONE
