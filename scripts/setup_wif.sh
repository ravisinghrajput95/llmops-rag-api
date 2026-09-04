#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Workload Identity Federation for GitHub Actions -- keyless deploys.
#
# Why bother: the alternative is `gcloud iam service-accounts keys create`,
# which produces a JSON key that never expires and that you then paste into
# GitHub. If it leaks, anyone can deploy to (and bill) your project until you
# notice. WIF issues credentials that live for minutes and are scoped to one
# repository. It is also free.
#
# Usage:
#   export PROJECT_ID=my-project GITHUB_REPO=owner/repo
#   ./scripts/setup_wif.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
GITHUB_REPO="${GITHUB_REPO:?set GITHUB_REPO as owner/repo}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-llmops-rag-api}"
POOL="github-pool"
PROVIDER="github-provider"
DEPLOYER_SA="${SERVICE}-deployer"
DEPLOYER_EMAIL="${DEPLOYER_SA}@${PROJECT_ID}.iam.gserviceaccount.com"
RUNTIME_EMAIL="${SERVICE}-run@${PROJECT_ID}.iam.gserviceaccount.com"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

say "Creating the workload identity pool"
gcloud iam workload-identity-pools create "$POOL" \
  --location=global --display-name="GitHub Actions" \
  --quiet 2>/dev/null || echo "  already exists, skipping"

say "Creating the GitHub OIDC provider"
# attribute-condition is mandatory. Without it, ANY GitHub repository on the
# internet could exchange a token for access to this project.
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
  --location=global \
  --workload-identity-pool="$POOL" \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository == '${GITHUB_REPO}'" \
  --quiet 2>/dev/null || echo "  already exists, skipping"

say "Creating the deployer service account"
gcloud iam service-accounts create "$DEPLOYER_SA" \
  --display-name="GitHub Actions deployer for $SERVICE" \
  --quiet 2>/dev/null || echo "  already exists, skipping"

say "Allowing only ${GITHUB_REPO} to impersonate it"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" \
  --quiet >/dev/null

say "Granting the deployer the minimum it needs"
# Scoped to this repository and this service, not project-wide admin.
gcloud artifacts repositories add-iam-policy-binding "$SERVICE" \
  --location="$REGION" \
  --member="serviceAccount:${DEPLOYER_EMAIL}" \
  --role="roles/artifactregistry.writer" --quiet >/dev/null

gcloud run services add-iam-policy-binding "$SERVICE" \
  --region="$REGION" \
  --member="serviceAccount:${DEPLOYER_EMAIL}" \
  --role="roles/run.admin" --quiet >/dev/null

# Needed to attach the runtime service account to a new revision.
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_EMAIL" \
  --member="serviceAccount:${DEPLOYER_EMAIL}" \
  --role="roles/iam.serviceAccountUser" --quiet >/dev/null

PROVIDER_RESOURCE="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"

say "Add these GitHub repository secrets"
cat <<SECRETS

  Settings > Secrets and variables > Actions > New repository secret

    GCP_PROJECT_ID                 = ${PROJECT_ID}
    GCP_REGION                     = ${REGION}
    GCP_SERVICE_NAME               = ${SERVICE}
    GCP_WORKLOAD_IDENTITY_PROVIDER = ${PROVIDER_RESOURCE}
    GCP_SERVICE_ACCOUNT            = ${DEPLOYER_EMAIL}

  Or with the gh CLI:

    gh secret set GCP_PROJECT_ID --body "${PROJECT_ID}"
    gh secret set GCP_REGION --body "${REGION}"
    gh secret set GCP_SERVICE_NAME --body "${SERVICE}"
    gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body "${PROVIDER_RESOURCE}"
    gh secret set GCP_SERVICE_ACCOUNT --body "${DEPLOYER_EMAIL}"

SECRETS
