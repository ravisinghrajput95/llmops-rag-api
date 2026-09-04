#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create a billing budget alert.
#
# A budget does NOT stop spending -- Google keeps your resources running past
# it. What it does is email you at 50%, 90% and 100%, which converts a silent
# drain into something you can act on. Set this up on day one.
#
# include-all-credits means the budget tracks GROSS usage, including what your
# trial credits absorb. That is the early warning you want on a fixed pot of
# expiring credits: you hear about the drawdown while it is happening.
# Switching to exclude-all-credits would instead track only out-of-pocket
# spend, which stays at zero until the credits are exhausted -- by which point
# the money is already gone.
#
#   PROJECT_ID=my-project BUDGET_INR=200 ./scripts/set_budget.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
BUDGET_INR="${BUDGET_INR:-200}"
: "${PROJECT_ID:?set PROJECT_ID}"

gcloud services enable billingbudgets.googleapis.com --project="$PROJECT_ID" --quiet

BILLING_ACCOUNT=$(gcloud billing projects describe "$PROJECT_ID" \
                  --format='value(billingAccountName)' 2>/dev/null || true)
if [ -z "$BILLING_ACCOUNT" ]; then
  echo "Could not read the billing account for ${PROJECT_ID}." >&2
  echo "Check permissions, or list accounts with: gcloud billing accounts list" >&2
  exit 1
fi
ACCOUNT_ID="${BILLING_ACCOUNT##*/}"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo "Creating a ₹${BUDGET_INR} budget on billing account ${ACCOUNT_ID}"

gcloud billing budgets create \
  --billing-account="$ACCOUNT_ID" \
  --display-name="llmops-trial-guard-${PROJECT_ID}" \
  --budget-amount="${BUDGET_INR}INR" \
  --filter-projects="projects/${PROJECT_NUMBER}" \
  --credit-types-treatment=include-all-credits \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0 \
  --threshold-rule=percent=1.0,basis=forecasted-spend \
  --quiet

cat <<DONE

  Budget created. Alerts go to the billing account admins by default; add more
  recipients in the console:
    https://console.cloud.google.com/billing/${ACCOUNT_ID}/budgets

  Remember: a budget alerts, it does not cap. To actually stop spending, run
  ./scripts/teardown.sh

DONE
