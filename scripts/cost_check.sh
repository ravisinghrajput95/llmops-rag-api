#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Audit everything in the project that could be drawing down credits.
#
# Read-only: this script never changes or deletes anything. Run it daily while
# the trial clock is ticking.
#
#   PROJECT_ID=my-project ./scripts/cost_check.sh
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
: "${PROJECT_ID:?set PROJECT_ID or run: gcloud config set project <id>}"

ok()   { printf '  \033[0;32m[ OK ]\033[0m %s\n' "$1"; }
warn() { printf '  \033[0;33m[WARN]\033[0m %s\n' "$1"; }
bad()  { printf '  \033[0;31m[COST]\033[0m %s\n' "$1"; }
head_() { printf '\n\033[1;34m== %s\033[0m\n' "$1"; }

echo "Cost audit for project: ${PROJECT_ID}"
FINDINGS=0

# --- Compute Engine --------------------------------------------------------
head_ "Compute Engine VMs (billed per second while RUNNING)"
VMS=$(gcloud compute instances list --project="$PROJECT_ID" \
      --format='value(name,zone,status,machineType)' 2>/dev/null || true)
if [ -z "$VMS" ]; then
  ok "No VM instances. This is the single biggest credit drain, so good."
else
  bad "VM instances found -- these bill 24/7:"
  echo "$VMS" | sed 's/^/         /'
  echo "         Delete with: gcloud compute instances delete NAME --zone=ZONE"
  FINDINGS=$((FINDINGS + 1))
fi

# --- Cloud Run -------------------------------------------------------------
head_ "Cloud Run services (free while min-instances = 0)"
SERVICES=$(gcloud run services list --project="$PROJECT_ID" --region="$REGION" \
           --format='value(metadata.name)' 2>/dev/null || true)
if [ -z "$SERVICES" ]; then
  ok "No Cloud Run services in ${REGION}."
else
  for svc in $SERVICES; do
    MIN=$(gcloud run services describe "$svc" --project="$PROJECT_ID" --region="$REGION" \
          --format='value(spec.template.metadata.annotations."autoscaling.knative.dev/minScale")' 2>/dev/null || true)
    MAX=$(gcloud run services describe "$svc" --project="$PROJECT_ID" --region="$REGION" \
          --format='value(spec.template.metadata.annotations."autoscaling.knative.dev/maxScale")' 2>/dev/null || true)
    MIN="${MIN:-0}"
    if [ "$MIN" = "0" ]; then
      ok "${svc}: min-instances=0 (no idle billing), max-instances=${MAX:-unset}"
    else
      bad "${svc}: min-instances=${MIN} -- BILLING 24/7 EVEN WITH NO TRAFFIC"
      echo "         Fix: gcloud run services update ${svc} --region=${REGION} --min-instances=0"
      FINDINGS=$((FINDINGS + 1))
    fi
    if [ -z "$MAX" ]; then
      warn "${svc}: no max-instances ceiling -- a traffic spike has no cost cap"
    fi
  done
fi

# --- GKE -------------------------------------------------------------------
head_ "GKE clusters (control plane billed hourly)"
CLUSTERS=$(gcloud container clusters list --project="$PROJECT_ID" \
           --format='value(name,location)' 2>/dev/null || true)
if [ -z "$CLUSTERS" ]; then
  ok "No GKE clusters."
else
  bad "GKE clusters found -- roughly \$0.10/hour each for the control plane:"
  echo "$CLUSTERS" | sed 's/^/         /'
  FINDINGS=$((FINDINGS + 1))
fi

# --- Cloud SQL -------------------------------------------------------------
head_ "Cloud SQL instances (billed hourly, no free tier)"
SQL=$(gcloud sql instances list --project="$PROJECT_ID" --format='value(name,tier)' 2>/dev/null || true)
if [ -z "$SQL" ]; then
  ok "No Cloud SQL instances."
else
  bad "Cloud SQL instances found -- these have NO free tier:"
  echo "$SQL" | sed 's/^/         /'
  FINDINGS=$((FINDINGS + 1))
fi

# --- Storage ---------------------------------------------------------------
head_ "Cloud Storage (Always Free: 5 GB in us-central1/us-east1/us-west1)"
BUCKETS=$(gcloud storage buckets list --project="$PROJECT_ID" --format='value(name)' 2>/dev/null || true)
if [ -z "$BUCKETS" ]; then
  ok "No buckets."
else
  for bucket in $BUCKETS; do
    SIZE=$(gcloud storage du "gs://${bucket}" --summarize --readable-sizes 2>/dev/null | awk '{print $1, $2}' || echo "unknown")
    LOC=$(gcloud storage buckets describe "gs://${bucket}" --format='value(location)' 2>/dev/null || echo "?")
    case "$LOC" in
      US-CENTRAL1|US-EAST1|US-WEST1)
        ok "${bucket}: ${SIZE} in ${LOC} (inside the free-tier region)" ;;
      *)
        warn "${bucket}: ${SIZE} in ${LOC} -- outside the Always Free storage regions" ;;
    esac
  done
fi

# --- Artifact Registry -----------------------------------------------------
head_ "Artifact Registry (Always Free up to 0.5 GB)"
REPOS=$(gcloud artifacts repositories list --project="$PROJECT_ID" \
        --format='value(name,sizeBytes)' 2>/dev/null || true)
if [ -z "$REPOS" ]; then
  ok "No repositories."
else
  echo "$REPOS" | while read -r name size; do
    if [ -n "${size:-}" ] && [ "$size" -gt 536870912 ] 2>/dev/null; then
      bad "$(basename "$name"): $((size / 1048576)) MiB -- over the free 0.5 GB"
    else
      ok "$(basename "$name"): $(( ${size:-0} / 1048576 )) MiB"
    fi
  done
fi

# --- Budget ----------------------------------------------------------------
head_ "Billing budget alerts"
BILLING=$(gcloud billing projects describe "$PROJECT_ID" \
          --format='value(billingAccountName)' 2>/dev/null || true)
if [ -z "$BILLING" ]; then
  warn "Could not read the billing account (needs billing.viewer)."
else
  ACCOUNT_ID="${BILLING##*/}"
  BUDGETS=$(gcloud billing budgets list --billing-account="$ACCOUNT_ID" \
            --format='value(displayName)' 2>/dev/null || true)
  if [ -z "$BUDGETS" ]; then
    bad "NO BUDGET ALERT CONFIGURED -- run ./scripts/set_budget.sh now"
    FINDINGS=$((FINDINGS + 1))
  else
    ok "Budgets: $(echo "$BUDGETS" | tr '\n' ' ')"
  fi
fi

# --- Credits ---------------------------------------------------------------
head_ "Remaining credits"
echo "  Not available from the CLI. Check:"
echo "    https://console.cloud.google.com/billing -> Credits"

head_ "Summary"
if [ "$FINDINGS" -eq 0 ]; then
  printf '  \033[0;32mNothing found that should meaningfully draw down credits.\033[0m\n\n'
else
  printf '  \033[0;31m%d issue(s) worth acting on -- see [COST] lines above.\033[0m\n\n' "$FINDINGS"
fi
