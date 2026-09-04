#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end check against a running instance: health -> ingest -> query.
#
# Costs a few tenths of a paisa in OpenAI tokens (nothing on GCP).
#
#   ./scripts/smoke_test.sh                          # against localhost:8080
#   ./scripts/smoke_test.sh https://my-svc.run.app   # against Cloud Run
#   API_KEY=secret ./scripts/smoke_test.sh <url>     # when APP_API_KEY is set
# ---------------------------------------------------------------------------
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"
AUTH_HEADER=()
[ -n "${API_KEY:-}" ] && AUTH_HEADER=(-H "X-API-Key: ${API_KEY}")

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
fail() { printf '\033[0;31mFAILED: %s\033[0m\n' "$1"; exit 1; }

command -v jq >/dev/null || fail "jq is required (brew install jq)"

say "1/4  GET /health"
curl -fsS "${BASE_URL}/health" | jq . || fail "health check"

say "2/4  POST /ingest"
curl -fsS -X POST "${BASE_URL}/ingest" \
  -H 'Content-Type: application/json' "${AUTH_HEADER[@]}" \
  -d '{
    "documents": [{
      "doc_id": "cloud-run-notes",
      "text": "Cloud Run is a serverless container platform on Google Cloud. It scales to zero when idle, so an unused service costs nothing. The free tier includes two million requests per month. The filesystem is read-only except for /tmp, which is an in-memory tmpfs.",
      "metadata": {"source": "smoke-test"}
    }]
  }' | jq . || fail "ingest"

say "3/4  POST /query (grounded)"
RESPONSE=$(curl -fsS -X POST "${BASE_URL}/query" \
  -H 'Content-Type: application/json' "${AUTH_HEADER[@]}" \
  -d '{"question": "What happens to Cloud Run when it is idle?"}') || fail "query"
echo "$RESPONSE" | jq '{answer, model, latency_ms, estimated_cost_usd, estimated_cost_inr, sources: (.sources | length), mlflow_run_id}'

echo "$RESPONSE" | jq -e '.answer | length > 0' >/dev/null || fail "empty answer"
echo "$RESPONSE" | jq -e '.sources | length > 0' >/dev/null || fail "no sources retrieved"

say "4/4  POST /query (ungrounded -- must not call the LLM)"
curl -fsS -X POST "${BASE_URL}/query" \
  -H 'Content-Type: application/json' "${AUTH_HEADER[@]}" \
  -d '{"question": "What is the airspeed velocity of an unladen swallow?"}' \
  | jq '{answer, cost: .estimated_cost_usd, prompt_tokens: .usage.prompt_tokens}'

printf '\n\033[0;32mAll smoke tests passed.\033[0m\n\n'
