# ---------------------------------------------------------------------------
# Workload Identity Federation for GitHub Actions.
#
# This is what lets CI deploy without a long-lived JSON service account key in
# GitHub secrets. GitHub mints a short-lived OIDC token, Google exchanges it
# for an access token valid for minutes, and there is no static credential to
# leak or rotate. All of this is free.
# ---------------------------------------------------------------------------

resource "google_iam_workload_identity_pool" "github" {
  project                   = var.project_id
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions"
  description               = "OIDC federation for GitHub Actions deploys"

  depends_on = [google_project_service.required]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.actor"      = "assertion.actor"
    "attribute.ref"        = "assertion.ref"
  }

  # Without this condition Google would accept tokens from ANY GitHub repo on
  # the internet. It scopes the trust to your repository only, and Google now
  # rejects providers that omit it.
  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# --- Deployer service account ---------------------------------------------
resource "google_service_account" "deployer" {
  project      = var.project_id
  account_id   = "${var.service_name}-deployer"
  display_name = "GitHub Actions deployer for ${var.service_name}"
  depends_on   = [google_project_service.required]
}

# Only this repository may impersonate the deployer.
resource "google_service_account_iam_member" "github_impersonation" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

# Push images to this repo only, not project-wide registry admin.
resource "google_artifact_registry_repository_iam_member" "deployer_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.deployer.email}"
}

# Deploy revisions of this service only.
resource "google_cloud_run_v2_service_iam_member" "deployer_admin" {
  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.admin"
  member   = "serviceAccount:${google_service_account.deployer.email}"
}

# Required for the deployer to set the runtime service account on a revision.
resource "google_service_account_iam_member" "deployer_acts_as_runtime" {
  service_account_id = google_service_account.runtime.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}
