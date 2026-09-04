output "service_url" {
  description = "Public URL of the Cloud Run service."
  value       = google_cloud_run_v2_service.api.uri
}

output "artifact_registry_repo" {
  description = "Docker repo path for image pushes."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "mlflow_artifact_bucket" {
  description = "GCS bucket holding MLflow artifacts."
  value       = "gs://${google_storage_bucket.mlflow_artifacts.name}"
}

output "runtime_service_account" {
  description = "Identity the Cloud Run service runs as."
  value       = google_service_account.runtime.email
}

# --- Values to paste into GitHub repository secrets ------------------------
output "github_secret_gcp_workload_identity_provider" {
  description = "GitHub secret: GCP_WORKLOAD_IDENTITY_PROVIDER"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "github_secret_gcp_service_account" {
  description = "GitHub secret: GCP_SERVICE_ACCOUNT"
  value       = google_service_account.deployer.email
}

output "github_secrets_summary" {
  description = "Everything CI needs, ready to copy into GitHub > Settings > Secrets."
  value       = <<-EOT

    Add these as GitHub repository secrets (Settings > Secrets and variables > Actions):

      GCP_PROJECT_ID                   = ${var.project_id}
      GCP_REGION                       = ${var.region}
      GCP_SERVICE_NAME                 = ${var.service_name}
      GCP_WORKLOAD_IDENTITY_PROVIDER   = ${google_iam_workload_identity_pool_provider.github.name}
      GCP_SERVICE_ACCOUNT              = ${google_service_account.deployer.email}
      GCP_RUNTIME_SERVICE_ACCOUNT      = ${google_service_account.runtime.email}

    Service URL: ${google_cloud_run_v2_service.api.uri}
  EOT
}
