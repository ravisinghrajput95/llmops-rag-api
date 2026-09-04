# ---------------------------------------------------------------------------
# Core infrastructure.
#
# Cost posture of everything created here:
#   Cloud Run              Always Free (2M req/mo) provided min_instances = 0
#   Artifact Registry      Always Free up to 0.5 GB, then billed  <- credits
#   Cloud Storage          Always Free up to 5 GB in US regions   <- then credits
#   Secret Manager         Always Free up to 6 versions
#   Cloud Logging          Always Free up to 50 GiB/project/month
#   Service accounts, IAM  Always free
# ---------------------------------------------------------------------------

locals {
  # Bucket names are globally unique; the project id makes collisions unlikely.
  mlflow_bucket_name = "${var.project_id}-mlflow-artifacts"

  common_labels = {
    app        = var.service_name
    managed-by = "terraform"
    cost-tier  = "free-tier-demo"
  }
}

# --- APIs ------------------------------------------------------------------
# Enabling an API costs nothing; only usage does.
resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "logging.googleapis.com",
  ])

  project = var.project_id
  service = each.value

  # Leave APIs enabled on destroy: disabling can break other things in the
  # project, and an enabled-but-unused API is free.
  disable_on_destroy = false
}

# --- Artifact Registry -----------------------------------------------------
resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = var.service_name
  description   = "Container images for ${var.service_name}"
  format        = "DOCKER"
  labels        = local.common_labels

  # Without this, every CI run leaves another image behind and storage creeps
  # past the free 0.5 GB into billed territory.
  cleanup_policies {
    id     = "keep-recent-releases"
    action = "KEEP"
    most_recent_versions {
      keep_count = var.artifact_retention_count
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  depends_on = [google_project_service.required]
}

# --- MLflow artifact bucket ------------------------------------------------
resource "google_storage_bucket" "mlflow_artifacts" {
  project  = var.project_id
  name     = local.mlflow_bucket_name
  location = upper(var.region)
  labels   = local.common_labels

  # STANDARD in a US region is what the 5 GB Always Free allowance covers.
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true

  # Lets `terraform destroy` remove the bucket even with objects in it --
  # important when you are racing a credit expiry deadline.
  force_destroy = true

  lifecycle_rule {
    condition {
      age = var.mlflow_artifact_retention_days
    }
    action {
      type = "Delete"
    }
  }

  # Versioning would silently multiply stored bytes against the free 5 GB.
  versioning {
    enabled = false
  }

  depends_on = [google_project_service.required]
}

# --- Runtime service account ----------------------------------------------
# The Cloud Run service runs as this identity rather than the over-privileged
# default compute service account.
resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "${var.service_name}-run"
  display_name = "Runtime identity for ${var.service_name}"
  depends_on   = [google_project_service.required]
}

resource "google_storage_bucket_iam_member" "runtime_artifacts" {
  bucket = google_storage_bucket.mlflow_artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_project_iam_member" "runtime_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# --- Secrets ---------------------------------------------------------------
resource "google_secret_manager_secret" "openai_api_key" {
  project   = var.project_id
  secret_id = "${var.service_name}-openai-api-key"
  labels    = local.common_labels

  replication {
    # Single region instead of automatic: cheaper and sufficient here.
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "openai_api_key" {
  secret      = google_secret_manager_secret.openai_api_key.id
  secret_data = var.openai_api_key
}

resource "google_secret_manager_secret_iam_member" "runtime_openai" {
  secret_id = google_secret_manager_secret.openai_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_secret_manager_secret" "app_api_key" {
  count = var.app_api_key == "" ? 0 : 1

  project   = var.project_id
  secret_id = "${var.service_name}-app-api-key"
  labels    = local.common_labels

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "app_api_key" {
  count = var.app_api_key == "" ? 0 : 1

  secret      = google_secret_manager_secret.app_api_key[0].id
  secret_data = var.app_api_key
}

resource "google_secret_manager_secret_iam_member" "runtime_app_key" {
  count = var.app_api_key == "" ? 0 : 1

  secret_id = google_secret_manager_secret.app_api_key[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

# --- Cloud Run -------------------------------------------------------------
resource "google_cloud_run_v2_service" "api" {
  project  = var.project_id
  name     = var.service_name
  location = var.region
  labels   = local.common_labels

  # Must be false or `terraform destroy` refuses to delete the service, which
  # is the last thing you want when clearing resources before credits expire.
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email
    timeout         = "${var.request_timeout_seconds}s"

    scaling {
      min_instance_count = var.min_instances # 0 => no idle charges
      max_instance_count = var.max_instances # hard cost ceiling
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
        # Bill CPU only while a request is in flight.
        cpu_idle = true
        # Free burst during cold start; it does not extend billed time.
        startup_cpu_boost = true
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "SERVICE_NAME"
        value = var.service_name
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }
      # /tmp is the only writable path on Cloud Run (in-memory, per-instance).
      env {
        name  = "CHROMA_DIR"
        value = "/tmp/chroma"
      }
      env {
        name  = "MLFLOW_TRACKING_URI"
        value = "sqlite:////tmp/mlflow.db"
      }
      env {
        name  = "MLFLOW_ARTIFACT_LOCATION"
        value = "gs://${google_storage_bucket.mlflow_artifacts.name}/mlflow"
      }
      env {
        name  = "MLFLOW_EXPERIMENT"
        value = "llmops-rag-demo"
      }

      env {
        name = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_api_key.secret_id
            version = "latest"
          }
        }
      }

      dynamic "env" {
        for_each = var.app_api_key == "" ? [] : [1]
        content {
          name = "APP_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.app_api_key[0].secret_id
              version = "latest"
            }
          }
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 6
        timeout_seconds       = 3
      }
    }
  }

  lifecycle {
    # GitHub Actions owns the deployed image after the first apply; without
    # this, every `terraform apply` would roll the service back.
    ignore_changes = [
      template[0].containers[0].image,
      client,
      client_version,
    ]
  }

  depends_on = [
    google_project_service.required,
    google_secret_manager_secret_iam_member.runtime_openai,
  ]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count = var.allow_unauthenticated ? 1 : 0

  project  = var.project_id
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
