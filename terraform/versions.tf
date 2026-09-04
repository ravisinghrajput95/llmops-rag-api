terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.12"
    }
  }

  # Local state by default: a GCS state bucket is one more thing to remember
  # to delete before the credits expire. Uncomment for team use.
  # backend "gcs" {
  #   bucket = "my-tfstate-bucket"
  #   prefix = "llmops-rag-api"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
