variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = <<-EOT
    Deployment region. Keep this at us-central1, us-east1 or us-west1: the
    Cloud Storage "Always Free" 5 GB allowance only applies in those three US
    regions. Any other region bills storage from the first byte.
  EOT
  type        = string
  default     = "us-central1"

  validation {
    condition     = contains(["us-central1", "us-east1", "us-west1"], var.region)
    error_message = "Use us-central1, us-east1 or us-west1 to stay inside the Always Free storage tier."
  }
}

variable "service_name" {
  description = "Cloud Run service name (also used for the Artifact Registry repo)."
  type        = string
  default     = "llmops-rag-api"
}

variable "github_repo" {
  description = "GitHub repository allowed to deploy, as 'owner/repo'."
  type        = string
}

variable "openai_api_key" {
  description = <<-EOT
    OpenAI API key, stored in Secret Manager. Pass it via the environment
    rather than a .tfvars file: TF_VAR_openai_api_key=sk-...
    Terraform state contains this value, so keep state out of git.
  EOT
  type        = string
  sensitive   = true
}

variable "app_api_key" {
  description = <<-EOT
    Shared secret required by /ingest and /query. Leave empty to run the
    service open to the internet -- only do that if you enjoy strangers
    spending your OpenAI balance.
  EOT
  type        = string
  sensitive   = true
  default     = ""
}

variable "container_image" {
  description = <<-EOT
    Image to deploy. Defaults to a public placeholder so `terraform apply`
    succeeds before the first CI build exists; GitHub Actions replaces it on
    every deploy. Terraform ignores later image changes (see lifecycle block).
  EOT
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

# --- Cost guardrails -------------------------------------------------------
variable "max_instances" {
  description = <<-EOT
    Hard ceiling on concurrent Cloud Run instances. This is the single most
    important cost control here: it caps the worst case if the endpoint is
    scraped or hit by a loop. 2 is plenty for a demo.
  EOT
  type        = number
  default     = 2
}

variable "min_instances" {
  description = <<-EOT
    MUST stay 0. Anything above zero keeps a container warm and bills CPU and
    memory 24/7, which would drain trial credits while you sleep.
  EOT
  type        = number
  default     = 0

  validation {
    condition     = var.min_instances == 0
    error_message = "min_instances must be 0 to avoid idle billing. Change this only deliberately."
  }
}

variable "cpu_limit" {
  description = "vCPU per instance."
  type        = string
  default     = "1"
}

variable "memory_limit" {
  description = <<-EOT
    Memory per instance. Chroma holds its index in memory and /tmp is a tmpfs
    that also counts against this, so 512Mi is the practical floor.
  EOT
  type        = string
  default     = "512Mi"
}

variable "request_timeout_seconds" {
  description = "Request timeout. Lower means a hung OpenAI call cannot bill for minutes."
  type        = number
  default     = 60
}

variable "allow_unauthenticated" {
  description = <<-EOT
    Make the service publicly reachable. Convenient for a demo; pair it with
    app_api_key so the endpoints that cost money still require a secret.
  EOT
  type        = bool
  default     = true
}

variable "artifact_retention_count" {
  description = "How many recent image versions to keep in Artifact Registry."
  type        = number
  default     = 3
}

variable "mlflow_artifact_retention_days" {
  description = "Delete MLflow artifacts older than this to stay under the 5 GB free allowance."
  type        = number
  default     = 30
}

# --- Budget alert ----------------------------------------------------------
variable "billing_account_id" {
  description = <<-EOT
    Billing account id (e.g. 01ABCD-234567-89EFGH) used to create a budget
    alert. Leave empty to skip; you can create the same budget in the console.
    Find it with: gcloud billing accounts list
  EOT
  type        = string
  default     = ""
}

variable "budget_amount_inr" {
  description = "Budget threshold in INR. Alerts fire at 50%, 90% and 100%."
  type        = number
  default     = 200
}

variable "budget_alert_email" {
  description = "Email for budget alerts. Defaults to the billing account admins."
  type        = string
  default     = ""
}
