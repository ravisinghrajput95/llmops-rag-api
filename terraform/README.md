# Terraform

Provisions the whole stack with the cost guardrails encoded as code rather than
as README prose.

## Usage

```bash
cp terraform.tfvars.example terraform.tfvars   # set project_id + github_repo

# Keep secrets out of tfvars: state files are not a secret store.
export TF_VAR_openai_api_key="sk-..."
export TF_VAR_app_api_key="$(openssl rand -hex 24)"

terraform init
terraform plan            # always read this first
terraform apply
terraform output github_secrets_summary
```

## What gets created

| Resource | Cost tier |
|---|---|
| `google_artifact_registry_repository` | Always Free ≤ 0.5 GB (cleanup policy keeps 3 images) |
| `google_storage_bucket` | Always Free ≤ 5 GB in US regions (30-day lifecycle rule) |
| `google_cloud_run_v2_service` | Always Free ≤ 2M req/mo at `min_instances = 0` |
| `google_secret_manager_secret` ×2 | Always Free ≤ 6 versions |
| `google_service_account` ×2 | Free |
| Workload identity pool + provider | Free |
| `google_billing_budget` | Free (only created when `billing_account_id` is set) |

## Guardrails worth knowing about

- **`min_instances` has a `validation` block that rejects any value but 0.**
  Anything higher bills CPU and memory 24/7 with no traffic. Overriding it is a
  deliberate act, not a typo.
- **`region` is validated** to `us-central1` / `us-east1` / `us-west1`, the only
  regions where the 5 GB Cloud Storage allowance applies.
- **`deletion_protection = false`** on Cloud Run, and **`force_destroy = true`**
  on the bucket, so `terraform destroy` actually works when you are racing a
  credit expiry.
- **`ignore_changes` on the container image**, because GitHub Actions owns the
  deployed image after the first apply. Without it every `terraform apply` would
  roll the service back to the placeholder.

## Teardown

```bash
terraform destroy
```

`terraform destroy` only removes what Terraform created. If you also provisioned
by hand, `../scripts/teardown.sh` sweeps by name and then runs a cost audit.
