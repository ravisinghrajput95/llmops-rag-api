# ---------------------------------------------------------------------------
# Billing budget alert.
#
# A budget does NOT cap spending -- Google will not stop your resources when
# you hit it. It emails you. That is still the single highest-value thing to
# set up when you are running on a fixed pot of trial credits, because it turns
# a silent drain into a notification you can act on.
#
# Requires the Cloud Billing Budget API and billing account permissions:
#   gcloud services enable billingbudgets.googleapis.com
# Skipped entirely when billing_account_id is empty.
# ---------------------------------------------------------------------------

resource "google_billing_budget" "trial_guard" {
  count = var.billing_account_id == "" ? 0 : 1

  billing_account = var.billing_account_id
  display_name    = "${var.service_name} trial credit guard"

  budget_filter {
    projects = ["projects/${data.google_project.current.number}"]
    # Only alert on real charges, not on credit-covered usage.
    credit_types_treatment = "EXCLUDE_ALL_CREDITS"
  }

  amount {
    specified_amount {
      currency_code = "INR"
      units         = tostring(var.budget_amount_inr)
    }
  }

  # Warn early, not just when the money is already gone.
  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }
  # Forecast-based: fires when Google predicts you WILL exceed the budget.
  threshold_rules {
    threshold_percent = 1.0
    spend_basis       = "FORECASTED_SPEND"
  }

  dynamic "all_updates_rule" {
    for_each = var.budget_alert_email == "" ? [] : [1]
    content {
      monitoring_notification_channels = [
        google_monitoring_notification_channel.budget_email[0].id,
      ]
      disable_default_iam_recipients = false
    }
  }
}

resource "google_monitoring_notification_channel" "budget_email" {
  count = var.budget_alert_email == "" || var.billing_account_id == "" ? 0 : 1

  project      = var.project_id
  display_name = "Budget alerts for ${var.service_name}"
  type         = "email"

  labels = {
    email_address = var.budget_alert_email
  }
}

data "google_project" "current" {
  project_id = var.project_id
}
