# Workload Identity Federation

Workload Identity Federation lets an external system authenticate to Google
Cloud without a long-lived service account key. GitHub Actions mints a
short-lived OIDC token, Google exchanges it for a credential valid for a few
minutes, and no private key is ever stored in the CI system.

A pool holds providers; a provider describes one external issuer. An attribute
condition restricts which external identities may exchange a token, and it is
the security boundary that matters most. Without one, any repository on the
issuer could impersonate the service account.

Pool identifiers are unique per project and cannot be reused for thirty days
after deletion, because deletion is soft. Attempting to recreate a pool with a
recently deleted identifier fails until the reservation expires.

The service account being impersonated must grant the Workload Identity User
role to the federated principal. Missing that binding is the usual cause of a
token exchange that authenticates but is then refused.
