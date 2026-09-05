# Secret Manager

A secret is a container; the sensitive value lives in a version beneath it.
Versions are immutable. Rotating a secret means adding a new version and
directing consumers at it, not editing the existing one.

Referencing a secret version as "latest" means a new version takes effect on
the next deployment without a configuration change. Pinning to a numbered
version is more predictable but requires a deployment to roll forward.

The free allowance covers six active secret versions and ten thousand access
operations per month. Access operations are counted per read, so a service
that reads a secret on every request rather than caching it at startup can
exhaust the allowance quickly.

Granting access requires the Secret Accessor role on the specific secret. Broad
project-level grants are the usual mistake; the role should be bound to the one
secret a service actually needs.
