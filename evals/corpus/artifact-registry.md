# Artifact Registry

Artifact Registry stores container images. The free allowance is half a
gigabyte of storage per month, measured across all repositories in the project.

Images accumulate faster than most people expect, because every build produces
a new layer set even when little has changed. A cleanup policy is what keeps a
repository inside the allowance. Two rules cover the common case: delete
untagged images after a short retention window, and keep only the most recent
few tagged releases.

Repositories are regional. A repository in one region cannot be read by a
Cloud Run service in another without cross-region data transfer charges, so
the repository and the service it feeds should share a region.

Pulling an image is not billed within the same region. Storage is the cost
that accrues, which is why retention policy matters more than pull volume.
