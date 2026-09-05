# Continuous delivery with GitHub Actions

A concurrency group prevents overlapping runs of the same workflow. Cancelling
in-progress runs suits tests, where only the latest commit matters, but is
wrong for deployments, where interrupting a half-finished rollout leaves an
undefined state.

Tagging an image with the commit SHA makes every deployment traceable to a
commit and makes rollback a matter of redeploying a previous tag. A mutable tag
alone cannot express which build is currently running.

A smoke test after deployment verifies that the new revision serves traffic
correctly. Pairing it with an automatic rollback means a broken release is
withdrawn without human intervention.

Steps that continue on error still report their real result through the step
outcome, while the conclusion is reported as success. Conditioning on outcome is
what allows a later step to react to a tolerated failure.
