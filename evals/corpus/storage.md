# Storage and durability

The Cloud Run filesystem is read-only except for `/tmp`, and `/tmp` is an
in-memory tmpfs private to a single container instance. Anything written there
counts against the container memory limit and disappears when the instance
goes away.

Cloud Storage offers five gigabytes of standard storage free each month, but
only in the us-central1, us-east1 and us-west1 regions. Buckets created in any
other region are billed from the first byte.

This project snapshots its Chroma vector store into Cloud Storage after each
ingest and restores it when a new instance starts, which is what allows
ingested documents to survive a scale-to-zero event.
