# Cloud Run and the GCP Free Tier

Cloud Run is a serverless container platform on Google Cloud. You give it a
container image and it handles scaling, TLS and traffic routing. When no
requests arrive, it scales to zero instances and you are charged nothing for
idle time.

## Free tier

The Cloud Run free tier resets every month and does not expire with trial
credits. It includes 2 million requests, 360,000 GiB-seconds of memory and
180,000 vCPU-seconds. A low-traffic demo service stays comfortably inside it.

## Minimum instances

Setting minimum instances above zero keeps containers warm to avoid cold
starts, but a warm instance bills for CPU and memory continuously, even with no
traffic. For a cost-constrained demo, minimum instances must stay at zero.

## Filesystem

The Cloud Run filesystem is read-only except for /tmp, which is an in-memory
tmpfs. Anything written there counts against the instance memory limit and is
lost when the instance is recycled. Durable state belongs in Cloud Storage or a
database.

## Artifact Registry

Artifact Registry stores container images. The first 0.5 GB of storage per
month is free; beyond that it bills per GB. Old image revisions accumulate
silently, so a cleanup policy is worth configuring early.
