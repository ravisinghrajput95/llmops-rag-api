# Cloud Run cost behaviour

Cloud Run bills for the CPU and memory a container uses while it is handling
a request. When `min-instances` is set to zero the service scales down to no
running containers once traffic stops, and an idle service therefore costs
nothing at all.

The Always Free allowance covers two million requests per month, plus 360,000
GiB-seconds of memory and 180,000 vCPU-seconds. A demo service that handles a
few hundred requests a month stays far inside every one of those limits.

Setting `min-instances` to one keeps a container warm around the clock. That
removes cold starts but bills roughly 720 instance-hours a month, which is the
single most expensive misconfiguration available in this project.

Cloud Run runs x86_64 containers only. An image built on an Apple Silicon Mac
without an explicit platform flag will be arm64 and fails to start with an
exec format error.
