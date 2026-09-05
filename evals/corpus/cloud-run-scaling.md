# Cloud Run scaling and concurrency

Each Cloud Run instance handles several requests at once. The concurrency
setting controls how many, and defaults to eighty. Lowering it to one gives
each request a dedicated instance, which suits CPU-bound work but multiplies
the instance count and therefore the bill.

Autoscaling reacts to concurrency utilisation. When existing instances are
saturated the service adds more, up to the maximum instance count. Setting a
maximum is what bounds the cost of a traffic spike; without one a runaway
client can scale the service far beyond what the budget tolerates.

A cold start is the delay incurred when a request arrives and no warm instance
exists. The container must be pulled, started, and pass its startup probe
before it serves traffic. Startup CPU boost allocates extra CPU during that
window to shorten it.

The startup probe determines when an instance is considered ready. A probe
that is too aggressive marks healthy instances as failed; one that is too
lenient sends traffic to instances that are not ready yet.
