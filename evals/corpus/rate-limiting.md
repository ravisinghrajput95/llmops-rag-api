# Rate limiting and quotas

A token bucket admits requests while tokens remain and refills at a fixed rate.
Burst size is the bucket capacity, which allows a short spike while bounding
sustained throughput.

The identity a limiter keys on determines what it actually protects. Keying on
client address is unreliable behind a proxy, where every request appears to
originate from the proxy and forwarded headers are caller-controlled. An
authenticated credential is a stronger key.

An in-process limiter holds state per instance, so the effective limit scales
with instance count and resets when an instance stops. A limit that must hold
exactly across a fleet requires shared state.

Returning a Retry-After header lets a well-behaved client back off correctly.
Without it, a client can only guess, and the usual guess is to retry
immediately.
