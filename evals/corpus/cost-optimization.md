# Controlling spend in a serverless deployment

Scaling to zero is the single largest lever. A service with a minimum instance
count of zero costs nothing while idle; raising the minimum to one bills roughly
720 instance-hours a month whether or not anyone sends a request.

A maximum instance count bounds the worst case. It converts an unbounded bill
under a traffic spike into a bounded one with degraded availability, which is
the correct trade for a demonstration service.

Right-sizing memory and CPU matters more than it appears, because serverless
billing multiplies allocated resources by time. Halving allocated memory halves
the memory component of every request that runs.

Provider bills are separate. Removing cloud infrastructure stops the cloud
provider's charges and none of a third-party language model provider's, which
must be bounded independently through a spend ceiling in the application.
