# Service observability

Structured logging is the foundation: fields rather than prose, so queries can
filter on values instead of matching substrings. Message text should stay
constant while the varying detail moves into fields.

A request identifier bound at the edge and attached to every subsequent log line
turns scattered entries into a coherent trace. Binding it in middleware rather
than passing it explicitly keeps the plumbing out of business logic.

Latency should be recorded per stage rather than only end to end. A request that
takes two seconds is diagnosed very differently depending on whether retrieval
or generation consumed the time.

Health and readiness serve different purposes. Health reports that the process
is alive; readiness reports that its dependencies are usable. Conflating them
means a service with a broken dependency continues receiving traffic.
