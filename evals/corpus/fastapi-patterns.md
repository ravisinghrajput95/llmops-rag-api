# Service structure with FastAPI

Dependency injection supplies collaborators to a route rather than having the
route construct them. Constructing expensive clients once at startup and
injecting them avoids rebuilding a connection on every request.

Lifespan handlers own startup and shutdown. Work that must happen before the
first request, such as opening a vector store, belongs there rather than at
module import time.

An exception handler converts a domain error into an appropriate status code in
one place. Mapping an upstream provider failure to 502 rather than 500 tells
the caller the service itself is healthy and its dependency is not.

Response models declare the shape of what an endpoint returns, which both
validates the response and generates accurate API documentation. Returning a
bare dictionary skips both.
