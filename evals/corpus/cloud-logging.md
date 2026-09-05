# Cloud Logging

Cloud Logging ingests structured JSON written to standard output. A log line
emitted as plain text is stored as an opaque string; the same line emitted as
JSON becomes queryable by field.

Severity is read from a top-level severity field. Without it every line is
recorded at default severity, which makes filtering for errors impossible.

Correlating the lines belonging to one request requires a trace identifier.
Cloud Run supplies one in the X-Cloud-Trace-Context header, and writing it to
the logging.googleapis.com/trace field groups those lines in the console.

The free allowance is fifty gibibytes of ingestion per project per month.
Debug-level logging in a busy service is the usual way that is exceeded, which
is why log level belongs in configuration rather than in code.
