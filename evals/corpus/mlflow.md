# Experiment tracking

MLflow records one run for every ingest and every query. Each run carries the
model name, the retrieval depth, measured latency split between retrieval and
generation, prompt and completion token counts, and an estimated cost in both
US dollars and Indian rupees.

The tracking backend is a SQLite database. Artifacts, which include the
question, the retrieved context and the final answer, are written to Cloud
Storage so they outlive any single container instance.

Cost is estimated from published list prices rather than measured from an
invoice, so a run records an approximation of what a call cost and not a
billed amount.
