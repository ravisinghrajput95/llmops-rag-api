# Embedding models

An embedding maps text to a fixed-length vector whose direction encodes
meaning. Two passages about the same topic point in similar directions
regardless of shared vocabulary, which is what separates semantic retrieval
from keyword search.

The text-embedding-3-small model produces 1,536-dimensional vectors and is
priced at two cents per million tokens. The large variant produces
3,072-dimensional vectors at thirteen cents per million tokens.

Embeddings from different models are not comparable. Changing the embedding
model requires re-embedding the entire corpus, because a query vector from one
model has no meaningful relationship to document vectors from another.

Embedding cost is dominated by ingestion rather than by querying. A query
embeds one short question; ingesting a corpus embeds every chunk of every
document.
