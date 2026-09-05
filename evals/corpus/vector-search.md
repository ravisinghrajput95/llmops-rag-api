# Vector search and similarity

Cosine similarity measures the angle between two vectors, ignoring their
magnitude. It is the appropriate metric for text embeddings, where direction
carries the meaning and length reflects little more than text length.

Cosine distance is one minus cosine similarity, giving a bounded range that
maps cleanly back onto a similarity score between zero and one.

A similarity floor discards retrieved chunks that score below a threshold. Its
purpose is twofold: fewer irrelevant chunks reach the prompt, which lowers
input token cost, and a query where every chunk falls below the floor can skip
the language model call entirely.

The correct floor is model-dependent and must be recalibrated whenever the
embedding model changes. Setting it too high causes answerable questions to be
refused; too low admits noise that encourages the model to answer from
irrelevant context.

HNSW is the index structure used for approximate nearest neighbour search. It
trades exact results for speed, which is the right trade at any meaningful
corpus size.
