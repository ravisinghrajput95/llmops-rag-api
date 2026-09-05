# Chunking strategy

A chunk is the unit that gets embedded and retrieved. Chunk size trades recall
against precision: large chunks retrieve more surrounding context but dilute
the embedding with unrelated text, while small chunks embed precisely but may
omit the sentence that makes an answer complete.

Splitting on paragraph boundaries preserves meaning better than splitting at a
fixed character count, because a paragraph is already a semantic unit. A
character limit is still needed as a fallback for paragraphs that exceed it.

Overlap repeats a portion of each chunk in the next one, so a fact spanning a
boundary appears whole in at least one chunk. Overlap costs storage and
embedding tokens proportional to the fraction repeated.

Chunk identifiers should be stable across re-ingestion of unchanged content.
Unstable identifiers cause duplicate entries in the vector store when the same
document is ingested twice.
