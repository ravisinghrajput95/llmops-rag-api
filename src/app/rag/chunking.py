"""Document splitting.

A paragraph-aware splitter: we pack whole paragraphs into a chunk until the
size budget is hit, and only fall back to hard character slicing for a single
paragraph that is itself oversized. Keeping paragraph boundaries intact gives
noticeably better retrieval than blind fixed-width slicing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"[ \t]+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: dict[str, str] = field(default_factory=dict)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size]]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = normalize(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    buffer = ""

    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if len(paragraph) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_hard_split(paragraph, chunk_size, chunk_overlap))
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            chunks.append(buffer)
            # Carry the tail of the previous chunk so a fact split across the
            # boundary is still retrievable from the following chunk.
            tail = buffer[-chunk_overlap:] if chunk_overlap else ""
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if buffer:
        chunks.append(buffer)
    return [c for c in (chunk.strip() for chunk in chunks) if c]


def chunk_document(
    text: str,
    doc_id: str | None = None,
    metadata: dict[str, str] | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    doc_id = doc_id or f"doc-{content_hash(text)}"
    metadata = metadata or {}
    pieces = split_text(text, chunk_size, chunk_overlap)
    return [
        Chunk(
            chunk_id=f"{doc_id}::{index}",
            doc_id=doc_id,
            text=piece,
            index=index,
            metadata={**metadata, "doc_id": doc_id, "chunk_index": str(index)},
        )
        for index, piece in enumerate(pieces)
    ]
