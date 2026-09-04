"""Chunking behaviour -- the part that decides how many tokens we pay for."""

from __future__ import annotations

import pytest

from app.rag.chunking import chunk_document, content_hash, normalize, split_text


def test_short_text_is_a_single_chunk():
    assert split_text("A short sentence.", chunk_size=800) == ["A short sentence."]


def test_blank_text_produces_no_chunks():
    assert split_text("   \n\n  \t ") == []


def test_every_chunk_respects_the_size_budget():
    text = "\n\n".join(f"Paragraph number {i} discussing serverless costs." for i in range(80))
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_oversized_paragraph_is_hard_split():
    chunks = split_text("x" * 1000, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_chunks_overlap_so_facts_spanning_a_boundary_stay_retrievable():
    text = "\n\n".join(
        f"Sentence {i} about retrieval augmented generation." for i in range(40)
    )
    chunks = split_text(text, chunk_size=200, chunk_overlap=60)

    # The tail of one chunk should reappear at the head of the next.
    assert any(chunks[i][-20:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=100, chunk_overlap=100)


def test_normalize_collapses_whitespace_but_keeps_paragraphs():
    assert normalize("a  \t b\r\n\r\nc") == "a b\n\nc"


def test_content_hash_is_stable_and_content_sensitive():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hello!")


def test_chunk_document_attaches_traceable_metadata():
    chunks = chunk_document("Some text.", doc_id="doc-1", metadata={"source": "readme"})

    assert chunks[0].chunk_id == "doc-1::0"
    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].metadata["source"] == "readme"
    assert chunks[0].metadata["chunk_index"] == "0"


def test_chunk_ids_are_unique_within_a_document():
    text = "\n\n".join(f"Paragraph {i}." for i in range(40))
    chunks = chunk_document(text, doc_id="doc-1", chunk_size=100, chunk_overlap=20)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
