"""Thin wrappers over the OpenAI SDK.

Everything the rest of the app needs from OpenAI goes through these two
protocols, which is what makes the test suite able to run with zero network
access and zero spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from openai import OpenAI

from app.config import Settings


@lru_cache
def _client(api_key: str, base_url: str | None, timeout: float, retries: int) -> OpenAI:
    # Cached so we reuse one HTTP connection pool across requests; creating a
    # client per request adds TLS handshakes to every call.
    return OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=retries,
    )


def build_openai_client(settings: Settings) -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or pass it via --set-secrets on Cloud Run."
        )
    return _client(
        settings.openai_api_key,
        settings.openai_base_url,
        settings.openai_timeout_seconds,
        settings.openai_max_retries,
    )


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class EmbeddingResult:
    vectors: list[list[float]] = field(default_factory=list)
    model: str = ""
    tokens: int = 0


class ChatClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> ChatResult: ...


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingResult: ...


class OpenAIChatClient:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._model = settings.chat_model
        self._max_tokens = settings.max_output_tokens
        self._temperature = settings.temperature

    def complete(self, system_prompt: str, user_prompt: str) -> ChatResult:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        usage = response.usage
        return ChatResult(
            text=(response.choices[0].message.content or "").strip(),
            model=response.model or self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class OpenAIEmbeddingClient:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._model = settings.embedding_model

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], model=self._model, tokens=0)
        response = self._client.embeddings.create(model=self._model, input=texts)
        # The API guarantees ordering matches the input, but sort defensively:
        # a mis-ordered batch would silently attach wrong vectors to chunks.
        items = sorted(response.data, key=lambda d: d.index)
        return EmbeddingResult(
            vectors=[list(item.embedding) for item in items],
            model=response.model or self._model,
            tokens=getattr(response.usage, "total_tokens", 0) or 0,
        )
