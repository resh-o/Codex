"""Tests for the embedding layer: batching + retry, no real network calls."""

from __future__ import annotations

import math

import pytest

from app.config import Settings
from app.embeddings.embedder import Embedder, content_input, rich_input
from app.embeddings.gemini_client import (
    TASK_DOCUMENT,
    TASK_QUERY,
    EmbeddingError,
    GeminiClient,
    TransientEmbeddingError,
)
from app.ingestion.models import Chunk


def _settings(**kw) -> Settings:
    base = dict(
        gemini_api_key="test-key",
        embedding_dim=4,
        embedding_batch_size=2,
        embedding_max_retries=3,
        embedding_base_delay=0.0,
    )
    base.update(kw)
    return Settings(**base)


def _chunk(name: str, content: str) -> Chunk:
    return Chunk(
        repo_url="https://example.com/r",
        commit_sha="deadbeef",
        file_path=f"{name}.py",
        language="python",
        chunk_type="function",
        name=name,
        qualified_name=name,
        parent_name=None,
        start_line=1,
        end_line=2,
        content=content,
        docstring=None,
    )


class RecordingEmbedFn:
    """Fake low-level embed function: records calls, returns deterministic vecs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, *, model, task_type, output_dim):
        self.calls.append(list(texts))
        # Return a distinct constant vector per text (length == output_dim).
        return [[float(i + 1)] * output_dim for i, _ in enumerate(texts)]


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def test_embedder_batches_by_batch_size() -> None:
    fn = RecordingEmbedFn()
    client = GeminiClient(_settings(), embed_fn=fn, sleep=lambda _: None)
    embedder = Embedder(client, batch_size=2)

    chunks = [_chunk(f"c{i}", f"content {i}") for i in range(5)]
    embedded, stats = embedder.embed_chunks(chunks)

    # 5 chunks, batch size 2 -> batches of [2, 2, 1].
    assert [len(c) for c in fn.calls] == [2, 2, 1]
    assert stats.batches == 3
    assert stats.chunks_embedded == 5
    assert stats.failures == 0
    assert len(embedded) == 5
    # Order preserved and each chunk paired with its own vector.
    assert [e.chunk.name for e in embedded] == [f"c{i}" for i in range(5)]
    assert all(len(e.embedding) == 4 for e in embedded)


def test_embedder_uses_document_task_type() -> None:
    seen = {}

    def fn(texts, *, model, task_type, output_dim):
        seen["task"] = task_type
        return [[1.0] * output_dim for _ in texts]

    client = GeminiClient(_settings(), embed_fn=fn, sleep=lambda _: None)
    Embedder(client, batch_size=10).embed_chunks([_chunk("a", "x")])
    assert seen["task"] == TASK_DOCUMENT


def test_embed_query_uses_query_task_type_and_normalizes() -> None:
    def fn(texts, *, model, task_type, output_dim):
        assert task_type == TASK_QUERY
        return [[3.0, 4.0, 0.0, 0.0]]  # norm 5 -> normalized (0.6, 0.8, 0, 0)

    client = GeminiClient(_settings(), embed_fn=fn, sleep=lambda _: None)
    vec = Embedder(client).embed_query("how does auth work?")
    assert vec == pytest.approx([0.6, 0.8, 0.0, 0.0])
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-6)


def test_input_builders() -> None:
    c = _chunk("validate", "def validate(): pass")
    object.__setattr__(c, "qualified_name", "Auth.validate")
    object.__setattr__(c, "docstring", "Validate a token.")
    assert content_input(c) == "def validate(): pass"
    rich = rich_input(c)
    assert "Auth.validate" in rich and "Validate a token." in rich and "def validate" in rich


# --------------------------------------------------------------------------- #
# Retry / backoff
# --------------------------------------------------------------------------- #


class FlakyEmbedFn:
    """Fails with a transient error N times, then succeeds."""

    def __init__(self, fails: int, exc: Exception) -> None:
        self.remaining = fails
        self.exc = exc
        self.attempts = 0

    def __call__(self, texts, *, model, task_type, output_dim):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.exc
        return [[1.0] * output_dim for _ in texts]


def test_retry_recovers_from_transient_errors() -> None:
    sleeps: list[float] = []
    fn = FlakyEmbedFn(2, TransientEmbeddingError("503 service unavailable"))
    client = GeminiClient(_settings(embedding_max_retries=3), embed_fn=fn, sleep=sleeps.append)

    vectors = client.embed(["hello"], task_type=TASK_DOCUMENT)
    assert len(vectors) == 1
    assert fn.attempts == 3  # 2 failures + 1 success
    assert len(sleeps) == 2  # backoff slept before each retry


def test_retry_classifies_rate_limit_string_as_transient() -> None:
    fn = FlakyEmbedFn(1, RuntimeError("429 Too Many Requests: rate limit exceeded"))
    client = GeminiClient(_settings(), embed_fn=fn, sleep=lambda _: None)
    vectors = client.embed(["x"], task_type=TASK_DOCUMENT)
    assert len(vectors) == 1
    assert fn.attempts == 2


def test_retry_gives_up_after_max_retries() -> None:
    fn = FlakyEmbedFn(99, RuntimeError("503 unavailable"))
    client = GeminiClient(_settings(embedding_max_retries=2), embed_fn=fn, sleep=lambda _: None)
    with pytest.raises(TransientEmbeddingError):
        client.embed(["x"], task_type=TASK_DOCUMENT)
    assert fn.attempts == 3  # initial + 2 retries


def test_non_transient_error_is_not_retried() -> None:
    fn = FlakyEmbedFn(99, ValueError("invalid API key: 400 bad request"))
    client = GeminiClient(_settings(), embed_fn=fn, sleep=lambda _: None)
    with pytest.raises(EmbeddingError):
        client.embed(["x"], task_type=TASK_DOCUMENT)
    assert fn.attempts == 1  # no retries for a non-transient failure


def test_failed_batch_is_recorded_not_fatal() -> None:
    """One failing batch is skipped and reported; others still succeed."""

    class SecondBatchFails:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, texts, *, model, task_type, output_dim):
            self.n += 1
            if self.n == 2:
                raise ValueError("400 permanent failure")
            return [[1.0] * output_dim for _ in texts]

    client = GeminiClient(_settings(), embed_fn=SecondBatchFails(), sleep=lambda _: None)
    embedder = Embedder(client, batch_size=1)
    chunks = [_chunk(f"c{i}", f"x{i}") for i in range(3)]
    embedded, stats = embedder.embed_chunks(chunks)

    assert stats.chunks_embedded == 2
    assert stats.failures == 1
    assert len(stats.errors) == 1
    assert len(embedded) == 2


def test_mismatched_vector_count_raises() -> None:
    def fn(texts, *, model, task_type, output_dim):
        return [[1.0] * output_dim]  # only one vector for many inputs

    client = GeminiClient(_settings(), embed_fn=fn, sleep=lambda _: None)
    with pytest.raises(EmbeddingError):
        client.embed(["a", "b"], task_type=TASK_DOCUMENT)
