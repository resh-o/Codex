"""Batch chunks into embeddings.

Turns Stage-1 :class:`Chunk` objects into ``(Chunk, vector)`` pairs by batching
requests through :class:`GeminiClient`.  The text that actually gets embedded is
produced by a pluggable *input builder* so we can later experiment with richer
representations (e.g. ``docstring + qualified_name + content``) without touching
the batching/retry machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from ..config import ConfigError
from ..ingestion.models import Chunk
from .gemini_client import TASK_DOCUMENT, TASK_QUERY, GeminiClient

logger = logging.getLogger(__name__)

# An input builder maps a Chunk to the string that should be embedded.
InputBuilder = Callable[[Chunk], str]


def content_input(chunk: Chunk) -> str:
    """Default: embed the raw source content of the chunk."""
    return chunk.content


def rich_input(chunk: Chunk) -> str:
    """Alternate representation (not used by default; here to show the seam).

    Prepends the qualified name and docstring so lexical signal from the symbol
    name and docs contributes to the embedding. Wire this in later to A/B it.
    """
    parts = [chunk.qualified_name]
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.content)
    return "\n\n".join(parts)


@dataclass
class EmbeddingStats:
    """Basic run stats for observability."""

    chunks_total: int = 0
    chunks_embedded: int = 0
    batches: int = 0
    failures: int = 0
    # Gemini's embed_content response does not currently expose token usage;
    # tracked here (approximate, chars/4) so the field exists for later.
    approx_tokens: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class EmbeddedChunk:
    """A chunk paired with its embedding vector."""

    chunk: Chunk
    embedding: list[float]


class Embedder:
    """Batches chunks through a :class:`GeminiClient`."""

    def __init__(
        self,
        client: GeminiClient,
        batch_size: int = 100,
        input_builder: InputBuilder = content_input,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._client = client
        self._batch_size = batch_size
        self._input_builder = input_builder

    def embed_chunks(
        self, chunks: Sequence[Chunk]
    ) -> tuple[list[EmbeddedChunk], EmbeddingStats]:
        """Embed all ``chunks``; returns pairs plus run stats.

        A batch that fails after retries is recorded in ``stats`` and skipped so
        one bad batch doesn't lose the whole run.
        """
        stats = EmbeddingStats(chunks_total=len(chunks))
        results: list[EmbeddedChunk] = []

        for batch in _batched(chunks, self._batch_size):
            texts = [self._input_builder(c) for c in batch]
            stats.batches += 1
            stats.approx_tokens += sum(len(t) for t in texts) // 4
            try:
                vectors = self._client.embed(texts, task_type=TASK_DOCUMENT)
            except ConfigError:
                # Missing/invalid config (e.g. no API key) fails every batch --
                # surface it immediately rather than reporting N "failures".
                raise
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                stats.failures += len(batch)
                stats.errors.append(str(exc))
                logger.error("Failed to embed a batch of %d chunks: %s", len(batch), exc)
                continue
            for chunk, vector in zip(batch, vectors):
                results.append(EmbeddedChunk(chunk=chunk, embedding=vector))
            stats.chunks_embedded += len(batch)

        return results, stats

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query (uses the QUERY task type)."""
        vectors = self._client.embed([query], task_type=TASK_QUERY)
        return vectors[0]


def _batched(items: Sequence[Chunk], size: int) -> Iterable[Sequence[Chunk]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
