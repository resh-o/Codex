"""Embeddings: Gemini client + chunk batching."""

from __future__ import annotations

from .embedder import (
    EmbeddedChunk,
    Embedder,
    EmbeddingStats,
    InputBuilder,
    content_input,
    rich_input,
)
from .gemini_client import (
    TASK_DOCUMENT,
    TASK_QUERY,
    EmbeddingError,
    GeminiClient,
    TransientEmbeddingError,
)

__all__ = [
    "GeminiClient",
    "EmbeddingError",
    "TransientEmbeddingError",
    "TASK_DOCUMENT",
    "TASK_QUERY",
    "Embedder",
    "EmbeddedChunk",
    "EmbeddingStats",
    "InputBuilder",
    "content_input",
    "rich_input",
]
