"""Thin wrapper around the Gemini embedding API.

Responsibilities kept here:
* Build the google-genai client from config (lazily).
* Issue a single embedding request for a list of texts.
* Retry transient failures (429 / 5xx / network) with exponential backoff.
* L2-normalize outputs -- required for gemini-embedding-001 at non-3072 dims.

Batching across many chunks lives one layer up in :mod:`embedder` so this class
stays a simple, retrying request primitive that is easy to mock in tests.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Callable, Optional, Protocol, Sequence

from ..config import ConfigError, Settings

logger = logging.getLogger(__name__)

# Task types (see Google's embeddings docs). Documents and queries are embedded
# with different task types for asymmetric retrieval quality.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class EmbeddingError(RuntimeError):
    """A non-recoverable embedding failure."""


class TransientEmbeddingError(EmbeddingError):
    """A retryable failure (rate limit, timeout, temporary server error)."""


class EmbedFn(Protocol):
    """Low-level embedding callable. Injectable so tests avoid the real SDK."""

    def __call__(
        self, texts: Sequence[str], *, model: str, task_type: str, output_dim: int
    ) -> list[list[float]]:
        ...


# Substrings that mark an SDK/HTTP error as worth retrying.
_TRANSIENT_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "rate limit",
    "resource_exhausted",
    "unavailable",
    "deadline",
    "timeout",
    "temporarily",
    "connection",
)


def _looks_transient(exc: Exception) -> bool:
    """Best-effort classification of an exception as transient/retryable."""
    if isinstance(exc, TransientEmbeddingError):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


class GeminiClient:
    """Retrying, normalizing front-end to the Gemini embedding API."""

    def __init__(
        self,
        settings: Settings,
        embed_fn: Optional[EmbedFn] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        # Inject a low-level embed function for tests; otherwise build the real
        # SDK-backed one lazily on first use.
        self._embed_fn = embed_fn
        self._explicit_fn = embed_fn is not None

    # -- public API ---------------------------------------------------------

    @property
    def model(self) -> str:
        return self._settings.embedding_model

    @property
    def output_dim(self) -> int:
        return self._settings.embedding_dim

    def embed(self, texts: Sequence[str], *, task_type: str) -> list[list[float]]:
        """Embed a list of texts in one request, with retries + normalization.

        Returns one unit-normalized vector per input text, in order.
        """
        if not texts:
            return []
        fn = self._get_embed_fn()
        vectors = self._call_with_retry(fn, list(texts), task_type)
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Gemini returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return [_normalize(v) for v in vectors]

    # -- internals ----------------------------------------------------------

    def _get_embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            self._embed_fn = _build_sdk_embed_fn(self._settings)
        return self._embed_fn

    def _call_with_retry(
        self, fn: EmbedFn, texts: list[str], task_type: str
    ) -> list[list[float]]:
        max_retries = self._settings.embedding_max_retries
        base = self._settings.embedding_base_delay
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                return fn(
                    texts,
                    model=self._settings.embedding_model,
                    task_type=task_type,
                    output_dim=self._settings.embedding_dim,
                )
            except ConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below
                last_exc = exc
                if not _looks_transient(exc) or attempt == max_retries:
                    if _looks_transient(exc):
                        raise TransientEmbeddingError(
                            f"Embedding failed after {max_retries} retries: {exc}"
                        ) from exc
                    raise EmbeddingError(f"Embedding request failed: {exc}") from exc
                # Exponential backoff with full jitter.
                delay = min(base * (2 ** attempt), 30.0)
                delay = random.uniform(0, delay)
                logger.warning(
                    "Transient embedding error (attempt %d/%d): %s; retrying in %.2fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                self._sleep(delay)

        # Unreachable, but keeps type-checkers happy.
        raise EmbeddingError(f"Embedding request failed: {last_exc}")


def _normalize(vector: Sequence[float]) -> list[float]:
    """L2-normalize a vector (required for gemini-embedding-001 < 3072 dims)."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def _build_sdk_embed_fn(settings: Settings) -> EmbedFn:
    """Construct the real google-genai-backed embedding function."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ConfigError(
            "google-genai is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    api_key = settings.require_gemini()
    client = genai.Client(api_key=api_key)

    def _embed(
        texts: Sequence[str], *, model: str, task_type: str, output_dim: int
    ) -> list[list[float]]:
        response = client.models.embed_content(
            model=model,
            contents=list(texts),
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=output_dim,
            ),
        )
        return [list(e.values) for e in response.embeddings]

    return _embed
