"""Flat (vector-only) similarity search.

Embeds a query with the same Gemini model used for documents, then delegates to
the repository's pgvector cosine search.  Deliberately thin -- hybrid/keyword
fusion is Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..embeddings.embedder import Embedder
from ..storage.repository import SearchHit, search_similar

DEFAULT_SNIPPET_CHARS = 400

# The repository search function, injectable for tests.
SearchFn = Callable[..., list[SearchHit]]


@dataclass
class SearchResult:
    """A single result formatted for API responses."""

    repo_url: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    similarity: float
    snippet: str


class VectorSearchService:
    """Embed-query -> cosine-search -> formatted results."""

    def __init__(
        self,
        embedder: Embedder,
        search_fn: SearchFn = search_similar,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    ) -> None:
        self._embedder = embedder
        self._search_fn = search_fn
        self._snippet_chars = snippet_chars

    def search(
        self, query: str, top_k: int = 10, repo_url: Optional[str] = None
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        top_k = max(1, min(top_k, 100))

        query_embedding = self._embedder.embed_query(query)
        hits = self._search_fn(query_embedding, top_k=top_k, repo_url=repo_url)
        return [self._to_result(hit) for hit in hits]

    def _to_result(self, hit: SearchHit) -> SearchResult:
        return SearchResult(
            repo_url=hit.repo_url,
            file_path=hit.file_path,
            language=hit.language,
            chunk_type=hit.chunk_type,
            name=hit.name,
            qualified_name=hit.qualified_name,
            start_line=hit.start_line,
            end_line=hit.end_line,
            similarity=hit.similarity,
            snippet=_snippet(hit.content, self._snippet_chars),
        )


def _snippet(content: str, max_chars: int) -> str:
    """Truncate content to a readable snippet on a line/word boundary."""
    content = content.strip()
    if len(content) <= max_chars:
        return content
    cut = content[:max_chars]
    # Prefer to end at the last newline, else the last space, for readability.
    boundary = max(cut.rfind("\n"), cut.rfind(" "))
    if boundary > max_chars // 2:
        cut = cut[:boundary]
    return cut.rstrip() + " …"


def cosine_rank(
    query: Sequence[float],
    candidates: Sequence[tuple[str, Sequence[float]]],
) -> list[tuple[str, float]]:
    """Reference cosine ranking (id, similarity), highest first.

    Not used by the production path (pgvector does this in SQL) but shared by
    tests as an independent oracle for expected ordering.
    """
    import numpy as np

    q = np.asarray(query, dtype=np.float64)
    qn = np.linalg.norm(q) or 1.0
    scored: list[tuple[str, float]] = []
    for cid, vec in candidates:
        v = np.asarray(vec, dtype=np.float64)
        vn = np.linalg.norm(v) or 1.0
        scored.append((cid, float(q @ v / (qn * vn))))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored
