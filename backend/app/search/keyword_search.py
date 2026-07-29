"""Keyword (Postgres full-text) search.

The lexical half of hybrid retrieval. Deliberately mirrors
:class:`~app.search.vector_search.VectorSearchService`: same constructor shape,
same ``search()`` signature, same :class:`SearchResult` output -- so fusion can
treat the two uniformly and either can be swapped out independently.

Unlike vector search this needs no embedding call, so keyword-only mode works
with no Gemini credentials at all.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..storage.repository import KeywordHit, search_keyword as _search_keyword
from .results import DEFAULT_SNIPPET_CHARS, SearchResult, snippet

# The repository search function, injectable for tests.
KeywordSearchFn = Callable[..., list[KeywordHit]]


class KeywordSearchService:
    """tsquery -> ranked chunks -> formatted results."""

    def __init__(
        self,
        search_fn: KeywordSearchFn = _search_keyword,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
    ) -> None:
        self._search_fn = search_fn
        self._snippet_chars = snippet_chars

    def search(
        self, query: str, top_k: int = 10, repo_url: Optional[str] = None
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        top_k = max(1, min(top_k, 100))

        hits = self._search_fn(query, top_k=top_k, repo_url=repo_url)
        return [self._to_result(hit) for hit in hits]

    def _to_result(self, hit: KeywordHit) -> SearchResult:
        return SearchResult(
            id=hit.id,
            repo_url=hit.repo_url,
            file_path=hit.file_path,
            language=hit.language,
            chunk_type=hit.chunk_type,
            name=hit.name,
            qualified_name=hit.qualified_name,
            start_line=hit.start_line,
            end_line=hit.end_line,
            snippet=snippet(hit.content, self._snippet_chars),
            score=hit.rank,
            keyword_score=hit.rank,
        )
