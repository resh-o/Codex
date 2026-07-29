"""Hybrid retrieval: vector + keyword, fused with RRF.

Orchestration only -- the ranking maths lives in :mod:`fusion` (pure) and the
SQL lives in the repository. This module's jobs are:

1. run the two retrievers **concurrently** (they are independent; running them
   in series would just add the slower one's latency to the faster one's),
2. **overfetch** from each before fusing, and
3. stitch the fused ids back into result objects carrying both retrievers'
   diagnostics.

On overfetching: fusing two lists that were each already truncated to ``top_k``
throws away exactly the signal RRF exists to exploit. A chunk ranked #1 by
keyword and #12 by vector is precisely the case hybrid should rescue, but with
``top_k = 10`` and no overfetch the vector list stops at #10 and the chunk looks
keyword-only. We fetch ``top_k * OVERFETCH_FACTOR`` from each instead.

Concurrency note: the storage layer is synchronous psycopg, so each retriever
is pushed to a worker thread with ``asyncio.to_thread`` and awaited together via
``asyncio.gather``. Both calls are I/O-bound and release the GIL while waiting
on the socket, so this is real overlap, not just interleaving.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Protocol, Sequence

from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .keyword_search import KeywordSearchService
from .results import SearchResult
from .vector_search import VectorSearchService

logger = logging.getLogger(__name__)

#: Multiplier applied to ``top_k`` when fetching candidates from each retriever
#: before fusion. 3 is a deliberately unremarkable default -- big enough that
#: fusion has a real candidate pool, small enough not to make either query
#: meaningfully slower. Tuning it is a post-Stage-5 job.
DEFAULT_OVERFETCH_FACTOR = 3

VALID_MODES = ("hybrid", "vector", "keyword")


class Retriever(Protocol):
    """What hybrid search needs from either half (see the two services)."""

    def search(
        self, query: str, top_k: int = 10, repo_url: Optional[str] = None
    ) -> list[SearchResult]: ...


class HybridSearchService:
    """Runs vector + keyword search concurrently and fuses them with RRF."""

    def __init__(
        self,
        vector: Retriever,
        keyword: Retriever,
        overfetch_factor: int = DEFAULT_OVERFETCH_FACTOR,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if overfetch_factor < 1:
            raise ValueError("overfetch_factor must be >= 1")
        self._vector = vector
        self._keyword = keyword
        self._overfetch_factor = overfetch_factor
        self._rrf_k = rrf_k

    async def search(
        self,
        query: str,
        top_k: int = 10,
        repo_url: Optional[str] = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Retrieve ``top_k`` results using ``mode``.

        ``mode="vector"`` and ``mode="keyword"`` bypass fusion entirely and
        return that retriever's own ranking -- kept as first-class options
        because Stage 5 needs to score hybrid *against* each of them.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if mode not in VALID_MODES:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {', '.join(VALID_MODES)}"
            )
        top_k = max(1, min(top_k, 100))

        if mode == "vector":
            return await asyncio.to_thread(
                self._vector.search, query, top_k=top_k, repo_url=repo_url
            )
        if mode == "keyword":
            return await asyncio.to_thread(
                self._keyword.search, query, top_k=top_k, repo_url=repo_url
            )

        candidates = top_k * self._overfetch_factor
        vector_results, keyword_results = await asyncio.gather(
            asyncio.to_thread(
                self._vector.search, query, top_k=candidates, repo_url=repo_url
            ),
            asyncio.to_thread(
                self._keyword.search, query, top_k=candidates, repo_url=repo_url
            ),
        )
        return self._fuse(vector_results, keyword_results)[:top_k]

    def search_sync(
        self,
        query: str,
        top_k: int = 10,
        repo_url: Optional[str] = None,
        mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Blocking wrapper, for callers outside an event loop (tests, scripts)."""
        return asyncio.run(
            self.search(query, top_k=top_k, repo_url=repo_url, mode=mode)
        )

    def _fuse(
        self,
        vector_results: Sequence[SearchResult],
        keyword_results: Sequence[SearchResult],
    ) -> list[SearchResult]:
        fused = reciprocal_rank_fusion(
            {"vector": vector_results, "keyword": keyword_results},
            k=self._rrf_k,
        )
        by_id_vector = {r.id: r for r in vector_results}
        by_id_keyword = {r.id: r for r in keyword_results}

        results: list[SearchResult] = []
        for item in fused:
            # Either copy carries identical chunk fields; prefer the vector one
            # arbitrarily, and merge in whichever scores each side supplied.
            source = by_id_vector.get(item.id) or by_id_keyword.get(item.id)
            if source is None:  # pragma: no cover - ids come from these lists
                continue
            vector_hit = by_id_vector.get(item.id)
            keyword_hit = by_id_keyword.get(item.id)
            results.append(
                SearchResult(
                    id=source.id,
                    repo_url=source.repo_url,
                    file_path=source.file_path,
                    language=source.language,
                    chunk_type=source.chunk_type,
                    name=source.name,
                    qualified_name=source.qualified_name,
                    start_line=source.start_line,
                    end_line=source.end_line,
                    snippet=source.snippet,
                    score=item.score,
                    similarity=vector_hit.similarity if vector_hit else None,
                    keyword_score=(
                        keyword_hit.keyword_score if keyword_hit else None
                    ),
                    vector_rank=item.ranks.get("vector"),
                    keyword_rank=item.ranks.get("keyword"),
                    fused_score=item.score,
                )
            )
        return results


def build_hybrid_service(
    vector: VectorSearchService,
    keyword: KeywordSearchService,
    overfetch_factor: int = DEFAULT_OVERFETCH_FACTOR,
    rrf_k: int = DEFAULT_RRF_K,
) -> HybridSearchService:
    """Small convenience constructor, mirroring how main.py wires things up."""
    return HybridSearchService(
        vector=vector,
        keyword=keyword,
        overfetch_factor=overfetch_factor,
        rrf_k=rrf_k,
    )
