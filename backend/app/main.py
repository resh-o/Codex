"""FastAPI app: Stages 1-4 (ingest, embed, search, answer).

Endpoints
---------
* ``POST /ingest`` — Stage 1: clone + chunk, return a summary (no persistence).
* ``POST /embed``  — Stage 2: ingest + embed + upsert into pgvector.
* ``POST /search`` — Stage 3: hybrid (vector + keyword, RRF-fused) search, with
  ``mode`` to fall back to either retriever on its own.
* ``POST /ask``    — Stage 4: retrieve, generate a cited answer, and validate
  every citation against the retrieved chunks before returning it.

Credential/connection problems (missing Gemini key, unreachable DB) surface as
clean HTTP errors rather than leaking raw tracebacks to the client.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answering import (
    AnswerGenerator,
    AnswerResult,
    AnswerService,
    GenerationError,
    NoContextError,
)
from .config import ConfigError, get_settings
from .embeddings import Embedder, EmbeddingError, GeminiClient
from .ingestion import Chunk, CloneError, ingest_repo
from .search import (
    HybridSearchService,
    KeywordSearchService,
    VectorSearchService,
)
from .storage import (
    StorageError,
    apply_schema,
    get_existing_keys,
    get_pool,
    upsert_chunks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Codex API",
    version="0.4.0",
    description=(
        "Stage 1 (AST chunking) + Stage 2 (embeddings) + Stage 3 (hybrid "
        "vector/keyword retrieval with RRF fusion) + Stage 4 (answers with "
        "mechanically-validated citations)."
    ),
)

SAMPLE_SIZE = 5


# --------------------------------------------------------------------------- #
# Lazily-built, process-wide singletons for the embedding + search stack.
# --------------------------------------------------------------------------- #

# Reentrant: _get_hybrid_service() builds the embedder via _get_embedder()
# while holding the lock, so the same thread must be able to re-acquire it.
_lock = threading.RLock()
_embedder: Embedder | None = None
_hybrid_service: HybridSearchService | None = None
_answer_search_service: HybridSearchService | None = None
_keyword_service: KeywordSearchService | None = None
_answer_service: AnswerService | None = None
_schema_ready = False


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                settings = get_settings()
                client = GeminiClient(settings)
                _embedder = Embedder(client, batch_size=settings.embedding_batch_size)
    return _embedder


def _get_keyword_service() -> KeywordSearchService:
    """Keyword search needs no embedder, so it is built independently.

    That is what lets ``mode=keyword`` work on a box with no Gemini key.
    """
    global _keyword_service
    if _keyword_service is None:
        with _lock:
            if _keyword_service is None:
                _keyword_service = KeywordSearchService()
    return _keyword_service


class _LazyVectorRetriever:
    """Defers building the embedder until a search actually needs a vector.

    Without this, constructing the hybrid service would demand a Gemini key up
    front and ``mode=keyword`` -- which never embeds anything -- would 503 on a
    box that only has a database. Errors from a missing key still surface, just
    at the point the vector half is genuinely used.
    """

    def __init__(self, snippet_chars: int | None = None) -> None:
        self._snippet_chars = snippet_chars

    def search(self, query: str, top_k: int = 10, repo_url: str | None = None):
        kwargs = {}
        if self._snippet_chars is not None:
            kwargs["snippet_chars"] = self._snippet_chars
        return VectorSearchService(_get_embedder(), **kwargs).search(
            query, top_k=top_k, repo_url=repo_url
        )


def _build_hybrid_service(snippet_chars: int | None = None) -> HybridSearchService:
    settings = get_settings()
    keyword = (
        _get_keyword_service()
        if snippet_chars is None
        else KeywordSearchService(snippet_chars=snippet_chars)
    )
    return HybridSearchService(
        vector=_LazyVectorRetriever(snippet_chars),
        keyword=keyword,
        overfetch_factor=settings.search_overfetch_factor,
        rrf_k=settings.rrf_k,
    )


def _get_hybrid_service() -> HybridSearchService:
    global _hybrid_service
    if _hybrid_service is None:
        with _lock:
            if _hybrid_service is None:
                _hybrid_service = _build_hybrid_service()
    return _hybrid_service


def _get_answer_search_service() -> HybridSearchService:
    """Same retrieval, bigger excerpts.

    /search returns snippets for a human to scan; /ask returns them to a model
    that must answer *from* them. The 400-char default cuts most functions in
    half, so the answering path retrieves with ``answer_context_chars`` instead.
    """
    global _answer_search_service
    if _answer_search_service is None:
        with _lock:
            if _answer_search_service is None:
                _answer_search_service = _build_hybrid_service(
                    get_settings().answer_context_chars
                )
    return _answer_search_service


def _get_answer_service() -> AnswerService:
    global _answer_service
    if _answer_service is None:
        with _lock:
            if _answer_service is None:
                settings = get_settings()
                _answer_service = AnswerService(
                    search=_get_answer_search_service(),
                    generator=AnswerGenerator(settings),
                )
    return _answer_service


def _ensure_storage_ready() -> None:
    """Open the pool and apply the schema once (idempotent)."""
    global _schema_ready
    get_pool()  # raises ConfigError/StorageError with a clean message
    if not _schema_ready:
        with _lock:
            if not _schema_ready:
                apply_schema()
                _schema_ready = True


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="A cloneable git URL (https or ssh).")


class ChunkSample(BaseModel):
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    parent_name: str | None
    start_line: int
    end_line: int
    docstring: str | None


class IngestResponse(BaseModel):
    repo_url: str
    commit_sha: str
    files_processed: int
    chunks_produced: int
    chunks_by_type: dict[str, int]
    chunks_by_language: dict[str, int]
    sample_chunks: list[ChunkSample]


class EmbedRequest(BaseModel):
    repo_url: str = Field(..., description="A cloneable git URL (https or ssh).")


class EmbedResponse(BaseModel):
    repo_url: str
    commit_sha: str
    files_processed: int
    chunks_total: int
    chunks_embedded: int
    chunks_skipped: int
    embedding_failures: int
    rows_inserted: int
    rows_updated: int
    embedding_model: str
    embedding_dim: int


SearchMode = Literal["hybrid", "vector", "keyword"]


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language or code query.")
    repo_url: str | None = Field(None, description="Optional: restrict to one repo.")
    top_k: int = Field(10, ge=1, le=100)
    mode: SearchMode = Field(
        "hybrid",
        description=(
            "'hybrid' (default) fuses vector + keyword with RRF. 'vector' and "
            "'keyword' run a single retriever, kept callable on their own so "
            "Stage 5 can score hybrid against each baseline."
        ),
    )


class SearchResultModel(BaseModel):
    id: str
    repo_url: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    snippet: str

    #: The score this result was ranked by in the requested mode.
    score: float

    # Diagnostics: populated per mode, null where inapplicable. Retained
    # deliberately -- the eventual UI won't show these, but Stage 5's eval and
    # any "why did this rank here?" debugging both need them.
    similarity: float | None = None
    keyword_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    repo_url: str | None
    mode: SearchMode
    count: int
    results: list[SearchResultModel]


class AskRequest(BaseModel):
    query: str = Field(..., description="A natural-language question about the code.")
    repo_url: str | None = Field(
        None,
        description=(
            "Optional: restrict retrieval to one repo. Left optional for "
            "consistency with /search; supply it whenever more than one "
            "repository has been embedded."
        ),
    )
    top_k: int = Field(10, ge=1, le=100, description="Chunks to retrieve and cite from.")


#: The response *is* AnswerResult: the answer plus the validation metadata
#: (attempts, per-claim verdicts, stripped claims, aggregate citation rates)
#: that Stage 5 will score on. Kept as one model so the eval harness and the API
#: consume the same shape rather than two that drift apart.
AskResponse = AnswerResult


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _sample(chunk: Chunk) -> ChunkSample:
    return ChunkSample(
        file_path=chunk.file_path,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        name=chunk.name,
        qualified_name=chunk.qualified_name,
        parent_name=chunk.parent_name,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        docstring=chunk.docstring,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Stage 1: clone a repo, chunk it, return a summary (no persistence)."""
    try:
        result = ingest_repo(request.repo_url)
    except CloneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Ingestion failed for %s", request.repo_url)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    chunks = result.chunks
    by_type = Counter(c.chunk_type for c in chunks)
    by_language = Counter(c.language for c in chunks)
    sample = _pick_sample(chunks)
    return IngestResponse(
        repo_url=result.repo_url,
        commit_sha=result.commit_sha,
        files_processed=len(result.files),
        chunks_produced=len(chunks),
        chunks_by_type=dict(sorted(by_type.items())),
        chunks_by_language=dict(sorted(by_language.items())),
        sample_chunks=[_sample(c) for c in sample],
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    """Stage 2: ingest, embed new chunks, and upsert them into pgvector."""
    settings = get_settings()
    try:
        embedder = _get_embedder()
        _ensure_storage_ready()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = ingest_repo(request.repo_url)
    except CloneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = result.chunks
    # Skip chunks already stored for this exact commit (idempotent re-runs).
    try:
        existing = get_existing_keys(result.repo_url, result.commit_sha)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    to_embed = [
        c
        for c in chunks
        if (c.file_path, c.start_line, c.end_line) not in existing
    ]

    try:
        embedded, stats = embedder.embed_chunks(to_embed)
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

    # If there was work to do and every batch failed, that's an error condition
    # (e.g. an invalid key or the API being down), not a 200 with a sad summary.
    if to_embed and stats.chunks_embedded == 0 and stats.failures > 0:
        detail = stats.errors[0] if stats.errors else "all embedding batches failed"
        raise HTTPException(status_code=502, detail=f"Embedding failed: {detail}")

    try:
        upserted = upsert_chunks(embedded)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return EmbedResponse(
        repo_url=result.repo_url,
        commit_sha=result.commit_sha,
        files_processed=len(result.files),
        chunks_total=len(chunks),
        chunks_embedded=stats.chunks_embedded,
        chunks_skipped=len(chunks) - len(to_embed),
        embedding_failures=stats.failures,
        rows_inserted=upserted.inserted,
        rows_updated=upserted.updated,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Stage 3: hybrid (RRF-fused vector + keyword) search over stored chunks.

    Async because hybrid mode fans the two retrievers out concurrently; the
    blocking psycopg work happens in worker threads.
    """
    try:
        service = _get_hybrid_service()
        _ensure_storage_ready()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        results = await service.search(
            request.query,
            top_k=request.top_k,
            repo_url=request.repo_url,
            mode=request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ConfigError as exc:
        # Deferred by _LazyVectorRetriever, so a missing key lands here rather
        # than at construction -- still a 503 (misconfigured), not a 502.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=f"Query embedding failed: {exc}") from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SearchResponse(
        query=request.query,
        repo_url=request.repo_url,
        mode=request.mode,
        count=len(results),
        results=[SearchResultModel(**vars(r)) for r in results],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AnswerResult:
    """Stage 4: answer a question, with every citation checked before it ships.

    Retrieval, generation, validation, one corrective retry, then stripping of
    anything still unsupported -- see :mod:`app.answering.answer_service`. The
    response carries the validation verdict alongside the answer, including any
    claims that were removed and why.
    """
    try:
        service = _get_answer_service()
        _ensure_storage_ready()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return await service.answer_question(
            request.query, repo_url=request.repo_url, top_k=request.top_k
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoContextError as exc:
        # Nothing retrieved: answering anyway would mean answering uncited, which
        # is the one thing this stage exists to prevent.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConfigError as exc:
        # Deferred: a missing Gemini key surfaces at the embed/generate call.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=502, detail=f"Query embedding failed: {exc}"
        ) from exc
    except GenerationError as exc:
        raise HTTPException(
            status_code=502, detail=f"Answer generation failed: {exc}"
        ) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _pick_sample(chunks: list[Chunk]) -> list[Chunk]:
    """Pick up to SAMPLE_SIZE chunks, favouring variety across chunk types."""
    if len(chunks) <= SAMPLE_SIZE:
        return chunks
    seen_types: set[str] = set()
    diverse: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_type not in seen_types:
            seen_types.add(chunk.chunk_type)
            diverse.append(chunk)
        if len(diverse) >= SAMPLE_SIZE:
            break
    for chunk in chunks:
        if len(diverse) >= SAMPLE_SIZE:
            break
        if chunk not in diverse:
            diverse.append(chunk)
    return diverse[:SAMPLE_SIZE]
