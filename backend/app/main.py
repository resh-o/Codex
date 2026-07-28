"""FastAPI app exposing the Stage-1 ingestion pipeline.

Stage 1 does not persist anything -- ``POST /ingest`` runs the
clone -> walk -> chunk pipeline in-process and returns a JSON summary so the
pipeline can be exercised end-to-end over HTTP.  Persistence (pgvector) arrives
in Stage 2.
"""

from __future__ import annotations

import logging
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ingestion import Chunk, CloneError, ingest_repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Codex Ingestion API",
    version="0.1.0",
    description="Stage 1: repo ingestion + AST chunking.",
)

SAMPLE_SIZE = 5


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Clone a repo, chunk it, and return a summary of what was produced."""
    try:
        result = ingest_repo(request.repo_url)
    except CloneError as exc:
        # Clone failures are user/input errors, not server faults.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected server fault
        logger.exception("Ingestion failed for %s", request.repo_url)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    chunks = result.chunks
    by_type = Counter(c.chunk_type for c in chunks)
    by_language = Counter(c.language for c in chunks)

    # A readable, representative sample rather than the full dump: spread across
    # the produced chunk types where possible.
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
    # Top up with the earliest chunks if we still have room.
    for chunk in chunks:
        if len(diverse) >= SAMPLE_SIZE:
            break
        if chunk not in diverse:
            diverse.append(chunk)
    return diverse[:SAMPLE_SIZE]
