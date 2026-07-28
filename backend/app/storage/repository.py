"""Chunk persistence + vector similarity queries.

Raw SQL via psycopg (no ORM) so pgvector's ``<=>`` operator and upsert semantics
stay explicit and legible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from psycopg_pool import ConnectionPool

from ..embeddings.embedder import EmbeddedChunk
from .db import StorageError, get_pool

# Natural key used for upsert de-duplication (matches the UNIQUE constraint).
NATURAL_KEY = ("repo_url", "commit_sha", "file_path", "start_line", "end_line")

_INSERT_COLUMNS = (
    "repo_url",
    "commit_sha",
    "file_path",
    "language",
    "chunk_type",
    "name",
    "qualified_name",
    "parent_name",
    "start_line",
    "end_line",
    "content",
    "docstring",
    "embedding",
)

_UPSERT_SQL = f"""
INSERT INTO chunks ({", ".join(_INSERT_COLUMNS)})
VALUES ({", ".join(["%s"] * len(_INSERT_COLUMNS))})
ON CONFLICT (repo_url, commit_sha, file_path, start_line, end_line)
DO UPDATE SET
    language       = EXCLUDED.language,
    chunk_type     = EXCLUDED.chunk_type,
    name           = EXCLUDED.name,
    qualified_name = EXCLUDED.qualified_name,
    parent_name    = EXCLUDED.parent_name,
    content        = EXCLUDED.content,
    docstring      = EXCLUDED.docstring,
    embedding      = EXCLUDED.embedding
RETURNING (xmax = 0) AS inserted
"""

_SELECT_COLUMNS = (
    "id",
    "repo_url",
    "commit_sha",
    "file_path",
    "language",
    "chunk_type",
    "name",
    "qualified_name",
    "parent_name",
    "start_line",
    "end_line",
    "content",
    "docstring",
)


@dataclass
class UpsertResult:
    inserted: int
    updated: int

    @property
    def total(self) -> int:
        return self.inserted + self.updated


@dataclass
class SearchHit:
    """One ranked search result (subset of chunk fields + similarity)."""

    id: str
    repo_url: str
    commit_sha: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    parent_name: Optional[str]
    start_line: int
    end_line: int
    content: str
    docstring: Optional[str]
    similarity: float


def _vec(embedding: Sequence[float]) -> np.ndarray:
    return np.asarray(embedding, dtype=np.float32)


def upsert_chunks(
    embedded_chunks: Sequence[EmbeddedChunk],
    pool: Optional[ConnectionPool] = None,
) -> UpsertResult:
    """Bulk insert-or-update chunks. Returns inserted vs updated counts.

    Uses the ``(xmax = 0)`` trick to distinguish freshly-inserted rows from rows
    that already existed and were updated -- so callers can report how many were
    genuinely new.
    """
    if not embedded_chunks:
        return UpsertResult(inserted=0, updated=0)

    pool = pool or get_pool()
    params = [
        (
            ec.chunk.repo_url,
            ec.chunk.commit_sha,
            ec.chunk.file_path,
            ec.chunk.language,
            ec.chunk.chunk_type,
            ec.chunk.name,
            ec.chunk.qualified_name,
            ec.chunk.parent_name,
            ec.chunk.start_line,
            ec.chunk.end_line,
            ec.chunk.content,
            ec.chunk.docstring,
            _vec(ec.embedding),
        )
        for ec in embedded_chunks
    ]

    inserted = 0
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # executemany with returning=True lets us read one result set
                # per row and tally inserts vs updates.
                cur.executemany(_UPSERT_SQL, params, returning=True)
                while True:
                    row = cur.fetchone()
                    if row is not None and row[0]:
                        inserted += 1
                    if not cur.nextset():
                        break
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        raise StorageError(f"Failed to upsert chunks: {exc}") from exc

    return UpsertResult(inserted=inserted, updated=len(params) - inserted)


def get_existing_keys(
    repo_url: str,
    commit_sha: str,
    pool: Optional[ConnectionPool] = None,
) -> set[tuple[str, int, int]]:
    """Return existing ``(file_path, start_line, end_line)`` keys for a commit."""
    pool = pool or get_pool()
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT file_path, start_line, end_line FROM chunks "
                "WHERE repo_url = %s AND commit_sha = %s",
                (repo_url, commit_sha),
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        raise StorageError(f"Failed to read existing chunks: {exc}") from exc
    return {(r[0], r[1], r[2]) for r in rows}


def count_chunks(
    repo_url: Optional[str] = None,
    pool: Optional[ConnectionPool] = None,
) -> int:
    """Count stored chunks, optionally filtered to a repo."""
    pool = pool or get_pool()
    try:
        with pool.connection() as conn:
            if repo_url:
                row = conn.execute(
                    "SELECT count(*) FROM chunks WHERE repo_url = %s", (repo_url,)
                ).fetchone()
            else:
                row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    except Exception as exc:  # noqa: BLE001
        raise StorageError(f"Failed to count chunks: {exc}") from exc
    return int(row[0]) if row else 0


def search_similar(
    query_embedding: Sequence[float],
    top_k: int = 10,
    repo_url: Optional[str] = None,
    pool: Optional[ConnectionPool] = None,
) -> list[SearchHit]:
    """Flat cosine-similarity search over stored chunks.

    Orders by pgvector's cosine distance (``<=>``); similarity is reported as
    ``1 - distance`` (1.0 == identical direction).
    """
    if top_k < 1:
        return []
    pool = pool or get_pool()
    qvec = _vec(query_embedding)

    where = "WHERE repo_url = %s" if repo_url else ""
    sql = f"""
        SELECT {", ".join(_SELECT_COLUMNS)},
               1 - (embedding <=> %s) AS similarity
        FROM chunks
        {where}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    # Placeholder order: SELECT-list distance param, then optional repo filter,
    # then ORDER BY distance param, then the LIMIT.
    ordered_params: list[object] = [qvec]
    if repo_url:
        ordered_params.append(repo_url)
    ordered_params.append(qvec)
    ordered_params.append(top_k)

    try:
        with pool.connection() as conn:
            rows = conn.execute(sql, ordered_params).fetchall()
    except Exception as exc:  # noqa: BLE001
        raise StorageError(f"Vector search failed: {exc}") from exc

    hits: list[SearchHit] = []
    for row in rows:
        data = dict(zip(_SELECT_COLUMNS, row[: len(_SELECT_COLUMNS)]))
        hits.append(
            SearchHit(
                id=str(data["id"]),
                repo_url=data["repo_url"],
                commit_sha=data["commit_sha"],
                file_path=data["file_path"],
                language=data["language"],
                chunk_type=data["chunk_type"],
                name=data["name"],
                qualified_name=data["qualified_name"],
                parent_name=data["parent_name"],
                start_line=data["start_line"],
                end_line=data["end_line"],
                content=data["content"],
                docstring=data["docstring"],
                similarity=float(row[len(_SELECT_COLUMNS)]),
            )
        )
    return hits
