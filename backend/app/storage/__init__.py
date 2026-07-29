"""Storage: Postgres/pgvector connection + chunk repository."""

from __future__ import annotations

from .db import StorageError, apply_schema, close_pool, get_pool
from .repository import (
    ChunkRow,
    KeywordHit,
    SearchHit,
    UpsertResult,
    count_chunks,
    get_existing_keys,
    search_keyword,
    search_similar,
    upsert_chunks,
)

__all__ = [
    "StorageError",
    "get_pool",
    "close_pool",
    "apply_schema",
    "ChunkRow",
    "SearchHit",
    "KeywordHit",
    "UpsertResult",
    "upsert_chunks",
    "get_existing_keys",
    "count_chunks",
    "search_similar",
    "search_keyword",
]
