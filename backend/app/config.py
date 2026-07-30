"""Central configuration, read from environment variables.

Kept deliberately tiny and dependency-light: a single frozen settings object
built from ``os.environ`` so every layer reads the same values and tests can
override them by constructing their own :class:`Settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    # --- Gemini embeddings -------------------------------------------------
    gemini_api_key: str | None = None
    # Verified against Google's docs (July 2026): gemini-embedding-001 is the
    # current GA embedding model on the Gemini Developer API. Overridable so a
    # future model (e.g. gemini-embedding-2 on Vertex) can be swapped in.
    embedding_model: str = "gemini-embedding-001"
    # 768 is a Google-recommended MRL dimension: far cheaper to store/search
    # than the 3072 default while retaining most retrieval quality. MUST match
    # the vector(N) column in schema.sql.
    embedding_dim: int = 768
    # gemini-embedding-001 accepts a list of contents per request; keep batches
    # modest to stay within request/token limits and make retries cheap.
    embedding_batch_size: int = 100
    embedding_max_retries: int = 5
    embedding_base_delay: float = 1.0

    # --- Database ----------------------------------------------------------
    database_url: str | None = None
    db_pool_min: int = 1
    db_pool_max: int = 10

    # --- Hybrid retrieval (Stage 3) ----------------------------------------
    # How many candidates to pull from each retriever before RRF fusion,
    # as a multiple of top_k. Fusing two already-truncated top_k lists loses
    # the cross-list signal RRF exists to capture.
    search_overfetch_factor: int = 3
    # RRF damping constant; 60 is the value from the original RRF paper.
    rrf_k: int = 60

    # --- Answer generation (Stage 4) ---------------------------------------
    # Verified against Google's model list (July 2026): gemini-3.6-flash is the
    # current GA workhorse and supports schema-constrained JSON output, which is
    # what makes the citation requirement structural. Same API key as
    # embeddings, so answering adds a model rather than a vendor.
    answer_model: str = "gemini-3.6-flash"
    # Grounded extraction, not prose. Near-zero temperature keeps the model
    # copying the line ranges it was given instead of improvising them.
    answer_temperature: float = 0.0
    # Retrieval truncates chunk content to a snippet. 400 chars is right for a
    # search result list and far too short to answer *from* -- most function
    # bodies get cut mid-way -- so /ask retrieves with a much larger budget.
    answer_context_chars: int = 2400

    @classmethod
    def from_env(cls) -> "Settings":
        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw else default

        def _float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw else default

        return cls(
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            embedding_model=os.environ.get("GEMINI_EMBEDDING_MODEL", cls.embedding_model),
            embedding_dim=_int("EMBEDDING_DIM", cls.embedding_dim),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", cls.embedding_batch_size),
            embedding_max_retries=_int("EMBEDDING_MAX_RETRIES", cls.embedding_max_retries),
            database_url=os.environ.get("DATABASE_URL"),
            db_pool_min=_int("DB_POOL_MIN", cls.db_pool_min),
            db_pool_max=_int("DB_POOL_MAX", cls.db_pool_max),
            search_overfetch_factor=_int(
                "SEARCH_OVERFETCH_FACTOR", cls.search_overfetch_factor
            ),
            rrf_k=_int("RRF_K", cls.rrf_k),
            answer_model=os.environ.get("GEMINI_ANSWER_MODEL", cls.answer_model),
            answer_temperature=_float("ANSWER_TEMPERATURE", cls.answer_temperature),
            answer_context_chars=_int(
                "ANSWER_CONTEXT_CHARS", cls.answer_context_chars
            ),
        )

    def require_gemini(self) -> str:
        if not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Add it to your environment or .env "
                "(see .env.example)."
            )
        return self.gemini_api_key

    def require_database_url(self) -> str:
        if not self.database_url:
            raise ConfigError(
                "DATABASE_URL is not set. Point it at your Supabase/Postgres "
                "instance (see .env.example)."
            )
        return self.database_url


def get_settings() -> Settings:
    """Return settings built from the current environment."""
    return Settings.from_env()
