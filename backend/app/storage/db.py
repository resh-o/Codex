"""Database connection handling.

A single lazily-created psycopg connection pool, shared process-wide, with the
pgvector type adapter registered on every pooled connection so ``vector`` values
round-trip as Python lists/numpy arrays.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ..config import ConfigError, Settings, get_settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class StorageError(RuntimeError):
    """Raised for database connectivity / operational failures."""


_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def _configure_connection(conn: psycopg.Connection) -> None:
    """Register the pgvector adapter on a freshly-opened pooled connection."""
    register_vector(conn)


def get_pool(settings: Optional[Settings] = None) -> ConnectionPool:
    """Return the shared connection pool, creating it on first use.

    Raises
    ------
    ConfigError
        If ``DATABASE_URL`` is not configured.
    StorageError
        If the pool cannot connect to the database.
    """
    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool
        settings = settings or get_settings()
        dsn = settings.require_database_url()
        try:
            pool = ConnectionPool(
                conninfo=dsn,
                min_size=settings.db_pool_min,
                max_size=settings.db_pool_max,
                configure=_configure_connection,
                open=True,
                timeout=10.0,
            )
            # Fail fast if the DB is unreachable / vector ext missing.
            pool.wait(timeout=10.0)
        except ConfigError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Could not connect to the database: {exc}") from exc
        _pool = pool
        logger.info("Database connection pool initialized")
        return _pool


def close_pool() -> None:
    """Close and drop the shared pool (used on shutdown / in tests)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def apply_schema(settings: Optional[Settings] = None) -> None:
    """Run schema.sql against the configured database (idempotent)."""
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    pool = get_pool(settings)
    try:
        with pool.connection() as conn:
            conn.execute(sql)
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        raise StorageError(f"Failed to apply schema: {exc}") from exc
