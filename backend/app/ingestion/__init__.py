"""Ingestion pipeline: clone -> walk -> chunk.

Public surface that later stages (embedding, retrieval) will import from.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .chunker import chunk_file, language_for_path
from .cloner import CloneError, CloneResult, clone_or_update
from .models import Chunk
from .walker import walk_repo

logger = logging.getLogger(__name__)

__all__ = [
    "Chunk",
    "CloneError",
    "CloneResult",
    "clone_or_update",
    "walk_repo",
    "chunk_file",
    "language_for_path",
    "ingest_repo",
    "IngestionResult",
]


class IngestionResult:
    """Aggregate output of an ingestion run."""

    __slots__ = ("repo_url", "commit_sha", "local_path", "files", "chunks")

    def __init__(
        self,
        repo_url: str,
        commit_sha: str,
        local_path: Path,
        files: list[str],
        chunks: list[Chunk],
    ) -> None:
        self.repo_url = repo_url
        self.commit_sha = commit_sha
        self.local_path = local_path
        self.files = files
        self.chunks = chunks


def ingest_repo(repo_url: str, workdir: Path | str | None = None) -> IngestionResult:
    """Run the full Stage-1 pipeline for a repo URL.

    Clones (or pulls) the repo, walks it for source files, and chunks each file.
    Files that fail to read or parse degrade to a fallback module chunk; a single
    unreadable file never aborts the run.
    """
    clone = clone_or_update(repo_url, workdir=workdir)
    files = walk_repo(clone.local_path)

    chunks: list[Chunk] = []
    for rel_path in files:
        abs_path = Path(clone.local_path) / rel_path
        try:
            source = abs_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("Skipping non-UTF-8 file: %s", rel_path)
            continue
        except OSError as exc:
            logger.warning("Could not read %s: %s", rel_path, exc)
            continue
        chunks.extend(
            chunk_file(
                rel_path,
                source,
                repo_url=clone.repo_url,
                commit_sha=clone.commit_sha,
            )
        )

    return IngestionResult(
        repo_url=clone.repo_url,
        commit_sha=clone.commit_sha,
        local_path=clone.local_path,
        files=files,
        chunks=chunks,
    )
